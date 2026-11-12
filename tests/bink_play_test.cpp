// bink_play_test: where a Bink movie is and when its frames are due
// (hle/bink_play.h), and the binkw32 calls that go by it.
//
// The rules, on their own:
// - BinkKeyFrame over an index of key frames, for every BINKGETKEY flag and
//   every frame of the movie, the ones before its first and after its last
//   included;
// - PlanBinkSeek: the frame a jump decodes from (the key frame before the
//   target, or the frame after the one the decoder holds if that is
//   closer; the target itself with BINKGOTOQUICK), and the frames whose
//   sound it queues, for movies with and without sound ahead of the picture;
// - BinkTooEarly, BinkWrapped: a frame due when first asked about, the next
//   a frame's time later; a movie far behind not caught up, and told so.
//
// Then BinkOpen, BinkGetKeyFrame, BinkGoto, BinkWait, BinkDoFrame,
// BinkNextFrame and BinkPause through their import thunks, in a 32-bit
// guest arena, on movies the test writes: 16 x 16 pixels, 30 frames a
// second, every tenth a key frame of one colour of its own, the frames
// between them copies of the frame before (so the picture tells which key
// frame it was built from), and a block of sound a frame, each block one
// level that names its frame (so the sound tells which frame it is from,
// and whether a block starts at its own level or blended from the block
// decoded before it). Some movies hold more blocks in their first frame, as
// Bink puts a movie's first 750 ms of sound there; some have a frame the
// decoder refuses or the file does not hold whole. The test's clock is the
// movie's; it is told of every frame the video decoder is done with and of
// the sound queued from every frame, and it plays the mixer itself to hear
// what was queued.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "runtime/audio/mixer.h"
#include "runtime/hle/bink_play.h"
#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/runtime.h"

using namespace recomp;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      g_failures++;                                                       \
    }                                                                     \
  } while (0)

using Bytes = std::vector<uint8_t>;
using Keys = std::vector<uint8_t>;

constexpr uint32_t kPrevious = hle::kBinkGetKeyPrevious, kNext = hle::kBinkGetKeyNext,
                   kClosest = hle::kBinkGetKeyClosest, kNotEqual = hle::kBinkGetKeyNotEqual;
constexpr uint32_t kQuick = hle::kBinkGotoQuick, kQuickSound = hle::kBinkGotoQuickSound;

// ---- the rules ----

void CheckKeyFrames() {
  // Ten frames; key frames 1, 4 and 8. The first frame is one whatever the
  // index says of it.
  const Keys keys = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0};
  struct Row {
    uint32_t flags;
    uint32_t want[12];  // for frames 0 to 11
  };
  const Row rows[] = {
      {kPrevious, {0, 1, 1, 1, 4, 4, 4, 4, 8, 8, 8, 8}},
      {kPrevious | kNotEqual, {0, 0, 1, 1, 1, 4, 4, 4, 4, 8, 8, 8}},
      {kNext, {1, 1, 4, 4, 4, 8, 8, 8, 8, 0, 0, 0}},
      {kNext | kNotEqual, {1, 4, 4, 4, 8, 8, 8, 8, 0, 0, 0, 0}},
      // As close on both sides (frame 6): the earlier one.
      {kClosest, {1, 1, 1, 4, 4, 4, 4, 8, 8, 8, 8, 8}},
      {kClosest | kNotEqual, {1, 4, 1, 4, 1, 4, 4, 8, 4, 8, 8, 8}},
  };
  for (const Row& row : rows) {
    for (uint32_t frame = 0; frame < 12; frame++) {
      const uint32_t got = hle::BinkKeyFrame(keys, frame, row.flags);
      if (got != row.want[frame])
        std::fprintf(stderr, "BinkKeyFrame(%u, %u) = %u, not %u\n", frame, row.flags, got,
                     row.want[frame]);
      CHECK(got == row.want[frame]);
    }
  }
  // A movie of one frame, and of none.
  CHECK(hle::BinkKeyFrame({0}, 1, kPrevious) == 1);
  CHECK(hle::BinkKeyFrame({0}, 1, kNext) == 1);
  CHECK(hle::BinkKeyFrame({0}, 1, kNext | kNotEqual) == 0);
  CHECK(hle::BinkKeyFrame({0}, 1, kClosest | kNotEqual) == 0);
  CHECK(hle::BinkKeyFrame({}, 1, kClosest) == 0);
  // Far past the end, to the largest number there is: none after it (not
  // frame 1, which the number after the largest would wrap to), the last
  // key frame before it.
  for (uint32_t frame : {11u, 1000u, 0xFFFFFFFEu, 0xFFFFFFFFu}) {
    CHECK(hle::BinkKeyFrame(keys, frame, kNext) == 0);
    CHECK(hle::BinkKeyFrame(keys, frame, kNext | kNotEqual) == 0);
    CHECK(hle::BinkKeyFrame(keys, frame, kPrevious) == 8);
    CHECK(hle::BinkKeyFrame(keys, frame, kPrevious | kNotEqual) == 8);
    CHECK(hle::BinkKeyFrame(keys, frame, kClosest) == 8);
    CHECK(hle::BinkKeyFrame(keys, frame, kClosest | kNotEqual) == 8);
  }
}

// A key frame every tenth frame of 60: 1, 11, ... 51.
Keys TenthKeys() {
  Keys keys(60, 0);
  for (size_t i = 0; i < keys.size(); i += 10) keys[i] = 1;
  return keys;
}

bool Same(const hle::BinkSeek& s, uint32_t key, uint32_t first, uint32_t audio_from,
          uint32_t drop, uint32_t silence) {
  if (s.key == key && s.first == first && s.audio_from == audio_from && s.drop == drop &&
      s.silence == silence)
    return true;
  std::fprintf(stderr, "seek: key %u first %u audio_from %u drop %u silence %u\n", s.key, s.first,
               s.audio_from, s.drop, s.silence);
  return false;
}

void CheckSeekPlans() {
  const Keys keys = TenthKeys();
  // The picture, for a movie whose sound is not ahead: from the key frame.
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, 0, 0), 41, 41, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 3, 47, 0, 0), 41, 41, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 60, 47, 0, 0), 41, 41, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 39, 47, 0, 0), 41, 41, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 47, 47, 0, 0), 41, 41, 47, 0, 0));  // decoded: again
  // On from the frame the decoder holds, if that is past the key frame.
  CHECK(Same(hle::PlanBinkSeek(keys, 40, 47, 0, 0), 41, 41, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 43, 47, 0, 0), 41, 44, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 46, 47, 0, 0), 41, 47, 47, 0, 0));
  // A key frame, the first frame, the last.
  CHECK(Same(hle::PlanBinkSeek(keys, 5, 41, 0, 0), 41, 41, 41, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 5, 1, 0, 0), 1, 1, 1, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 5, 60, 0, 0), 51, 51, 60, 0, 0));
  // BINKGOTOQUICK: straight to the frame.
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, kQuick, 0), 47, 47, 47, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 60, 47, kQuick, 0), 47, 47, 47, 0, 0));
  // Sound 22 frames ahead of the picture: from that many frames back,
  // whatever the key frame; near the start, from the first frame less
  // what it holds before the target's time.
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, 0, 22), 41, 41, 25, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, kQuick, 22), 47, 47, 25, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 24, 0, 22), 21, 21, 2, 0, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 23, 0, 22), 21, 21, 1, 22, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 5, 0, 22), 1, 1, 1, 4, 0));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 1, 0, 22), 1, 1, 1, 0, 0));
  // BINKGOTOQUICKSOUND: none of it, and silence for as long (the first
  // frame's own sound starts with the movie).
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, kQuickSound, 22), 41, 41, 47, 0, 22));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 47, kQuick | kQuickSound, 22), 47, 47, 47, 0, 22));
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 1, kQuickSound, 22), 1, 1, 1, 0, 0));
  // A decoder that holds no frame's picture (decoded: 0) starts from the
  // key frame, however near the target the movie is.
  CHECK(Same(hle::PlanBinkSeek(keys, 0, 48, 0, 0), 41, 41, 48, 0, 0));
  // A movie with no index at all.
  CHECK(Same(hle::PlanBinkSeek({}, 0, 1, 0, 0), 1, 1, 1, 0, 0));

  // From anywhere to anywhere: never from before the key frame, never past
  // the target, and so never more than the nine frames between two keys.
  uint32_t most = 0, from_key = 0, onward = 0;
  for (uint32_t decoded = 0; decoded <= 60; decoded++) {
    for (uint32_t target = 1; target <= 60; target++) {
      const hle::BinkSeek s = hle::PlanBinkSeek(keys, decoded, target, 0, 0);
      const uint32_t key = (target - 1) / 10 * 10 + 1;
      CHECK(s.key == key);
      CHECK(s.first == key || s.first == decoded + 1);
      CHECK(s.first >= key && s.first <= target);
      most = std::max(most, target - s.first);
      (s.first == key ? from_key : onward)++;
    }
  }
  CHECK(most == 9);
  CHECK(from_key > 0 && onward > 0);  // both kinds were met
}

void CheckPace() {
  const double fps = 30, frame = 1 / fps;
  // Not synced: due when first asked about, wherever in the movie.
  {
    hle::BinkPace pace;
    bool behind = false;
    CHECK(!hle::BinkTooEarly(&pace, 4700, fps, 50.0, &behind));
    CHECK(pace.synced && pace.at == 50.0 && pace.frame == 4700);
    CHECK(!behind);  // a start is not a movie that fell behind
    CHECK(!hle::BinkTooEarly(&pace, 4700, fps, 50.0));  // still due
    CHECK(hle::BinkTooEarly(&pace, 4701, fps, 50.0));   // the next: not yet
    CHECK(hle::BinkTooEarly(&pace, 4701, fps, 50.0 + 0.9 * frame));
    CHECK(!hle::BinkTooEarly(&pace, 4701, fps, 50.0 + 1.1 * frame));
    CHECK(pace.at == 50.0 && pace.frame == 4700);  // paced from where it was
    CHECK(hle::BinkTooEarly(&pace, 4710, fps, 50.0 + 9.9 * frame));
    CHECK(!hle::BinkTooEarly(&pace, 4710, fps, 50.0 + 10.1 * frame, &behind));
    CHECK(!behind);
  }
  // Behind by less than kBinkMaxLag: the frames are still due when they
  // were, so the ones missed are due at once and the movie catches up.
  {
    hle::BinkPace pace = {true, 10.0, 1};
    const double now = 10.0 + 3.5 * frame;
    bool behind = false;
    CHECK(!hle::BinkTooEarly(&pace, 1, fps, now, &behind));
    CHECK(pace.at == 10.0 && pace.frame == 1 && !behind);
    CHECK(!hle::BinkTooEarly(&pace, 4, fps, now, &behind));
    CHECK(hle::BinkTooEarly(&pace, 5, fps, now, &behind));
    CHECK(pace.at == 10.0 && pace.frame == 1 && !behind);
    // Just under the limit, and just over it.
    const double under = 10.0 + hle::kBinkMaxLag - 0.01, over = 10.0 + hle::kBinkMaxLag + 0.01;
    CHECK(!hle::BinkTooEarly(&pace, 1, fps, under, &behind));
    CHECK(pace.at == 10.0 && !behind);
    CHECK(!hle::BinkTooEarly(&pace, 1, fps, over, &behind));
    CHECK(pace.at == over && behind);
  }
  // Behind by more: nothing to catch up, the frame is due now and the next
  // a frame's time from now; the caller is told the movie fell behind.
  {
    hle::BinkPace pace = {true, 10.0, 1};
    const double now = 10.0 + hle::kBinkMaxLag + 2 * frame;
    bool behind = false;
    CHECK(!hle::BinkTooEarly(&pace, 1, fps, now, &behind));
    CHECK(behind);
    CHECK(pace.at == now && pace.frame == 1);
    behind = false;
    CHECK(hle::BinkTooEarly(&pace, 2, fps, now, &behind));
    CHECK(hle::BinkTooEarly(&pace, 2, fps, now + 0.9 * frame, &behind));
    CHECK(!hle::BinkTooEarly(&pace, 2, fps, now + 1.1 * frame, &behind));
    CHECK(!behind);  // once, not for the frames after
    hle::BinkPace hour = {true, 10.0, 1};
    CHECK(!hle::BinkTooEarly(&hour, 1, fps, 3610.0));
    CHECK(hle::BinkTooEarly(&hour, 2, fps, 3610.0));  // one frame, not 108000
  }
  // Round from the last frame: the first is due a frame after it.
  {
    hle::BinkPace pace = {true, 10.0, 41};  // 41 at 10: 60 is due 19 frames later
    hle::BinkWrapped(&pace, 60, fps);
    CHECK(pace.synced && pace.frame == 1);
    CHECK(hle::BinkTooEarly(&pace, 1, fps, 10.0 + 19.9 * frame));
    CHECK(!hle::BinkTooEarly(&pace, 1, fps, 10.0 + 20.1 * frame));
    hle::BinkPace none;
    hle::BinkWrapped(&none, 60, fps);
    CHECK(!none.synced);
  }
}

// ---- the calls ----

constexpr uint32_t kReturn = 0x00401000;  // never run: the handlers return here
constexpr uint32_t kNoFrameBuffers = 0x400;

Ctx* g_ctx = nullptr;  // the guest thread's
uint32_t g_stack_top = 0, g_next = 0, g_scratch_end = 0;

uint32_t Alloc(uint32_t size) {
  uint32_t p = g_next;
  g_next += (size + 15) & ~15u;
  if (g_next > g_scratch_end) std::abort();
  std::memset(GPtr(p), 0, size);
  return p;
}

uint32_t Str(const std::string& s) {
  uint32_t p = Alloc(uint32_t(s.size() + 1));
  std::memcpy(GPtr(p), s.data(), s.size());
  return p;
}

// A stdcall to binkw32!name with `args`, as guest code makes it.
uint32_t Call(const char* name, const std::vector<uint32_t>& args) {
  uint32_t sp = g_stack_top - 4 * uint32_t(args.size());
  for (size_t i = 0; i < args.size(); i++) Wr32(sp + 4 * uint32_t(i), args[i]);
  sp -= 4;
  Wr32(sp, kReturn);
  g_ctx->esp = sp;
  hle::CallThunk(*g_ctx, uint32_t(hle::ImportThunk("binkw32.dll", name)));
  CHECK(g_ctx->esp == g_stack_top);  // all arguments popped
  return g_ctx->eax;
}

uint32_t KeyFrame(uint32_t b, uint32_t frame, uint32_t flags) {
  return Call("_BinkGetKeyFrame@12", {b, frame, flags});
}
void Goto(uint32_t b, uint32_t frame, uint32_t flags) { Call("_BinkGoto@12", {b, frame, flags}); }
uint32_t Wait(uint32_t b) { return Call("_BinkWait@4", {b}); }
void DoFrame(uint32_t b) { Call("_BinkDoFrame@4", {b}); }
void NextFrame(uint32_t b) { Call("_BinkNextFrame@4", {b}); }
uint32_t FrameNum(uint32_t b) { return Rd32(b + 0x0C); }
uint32_t LastFrameNum(uint32_t b) { return Rd32(b + 0x10); }

constexpr uint32_t kFrames = 60, kKeyEvery = 10, kSide = 16, kFps = 30;
// One block of sound a frame: 480 sample frames of a 512-point transform,
// at 480 * 30 a second, mono.
constexpr uint32_t kBlockSamples = 480, kRate = kBlockSamples * kFps, kBlockBytes = 72;
constexpr uint32_t kAudioDct = 0x1000;
// The mixer's rate, and a block's length there.
constexpr int kMixRate = 48000, kMixBlock = kMixRate / int(kFps);

// The level of the block of sound in frame `frame`'s packet: the frame's
// number. The further blocks of a first frame: 100 more than their place.
constexpr uint32_t kFurther = 100;

using Outcome = hle::BinkPicture;
struct Seen {
  uint32_t frame;
  Outcome picture;
  bool shown;
  bool operator==(const Seen&) const = default;
};
// Sound queued from a frame's packets (frame 0: silence): how much, its
// level (0 for silence) and whether it starts at that level: a block
// decoded after another starts blended from the end of that one.
struct Heard {
  uint32_t frame;
  int count;
  long level;
  bool fresh;
};

// The clock, and what the movie did since Forget().
double g_now = 1000.0;
std::vector<Seen> g_video;
std::vector<Heard> g_audio;
float g_unit = 0;  // the samples of level 1: those of the first block heard
void Forget() {
  g_video.clear();
  g_audio.clear();
}

void OnVideo(uint32_t frame, Outcome picture, bool shown) {
  g_video.push_back({frame, picture, shown});
}
void OnAudio(uint32_t frame, const float* samples, int count) {
  if (!samples) {
    g_audio.push_back({frame, count, 0, false});
    return;
  }
  const float body = samples[count / 2];
  if (g_unit == 0) g_unit = body;
  g_audio.push_back({frame, count, std::lround(body / g_unit),
                     std::fabs(samples[0] - body) < std::fabs(g_unit) / 4});
}

// The frames the video decoder decoded were first..last, in order, none of
// them shown.
bool DecodedUnseen(uint32_t first, uint32_t last) {
  std::vector<Seen> want;
  for (uint32_t f = first; f <= last; f++) want.push_back({f, Outcome::Decoded, false});
  if (g_video == want) return true;
  std::fprintf(stderr, "decoded:");
  for (const Seen& v : g_video)
    std::fprintf(stderr, " %u%s%s", v.frame, v.picture == Outcome::Decoded ? "" : "!",
                 v.shown ? "+" : "");
  std::fprintf(stderr, "\n");
  return false;
}
// Exactly one frame went through the decoder: decoded and shown.
bool Shown(uint32_t frame) {
  return g_video == std::vector<Seen>{{frame, Outcome::Decoded, true}};
}
// The sample frames queued from frame `frame`'s packets.
int Sound(uint32_t frame) {
  int n = 0;
  for (const Heard& a : g_audio)
    if (a.frame == frame) n += a.count;
  return n;
}
int SoundBefore(uint32_t frame) {
  int n = 0;
  for (const Heard& a : g_audio)
    if (a.frame && a.frame < frame) n += a.count;
  return n;
}
// The levels of the blocks queued, in order.
std::vector<long> Levels() {
  std::vector<long> levels;
  for (const Heard& a : g_audio) levels.push_back(a.level);
  return levels;
}
std::vector<long> Range(long first, long last) {
  std::vector<long> r;
  for (long v = first; v <= last; v++) r.push_back(v);
  return r;
}
std::vector<long> operator+(std::vector<long> a, const std::vector<long>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// The mixer plays `frames` of its output frames; the left channel.
std::vector<float> Mixed(int frames) {
  const size_t n = size_t(frames);
  std::vector<float> out(n * 2), left(n);
  audio::GetMixer().Mix(out.data(), frames);
  for (int i = 0; i < frames; i++) left[size_t(i)] = out[size_t(i) * 2];
  return left;
}

// The mixer plays until the movie's queue has run out: the levels of the
// blocks heard, in order (stretches of one level of half a block or longer,
// counted in blocks; the blend from a block to the next is shorter).
float g_mix_unit = 0;  // the mixer's output for level 1
std::vector<long> Played() {
  std::vector<float> all;
  for (int quiet = 0; quiet < 8 && all.size() < size_t(kMixRate) * 60;) {
    const std::vector<float> part = Mixed(kMixBlock);
    const bool silent = std::all_of(part.begin(), part.end(), [](float v) { return v == 0; });
    quiet = silent ? quiet + 1 : 0;
    all.insert(all.end(), part.begin(), part.end());
  }
  if (g_mix_unit == 0)
    for (float v : all)
      if (std::fabs(v) > std::fabs(g_mix_unit)) g_mix_unit = v;
  std::vector<long> blocks;
  if (g_mix_unit == 0) return blocks;
  for (size_t i = 0; i < all.size();) {
    const long level = std::lround(all[i] / g_mix_unit);
    size_t end = i;
    while (end < all.size() && std::lround(all[end] / g_mix_unit) == level) end++;
    const long n = std::lround(double(end - i) / kMixBlock);
    if (level != 0) blocks.insert(blocks.end(), size_t(n), level);
    i = end;
  }
  return blocks;
}
bool Is(const std::vector<long>& got, const std::vector<long>& want) {
  if (got == want) return true;
  std::fprintf(stderr, "levels:");
  for (long v : got) std::fprintf(stderr, " %ld", v);
  std::fprintf(stderr, "\n");
  return false;
}
// `got` ends with `want`, with at most `before` blocks before it.
bool EndsWith(const std::vector<long>& got, const std::vector<long>& want, size_t before) {
  if (got.size() >= want.size() && got.size() - want.size() <= before &&
      std::equal(want.begin(), want.end(), got.end() - long(want.size())))
    return true;
  return Is(got, want);
}

// What `f` writes to the standard error stream (and it is written there
// afterwards).
std::string g_capture;  // a file
template <class F>
std::string Stderr(F f) {
  std::fflush(stderr);
  const int saved = ::dup(2);
  const int fd = ::open(g_capture.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (saved < 0 || fd < 0) std::abort();
  ::dup2(fd, 2);
  ::close(fd);
  f();
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  std::string text;
  if (FILE* in = std::fopen(g_capture.c_str(), "rb")) {
    char buf[512];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, in)) > 0;) text.append(buf, n);
    std::fclose(in);
  }
  std::fputs(text.c_str(), stderr);
  return text;
}

void Put32(Bytes* to, uint32_t v) {
  for (int i = 0; i < 4; i++) to->push_back(uint8_t(v >> (8 * i)));
}

// Bits as the Bink decoder reads them: from the low bit of each byte.
struct Bits {
  Bytes bytes;
  uint32_t n = 0;
  void Put(uint32_t v, uint32_t bits) {
    for (uint32_t i = 0; i < bits; i++, n++) {
      if (n % 8 == 0) bytes.push_back(0);
      bytes.back() |= uint8_t(((v >> i) & 1) << (n % 8));
    }
  }
  void Align32() {
    while (n % 32) Put(0, 1);
  }
};

// The colours of the key frame that starts group `g` of ten frames.
uint8_t Luma(uint32_t g) { return uint8_t(40 + 10 * g); }
uint8_t Cr(uint32_t g) { return uint8_t(60 + g); }
uint8_t Cb(uint32_t g) { return uint8_t(90 + g); }

// A frame of revision 'k' video. A key frame: each plane (Y, cR, cB) a
// flag and the value it is filled with. Any other: each plane as blocks of
// one type (no fill; the nine trees, none given; the count of block types,
// which revision 'k' stores xor 0xBB, all of one value; no values for the
// eight other kinds): type 0, "as in the frame before", or with `refused`
// type 15, which there is none of, so the decoder refuses the frame.
Bytes VideoFrame(bool key, uint32_t group, bool refused) {
  Bits b;
  b.Put(0, 32);
  if (key) {
    for (uint8_t fill : {Luma(group), Cr(group), Cb(group)}) {
      b.Put(1, 1);
      b.Put(fill, 8);
      b.Align32();
    }
  } else {
    for (uint32_t blocks : {4u, 1u, 1u}) {
      b.Put(0, 1);
      b.Put(0, 92);
      b.Put(blocks ^ 0xBB, 10);
      b.Put(1, 1);
      b.Put(refused ? 15 : 0, 4);
      b.Put(0, 80);
      b.Align32();
    }
  }
  return b.bytes;
}

// A block of sound of one level: the transform's first coefficient (a
// float: 5 bits of exponent, 23 of mantissa, a sign) and nothing else.
Bytes SoundBlock(uint32_t level) {
  Bits b;
  b.Put(0, 2);
  b.Put(24, 5);
  b.Put(level, 23);
  b.Put(0, 1);
  b.bytes.resize(kBlockBytes, 0);
  return b.bytes;
}

// A movie: `frames` frames, with a sound track if `sound`: a block in every
// frame's packet, `lead` more in the first frame's and none in the last
// `lead` frames'. One frame, `at`, may be at fault.
struct Spec {
  uint32_t frames = kFrames;
  bool sound = true;
  uint32_t lead = 0;
  enum Fault {
    None,
    Refused,  // its picture is one the decoder refuses
    PastEnd,  // its sound packet is a byte longer than the frame
    Wraps,    // its sound packet's length and the four bytes of it wrap round
    Stub,     // it is two bytes: no room for a sound packet's length
    Cut,      // the file ends four bytes into it
  } fault = None;
  uint32_t at = 0;
};
Bytes Movie(const Spec& spec) {
  std::vector<Bytes> frames;
  uint32_t largest = 0;
  for (uint32_t i = 0; i < spec.frames; i++) {
    const bool bad = spec.fault != Spec::None && i + 1 == spec.at;
    Bytes f;
    if (spec.sound) {
      uint32_t blocks = i == 0 ? 1 + spec.lead : i < spec.frames - spec.lead ? 1 : 0;
      // The frame after a stub has no sound: its first bytes are zero.
      if (spec.fault == Spec::Stub && i == spec.at) blocks = 0;
      Put32(&f, blocks ? 4 + blocks * kBlockBytes : 0);
      if (blocks) {
        Put32(&f, blocks * kBlockSamples * 2);  // decoded: 16-bit samples
        for (uint32_t k = 0; k < blocks; k++) {
          const Bytes block = SoundBlock(k ? kFurther + k : i + 1);
          f.insert(f.end(), block.begin(), block.end());
        }
      }
    }
    const Bytes v =
        VideoFrame(i % kKeyEvery == 0, i / kKeyEvery, bad && spec.fault == Spec::Refused);
    f.insert(f.end(), v.begin(), v.end());
    if (bad && (spec.fault == Spec::PastEnd || spec.fault == Spec::Wraps)) {
      const uint32_t length = spec.fault == Spec::Wraps ? 0xFFFFFFFF : uint32_t(f.size()) - 4 + 1;
      for (int k = 0; k < 4; k++) f[size_t(k)] = uint8_t(length >> (8 * k));
    }
    if (bad && spec.fault == Spec::Stub) f.assign(2, 0);
    largest = std::max(largest, uint32_t(f.size()));
    frames.push_back(std::move(f));
  }
  Bytes b = {'B', 'I', 'K', 'k'};
  Put32(&b, 0);  // the length of the rest of the file, below
  Put32(&b, spec.frames);
  Put32(&b, largest);
  Put32(&b, spec.frames);
  Put32(&b, kSide);
  Put32(&b, kSide);
  Put32(&b, kFps);
  Put32(&b, 1);
  Put32(&b, 0);  // video flags
  Put32(&b, spec.sound ? 1 : 0);
  Put32(&b, 0);  // revision 'k'
  if (spec.sound) {
    Put32(&b, largest);               // the most decoded
    Put32(&b, kRate | kAudioDct << 16);
    Put32(&b, 0);                     // the track's ID
  }
  uint32_t pos = uint32_t(b.size()) + 4 * spec.frames, cut = 0;
  for (uint32_t i = 0; i < spec.frames; i++) {
    // Bit 0: a key frame. Not set for the first frame, which is one anyway.
    Put32(&b, pos | (i && i % kKeyEvery == 0 ? 1 : 0));
    if (spec.fault == Spec::Cut && i + 1 == spec.at) cut = pos + 4;
    pos += uint32_t(frames[i].size());
  }
  for (const Bytes& f : frames) b.insert(b.end(), f.begin(), f.end());
  const uint32_t rest = uint32_t(b.size()) - 8;
  for (int i = 0; i < 4; i++) b[4 + i] = uint8_t(rest >> (8 * i));
  if (cut) b.resize(cut);  // the header still says how long the movie was
  return b;
}

void WriteFile(const std::filesystem::path& path, const Bytes& b) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) std::abort();
  std::fclose(f);
}

// The game's frame buffers: BINKFRAMEBUFFERS with two sets of planes.
constexpr uint32_t kFbFrameNum = 20, kFbSets = 24, kFbSetSize = 48;
uint32_t RegisterBuffers(uint32_t b) {
  const uint32_t fb = Alloc(kFbSets + 2 * kFbSetSize);
  Call("_BinkGetFrameBuffersInfo@8", {b, fb});
  CHECK(Rd32(fb) == 2 && Rd32(fb + 4) == kSide && Rd32(fb + 12) == kSide / 2);
  for (uint32_t set = 0; set < 2; set++) {
    for (uint32_t plane = 0; plane < 3; plane++) {
      const uint32_t at = fb + kFbSets + set * kFbSetSize + plane * 12;
      const uint32_t side = plane ? kSide / 2 : kSide;
      Wr32(at + 4, Alloc(side * side));
      Wr32(at + 8, side);
    }
  }
  Call("_BinkRegisterFrameBuffers@8", {b, fb});
  return fb;
}

// A set of the game's planes is all of one value each: that of the key
// frame of group `g`.
bool PictureIn(uint32_t fb, uint32_t set_index, uint32_t g) {
  const uint32_t set = fb + kFbSets + set_index * kFbSetSize;
  const uint8_t want[3] = {Luma(g), Cr(g), Cb(g)};
  for (uint32_t plane = 0; plane < 3; plane++) {
    const uint32_t side = plane ? kSide / 2 : kSide;
    const uint8_t* p = GPtr(Rd32(set + plane * 12 + 4));
    for (uint32_t i = 0; i < side * side; i++) {
      if (p[i] == want[plane]) continue;
      std::fprintf(stderr, "picture: plane %u is %u at %u, not %u\n", plane, p[i], i, want[plane]);
      return false;
    }
  }
  return true;
}
// The picture the game was given last: the set FrameNum names.
uint32_t PictureSet(uint32_t fb) { return Rd32(fb + kFbFrameNum); }
bool Picture(uint32_t fb, uint32_t g) { return PictureIn(fb, PictureSet(fb), g); }

uint32_t Open(const char* name) {
  const uint32_t b = Call("_BinkOpen@8", {Str(std::string("C:\\Test\\") + name), kNoFrameBuffers});
  CHECK(b != 0);
  return b;
}
void Close(uint32_t b) { Call("_BinkClose@4", {b}); }

const double kFrame = 1.0 / kFps;
const int kBlock = int(kBlockSamples);

// ---- sound with the picture ----
void CheckPlayAndJump() {
  uint32_t b = Open("even.bik");
  if (!b) return;
  CHECK(Rd32(b + 0x08) == kFrames && FrameNum(b) == 1 && LastFrameNum(b) == 0);
  uint32_t fb = RegisterBuffers(b);

  // Key frames, from the movie's index.
  CHECK(KeyFrame(b, 23, kPrevious) == 21);
  CHECK(KeyFrame(b, 21, kPrevious) == 21);
  CHECK(KeyFrame(b, 21, kPrevious | kNotEqual) == 11);
  CHECK(KeyFrame(b, 23, kNext) == 31);
  CHECK(KeyFrame(b, 31, kNext | kNotEqual) == 41);
  CHECK(KeyFrame(b, 26, kClosest) == 21);
  CHECK(KeyFrame(b, 27, kClosest) == 31);
  CHECK(KeyFrame(b, 1, kPrevious) == 1);
  CHECK(KeyFrame(b, 1, kPrevious | kNotEqual) == 0);
  CHECK(KeyFrame(b, kFrames, kPrevious) == 51);
  CHECK(KeyFrame(b, 52, kNext) == 0);  // none after the last key frame
  CHECK(KeyFrame(b, kFrames, kNext) == 0);
  CHECK(KeyFrame(b, 0xFFFFFFFF, kNext | kNotEqual) == 0);
  CHECK(KeyFrame(b, 0xFFFFFFFF, kPrevious) == 51);
  for (uint32_t f = 1; f <= kFrames; f++)  // not the frame asked about, but for the keys
    CHECK((KeyFrame(b, f, kPrevious) == f) == (f % kKeyEvery == 1));

  // From the start: the first frame is due at once, the next a frame later.
  // The first picture goes into the first set of the game's planes.
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(1) && Picture(fb, 0) && PictureSet(fb) == 0);
  CHECK(Sound(1) == kBlock && g_audio.size() == 1);
  CHECK(g_audio.size() == 1 && g_audio[0].fresh);  // the movie's first block
  CHECK(FrameNum(b) == 1 && LastFrameNum(b) == 1);  // decompressed
  // What the mixer plays of it: that block. Its level is the measure of
  // the ones heard from here on.
  CHECK(Is(Played(), {1}));
  DoFrame(b);  // again: nothing more
  CHECK(Shown(1) && g_audio.size() == 1);
  NextFrame(b);
  CHECK(FrameNum(b) == 2 && LastFrameNum(b) == 1);
  CHECK(Wait(b) == 1);
  g_now += 0.9 * kFrame;
  CHECK(Wait(b) == 1);
  g_now += 0.2 * kFrame;
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(2) && Picture(fb, 0) && Sound(2) == kBlock);
  CHECK(PictureSet(fb) == 1);  // the other set
  // A block decoded after another starts blended from that one's end: what
  // tells a decoder that started over from one that went on.
  CHECK(g_audio.size() == 1 && g_audio[0].level == 2 && !g_audio[0].fresh);
  NextFrame(b);

  // A jump to a late frame while frame 3 is not due yet: frames 41 (the key
  // frame) to 46 decoded and not shown, none before them, and no sound
  // queued for them.
  Forget();
  Goto(b, 47, 0);
  CHECK(DecodedUnseen(41, 46));
  CHECK(g_audio.empty());
  CHECK(FrameNum(b) == 47 && LastFrameNum(b) == 46);
  CHECK(Picture(fb, 0) && PictureSet(fb) == 1);  // the game's buffers: as they were
  // Due at once: not 46 frames after the jump, nor 46 after the start.
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(47) && Picture(fb, 4) && PictureSet(fb) == 0);
  CHECK(PictureIn(fb, 1, 0));  // and the set before it still holds frame 2's
  CHECK(Sound(47) == kBlock && g_audio.size() == 1);
  // The sound decoder started over, and frame 2's sound, which was queued
  // and not played, is gone: frame 47's is all there is to hear.
  CHECK(g_audio.size() == 1 && g_audio[0].level == 47 && g_audio[0].fresh);
  CHECK(Is(Played(), {47}));
  NextFrame(b);
  CHECK(FrameNum(b) == 48 && LastFrameNum(b) == 47);
  CHECK(Wait(b) == 1);  // and paced from there
  g_now += 0.9 * kFrame;
  CHECK(Wait(b) == 1);
  g_now += 0.2 * kFrame;
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(48) && Picture(fb, 4) && Sound(48) == kBlock);

  // Back: from key frame 21. The picture was group 4's, and frame 23 only
  // copies the frame before it.
  Forget();
  Goto(b, 23, 0);
  CHECK(DecodedUnseen(21, 22) && g_audio.empty());
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(23) && Picture(fb, 2) && Sound(23) == kBlock);

  // BINKGOTOQUICK: nothing decoded, so frame 35 copies what was there, the
  // picture of group 2.
  Forget();
  Goto(b, 35, kQuick);
  CHECK(g_video.empty() && g_audio.empty() && FrameNum(b) == 35 && LastFrameNum(b) == 34);
  DoFrame(b);
  CHECK(Shown(35) && Picture(fb, 2));
  // The decoder's picture is no frame's now. A jump a little forward does
  // not decode on from it: it starts at key frame 31, and frame 38 is right.
  Forget();
  Goto(b, 38, 0);
  CHECK(DecodedUnseen(31, 37) && g_audio.empty());
  DoFrame(b);
  CHECK(g_video.back() == (Seen{38, Outcome::Decoded, true}) && Picture(fb, 3));
  // BINKGOTOQUICK to the frame after the one the decoder holds leaves its
  // picture right: the next jump, to the frame after, has nothing to
  // decode (from the key frame it would be 31 to 39).
  Forget();
  Goto(b, 39, kQuick);
  DoFrame(b);
  CHECK(Shown(39) && Picture(fb, 3));
  Forget();
  Goto(b, 40, 0);
  CHECK(g_video.empty());
  // To the frame the movie is at, twice: nothing to decode, then all again.
  Forget();
  Goto(b, 40, 0);
  CHECK(g_video.empty());
  DoFrame(b);
  CHECK(Shown(40) && Picture(fb, 3));
  Forget();
  Goto(b, 40, 0);
  CHECK(DecodedUnseen(31, 39));
  // To a key frame: nothing before it. Then a little forward: on from the
  // frame the movie is at, not from the key frame.
  Forget();
  Goto(b, 41, 0);
  CHECK(g_video.empty());
  DoFrame(b);
  CHECK(Shown(41) && Picture(fb, 4));
  Forget();
  Goto(b, 44, 0);
  CHECK(DecodedUnseen(42, 43) && g_audio.empty());
  // To the first frame: nothing before it.
  Forget();
  Goto(b, 1, 0);
  CHECK(g_video.empty() && FrameNum(b) == 1 && LastFrameNum(b) == 0);
  DoFrame(b);
  CHECK(Shown(1) && Picture(fb, 0) && Sound(1) == kBlock);

  // Past the end: the last frame. Then round to the first, which is due a
  // frame after the last.
  Forget();
  Goto(b, 1000, 0);
  CHECK(FrameNum(b) == kFrames && DecodedUnseen(51, 59));
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(kFrames) && Picture(fb, 5));
  CHECK(Sound(kFrames) == kBlock);
  NextFrame(b);
  // The last frame decompressed is the movie's last, not none.
  CHECK(FrameNum(b) == 1 && LastFrameNum(b) == kFrames);
  CHECK(Wait(b) == 1);
  g_now += 0.9 * kFrame;
  CHECK(Wait(b) == 1);
  g_now += 0.2 * kFrame;
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(1) && Picture(fb, 0) && Sound(1) == kBlock);
  // The first frame's sound starts the movie again: at its own level, not
  // blended from the last frame's.
  CHECK(g_audio.size() == 1 && g_audio[0].level == 1 && g_audio[0].fresh);
  CHECK(LastFrameNum(b) == 1);
  NextFrame(b);

  // The game stops for an hour. Frame 2 is due, and no more than it: one
  // frame decoded, the next a frame's time away.
  g_now += 3600;
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(2));
  NextFrame(b);
  CHECK(FrameNum(b) == 3 && Wait(b) == 1);
  g_now += 1.1 * kFrame;
  CHECK(Wait(b) == 0);
  // It stops for three and a half frames: frames 3 to 6 are due at once,
  // each decoded and shown, and frame 7 when it would have been.
  g_now += 3.5 * kFrame;
  for (uint32_t f = 3; f <= 6; f++) {
    CHECK(Wait(b) == 0);
    Forget();
    DoFrame(b);
    CHECK(Shown(f) && Sound(f) == kBlock && FrameNum(b) == f);
    NextFrame(b);
  }
  CHECK(Wait(b) == 1);
  g_now += 0.6 * kFrame;
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(7));
  // The game moves on two frames before it decodes again: the one between
  // is decoded for its sound and for the picture after it, and not shown.
  NextFrame(b);
  NextFrame(b);
  CHECK(LastFrameNum(b) == 7);  // nothing decompressed since
  Forget();
  DoFrame(b);
  CHECK(g_video == (std::vector<Seen>{{8, Outcome::Decoded, false}, {9, Outcome::Decoded, true}}));
  CHECK(Sound(8) == kBlock && Sound(9) == kBlock);
  CHECK(LastFrameNum(b) == 9);

  // Paused: never due; afterwards as far from due as before (frame 10 is
  // 2.8 frames away).
  NextFrame(b);
  CHECK(Wait(b) == 1);
  CHECK(Call("_BinkPause@8", {b, 1}) == 1);
  g_now += 50;
  CHECK(Wait(b) == 1);
  CHECK(Call("_BinkPause@8", {b, 0}) == 0);
  CHECK(Wait(b) == 1);
  g_now += 2.5 * kFrame;
  CHECK(Wait(b) == 1);
  g_now += 0.5 * kFrame;
  CHECK(Wait(b) == 0);
  Close(b);

  // What games with a movie in a loop do: open, look for a key frame, go
  // there, and only then wait.
  b = Open("even.bik");
  if (!b) return;
  fb = RegisterBuffers(b);
  Forget();
  Goto(b, KeyFrame(b, 57, kPrevious), 0);
  CHECK(FrameNum(b) == 51 && g_video.empty() && g_audio.empty());
  CHECK(Wait(b) == 0);
  DoFrame(b);
  CHECK(Shown(51) && Picture(fb, 5) && Sound(51) == kBlock);
  NextFrame(b);
  CHECK(Wait(b) == 1);
  g_now += 1.1 * kFrame;
  CHECK(Wait(b) == 0);
  Close(b);
}

// ---- sound three frames ahead of the picture ----
void CheckSoundAhead() {
  const uint32_t b = Open("ahead.bik");
  if (!b) return;
  const uint32_t fb = RegisterBuffers(b);
  Forget();
  DoFrame(b);
  CHECK(Shown(1) && Sound(1) == 4 * kBlock);
  CHECK(Is(Levels(), {1, kFurther + 1, kFurther + 2, kFurther + 3}));
  // A jump: the sound of frame 47's time on is in frames 44 to 46, and that
  // is what is queued, nothing of frames 41 to 43. The first frame's four
  // blocks were queued and not played: they are not heard.
  Forget();
  Goto(b, 47, 0);
  CHECK(DecodedUnseen(41, 46));
  CHECK(SoundBefore(44) == 0);
  CHECK(Sound(44) == kBlock && Sound(45) == kBlock && Sound(46) == kBlock && g_audio.size() == 3);
  CHECK(Is(Levels(), {44, 45, 46}));  // each block is its frame's
  CHECK(g_audio.size() == 3 && g_audio[0].fresh && !g_audio[1].fresh);
  CHECK(Is(Played(), {44, 45, 46}));
  Forget();
  DoFrame(b);
  CHECK(Shown(47) && Picture(fb, 4) && Sound(47) == kBlock && g_audio.size() == 1);
  // Near the start: the first frame's sound, less the two frames' worth
  // that is before frame 3's time.
  Forget();
  Goto(b, 3, 0);
  CHECK(DecodedUnseen(1, 2));
  CHECK(Sound(1) == 2 * kBlock && Sound(2) == kBlock && Sound(0) == 0);
  CHECK(Is(Levels(), {kFurther + 2, kFurther + 3, 2}));
  CHECK(Is(Played(), {kFurther + 2, kFurther + 3, 2}));
  Forget();
  DoFrame(b);
  CHECK(Shown(3) && Sound(3) == kBlock);
  // BINKGOTOQUICKSOUND: none of it, and three frames of silence before the
  // target's own.
  Forget();
  Goto(b, 47, kQuickSound);
  CHECK(DecodedUnseen(41, 46) && g_audio.empty());
  DoFrame(b);
  CHECK(Sound(0) == 3 * kBlock && Sound(47) == kBlock && g_audio.size() == 2);
  CHECK(g_audio.front().frame == 0);  // the silence first
  // The last three frames have no sound of their own.
  Forget();
  Goto(b, kFrames, 0);
  CHECK(DecodedUnseen(51, 59));
  CHECK(Sound(57) == kBlock && Sound(58) == 0 && Sound(59) == 0 && g_audio.size() == 1);
  DoFrame(b);
  CHECK(Sound(kFrames) == 0 && Picture(fb, 5));
  // Round the end with silence still owed from a jump (the last frame has
  // no sound to put it before): the first frame's sound is the movie's
  // start, all four blocks and no silence before them.
  Forget();
  Goto(b, kFrames, kQuickSound);
  DoFrame(b);
  CHECK(g_audio.empty());
  NextFrame(b);
  CHECK(FrameNum(b) == 1);
  DoFrame(b);
  CHECK(Sound(0) == 0 && Sound(1) == 4 * kBlock);
  CHECK(!g_audio.empty() && g_audio[0].level == 1 && g_audio[0].fresh);
  Close(b);
}

// ---- no sound at all ----
void CheckSilent() {
  const uint32_t b = Open("silent.bik");
  if (!b) return;
  const uint32_t fb = RegisterBuffers(b);
  Forget();
  Goto(b, 30, 0);
  CHECK(DecodedUnseen(21, 29) && g_audio.empty());
  CHECK(Wait(b) == 0);
  Forget();
  DoFrame(b);
  CHECK(Shown(30) && Picture(fb, 2) && g_audio.empty());
  Close(b);
}

// ---- a movie that fell behind: its sound goes on with its picture ----
// 100 frames, the sound 30 frames (a second) ahead: the mixer has that much
// queued, and plays on while the game does not come for frames.
constexpr uint32_t kFarFrames = 100, kFarLead = 30;
void CheckFellBehind() {
  const uint32_t b = Open("far.bik");
  if (!b) return;
  RegisterBuffers(b);
  Played();  // nothing left of the movies before
  // From frame 50: the sound of frames 20 to 49 is queued. Three frames
  // play, the mixer with them, a frame's time each.
  Goto(b, 50, 0);
  CHECK(Wait(b) == 0);
  const double start = g_now;
  for (uint32_t f = 50; f <= 52; f++) {
    CHECK(Wait(b) == 0);
    DoFrame(b);
    NextFrame(b);
    Mixed(kMixBlock);
    g_now = start + (f - 49) * kFrame + 1e-4;
  }
  // The game stops for 16 frames' time: more than kBinkMaxLag, less than
  // the sound's lead. The mixer plays on: of the 33 blocks queued it has
  // played 19 by now, and frames 39 to 52's wait.
  const uint32_t stop = 16;
  CHECK(stop * kFrame > hle::kBinkMaxLag && stop + 3 + 8 < kFarLead);
  Mixed(int(stop) * kMixBlock);
  g_now += stop * kFrame;
  // Frame 53 is due, and goes on from here. What is queued now is the sound
  // of its time on, frames 23 to 52, from a decoder that started over.
  Forget();
  CHECK(FrameNum(b) == 53 && Wait(b) == 0);
  CHECK(g_video.empty());
  CHECK(Is(Levels(), Range(23, 52)));
  CHECK(!g_audio.empty() && g_audio[0].fresh);
  Forget();
  DoFrame(b);
  CHECK(Shown(53) && Is(Levels(), {53}));
  // And that is what the mixer plays, not frames 39 to 52 and then 53 (the
  // sound 16 frames ahead of the picture), nor those before these. Of the
  // old sound no more than what the mixer had already taken to play is
  // heard first (under three blocks).
  CHECK(EndsWith(Played(), Range(23, 53), 3));
  // The frame after it is paced from here.
  NextFrame(b);
  CHECK(Wait(b) == 1);
  g_now += 1.1 * kFrame;
  CHECK(Wait(b) == 0);

  // A game that waits after it decodes: frame 54 is decoded when it stops.
  // Its sound is of its time on too, and is queued again with the rest.
  DoFrame(b);
  g_now += 1;
  Forget();
  CHECK(Wait(b) == 0);
  CHECK(g_video.empty() && Is(Levels(), Range(24, 54)));
  CHECK(Is(Played(), Range(24, 54)));
  // A game that moved on two frames before it stopped: the frame between
  // is decoded for the picture, and its sound queued once.
  NextFrame(b);
  NextFrame(b);
  g_now += 1;
  Forget();
  CHECK(FrameNum(b) == 56 && Wait(b) == 0);
  CHECK(g_video == (std::vector<Seen>{{55, Outcome::Decoded, false}}));
  CHECK(Is(Levels(), Range(26, 55)));
  DoFrame(b);
  CHECK(g_video.back() == (Seen{56, Outcome::Decoded, true}));
  CHECK(Is(Levels(), Range(26, 56)));
  CHECK(Is(Played(), Range(26, 56)));

  // Less than kBinkMaxLag behind: nothing is queued again.
  NextFrame(b);
  g_now += 5 * kFrame;
  Forget();
  CHECK(5 * kFrame < hle::kBinkMaxLag && Wait(b) == 0);
  CHECK(g_audio.empty() && g_video.empty());
  // Paused for longer than that, and a jump meanwhile: not behind either.
  // The jump's sound is queued once, and the frame is due when the movie
  // goes on.
  CHECK(Call("_BinkPause@8", {b, 1}) == 1);
  g_now += 10;
  Forget();
  Goto(b, 60, 0);
  CHECK(Is(Levels(), Range(30, 59)));
  CHECK(Wait(b) == 1);
  g_now += 10;
  CHECK(Call("_BinkPause@8", {b, 0}) == 0);
  CHECK(Wait(b) == 0);
  CHECK(Is(Levels(), Range(30, 59)));
  CHECK(Is(Played(), Range(30, 59)));
  Close(b);
}

// ---- more sound in the first frame than a movie is taken to hold ----
// 100 frames, the first with the sound of 70 more: over two seconds'. The
// sound is taken to be two seconds (60 frames) ahead, no more.
constexpr uint32_t kLongFrames = 100, kLongLead = 70, kMostLead = 2 * kFps;
void CheckLeadCap() {
  const uint32_t b = Open("long.bik");
  if (!b) return;
  RegisterBuffers(b);
  CHECK(kLongLead > kMostLead);
  // The last frame with sound is 30. A jump to frame 85 queues from frame
  // 25 (60 before it), not from frame 15 (70 before it).
  Forget();
  Goto(b, 85, 0);
  CHECK(Is(Levels(), Range(25, 30)));
  CHECK(SoundBefore(25) == 0);
  // To frame 95: from frame 35, where there is none; not frames 25 to 30.
  Forget();
  Goto(b, 95, 0);
  CHECK(g_audio.empty());
  Close(b);
}

// ---- a frame the decoder refuses ----
void CheckRefused() {
  const uint32_t bad = 25;
  uint32_t b = Open("refused.bik");
  if (!b) return;
  uint32_t fb = RegisterBuffers(b);
  Goto(b, bad - 1, 0);
  Forget();
  CHECK(Stderr([&] { DoFrame(b); }).empty());  // a frame that is right: no warning
  CHECK(Shown(bad - 1) && Picture(fb, 2));
  const uint32_t set = PictureSet(fb);
  NextFrame(b);
  // It is a frame all the same: counted, its sound queued, the picture and
  // the game's planes as they were, and a warning.
  Forget();
  CHECK(!Stderr([&] { DoFrame(b); }).empty());
  CHECK(g_video == (std::vector<Seen>{{bad, Outcome::Refused, false}}));
  CHECK(Sound(bad) == kBlock);
  CHECK(FrameNum(b) == bad && LastFrameNum(b) == bad);
  CHECK(Picture(fb, 2) && PictureSet(fb) == set);
  uint32_t summary = Alloc(0x68);
  Call("_BinkGetSummary@8", {b, summary});
  CHECK(Rd32(summary + 0x24) == 1);  // frames played: the one before it
  // The frame after it is frame 26, not this one again.
  NextFrame(b);
  Forget();
  DoFrame(b);
  CHECK(Shown(bad + 1) && Sound(bad + 1) == kBlock && SoundBefore(bad + 1) == 0);
  // And the frames stay the file's: frame 31 is the key frame of group 3.
  for (uint32_t f = bad + 2; f <= 31; f++) {
    NextFrame(b);
    Forget();
    DoFrame(b);
    CHECK(Shown(f) && Picture(fb, (f - 1) / kKeyEvery));
  }
  Close(b);

  // A jump over it decodes it as one of the frames between. The decoder's
  // picture is no frame's after it: the next jump, a frame on, starts from
  // the key frame, not from where the movie is.
  b = Open("refused.bik");
  if (!b) return;
  fb = RegisterBuffers(b);
  Forget();
  Goto(b, 28, 0);
  std::vector<Seen> want;
  for (uint32_t f = 21; f <= 27; f++)
    want.push_back({f, f == bad ? Outcome::Refused : Outcome::Decoded, false});
  CHECK(g_video == want);
  CHECK(FrameNum(b) == 28);
  Forget();
  Goto(b, 29, 0);
  want.push_back({28, Outcome::Decoded, false});
  CHECK(g_video == want);
  // From a key frame on, it is right again: the next jump goes on.
  Goto(b, 33, 0);
  Forget();
  Goto(b, 35, 0);
  CHECK(DecodedUnseen(33, 34));
  Close(b);
}

// ---- frames the movie's bytes do not hold ----
// Frame `bad` of movie `name` is not whole in it (and with `rest` neither
// are the frames after it): nothing of it is given to a decoder. It is a
// frame all the same.
void CheckUnread(const char* name, uint32_t bad, bool rest) {
  const uint32_t b = Open(name);
  if (!b) return;
  const uint32_t fb = RegisterBuffers(b);
  const uint32_t group = (bad - 2) / kKeyEvery;  // of the frame before it
  Goto(b, bad - 1, 0);
  Forget();
  CHECK(Stderr([&] { DoFrame(b); }).empty());
  CHECK(Shown(bad - 1) && Picture(fb, group));
  NextFrame(b);
  Forget();
  CHECK(!Stderr([&] { DoFrame(b); }).empty());  // a warning
  CHECK(g_video == (std::vector<Seen>{{bad, Outcome::Unread, false}}));
  CHECK(g_audio.empty());
  CHECK(FrameNum(b) == bad && LastFrameNum(b) == bad && Picture(fb, group));
  NextFrame(b);
  Forget();
  CHECK(Stderr([&] { DoFrame(b); }).empty());  // said once a movie
  CHECK(FrameNum(b) == bad + 1 && LastFrameNum(b) == bad + 1);
  if (rest) {
    CHECK(g_video == (std::vector<Seen>{{bad + 1, Outcome::Unread, false}}));
    CHECK(g_audio.empty() && Picture(fb, group));
  } else {
    CHECK(Shown(bad + 1) && Picture(fb, group));
  }
  // A jump over it: one of the frames between, not decoded, and nothing
  // queued from it.
  Forget();
  Goto(b, bad + 1, 0);
  CHECK(g_video.size() == bad - (group * kKeyEvery + 1) + 1);
  CHECK(!g_video.empty() && g_video.back() == (Seen{bad, Outcome::Unread, false}));
  CHECK(Sound(bad) == 0);
  Close(b);
}

void CheckCalls() {
  hle::g_bink_play.now = [] { return g_now; };
  hle::g_bink_play.video = OnVideo;
  hle::g_bink_play.audio = OnAudio;

  // A sound system, so the movies' sound is decoded: a mixer the test
  // plays.
  audio::GetMixer().OpenOffline(kMixRate);
  Call("_BinkSetSoundSystem@8", {uint32_t(hle::ThunkFor("binkw32.dll", "_BinkOpenDirectSound@4")), 0});

  CheckPlayAndJump();
  CheckSoundAhead();
  CheckSilent();
  CheckFellBehind();
  CheckLeadCap();
  CheckRefused();
  // The file ends in frame 59: that one's bytes run past the end, frame
  // 60's start there. A sound packet a byte longer than its frame; one
  // whose length wraps round; a frame with no room for a length.
  CheckUnread("cut.bik", 59, true);
  CheckUnread("pastend.bik", 25, false);
  CheckUnread("wraps.bik", 25, false);
  CheckUnread("stub.bik", 25, false);
}

}  // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bink_play_test <directory for its files>\n");
    return 2;
  }
  CheckKeyFrames();
  CheckSeekPlans();
  CheckPace();

  const fs::path dir = fs::path(argv[1]) / ("bink_play_test_" + std::to_string(::getpid()));
  fs::create_directories(dir / "data");
  ::setenv("XDG_DATA_HOME", (dir / "data").c_str(), 1);
  g_capture = (dir / "stderr.txt").string();
  WriteFile(dir / "even.bik", Movie({}));
  WriteFile(dir / "ahead.bik", Movie({.lead = 3}));
  WriteFile(dir / "silent.bik", Movie({.sound = false}));
  WriteFile(dir / "far.bik", Movie({.frames = kFarFrames, .lead = kFarLead}));
  WriteFile(dir / "long.bik", Movie({.frames = kLongFrames, .lead = kLongLead}));
  WriteFile(dir / "refused.bik", Movie({.fault = Spec::Refused, .at = 25}));
  WriteFile(dir / "cut.bik", Movie({.fault = Spec::Cut, .at = 59}));
  WriteFile(dir / "pastend.bik", Movie({.fault = Spec::PastEnd, .at = 25}));
  WriteFile(dir / "wraps.bik", Movie({.fault = Spec::Wraps, .at = 25}));
  WriteFile(dir / "stub.bik", Movie({.fault = Spec::Stub, .at = 25}));

  // A 32-bit process with its heap and scratch memory.
  g_proc = new Process;
  g_proc->game.id = "binktest";
  g_proc->game.guest_dir = "C:\\Test";
  g_proc->game.exe_name = "test.exe";
  g_proc->game_dir = dir.string();
  if (!g_proc->mem.Reserve(32)) return 1;
  g_proc->mem.Commit(kThunkBase, kThunkRegionSize, Prot::R);
  g_proc->mem.Commit(kPebVa, 0x1000, Prot::RW);
  g_proc->vm.Init(&g_proc->mem, 0x00010000, kThunkBase);
  InitKernel32State();
  g_next = uint32_t(g_proc->vm.Reserve(0, 0x100000));
  g_proc->vm.Commit(g_next, 0x100000, Prot::RW);
  g_scratch_end = g_next + 0x100000;

  // The calls run as a guest thread (opening a file sets its last error).
  Thread* t = CreateGuestThread(0x100000);
  g_ctx = &t->ctx;
  g_stack_top = uint32_t(t->ctx.esp) - 0x100;
  std::promise<void> done;
  StartServiceThread(t, [&done] {
    CheckCalls();
    done.set_value();
  });
  done.get_future().wait();

  fs::remove_all(dir);
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("bink_play_test: ok\n");
  return 0;
}
