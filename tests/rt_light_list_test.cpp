// The frame's local lights between the recording and encoding sides
// (runtime/gfx/rt_light_list.h), with the device's recorded work stood in
// for by a queue of closures run in order: a list given is what the
// frame's work reads on the encoding side; frames recorded ahead keep
// their own lists; a frame without one reads none; the last of several
// calls wins for the work after it; the cap, the invalid lights dropped and
// the counts; lists reused; the same with a second thread encoding while
// the first records, as the device runs them; the device's wiring
// (RtLightsGive, RtLightsPresent): nothing recorded with ray tracing off,
// and with its work run at once (no encoding thread); and the device that
// takes the lights (RtLightTaker) unregistered only by itself.
//
// The Metal device itself is not made here: it needs a window and a GPU
// that traces rays, which these portable tests do not have; what it does
// with the lights is RtLightsGive and RtLightsPresent with its Rec.
//
//   rt_light_list_test

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/gfx/rt_light_list.h"

namespace {

using recomp::gfx::kRtLightsMax;
using recomp::gfx::RtLight;
using recomp::gfx::RtLightCounts;
using recomp::gfx::RtLightFrames;
using recomp::gfx::RtLightKind;
using recomp::gfx::RtLightsGive;
using recomp::gfx::RtLightsPresent;
using recomp::gfx::RtLightTaker;

int failures = 0;
void Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  failures += ok ? 0 : 1;
}

// A light named by `key`, at x = key.
RtLight Light(uint64_t key, RtLightKind kind = RtLightKind::Point) {
  RtLight l{};
  l.position[0] = float(key);
  l.radius = 5;
  l.color[0] = l.color[1] = l.color[2] = 1;
  l.kind = kind;
  if (kind == RtLightKind::Spot) {
    l.direction[2] = -1;
    l.cone_offset = 0.5f;
    l.cone_scale = 4;
  }
  l.key = key;
  return l;
}

// `n` lights of keys base, base + 1, ...
std::vector<RtLight> Lights(uint64_t base, size_t n) {
  std::vector<RtLight> v;
  for (size_t i = 0; i < n; i++) v.push_back(Light(base + i));
  return v;
}

// The keys the encoding side reads now.
std::vector<uint64_t> Seen(const RtLightFrames& f) {
  std::vector<uint64_t> k;
  for (const RtLight& l : f.Current()) k.push_back(l.key);
  return k;
}

std::vector<uint64_t> Keys(const std::vector<RtLight>& v) {
  std::vector<uint64_t> k;
  for (const RtLight& l : v) k.push_back(l.key);
  return k;
}

// The device's recording side, single-threaded: SetRtLights and Present
// as the Metal device records them (RtLightsGive, RtLightsPresent), a
// consumer's read as a recorded op; Encode runs what was recorded so far,
// in order. `immediate`: each op run as it is recorded (the device without
// an encoding thread); `on`: ray tracing on.
struct Recorder {
  RtLightFrames frames;
  std::vector<std::function<void()>> ops;
  RtLightCounts last;
  bool immediate = false, on = true;

  void Record(std::function<void()> f) {
    if (immediate)
      f();
    else
      ops.push_back(std::move(f));
  }
  bool Set(const std::vector<RtLight>& v) {
    return RtLightsGive(frames, on, v.data(), v.size(), [this](auto f) { Record(f); }, &last);
  }
  void Present() {
    RtLightsPresent(frames, on, [this](auto f) { Record(f); });
  }
  void Read(std::vector<uint64_t>* out) {
    Record([this, out] { *out = Seen(frames); });
  }
  void Encode() {
    for (auto& o : ops) o();
    ops.clear();
  }
};

void OneFrame() {
  Recorder r;
  const std::vector<RtLight> v = Lights(100, 3);
  std::vector<uint64_t> seen;
  r.Set(v);
  r.Read(&seen);
  r.Present();
  r.Encode();
  Check(seen == Keys(v), "a list given is what the frame's work reads, in order");
  Check(r.frames.Current().empty(), "after the presented frame's end nothing is read");
}

// Three frames recorded before the first is encoded (the device keeps up to
// four in flight): each frame's work reads its own list.
void FramesInFlight() {
  Recorder r;
  std::vector<uint64_t> seen[3];
  std::vector<RtLight> v[3] = {Lights(10, 2), Lights(20, 5), Lights(30, 1)};
  for (int k = 0; k < 3; k++) {
    r.Set(v[k]);
    r.Read(&seen[k]);
    r.Present();
  }
  r.Encode();
  bool own = true;
  for (int k = 0; k < 3; k++) own = own && seen[k] == Keys(v[k]);
  Check(own, "frames recorded ahead of the encoding keep their own lists");
  Check(r.frames.Lists() == 3, "one list per frame in flight");
}

void EmptyFrame() {
  Recorder r;
  std::vector<uint64_t> before{1}, with, without{1}, again;
  r.Read(&before);  // no call yet
  r.Present();
  r.Set(Lights(40, 4));
  r.Read(&with);
  r.Present();
  r.Read(&without);  // a frame without a call, after one with
  r.Present();
  r.Set({});
  r.Read(&again);  // an empty list given
  r.Present();
  r.Encode();
  Check(before.empty(), "no call: an empty list");
  Check(with.size() == 4, "the frame with a list reads it");
  Check(without.empty(), "a frame without a call after one with: an empty list, not the last frame's");
  Check(again.empty(), "an empty list given reads empty");
}

void LastCallWins() {
  Recorder r;
  const std::vector<RtLight> a = Lights(50, 2), b = Lights(60, 3);
  std::vector<uint64_t> between, after, next;
  r.Set(a);
  r.Read(&between);
  r.Set(b);
  r.Read(&after);
  r.Present();
  r.Read(&next);
  r.Present();
  r.Encode();
  Check(between == Keys(a), "work recorded between two calls reads the first");
  Check(after == Keys(b), "work recorded after the last call reads it");
  Check(next.empty(), "the next frame does not inherit it");
}

void CapAndValidity() {
  RtLightFrames f;
  RtLightCounts n;
  const std::vector<RtLight> many = Lights(1000, kRtLightsMax + 60);
  RtLightFrames::List* l = f.Give(many.data(), many.size(), &n);
  f.Publish(l);
  const auto cur = f.Current();
  Check(cur.size() == kRtLightsMax && cur.front().key == 1000 && cur.back().key == 1000 + kRtLightsMax - 1,
        "a longer list: its first kRtLightsMax kept");
  Check(n.given == kRtLightsMax + 60 && n.kept == kRtLightsMax && n.point == kRtLightsMax && n.spot == 0,
        "counts: given, kept, by kind");
  f.EndFrame();

  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  std::vector<RtLight> mixed;
  mixed.push_back(Light(1));
  mixed.push_back(Light(2, RtLightKind::Spot));
  RtLight bad = Light(3);
  bad.position[1] = nan;
  mixed.push_back(bad);
  bad = Light(4);
  bad.radius = 0;
  mixed.push_back(bad);
  bad = Light(5);
  bad.radius = -1;
  mixed.push_back(bad);
  bad = Light(6);
  bad.color[2] = inf;
  mixed.push_back(bad);
  bad = Light(7, RtLightKind::Spot);
  bad.cone_scale = nan;
  mixed.push_back(bad);
  bad = Light(8);
  bad.kind = RtLightKind(7);
  mixed.push_back(bad);
  bad = Light(9);
  bad.radius = inf;
  mixed.push_back(bad);
  bad = Light(11);
  bad.color[1] = -0.5f;
  mixed.push_back(bad);
  mixed.push_back(Light(10, RtLightKind::Spot));
  RtLight dim = Light(12);
  dim.color[0] = dim.color[1] = dim.color[2] = 0;
  mixed.push_back(dim);
  l = f.Give(mixed.data(), mixed.size(), &n);
  f.Publish(l);
  Check(Seen(f) == std::vector<uint64_t>({1, 2, 10, 12}),
        "dropped: a non-finite number, a radius of 0, below 0 or infinite, a colour channel below 0, an unknown "
        "kind (a colour of 0 kept)");
  Check(n.given == 12 && n.kept == 4 && n.point == 2 && n.spot == 2, "counts with lights dropped");
  f.EndFrame();

  // The cap counts lights kept: invalid ones ahead of it do not use it up.
  std::vector<RtLight> ahead(5, bad);
  const std::vector<RtLight> full = Lights(5000, kRtLightsMax);
  ahead.insert(ahead.end(), full.begin(), full.end());
  l = f.Give(ahead.data(), ahead.size(), &n);
  f.Publish(l);
  Check(n.kept == kRtLightsMax && f.Current().back().key == 5000 + kRtLightsMax - 1,
        "invalid lights do not take places under the cap");
  f.EndFrame();

  l = f.Give(nullptr, 12, &n);
  f.Publish(l);
  Check(f.Current().empty() && n.given == 0 && n.kept == 0, "a null list: empty");
}

// The device's wiring: with ray tracing off nothing is given or recorded;
// with its work run at once, a list is read from the call to the frame's
// end.
void Wiring() {
  Recorder off;
  off.on = false;
  std::vector<uint64_t> seen{1};
  Check(!off.Set(Lights(70, 3)), "ray tracing off: nothing given");
  off.Present();
  Check(off.ops.empty() && off.frames.Lists() == 0, "ray tracing off: nothing recorded, no list made");
  off.Read(&seen);
  off.Encode();
  Check(seen.empty(), "ray tracing off: the encoding side reads none");

  Recorder now;
  now.immediate = true;
  const std::vector<RtLight> a = Lights(80, 2), b = Lights(90, 4);
  std::vector<uint64_t> first, second, after, next;
  Check(now.Set(a), "run at once: given");
  now.Read(&first);
  now.Set(b);
  now.Read(&second);
  now.Present();
  now.Read(&after);
  now.Set(a);
  now.Read(&next);
  now.Present();
  Check(now.ops.empty(), "run at once: nothing left to encode");
  Check(first == Keys(a) && second == Keys(b), "run at once: each call read until the next");
  Check(after.empty(), "run at once: none after the presented frame's end");
  Check(next == Keys(a) && now.frames.Current().empty(), "run at once: the next frame's own, then none");
  Check(now.frames.Lists() <= 2, "run at once: lists reused");
}

// The device that takes the lights: unregistered only by itself.
void Taker() {
  struct Dev {};
  Dev a, b;
  RtLightTaker<Dev> t;
  Check(t.Get() == nullptr, "taker: none at first");
  t.Note(&a, true);
  Check(t.Get() == &a, "taker: a registered");
  t.Note(&b, true);
  Check(t.Get() == &b, "taker: b took over");
  t.Note(&a, false);
  Check(t.Get() == &b, "taker: a going leaves b");
  t.Note(&b, false);
  Check(t.Get() == nullptr, "taker: b going unregisters it");
  t.Note(&a, false);
  Check(t.Get() == nullptr, "taker: unregistering one not registered changes nothing");
}

// Lists come back once their frame is over: as many as frames in flight.
void Reuse() {
  Recorder r;
  for (int frame = 0; frame < 500; frame++) {
    r.Set(Lights(uint64_t(frame) * 10, 3));
    r.Present();
    if (frame % 3 == 2) r.Encode();  // three frames ahead at most
  }
  r.Encode();
  Check(r.frames.Lists() <= 3, "lists reused: no more than the frames in flight");
}

// The device's two threads: one records frames (a list a frame, keys
// telling the frame, some frames none or two), the other encodes the
// batches as they come, at most four frames behind; each frame's reads
// must see that frame's last list.
void Threads() {
  constexpr int kFrames = 3000, kAhead = 4;
  RtLightFrames frames;
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::vector<std::function<void()>>> batches;
  int encoded = 0;  // frames whose batch is done
  bool done = false;
  std::vector<int> bad(kFrames, 0);

  std::thread encoder([&] {
    for (;;) {
      std::vector<std::function<void()>> b;
      {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [&] { return !batches.empty() || done; });
        if (batches.empty()) return;
        b = std::move(batches.front());
        batches.pop_front();
      }
      for (auto& o : b) o();
      {
        std::lock_guard<std::mutex> l(mu);
        encoded++;
      }
      cv.notify_all();
    }
  });

  for (int frame = 0; frame < kFrames; frame++) {
    {
      std::unique_lock<std::mutex> l(mu);
      cv.wait(l, [&] { return frame - encoded < kAhead; });
    }
    std::vector<std::function<void()>> ops;
    const int calls = frame % 7 == 3 ? 0 : frame % 5 == 1 ? 2 : 1;
    uint64_t expect_key = 0;
    size_t expect_n = 0;
    for (int c = 0; c < calls; c++) {
      const size_t n = 1 + size_t(frame + c) % 40;
      const std::vector<RtLight> v = Lights(uint64_t(frame) * 1000 + uint64_t(c) * 100, n);
      RtLightFrames::List* list = frames.Give(v.data(), v.size());
      ops.push_back([&frames, list] { frames.Publish(list); });
      expect_key = v.front().key;
      expect_n = n;
    }
    ops.push_back([&frames, &bad, frame, expect_key, expect_n] {
      const auto cur = frames.Current();
      const bool ok = expect_n ? cur.size() == expect_n && cur.front().key == expect_key &&
                                     cur.back().key == expect_key + expect_n - 1
                               : cur.empty();
      if (!ok) bad[size_t(frame)] = 1;
    });
    ops.push_back([&frames] { frames.EndFrame(); });
    {
      std::lock_guard<std::mutex> l(mu);
      batches.push_back(std::move(ops));
    }
    cv.notify_all();
  }
  {
    std::lock_guard<std::mutex> l(mu);
    done = true;
  }
  cv.notify_all();
  encoder.join();
  int wrong = 0;
  for (int b : bad) wrong += b;
  if (wrong) std::printf("  %d frames read another list\n", wrong);
  Check(wrong == 0, "two threads: every frame's work reads its own last list, or none");
  Check(frames.Lists() <= size_t(kAhead) * 2 + 1, "two threads: lists bounded by the frames in flight");
}

}  // namespace

int main() {
  OneFrame();
  FramesInFlight();
  EmptyFrame();
  LastCallWins();
  CapAndValidity();
  Reuse();
  Threads();
  Wiring();
  Taker();
  if (failures) std::printf("%d failed\n", failures);
  return failures ? 1 : 0;
}
