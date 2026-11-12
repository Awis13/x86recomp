#pragma once
// Where a Bink movie is and when its frames are due: key frames
// (BinkGetKeyFrame), what a jump has to decode (BinkGoto) and the pacing of
// BinkWait. Frame numbers are Bink's: the first is 1.
// Nothing here touches a movie; binkw32.cpp does what these say.

#include <cstdint>
#include <vector>

namespace recomp::hle {

// bink.h: BINKGOTO* and BINKGETKEY*.
inline constexpr uint32_t kBinkGotoQuick = 1, kBinkGotoQuickSound = 2;
inline constexpr uint32_t kBinkGetKeyPrevious = 0, kBinkGetKeyNext = 1, kBinkGetKeyClosest = 2,
                          kBinkGetKeyNotEqual = 128;

// BinkGetKeyFrame(frame, flags) over a movie's key frames (keys[i] nonzero:
// frame i + 1 is one; the first frame of a movie always is): the first key
// frame before or equal to `frame` (BINKGETKEYPREVIOUS), after or equal to
// it (BINKGETKEYNEXT) or the closer of the two (BINKGETKEYCLOSEST: the
// earlier one if they are as close), and with BINKGETKEYNOTEQUAL never
// `frame` itself. 0 if there is none: before the first frame, after the
// last key frame, and so for any `frame` past the end of the movie when a
// later one is asked for.
uint32_t BinkKeyFrame(const std::vector<uint8_t>& keys, uint32_t frame, uint32_t flags);

// What BinkGoto(target, flags) does to a movie whose video decoder holds
// the picture of frame `decoded`, so that the next BinkDoFrame decodes
// `target`. `decoded` is 0 when the decoder holds no frame's picture: none
// decoded yet, or the last one was not built on the frame before it (after
// BINKGOTOQUICK, or a frame the decoder refused or the movie's bytes do
// not hold, until a key frame has been decoded).
//
// The picture: frames first..target-1 are decoded and not shown. Without
// flags `first` is the key frame the target is built from, or the frame
// after `decoded` if that is closer (a short jump forward decodes on from
// where the movie is). With BINKGOTOQUICK nothing is decoded: Bink "jumps
// directly to the frame", which shows wrong until the next key frame.
//
// A jump costs the frames from the key frame before the target, decoded on
// the caller's thread: bounded by the distance between two key frames, not
// by the movie's length. A movie whose only key frame is its first has no
// such bound: a jump in it decodes from frame 1.
//
// The sound: Bink stores the first 750 ms of a movie's sound in its first
// frame, so every later frame holds the sound of that much later: `lead`
// frames later here. The sound of the target's time on is in the frames
// from target - lead: the sound of frames audio_from..target-1 is queued
// (and nothing of the frames only decoded for the picture), less the first
// `drop` frames' worth of it, which a jump to less than `lead` frames into
// the movie finds in the first frame before what it wants. With
// BINKGOTOQUICKSOUND Bink "just doesn't skip back", and the first 750 ms
// are silence: `silence` frames' worth is queued instead.
//
// The sound is planned the same way for a movie that goes on at `target`
// after it fell behind (BinkTooEarly).
struct BinkSeek {
  uint32_t key, first, audio_from, drop, silence;
};
BinkSeek PlanBinkSeek(const std::vector<uint8_t>& keys, uint32_t decoded, uint32_t target,
                      uint32_t flags, uint32_t lead);

// When frames are due, on a clock of seconds: frame `frame` at `at`, the
// ones after it 1/fps apart. Not `synced` when a movie is opened and after
// a jump: the next BinkWait takes the frame it is asked about as due then.
struct BinkPace {
  bool synced = false;
  double at = 0;
  uint32_t frame = 1;
};

// A movie this far behind is not caught up: its frame is due now, and the
// ones after it are paced from now (a game that stopped for a while, in a
// loading screen or a debugger, gets the movie where it left it, not as
// many frames at once as it missed). What Bink does then is not in its
// documentation; the threshold and going on from where the movie is are
// this module's choice.
inline constexpr double kBinkMaxLag = 0.5;

// BinkWait at `now` for frame `frame_num`: true while it is too early.
// *fell_behind (if given) is set when the movie was more than kBinkMaxLag
// behind and its pace started over from this frame: the sound queued for
// it ran on without the picture meanwhile, and has to be queued again as
// for a jump to this frame.
bool BinkTooEarly(BinkPace* pace, uint32_t frame_num, double fps, double now,
                  bool* fell_behind = nullptr);

// BinkNextFrame on the last of `frames` frames: frame 1 again, due one
// frame after the last.
void BinkWrapped(BinkPace* pace, uint32_t frames, double fps);

// What became of a frame's picture: decoded; refused by the decoder; or not
// given to it, because the movie's bytes do not hold the frame as its index
// and its packets' lengths say.
enum class BinkPicture { Decoded, Refused, Unread };

// What binkw32 asks of the world while it plays, and tells it: the clock
// (seconds, steady); each frame as the video decoder is done with it
// (`shown`: its picture went to the game); and the sound queued from a
// frame's packets: `count` sample frames at `samples`, interleaved (frame
// 0: silence, no samples). A test puts its own clock here and records the
// rest; the last two are null otherwise.
struct BinkPlayCalls {
  double (*now)();
  void (*video)(uint32_t frame, BinkPicture picture, bool shown);
  void (*audio)(uint32_t frame, const float* samples, int count);
};
extern BinkPlayCalls g_bink_play;

}  // namespace recomp::hle
