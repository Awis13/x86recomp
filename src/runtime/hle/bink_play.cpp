#include "runtime/hle/bink_play.h"

#include <algorithm>
#include <chrono>

namespace recomp::hle {
namespace {

double SteadyNow() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool IsKey(const std::vector<uint8_t>& keys, uint32_t frame) {
  return frame == 1 || keys[frame - 1] != 0;
}

double Lag(const BinkPace& pace, uint32_t frame_num, double fps, double now) {
  return now - (pace.at + (double(frame_num) - double(pace.frame)) / fps);
}

}  // namespace

BinkPlayCalls g_bink_play = {SteadyNow, nullptr, nullptr};

uint32_t BinkKeyFrame(const std::vector<uint8_t>& keys, uint32_t frame, uint32_t flags) {
  const uint32_t frames = uint32_t(keys.size());
  const bool not_equal = flags & kBinkGetKeyNotEqual;
  uint32_t before = 0, after = 0;
  if (frame > (not_equal ? 1u : 0u)) {
    for (uint32_t f = std::min(not_equal ? frame - 1 : frame, frames); f >= 1 && !before; f--)
      if (IsKey(keys, f)) before = f;
  }
  // No frame past the last: nothing to look for (and no frame + 1 to wrap).
  if (frame <= frames) {
    for (uint32_t f = std::max(not_equal ? frame + 1 : frame, 1u); f <= frames && !after; f++)
      if (IsKey(keys, f)) after = f;
  }
  switch (flags & 3) {
    case kBinkGetKeyPrevious: return before;
    case kBinkGetKeyNext: return after;
    default:
      if (!before || !after) return before ? before : after;
      return after - frame < frame - before ? after : before;
  }
}

BinkSeek PlanBinkSeek(const std::vector<uint8_t>& keys, uint32_t decoded, uint32_t target,
                      uint32_t flags, uint32_t lead) {
  BinkSeek s = {};
  if (flags & kBinkGotoQuick) {
    s.key = s.first = target;
  } else {
    s.key = std::max(BinkKeyFrame(keys, target, kBinkGetKeyPrevious), 1u);
    const uint32_t next = decoded + 1;
    s.first = next >= s.key && next <= target ? next : s.key;
  }
  if (flags & kBinkGotoQuickSound) {
    s.audio_from = target;
    s.silence = target > 1 ? lead : 0;
  } else if (target > lead + 1) {
    s.audio_from = target - lead;
  } else {
    // The first frame holds the sound from the movie's start on.
    s.audio_from = 1;
    s.drop = target - 1;
  }
  return s;
}

bool BinkTooEarly(BinkPace* pace, uint32_t frame_num, double fps, double now,
                  bool* fell_behind) {
  const double lag = pace->synced ? Lag(*pace, frame_num, fps, now) : 0;
  if (lag < 0) return true;
  if (fell_behind && lag > kBinkMaxLag) *fell_behind = true;
  if (!pace->synced || lag > kBinkMaxLag) *pace = {true, now, frame_num};
  return false;
}

void BinkWrapped(BinkPace* pace, uint32_t frames, double fps) {
  if (!pace->synced) return;
  pace->at += (double(frames) - double(pace->frame) + 1) / fps;
  pace->frame = 1;
}

}  // namespace recomp::hle
