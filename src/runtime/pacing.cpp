// Frame pacing; see pacing.h.

#include "runtime/pacing.h"

#include "runtime/platform.h"

namespace recomp {

void FramePacer::Wait(double fps) {
  if (!(fps > 0)) {
    next_ = 0;
    return;
  }
  const uint64_t period = uint64_t(1e9 / fps + 0.5);
  uint64_t now = platform::MonotonicNs();
  if (!next_ || period != period_) {
    next_ = now;
  } else {
    if (now < next_) {
      platform::SleepUntilNs(next_);
      now = platform::MonotonicNs();
    }
    // Work and the wait itself can both overrun. Rebase before handing
    // the frame to the caller, so a late wake-up cannot cause a burst.
    if (now >= next_ + period) next_ = now;
  }
  period_ = period;
  next_ += period;
}

}  // namespace recomp
