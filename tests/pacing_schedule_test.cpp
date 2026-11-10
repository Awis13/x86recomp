// Exercise the real FramePacer with a controlled platform clock. Work and
// wake-up delays must not depend on the host scheduler in these checks.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "runtime/pacing.h"
#include "runtime/platform.h"

namespace {
uint64_t now = 1000000000;
uint64_t wake_delay = 0;
std::vector<uint64_t> deadlines;
int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

void Reset() {
  now = 1000000000;
  wake_delay = 0;
  deadlines.clear();
}

void Rate(double fps) {
  Reset();
  recomp::FramePacer pacer;
  const auto period = static_cast<uint64_t>(1e9 / fps + 0.5);
  const uint64_t origin = now;
  pacer.Wait(fps);
  Check(deadlines.empty(), "first frame starts immediately");
  for (int i = 1; i < 500; ++i) {
    now += period * (i % 6) / 10;
    wake_delay = static_cast<uint64_t>(i % 7) * 10000;
    pacer.Wait(fps);
    Check(deadlines.size() == static_cast<size_t>(i), "one sleep per frame");
    Check(!deadlines.empty() && deadlines.back() == origin + i * period,
          "fractional frame periods and wake-up latency do not accumulate drift");
    Check(now == origin + i * period + wake_delay, "frame starts at wake-up");
  }
}

void LateWork() {
  Reset();
  recomp::FramePacer pacer;
  constexpr uint64_t period = 5000000;
  pacer.Wait(200);
  const uint64_t origin = now;
  now += period + period / 4;
  pacer.Wait(200);
  Check(deadlines.empty(), "slightly late work does not sleep");
  pacer.Wait(200);
  Check(now == origin + 2 * period, "slightly late work keeps the grid");
  now += 3 * period;
  const uint64_t late = now;
  const auto sleeps = deadlines.size();
  pacer.Wait(200);
  Check(now == late && deadlines.size() == sleeps, "long work starts next frame immediately");
  pacer.Wait(200);
  Check(now == late + period, "long work rebases without a catch-up burst");
}

void LateWake(uint64_t delay) {
  Reset();
  recomp::FramePacer pacer;
  constexpr uint64_t period = 5000000;
  pacer.Wait(200);
  const uint64_t origin = now;
  wake_delay = delay;
  pacer.Wait(200);
  const uint64_t woke = now;
  Check(woke == origin + period + delay, "injected delay occurs inside the pacer's wait");
  wake_delay = 0;
  pacer.Wait(200);
  const uint64_t expected = delay >= period ? woke + period : origin + 2 * period;
  Check(now == expected, "late wake-up keeps the grid or rebases before the next frame");
  Check(deadlines.size() == 2, "late wake-up is not followed by an immediate catch-up frame");
}

void RateChange() {
  Reset();
  recomp::FramePacer pacer;
  pacer.Wait(144);
  pacer.Wait(240);
  Check(deadlines.empty(), "changing rate starts a new grid immediately");
  const uint64_t origin = now;
  pacer.Wait(240);
  Check(now == origin + 4166667, "new rate uses its rounded nanosecond period");
  pacer.Wait(0);
  now += 123;
  const uint64_t resumed = now;
  pacer.Wait(240);
  Check(now == resumed && deadlines.size() == 1, "unlimited mode resets the grid");
  pacer.Wait(240);
  Check(now == resumed + 4166667, "limiting resumes from the new grid");
}
}  // namespace

namespace recomp::platform {
uint64_t MonotonicNs() { return now; }
void SleepUntilNs(uint64_t deadline) {
  Check(deadline > now, "sleep uses a future absolute deadline");
  deadlines.push_back(deadline);
  now = deadline + wake_delay;
}
}  // namespace recomp::platform

int main() {
  Rate(144);
  Rate(240);
  LateWork();
  LateWake(100000);
  LateWake(4999999);
  LateWake(5000000);
  LateWake(15000000);
  RateChange();
  if (failures) return 1;
  std::puts("ok");
  return 0;
}
