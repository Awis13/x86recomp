// AudioDump (runtime/audio/dump.h): the audio thread's Write does not wait
// for a sink that stalls (a disk flushing), every frame reaches the sink in
// order, and a ring too small for a stall drops frames and counts them
// rather than wait; portable.

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "runtime/audio/dump.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

using Clock = std::chrono::steady_clock;

// 100 writes of 441 six-float frames, 10 ms apart, into a sink that takes
// 150 ms on its first call: the longest Write, and what reached the sink.
double Run(int ring_frames, std::vector<float>* got, uint64_t* dropped) {
  recomp::audio::AudioDump d;
  bool first = true;
  d.OpenSink(
      [&](const float* p, int n) {
        if (first) std::this_thread::sleep_for(std::chrono::milliseconds(150));
        first = false;
        got->insert(got->end(), p, p + 6 * n);
      },
      6, ring_frames);
  std::vector<float> buf(6 * 441);
  double longest = 0;
  float k = 0;
  for (int w = 0; w < 100; w++) {
    for (float& x : buf) x = k++;
    const auto a = Clock::now();
    d.Write(buf.data(), 441);
    longest = std::max(longest, std::chrono::duration<double, std::milli>(Clock::now() - a).count());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  d.Close();
  *dropped = d.dropped();
  return longest;
}

}  // namespace

int main() {
  std::vector<float> got;
  uint64_t dropped = 0;
  const double longest = Run(48000, &got, &dropped);
  std::printf("longest Write %.3f ms over a 150 ms stall\n", longest);
  Check(longest < 20.0, "Write does not wait for a stalled sink");
  Check(dropped == 0 && got.size() == size_t(6 * 441 * 100), "with room, every frame reaches the sink");
  bool order = true;
  for (size_t i = 0; i < got.size(); i++) order &= got[i] == float(i);
  Check(order, "in order");
  got.clear();
  const double small = Run(1024, &got, &dropped);
  Check(small < 20.0, "a small ring does not wait either");
  Check(dropped > 0 && got.size() + 6 * dropped == size_t(6 * 441 * 100), "a full ring drops frames and counts them");
  if (g_failures) return 1;
  std::printf("ok\n");
  return 0;
}
