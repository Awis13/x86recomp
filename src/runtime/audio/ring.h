#pragma once
// A lock-free ring of float frames from the mixer thread (the only
// writer) to the device callback (the only reader): the mixer fills it a
// period at a time, the callback copies it out, and neither takes a lock
// or touches guest state. See mixer.h for why.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace recomp::audio {

// Sizes and counts are in frames (a float per channel: two, or six for a
// 5.1 mix), not floats or bytes.
class FrameRing {
 public:
  // Makes room for `frames` frames of `channels` floats, rounded up to a
  // power of two, and empties it. Only while no producer or consumer runs.
  void Reset(int frames, int channels = 2);
  int Capacity() const { return capacity_; }
  int channels() const { return channels_; }

  // Frames the producer may append now: the producer's side only.
  int Free() const {
    const uint32_t w = write_.load(std::memory_order_relaxed);
    const uint32_t r = read_.load(std::memory_order_acquire);
    return capacity_ - int(w - r);
  }
  // Frames the consumer may take now: the consumer's side only.
  int Filled() const {
    const uint32_t w = write_.load(std::memory_order_acquire);
    const uint32_t r = read_.load(std::memory_order_relaxed);
    return int(w - r);
  }

  // Producer: appends n frames (at most Free()).
  void Write(const float* src, int n);
  // Consumer: takes min(n, Filled()) frames into dst; returns how many.
  int Read(float* dst, int n);

 private:
  // A frame's slot: the counter masked, the capacity being a power of two
  // (no modulo, and the counters compare right across the uint32_t wrap).
  float* At(uint32_t frame) {
    return &data_[size_t(channels_) * size_t(frame & uint32_t(capacity_ - 1))];
  }
  const float* At(uint32_t frame) const {
    return &data_[size_t(channels_) * size_t(frame & uint32_t(capacity_ - 1))];
  }

  std::vector<float> data_;  // interleaved frames
  int capacity_ = 0, channels_ = 2;
  // Written by one side each: the producer reads read_ and publishes
  // write_ (release), the consumer the other way round, so a frame is
  // whole before the other side sees it.
  std::atomic<uint32_t> read_{0}, write_{0};
};

inline void FrameRing::Reset(int frames, int channels) {
  int capacity = 1;
  while (capacity < frames) capacity *= 2;
  capacity_ = frames > 0 ? capacity : 0;
  channels_ = channels > 0 ? channels : 2;
  data_.assign(size_t(channels_) * size_t(capacity_), 0.0f);
  read_.store(0, std::memory_order_relaxed);
  write_.store(0, std::memory_order_relaxed);
}

inline void FrameRing::Write(const float* src, int n) {
  if (!capacity_ || n <= 0) return;
  const uint32_t w = write_.load(std::memory_order_relaxed);
  const int first = std::min(n, capacity_ - int(w & uint32_t(capacity_ - 1)));
  const size_t ch = size_t(channels_);
  std::memcpy(At(w), src, sizeof(float) * ch * size_t(first));
  if (n > first)
    std::memcpy(At(w + uint32_t(first)), src + ch * size_t(first),
                sizeof(float) * ch * size_t(n - first));
  write_.store(w + uint32_t(n), std::memory_order_release);
}

inline int FrameRing::Read(float* dst, int n) {
  if (!capacity_ || n <= 0) return 0;
  const uint32_t r = read_.load(std::memory_order_relaxed);
  const uint32_t w = write_.load(std::memory_order_acquire);
  n = std::min(n, int(w - r));
  if (n <= 0) return 0;
  const int first = std::min(n, capacity_ - int(r & uint32_t(capacity_ - 1)));
  const size_t ch = size_t(channels_);
  std::memcpy(dst, At(r), sizeof(float) * ch * size_t(first));
  if (n > first)
    std::memcpy(dst + ch * size_t(first), At(r + uint32_t(first)),
                sizeof(float) * ch * size_t(n - first));
  read_.store(r + uint32_t(n), std::memory_order_release);
  return n;
}

// The fill the mixer thread keeps a ring at: two device requests (of
// `request` frames) and a period ahead of what the device has taken, so
// that a stall of up to that is caught up without silence; within a ring
// of `capacity` frames, a period to spare.
inline int FillTarget(int request, int period, int capacity) {
  return std::min(2 * request + period, capacity - period);
}

// The room a ring needs before it is rounded up to a power of two: its
// fill and a period; where the device may change as it runs, the fill for
// the largest request a device may make (0: none) and a period.
inline int RingRoom(int request, int period, int largest_request) {
  return std::max(2 * request, 2 * largest_request) + 2 * period;
}

}  // namespace recomp::audio
