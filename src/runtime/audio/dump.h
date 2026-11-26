#pragma once
// RECOMP_AUDIO_DUMP's files written off the audio threads. A real-time
// audio thread that writes a file itself waits whenever the disk does: the
// system flushes written data about every 30 s, and a write then takes 50
// to 80 ms (measured on a USB disk), far longer than the mixer's ring
// holds, so the dump put silence into what it recorded. Here Write copies
// the frames into a lock-free ring and a thread of its own writes them to
// the file; a ring that is full drops frames (counted, said at Close)
// rather than wait.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "runtime/audio/ring.h"

namespace recomp::audio {

class AudioDump {
 public:
  // Where frames go: a file by default, or for a test anything.
  using Sink = std::function<void(const float* frames, int n)>;

  AudioDump() = default;
  AudioDump(const AudioDump&) = delete;
  AudioDump& operator=(const AudioDump&) = delete;
  ~AudioDump() { Close(); }

  // RECOMP_AUDIO_DUMP=dir: dir/name, frames of `channels` floats. False
  // (and nothing written) without the variable or the file.
  bool Open(const char* name, int channels) {
    const char* dir = std::getenv("RECOMP_AUDIO_DUMP");
    if (!dir || !*dir) return false;
    FILE* f = std::fopen((std::string(dir) + "/" + name).c_str(), "wb");
    if (!f) return false;
    file_ = f;
    return OpenSink([f, channels](const float* p, int n) { std::fwrite(p, sizeof(float), size_t(n) * channels, f); },
                    channels);
  }
  bool OpenSink(Sink sink, int channels, int ring_frames = 4 * 48000) {
    sink_ = std::move(sink);
    channels_ = channels;
    ring_.Reset(ring_frames, channels);
    stop_.store(false);
    thread_ = std::thread([this] { Run(); });
    open_.store(true, std::memory_order_release);
    return true;
  }
  bool is_open() const { return open_.load(std::memory_order_acquire); }

  // The audio thread: n frames, without waiting.
  void Write(const float* frames, int n) {
    if (!is_open()) return;
    const int take = std::min(n, ring_.Free());
    ring_.Write(frames, take);
    if (take < n) dropped_.fetch_add(uint64_t(n - take), std::memory_order_relaxed);
  }
  uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

  // Writes what is left and closes the file.
  void Close() {
    if (!thread_.joinable()) return;
    open_.store(false, std::memory_order_release);
    stop_.store(true);
    thread_.join();
    if (file_) std::fclose(file_);
    file_ = nullptr;
    if (dropped()) std::fprintf(stderr, "audio: a dump dropped %llu frames\n", (unsigned long long)dropped());
  }

 private:
  void Run() {
    std::vector<float> buf(size_t(4096) * size_t(channels_));
    for (;;) {
      const bool last = stop_.load();
      for (int got; (got = ring_.Read(buf.data(), 4096)) > 0;) sink_(buf.data(), got);
      if (last) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  Sink sink_;
  FILE* file_ = nullptr;
  int channels_ = 2;
  FrameRing ring_;
  std::thread thread_;
  std::atomic<bool> stop_{false}, open_{false};
  std::atomic<uint64_t> dropped_{0};
};

}  // namespace recomp::audio
