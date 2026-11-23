#pragma once
// What a Direct3D buffer's contents are, for a graphics device that keeps
// work derived from them (gfx::Device::TracksBufferContents: acceleration
// structures for ray tracing): a version that changes whenever they may
// have (gfx::VertexStream::version), and whether they are rewritten frame
// after frame. Locks and unlocks may come from another thread than draws.
// Header only, so tests can run it without a device.

#include <atomic>
#include <cstdint>

namespace recomp::d3d9 {

// A new number of the series buffer identities and versions share: never
// 0, never reused.
inline uint64_t NewBufferVersion() {
  static std::atomic<uint64_t> last{0};
  return last.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Whether any device tracks buffer contents: unlocks do nothing otherwise.
inline std::atomic<bool>& BufferContentsTracked() {
  static std::atomic<bool> on{false};
  return on;
}

struct BufferContents {
  uint64_t id = 0;  // the buffer (gfx::VertexStream::buffer); 0: not tracked
  std::atomic<uint64_t> version{0};
  std::atomic<uint32_t> writing{0};  // locks that can write, not unlocked yet
  // Writing locks in consecutive frames, up to the last one's frame.
  std::atomic<uint64_t> write_frame{0};
  std::atomic<uint32_t> rewrites{0};

  // Tracked from now on (the buffer is new).
  void Track() {
    id = NewBufferVersion();
    version.store(NewBufferVersion(), std::memory_order_release);
  }
  uint64_t Version() const { return version.load(std::memory_order_acquire); }

  // A lock that can write, in graphics frame `frame`: the contents may
  // change from now on (renamed or not: a lock after the GPU is done writes
  // in place).
  void WriteLock(uint64_t frame) {
    if (!id) return;
    version.store(NewBufferVersion(), std::memory_order_release);
    writing.fetch_add(1, std::memory_order_relaxed);
    const uint64_t last = write_frame.load(std::memory_order_relaxed);
    if (frame != last) {
      rewrites.store(frame == last + 1 ? rewrites.load(std::memory_order_relaxed) + 1 : 1, std::memory_order_relaxed);
      write_frame.store(frame, std::memory_order_relaxed);
    }
  }
  // An unlock: one after a writing lock (nested ones each count) leaves
  // what was written, contents of their own; a draw between the lock and
  // the unlock is not taken for them.
  void Unlock() {
    uint32_t n = writing.load(std::memory_order_relaxed);
    while (n && !writing.compare_exchange_weak(n, n - 1, std::memory_order_relaxed)) {
    }
    if (n) version.store(NewBufferVersion(), std::memory_order_release);
  }
  // Written in three frames running, the last of them `frame` or the one
  // before.
  bool Rewritten(uint64_t frame) const {
    return rewrites.load(std::memory_order_relaxed) >= 3 && frame <= write_frame.load(std::memory_order_relaxed) + 1;
  }
};

}  // namespace recomp::d3d9
