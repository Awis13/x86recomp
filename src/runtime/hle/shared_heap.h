#pragma once
// A heap of guest memory a graphics device reads in place (vertex and
// index buffers, texture uploads): reserved and committed whole, as the
// device maps one range.
//
// The GPU reads a block for a few frames after the game is done with it,
// so blocks are not freed but retired, with the frame of their last use,
// and freed once the device has finished that frame. They are collected
// whenever the device has finished another frame, not when the heap is
// full: what is in use then is what the game holds plus the last frames'
// blocks, and that is what the heap's size has to cover. Every megabyte
// of it comes out of a 32-bit guest's 2 GiB.
//
// Each device maps the whole heap and counts its own frames, so a block
// is freed only by the device that retired it (its frame over), or by
// Forget once that device is gone.
//
// The device is named by a pointer and says itself which frame it has
// finished, so nothing here knows a graphics API
// (tests/shared_heap_test.cpp).

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime/guest_alloc.h"

namespace recomp {

class SharedHeap {
 public:
  // The heap's size when nothing sets it: for a 32-bit guest an eighth of
  // its address space, for a 64-bit one, which has the room, twice that.
  static constexpr uint64_t kDefault32 = 256ull << 20, kDefault64 = 512ull << 20;

  // Reserves `size` bytes of `vm` (16 MB at least: size() says); false if
  // it has no such gap. Once the heap exists, true and nothing else: the
  // devices already mapping it keep it (devices made on two threads).
  bool Init(GuestAllocator& vm, uint64_t size);
  explicit operator bool() const { return heap_ != nullptr; }
  uint64_t base() const { return heap_ ? heap_->handle() : 0; }
  uint64_t size() const { return size_; }

  // A block, or 0 if the heap is full. `user` is the device that will read
  // it and `done` the last frame that device has finished: its retired
  // blocks up to that frame are freed first. Other devices' blocks stay.
  uint64_t Alloc(const void* user, uint64_t done, uint64_t size, bool zero);
  // Gives back a block `user` reads until its frame `serial` is over
  // (`user` null: nobody reads it). Freed at once if `done` says it is.
  void Retire(const void* user, uint64_t done, uint64_t mem, uint64_t serial);
  // Frees every block `user` retired, whatever its frame: for a device
  // that is gone, once its GPU work is over.
  void Forget(const void* user);

  struct Stats {
    uint64_t used = 0;          // bytes in blocks, retired ones included
    uint64_t peak = 0;          // the most `used` has been
    uint64_t largest_free = 0;  // bytes
    uint64_t waiting = 0;       // retired blocks not yet freed
    uint64_t waiting_bytes = 0;
  };
  Stats GetStats() const;

 private:
  // Frees the blocks `user` retired up to its frame `done` (all of them
  // with `all`).
  void ReclaimLocked(const void* user, uint64_t done, bool all = false);

  struct Retired {
    uint64_t mem, size, serial;
    const void* user;
  };
  mutable std::mutex mu_;
  std::unique_ptr<GuestHeap> heap_;
  uint64_t size_ = 0;
  std::vector<Retired> retired_;
  uint64_t retired_bytes_ = 0;
  uint64_t peak_ = 0;
  // The last collection. Until either changes, an allocation that the
  // heap serves collects nothing; one it cannot serve collects again (a
  // block may have been retired since with a frame already over).
  const void* reclaimed_user_ = nullptr;
  uint64_t reclaimed_done_ = 0;
};

}  // namespace recomp
