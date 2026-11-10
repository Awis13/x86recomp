#include "runtime/hle/shared_heap.h"

#include <algorithm>

namespace recomp {

bool SharedHeap::Init(GuestAllocator& vm, uint64_t size) {
  std::lock_guard<std::mutex> l(mu_);
  if (heap_) return true;
  auto heap = std::make_unique<GuestHeap>(vm, size, size);
  if (!heap->handle()) return false;
  heap_ = std::move(heap);
  size_ = heap_->GetStats().segments;  // (a heap is never under one segment, 16 MB)
  return true;
}

void SharedHeap::ReclaimLocked(const void* user, uint64_t done, bool all) {
  std::erase_if(retired_, [&](const Retired& r) {
    if (r.user != user || (!all && r.serial > done)) return false;
    heap_->Free(r.mem);
    retired_bytes_ -= r.size;
    return true;
  });
  if (all) return;
  reclaimed_user_ = user;
  reclaimed_done_ = done;
}

void SharedHeap::Forget(const void* user) {
  std::lock_guard<std::mutex> l(mu_);
  if (heap_ && user) ReclaimLocked(user, 0, true);
}

uint64_t SharedHeap::Alloc(const void* user, uint64_t done, uint64_t size, bool zero) {
  std::lock_guard<std::mutex> l(mu_);
  if (!heap_) return 0;
  if (!retired_.empty() && (done != reclaimed_done_ || user != reclaimed_user_))
    ReclaimLocked(user, done);
  uint64_t p = heap_->Alloc(size, zero);
  if (!p && !retired_.empty()) {
    ReclaimLocked(user, done);
    p = heap_->Alloc(size, zero);
  }
  if (p) peak_ = std::max(peak_, heap_->GetStats().used);
  return p;
}

void SharedHeap::Retire(const void* user, uint64_t done, uint64_t mem, uint64_t serial) {
  std::lock_guard<std::mutex> l(mu_);
  if (!heap_) return;
  if (!user || serial <= done) {
    heap_->Free(mem);
    return;
  }
  const uint64_t size = heap_->Size(mem);
  if (size == UINT64_MAX) return;  // not a block of this heap
  retired_.push_back({mem, size, serial, user});
  retired_bytes_ += size;
}

SharedHeap::Stats SharedHeap::GetStats() const {
  std::lock_guard<std::mutex> l(mu_);
  Stats s;
  if (!heap_) return s;
  const GuestHeap::Stats h = heap_->GetStats();
  s.used = h.used;
  s.peak = peak_;
  s.largest_free = h.largest_free;
  s.waiting = retired_.size();
  s.waiting_bytes = retired_bytes_;
  return s;
}

}  // namespace recomp
