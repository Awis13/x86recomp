// shared_heap_test: the heap a graphics device reads in place
// (runtime/hle/shared_heap.h) over real guest memory, and the statistics
// <ID>_MEM_REPORT prints (GuestAllocator::GetStats, GuestHeap::GetStats).
// Checks that:
//   - the address space's reserved and free bytes and its largest gap
//     follow reservations and releases, and a heap's segments, bytes in
//     blocks and largest free block follow Alloc, Free and ReAlloc;
//   - a heap still grows in an address space with no gap of a whole
//     segment (16 MB) left, by a segment just big enough;
//   - the shared heap takes its whole size from the address space once, and
//     nothing more whatever is allocated; a size with no gap for it fails;
//   - a block retired in a frame the device has finished, or with no
//     device, is free at once; one of a later frame stays (its bytes
//     untouched) until the device finishes that frame, and is then freed by
//     the next Alloc, without the heap being full;
//   - a game that renames a buffer every frame (Alloc, Retire the old one)
//     keeps the heap at what it holds plus the frames in flight, for any
//     number of frames (collected only when full, the heap would fill);
//   - a full heap gives 0 while the frames are in flight and a block again
//     once they are over;
//   - an allocation for one device leaves another device's retired blocks
//     alone (their bytes too), whatever frames the first has finished;
//     Forget frees them;
//   - a block retired with a frame already over by the time of the last
//     collection is collected when the heap is full (the done the caller
//     read may be older than the one the heap last collected at);
//   - Init once the heap exists keeps it (two devices made at once);
//   - Rename gives a block that a frame in flight reads a new one, with its
//     bytes when asked: in the heap while it has room, then in the
//     fallback heap, and 0 (the old block untouched, not retired) when
//     neither has.
// Scenarios come from raw mt19937 output, which the standard fixes.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "runtime/guest_alloc.h"
#include "runtime/guest_memory.h"
#include "runtime/hle/shared_heap.h"

using namespace recomp;

namespace {

GuestMemory g_mem;
int g_failures = 0;

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                  \
      g_failures++;                                                         \
    }                                                                       \
  } while (0)

constexpr uint64_t kMb = 1 << 20;
constexpr uint64_t kLo = 0x10000, kHi = 0x80000000u;

void TestStats() {
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  GuestAllocator::Stats s = vm.GetStats();
  CHECK(s.reserved == 0 && s.free == kHi - kLo && s.largest_free == kHi - kLo);

  // An image in the middle splits the space in two gaps.
  vm.MarkReserved(0x40000000, 64 * kMb);
  s = vm.GetStats();
  CHECK(s.reserved == 64 * kMb && s.free == kHi - kLo - 64 * kMb);
  CHECK(s.largest_free == 0x40000000 - kLo);

  const uint64_t a = vm.Reserve(0, 256 * kMb), b = vm.Reserve(0, 16 * kMb);
  CHECK(a && b);
  s = vm.GetStats();
  CHECK(s.reserved == (64 + 256 + 16) * kMb);
  CHECK(s.reserved + s.free == kHi - kLo);
  CHECK(s.largest_free == kHi - 0x40000000 - 64 * kMb);  // above the image now
  CHECK(vm.Release(a));
  s = vm.GetStats();
  CHECK(s.reserved == (64 + 16) * kMb && s.reserved + s.free == kHi - kLo);
  // The freed 256 MB is a gap of its own, below b.
  CHECK(vm.Reserve(0, 256 * kMb) == a);

  // A heap: 16 MB segments, bytes in blocks, the largest free block.
  {
    GuestHeap h(vm, kMb, 0);
    GuestHeap::Stats hs = h.GetStats();
    CHECK(hs.segments == 16 * kMb && hs.used == 0 && hs.largest_free == 16 * kMb);
    const uint64_t p = h.Alloc(1000, false), q = h.Alloc(4 * kMb, false);
    hs = h.GetStats();
    CHECK(hs.used == 1008 + 4 * kMb);  // blocks are rounded to 16 bytes
    CHECK(hs.largest_free == 16 * kMb - hs.used);
    CHECK(h.ReAlloc(p, 3000, false, false));
    hs = h.GetStats();
    CHECK(hs.used >= 3008 + 4 * kMb && hs.used < 8000 + 4 * kMb);
    CHECK(h.Free(q));
    CHECK(h.GetStats().used == hs.used - 4 * kMb);
    CHECK(h.Alloc(20 * kMb, false));  // a second segment
    hs = h.GetStats();
    CHECK(hs.segments == 36 * kMb);
    CHECK(vm.GetStats().reserved == (64 + 16 + 256 + 36) * kMb);
  }
  CHECK(vm.GetStats().reserved == (64 + 16 + 256) * kMb);  // the heap gave its segments back
}

// An address space whose gaps are all 8 MB: a growable heap made there,
// and its allocations up to that size, take segments just big enough.
void TestFragmented() {
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  std::vector<uint64_t> chunks;
  while (uint64_t c = vm.Reserve(0, 8 * kMb)) chunks.push_back(c);
  // Every other one back (the odd ones: the rest at the top, under 8 MB,
  // stays a gap of its own).
  for (size_t i = 1; i < chunks.size(); i += 2) CHECK(vm.Release(chunks[i]));
  const GuestAllocator::Stats before = vm.GetStats();
  CHECK(before.largest_free == 8 * kMb);
  {
    GuestHeap h(vm, kMb, 0);
    CHECK(h.handle() != 0);
    CHECK(h.GetStats().segments == kMb);
    const uint64_t p = h.Alloc(512 << 10, true), q = h.Alloc(6 * kMb, true);
    CHECK(p && q);
    CHECK(h.GetStats().segments == 7 * kMb);
    CHECK(!h.Alloc(9 * kMb, false));  // no gap that big
    CHECK(h.Free(p) && h.Free(q));
  }
  CHECK(vm.GetStats().reserved == before.reserved);
}

void Fill(uint64_t p, uint8_t v, size_t n) { std::memset(g_mem.ptr(p), v, n); }
bool AllEqual(uint64_t p, uint8_t v, size_t n) {
  const uint8_t* q = g_mem.ptr(p);
  for (size_t i = 0; i < n; i++)
    if (q[i] != v) return false;
  return true;
}

void TestRetire() {
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  {
    SharedHeap none;
    CHECK(!none && none.Alloc(nullptr, 0, 16, false) == 0);
    CHECK(!none.Init(vm, 3072 * kMb));  // more than the guest has
    CHECK(!none && vm.GetStats().reserved == 0);
    // Never under a heap segment, and size() is what was reserved.
    CHECK(none.Init(vm, kMb) && none.size() == 16 * kMb);
    CHECK(vm.GetStats().reserved == 16 * kMb);
  }
  CHECK(vm.GetStats().reserved == 0);
  SharedHeap h;
  CHECK(h.Init(vm, 32 * kMb));
  CHECK(h && h.size() == 32 * kMb && h.base() >= kLo);
  CHECK(vm.GetStats().reserved == 32 * kMb);
  int dev_a = 0, dev_b = 0;  // two devices, by address

  // Finished frames, or no device: free at once.
  uint64_t p = h.Alloc(&dev_a, 0, kMb, true);
  CHECK(p >= h.base() && p + kMb <= h.base() + h.size());
  CHECK(AllEqual(p, 0, kMb));
  h.Retire(&dev_a, 5, p, 5);
  CHECK(h.GetStats().used == 0 && h.GetStats().waiting == 0);
  p = h.Alloc(&dev_a, 5, kMb, false);
  h.Retire(nullptr, 0, p, 99);
  CHECK(h.GetStats().used == 0);

  // A frame in flight: the block stays, with its bytes.
  p = h.Alloc(&dev_a, 5, kMb, false);
  Fill(p, 0x5A, kMb);
  h.Retire(&dev_a, 5, p, 7);
  SharedHeap::Stats s = h.GetStats();
  CHECK(s.used == kMb && s.peak == kMb && s.waiting == 1 && s.waiting_bytes == kMb);
  // Allocations while frame 7 is not over do not get it...
  uint64_t q = h.Alloc(&dev_a, 6, kMb, false);
  CHECK(q && q != p && AllEqual(p, 0x5A, kMb));
  CHECK(h.GetStats().waiting == 1);
  // ... the first one after it is does (the heap is far from full).
  uint64_t r = h.Alloc(&dev_a, 7, 64, false);
  s = h.GetStats();
  CHECK(r && s.waiting == 0 && s.waiting_bytes == 0 && s.used == kMb + 64);
  h.Retire(&dev_a, 7, q, 0);
  h.Retire(&dev_a, 7, r, 0);
  CHECK(h.GetStats().used == 0);

  // Full: 0 while the frames are in flight, a block when they are over.
  std::vector<uint64_t> blocks;
  for (int i = 0; i < 8; i++) blocks.push_back(h.Alloc(&dev_a, 7, 4 * kMb, false));
  for (uint64_t b : blocks) CHECK(b);
  CHECK(h.Alloc(&dev_a, 7, kMb, false) == 0);
  CHECK(h.GetStats().largest_free == 0);
  for (uint64_t b : blocks) h.Retire(&dev_a, 7, b, 9);
  CHECK(h.GetStats().waiting == 8);
  CHECK(h.Alloc(&dev_a, 8, kMb, false) == 0);
  p = h.Alloc(&dev_a, 9, kMb, false);
  CHECK(p && h.GetStats().waiting == 0 && h.GetStats().used == kMb);
  CHECK(h.GetStats().peak == 32 * kMb);  // it was full

  // Another device draws from the same heap: its allocations, at any of
  // its frames, leave dev_a's blocks waiting with their bytes; the heap
  // being full does not change that.
  Fill(p, 0xA5, kMb);
  h.Retire(&dev_a, 9, p, 1000);
  CHECK(h.GetStats().waiting == 1);
  uint64_t b1 = h.Alloc(&dev_b, 0, kMb, false);
  uint64_t b2 = h.Alloc(&dev_b, 5000, kMb, false);
  CHECK(b1 && b2 && b1 != p && b2 != p);
  CHECK(h.GetStats().waiting == 1 && AllEqual(p, 0xA5, kMb));
  std::vector<uint64_t> rest;
  while (uint64_t b = h.Alloc(&dev_b, 5000, kMb, false)) rest.push_back(b);
  CHECK(h.GetStats().waiting == 1 && AllEqual(p, 0xA5, kMb));
  CHECK(std::find(rest.begin(), rest.end(), p) == rest.end());
  for (uint64_t b : rest) h.Retire(&dev_b, 5000, b, 0);
  h.Retire(&dev_b, 5000, b1, 0);
  h.Retire(&dev_b, 5000, b2, 0);
  CHECK(h.GetStats().used == kMb);
  // dev_a is gone (its GPU work over): Forget frees its blocks, and only
  // its own.
  h.Retire(&dev_b, 0, h.Alloc(&dev_b, 0, kMb, false), 3);
  h.Forget(&dev_a);
  CHECK(h.GetStats().waiting == 1 && h.GetStats().used == kMb);
  h.Forget(&dev_b);
  CHECK(h.GetStats().waiting == 0 && h.GetStats().used == 0);
  CHECK(vm.GetStats().reserved == 32 * kMb);  // never more than its size

  // Init again (a second device made at the same time): the same heap.
  const uint64_t base = h.base();
  q = h.Alloc(&dev_a, 0, kMb, false);
  Fill(q, 0x3C, kMb);
  CHECK(h.Init(vm, 64 * kMb));
  CHECK(h.base() == base && h.size() == 32 * kMb);
  CHECK(vm.GetStats().reserved == 32 * kMb);
  CHECK(AllEqual(q, 0x3C, kMb) && h.GetStats().used == kMb);
  h.Retire(nullptr, 0, q, 0);
}

// A block retired with a `done` older than the one the heap last
// collected at (two threads: one read the device's frame, the other
// collected at a later one first) is collected when the heap is full.
void TestStaleDone() {
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  SharedHeap h;
  CHECK(h.Init(vm, 32 * kMb));
  int dev = 0;
  std::vector<uint64_t> blocks;
  for (int i = 0; i < 8; i++) blocks.push_back(h.Alloc(&dev, 5, 4 * kMb, false));
  for (uint64_t b : blocks) CHECK(b);
  h.Retire(&dev, 5, blocks[0], 6);
  h.Retire(&dev, 5, blocks[1], 7);
  // The device has finished frame 6: blocks[0] goes to this allocation,
  // and the heap has collected at 6.
  const uint64_t q = h.Alloc(&dev, 6, 4 * kMb, false);
  CHECK(q == blocks[0]);
  // Retired with done 5 read before that collection: it waits.
  h.Retire(&dev, 5, q, 6);
  CHECK(h.GetStats().waiting == 2);
  // Full, at the frame it last collected at: frame 6 is over, so q is
  // free to take.
  CHECK(h.Alloc(&dev, 6, 4 * kMb, false) == q);
  CHECK(h.GetStats().waiting == 1);
}

void TestRename() {
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  SharedHeap h;
  CHECK(h.Init(vm, 16 * kMb));
  GuestHeap fallback(vm, 16 * kMb, 16 * kMb);
  CHECK(fallback.handle());
  int dev = 0;
  auto in_heap = [&](uint64_t p) { return p >= h.base() && p < h.base() + h.size(); };

  // Room in the heap: a block of it, the bytes copied, the old one waiting
  // for frame 3 with its bytes.
  const uint64_t a = h.Alloc(&dev, 0, 4 * kMb, false);
  Fill(a, 0x11, 4 * kMb);
  SharedHeap::Renamed n = h.Rename(&dev, 0, a, 4 * kMb, 3, true, fallback);
  CHECK(n.mem && n.shared && in_heap(n.mem) && n.mem != a);
  CHECK(AllEqual(n.mem, 0x11, 4 * kMb) && AllEqual(a, 0x11, 4 * kMb));
  CHECK(h.GetStats().waiting == 1 && h.GetStats().waiting_bytes == 4 * kMb);
  const uint64_t live = n.mem;

  // Full (two blocks of 4 MB waiting, two live): the fallback heap.
  const uint64_t b = h.Alloc(&dev, 0, 4 * kMb, false), c = h.Alloc(&dev, 0, 4 * kMb, false);
  CHECK(b && c && !h.Alloc(&dev, 0, kMb, false));
  Fill(b, 0x22, 4 * kMb);
  n = h.Rename(&dev, 0, b, 4 * kMb, 3, true, fallback);
  CHECK(n.mem && !n.shared && !in_heap(n.mem));
  CHECK(n.mem >= fallback.handle() && n.mem < fallback.handle() + 16 * kMb);
  CHECK(AllEqual(n.mem, 0x22, 4 * kMb) && AllEqual(b, 0x22, 4 * kMb));
  CHECK(h.GetStats().waiting == 2);
  // Without keep (DISCARD) nothing is copied, still a fresh block.
  Fill(c, 0x33, 4 * kMb);
  const uint64_t m = h.Rename(&dev, 0, c, 4 * kMb, 3, false, fallback).mem;
  CHECK(m && !in_heap(m) && m != n.mem);
  CHECK(h.GetStats().waiting == 3);

  // Neither heap has room (the shared one is full of waiting blocks and
  // `live`): 0, and the old block stays live, unretired.
  while (fallback.Alloc(kMb, false)) {
  }
  Fill(live, 0x44, 4 * kMb);
  n = h.Rename(&dev, 0, live, 4 * kMb, 3, true, fallback);
  CHECK(!n.mem && !n.shared);
  CHECK(h.GetStats().waiting == 3 && AllEqual(live, 0x44, 4 * kMb));
  CHECK(h.GetStats().used == 16 * kMb);

  // Frame 3 over: the retired blocks are what the next allocation gets.
  CHECK(h.Alloc(&dev, 3, 4 * kMb, false));
  CHECK(h.GetStats().waiting == 0);
}

// A game renaming buffers: each frame it locks some of its buffers with
// DISCARD (a new block, the old one retired with the current frame), and
// the device finishes frames `lag` behind.
void TestRenames(uint32_t seed) {
  std::mt19937 rng(seed);
  GuestAllocator vm;
  vm.Init(&g_mem, kLo, kHi);
  SharedHeap h;
  CHECK(h.Init(vm, 16 * kMb));
  int dev = 0;
  struct Buf {
    uint64_t mem;
    uint32_t size;
    uint8_t value;
  };
  std::vector<Buf> bufs;
  uint64_t held = 0;
  for (int i = 0; i < 40; i++) {
    Buf b{0, 4096u << (rng() % 5), uint8_t(rng())};  // 4 to 64 KiB
    b.mem = h.Alloc(&dev, 0, b.size, false);
    CHECK(b.mem);
    Fill(b.mem, b.value, b.size);
    held += b.size;
    bufs.push_back(b);
  }
  const uint64_t lag = 3;
  uint64_t peak = 0, peak_frame_bytes = 0;
  for (uint64_t frame = 1; frame <= 3000; frame++) {
    const uint64_t done = frame > lag ? frame - lag : 0;
    uint64_t frame_bytes = 0;
    for (uint32_t n = 4 + rng() % 8; n; n--) {
      Buf& b = bufs[rng() % bufs.size()];
      CHECK(AllEqual(b.mem, b.value, b.size));  // nobody else got its block
      const uint64_t fresh = h.Alloc(&dev, done, b.size, false);
      CHECK(fresh);
      if (!fresh) return;
      h.Retire(&dev, done, b.mem, frame);
      b.mem = fresh;
      b.value = uint8_t(rng());
      Fill(b.mem, b.value, b.size);
      frame_bytes += b.size;
    }
    peak_frame_bytes = std::max(peak_frame_bytes, frame_bytes);
    peak = std::max(peak, h.GetStats().used);
  }
  // What the game holds, and the blocks of the frames not finished (the
  // current one and `lag` before it, and one more collected a call late).
  CHECK(peak <= held + (lag + 2) * peak_frame_bytes + 40 * 16);
  CHECK(peak < 16 * kMb / 2);
  for (const Buf& b : bufs) CHECK(AllEqual(b.mem, b.value, b.size));
}

}  // namespace

int main() {
  if (!g_mem.Reserve()) return 1;
  TestStats();
  TestFragmented();
  TestRetire();
  TestStaleDone();
  TestRename();
  for (uint32_t seed = 1; seed <= 4; seed++) TestRenames(seed);
  if (g_failures) {
    std::fprintf(stderr, "shared_heap_test: %d failures\n", g_failures);
    return 1;
  }
  std::printf("shared_heap_test: ok\n");
  return 0;
}
