// Use the real guest VM and heap to separate growth policy from exhaustion.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/guest_alloc.h"

using namespace recomp;

namespace {
void Check(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "Expected a heap growth scenario");
  const std::string scenario = argv[1];
  Check(scenario == "tail" || scenario == "vm_fragmented" ||
            scenario == "heap_fragmented" || scenario == "fixed" ||
            scenario == "exhausted" || scenario == "realloc",
        "Unknown heap growth scenario");

  constexpr uint64_t mib = 1ull << 20;
  // The stock streaming path failed this request despite a 14 MiB VM gap.
  constexpr uint64_t request = 175104;
  GuestMemory memory;
  Check(memory.Reserve(), "Cannot reserve guest arena");
  GuestAllocator vm;
  vm.Init(&memory, 0x10000,
          scenario == "heap_fragmented" ? 0x1010000 : 0x4000000);
  GuestHeap heap(vm, mib, scenario == "fixed" ? 16 * mib : 0);

  if (scenario == "heap_fragmented") {
    std::vector<uint64_t> blocks;
    for (int i = 0; i < 256; ++i) {
      const auto block = heap.Alloc(0x10000, false);
      Check(block != 0, "Cannot fill first segment");
      blocks.push_back(block);
    }
    for (int i = 0; i < 256; i += 2)
      Check(heap.Free(blocks[i]), "Cannot create heap fragmentation");
    Check(heap.Alloc(request, false) == 0,
          "Separate free blocks cannot satisfy a contiguous allocation");
  } else {
    const auto first = heap.Alloc(16 * mib, false);
    Check(first != 0, "Cannot fill first segment");
    if (scenario == "tail" || scenario == "realloc")
      Check(vm.Reserve(0x2000000, 32 * mib) != 0,
            "Cannot bound the remaining VM gap below 16 MiB");
    if (scenario == "vm_fragmented") {
      for (uint64_t address = 0x1400000; address < 0x4000000;
           address += 4 * mib)
        Check(vm.Reserve(address, 0x10000) != 0, "Cannot fragment guest VM");
      Check(heap.Alloc(8 * mib, false) == 0,
            "Aggregate VM space cannot replace a fitting gap");
    }
    if (scenario == "exhausted")
      Check(vm.Reserve(0x1010000, 0x2ff0000) != 0,
            "Cannot exhaust remaining VM");

    uint64_t block = 0;
    if (scenario == "realloc") {
      // A move must preserve data, zero the extension and release the old block.
      Check(heap.Free(first), "Cannot release setup block");
      const auto original = heap.Alloc(0x10000, false);
      Check(original != 0, "Cannot allocate realloc source");
      std::memset(memory.ptr(original), 0x5a, 0x10000);
      Check(heap.Alloc(16 * mib - 0x10000, false) != 0,
            "Cannot block in-place growth");
      Check(heap.ReAlloc(original, request, true, true) == 0,
            "In-place-only realloc must not move");
      block = heap.ReAlloc(original, request, true, false);
      Check(block != 0 && block != original, "Realloc did not use a smaller segment");
      Check(heap.Size(original) == UINT64_MAX, "Realloc retained the old allocation");
      for (uint64_t i = 0; i < request; ++i)
        Check(memory.ptr(block)[i] == (i < 0x10000 ? 0x5a : 0),
              "Realloc lost data or did not zero its extension");
    } else {
      block = heap.Alloc(request, true);
      if (scenario == "fixed" || scenario == "exhausted") {
        Check(block == 0, "Fixed heaps and real exhaustion must still fail");
        Check(heap.Size(first) == 16 * mib, "Failed growth changed a live block");
      } else {
        Check(block != 0, "Small request must fit the available VM gap");
        for (uint64_t i = 0; i < request; ++i)
          Check(memory.ptr(block)[i] == 0, "New allocation was not zeroed");
      }
    }
    if (block) {
      Check(heap.Size(block) == request, "HeapSize lost the requested size");
      std::memset(memory.ptr(block), 0xa5, request);
      Check(heap.Free(block), "Cannot free fallback allocation");
      Check(heap.Alloc(request, false) == block, "Cannot reuse fallback storage");
    }
  }
  std::printf("%s passed\n", scenario.c_str());
}
