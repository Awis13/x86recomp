// Load a synthetic PE through the real process loader, then exercise the
// GlobalAlloc path for resource allocations above the legacy address range.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>
#include <unistd.h>

#include "runtime/hle/hle.h"
#include "runtime/lifted.h"
#include "runtime/runtime.h"

using namespace recomp;

namespace {
bool large_address_aware = false;
std::filesystem::path fixture;

void Check(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "%s\n", message);
  std::filesystem::remove_all(fixture);
  std::_Exit(1);
}

void HighStackEntry(Ctx& c) {
  Check(c.esp >= 0x80000000 && Rd32(c.esp + 4) == 0xA1B2C3D4,
        "Guest call lost its high stack or argument");
  c.eax = 0x87654321;
  c.esp += 8;
}

void Entry(Ctx& c) {
  const auto stack = c.esp - 256;
  auto call = [&](const char* name, std::initializer_list<uint32_t> args) {
    c.esp = stack;
    Wr32(stack, kReturnToHost);
    uint32_t offset = 4;
    for (const auto arg : args) {
      Wr32(stack + offset, arg);
      offset += 4;
    }
    hle::CallThunk(c, hle::ThunkFor("kernel32.dll", name));
    return c.eax;
  };

  const uint32_t output = stack + 128;
  call("GetSystemInfo", {output});
  Check(Rd32(output + 12) == (large_address_aware ? 0xFFFEFFFFu : 0x7FFEFFFFu),
        "GetSystemInfo does not match the executable's address range");
  call("GlobalMemoryStatus", {output});
  Check(Rd32(output + 24) == (large_address_aware ? 0xFFFE0000u : 0x7FFE0000u),
        "GlobalMemoryStatus does not match the executable's address range");
  Wr32(output, 64);
  Check(call("GlobalMemoryStatusEx", {output}) == 1, "Cannot query MEMORYSTATUSEX");
  Check(Rd64(output + 40) == (large_address_aware ? 0xFFFE0000ull : 0x7FFE0000ull),
        "GlobalMemoryStatusEx does not match the executable's address range");
  for (const uint32_t guard : {uint32_t(kThunkBase), kPebVa & ~0xFFFFu, 0xFFFF0000u})
    Check(call("VirtualAlloc", {guard, 0x10000, 0x2000, 4}) == 0,
          "VirtualAlloc overwrote the runtime or the upper guard");

  // Reserve all remaining low VM without committing/touching 2 GiB. Live
  // image, process heap, stack and runtime reservations stay intact.
  auto& vm = g_proc->vm;
  for (uint64_t address = 0x10000; address < 0x80000000;) {
    GuestAllocator::Region region;
    if (vm.Find(address, &region)) {
      address = region.base + region.size;
      continue;
    }
    uint64_t base = 0, size = 0;
    vm.FreeGap(address, &base, &size);
    if (!size) break;
    const auto end = std::min(base + size, uint64_t(0x80000000));
    base = (base + 0xFFFF) & ~uint64_t(0xFFFF);
    if (end > base)
      Check(vm.Reserve(base, end - base) == base, "Cannot fill low address space");
    address = end;
  }

  bool high = false, failed = false;
  std::vector<uint32_t> blocks;
  constexpr uint32_t request = 4194304;
  for (int i = 0; i < 8; ++i) {
    const auto block = call("GlobalAlloc", {0, request});
    if (!block) { failed = true; break; }
    high |= block >= 0x80000000;
    Check(call("GlobalLock", {block}) == block, "GlobalLock truncated a pointer");
    Check(call("GlobalSize", {block}) == request, "GlobalSize lost allocation size");
    Check(call("VirtualQuery", {block, output, 28}) == 28 &&
              Rd32(output + 16) == 0x1000,
          "VirtualQuery did not report committed resource storage");
    Wr32(block, 0x12345678);
    Wr32(block + request - 4, 0x87654321);
    Check(Rd32(block) == 0x12345678 && Rd32(block + request - 4) == 0x87654321,
          "Cannot access both ends of the allocation");
    blocks.push_back(block);
  }
  std::printf("%s: GlobalAlloc(4194304) failed=%d high_pointer=%d\n",
              large_address_aware ? "LAA" : "non-LAA", failed, high);
  std::fflush(stdout);
  Check(large_address_aware ? !failed && high : failed && !high,
        "RESC10: LAA resource allocation must continue above 2 GiB; non-LAA must stay below it");
  for (const auto block : blocks) Check(call("GlobalFree", {block}) == 0, "Cannot free resource");
  if (large_address_aware) {
    const auto top = call("VirtualAlloc", {0, request, 0x103000, 4});
    Check(top >= 0x80000000 && uint64_t(top) + request <= 0xFFFF0000ull,
          "MEM_TOP_DOWN did not use valid high storage");
    Wr32(top + request - 4, 0xABCDEF01);
    Check(call("VirtualFree", {top, 0, 0x8000}) == 1, "Cannot release high reservation");
    auto* thread = CreateGuestThread(0x100000);
    Check(thread->stack_lo >= 0x80000000 && thread->teb < kTebTop &&
              Rd32(thread->teb + 0x18) == thread->teb &&
              Rd32(thread->teb + 0x30) == kPebVa,
          "High thread stack or reserved TEB was corrupted");
    Check(CallGuest(thread->ctx, 0x401010, {0xA1B2C3D4}) == 0x87654321,
          "Guest call did not return from a high stack");
  }
  std::filesystem::remove_all(fixture);
  std::puts("loader and resource allocation passed");
  std::fflush(stdout);
  std::_Exit(0);  // Process heaps live until process exit, as in the game.
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "Expected laa or standard");
  large_address_aware = std::string(argv[1]) == "laa";
  Check(large_address_aware || std::string(argv[1]) == "standard", "Unknown scenario");
  std::array<char, 64> directory{};
  std::strcpy(directory.data(), "/tmp/recomp-address-space-XXXXXX");
  Check(mkdtemp(directory.data()) != nullptr, "Cannot create fixture directory");
  fixture = directory.data();
  setenv("XDG_DATA_HOME", directory.data(), 1);

  // One .text section and a registered host entry; no retail game data.
  std::array<char, 0x400> pe{};
  auto put = [&]<typename T>(size_t offset, T value) {
    std::memcpy(pe.data() + offset, &value, sizeof value);
  };
  put(0, uint16_t(0x5A4D));
  put(0x3C, uint32_t(0x80));
  put(0x80, uint32_t(0x4550));
  put(0x84, uint16_t(0x14C));
  put(0x86, uint16_t(1));
  put(0x94, uint16_t(0xE0));
  put(0x96, uint16_t(large_address_aware ? 0x122 : 0x102));
  constexpr size_t optional = 0x98;
  put(optional, uint16_t(0x10B));
  put(optional + 16, uint32_t(0x1000));
  put(optional + 28, uint32_t(0x400000));
  put(optional + 56, uint32_t(0x2000));
  put(optional + 60, uint32_t(0x200));
  put(optional + 72, uint32_t(0x100000));
  constexpr size_t section = optional + 0xE0;
  std::memcpy(pe.data() + section, ".text", 5);
  put(section + 8, uint32_t(0x1000));
  put(section + 12, uint32_t(0x1000));
  put(section + 16, uint32_t(0x200));
  put(section + 20, uint32_t(0x200));
  put(section + 36, uint32_t(0x60000020));
  const auto executable = fixture / "fixture.exe";
  { std::ofstream output(executable, std::ios::binary); output.write(pe.data(), pe.size()); }
  const LiftedFn functions[] = {{0x401000, Entry}, {0x401010, HighStackEntry}};
  RegisterLiftedFunctions(functions, 2);
  GameConfig game;
  game.id = "address-space-test";
  game.exe_name = "fixture.exe";
  game.guest_dir = "C:\\fixture";
  return RunProcess(game, executable.string(), fixture.string());
}
