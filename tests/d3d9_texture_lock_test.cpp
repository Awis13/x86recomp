// Exercise the real HLE with an exhausted guest heap, without a GPU or game data.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "runtime/hle/d3d9_impl.h"
#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/runtime.h"

using namespace recomp;

namespace {
void Check(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "%s\n", message);
  std::_Exit(1);
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "Expected texture, cube, or surface");
  const std::string kind = argv[1];
  Check(kind == "texture" || kind == "cube" || kind == "surface", "Unknown lock path");

  // Kernel32 keeps its heaps for the process lifetime. This fixture exits with
  // _Exit so their destructors cannot outlive the guest address space.
  auto* process = new Process;
  g_proc = process;
  process->game.id = "texture-lock-test";
  Check(process->mem.Reserve(), "Cannot reserve guest arena");
  process->vm.Init(&process->mem, 0x10000, 0x2000000);
  Check(process->mem.Commit(kPebVa, 0x1000, Prot::RW), "Cannot commit PEB");
  InitKernel32State();

  const auto stack = uint32_t(HeapAllocProcess(4096, true));
  Check(stack != 0, "Cannot allocate fixture stack");
  const uint32_t output = stack + 256;
  const bool cube = kind == "cube";
  d3d9::Device device;
  uint32_t texture = 0;
  Check(d3d9::NewTexture(device, 0, cube, 512, 512, 1, 0, d3d9::D3DFMT_DXT1,
                        d3d9::D3DPOOL_SYSTEMMEM, &texture) == d3d9::D3D_OK,
        "Cannot create fixture texture");

  Ctx context{};
  auto call = [&](const std::string& name, std::initializer_list<uint32_t> args) {
    context.esp = stack;
    Wr32(stack, kReturnToHost);
    uint32_t offset = 4;
    for (uint32_t arg : args) {
      Wr32(stack + offset, arg);
      offset += 4;
    }
    hle::CallThunk(context, hle::ThunkFor("d3d9.dll", name));
    return context.eax;
  };

  uint32_t surface = 0;
  if (kind == "surface") {
    Check(call("IDirect3DTexture9::GetSurfaceLevel", {texture, 0, output}) == d3d9::D3D_OK,
          "Cannot obtain texture surface");
    surface = Rd32(output);
  }
  auto lock = [&] {
    if (cube)
      return call("IDirect3DCubeTexture9::LockRect", {texture, 0, 0, output, 0, 0});
    if (surface)
      return call("IDirect3DSurface9::LockRect", {surface, output, 0, 0});
    return call("IDirect3DTexture9::LockRect", {texture, 0, output, 0, 0});
  };
  auto unlock = [&] {
    if (cube) return call("IDirect3DCubeTexture9::UnlockRect", {texture, 0, 0});
    if (surface) return call("IDirect3DSurface9::UnlockRect", {surface});
    return call("IDirect3DTexture9::UnlockRect", {texture, 0});
  };

  std::vector<gaddr> blocks;
  while (const auto block = HeapAllocProcess(4096, false)) blocks.push_back(block);
  auto& level = com::As<d3d9::Texture>(texture)->At(0, 0);
  Check(level.mem == 0, "Fixture already has lock storage");
  Wr32(output, 0x12345678);
  Wr32(output + 4, 0x87654321);
  const uint32_t result = lock();
  std::printf("%s: exhausted LockRect HRESULT=%08x pitch=%08x pBits=%08x\n",
              kind.c_str(), result, Rd32(output), Rd32(output + 4));
  std::fflush(stdout);
  Check(result == 0x8007000E, "LockRect must report E_OUTOFMEMORY, not success with null storage");
  Check(!level.locked && !level.mem, "Failed lock changed the level state");
  Check(Rd32(output) == 0x12345678 && Rd32(output + 4) == 0x87654321,
        "Failed lock published an invalid output");

  for (const auto block : blocks) HeapFreeProcess(block);
  Check(lock() == d3d9::D3D_OK, "Lock cannot be retried after allocation failure");
  const uint32_t pixels = Rd32(output + 4);
  Check(pixels != 0 && pixels == level.mem && Rd32(output) == level.pitch,
        "Successful retry did not publish its storage");
  for (uint64_t page = pixels & ~uint64_t(4095); page < uint64_t(pixels) + level.size; page += 4096)
    Check(process->mem.QueryProt(page) == Prot::RW, "Retry storage is not readable and writable");
  std::memset(GPtr(pixels), 0x5A, level.size);
  Check(unlock() == d3d9::D3D_OK && !level.locked, "Retry cannot be unlocked");
  Check(lock() == d3d9::D3D_OK && Rd32(output + 4) == pixels && Rd8(pixels) == 0x5A,
        "Existing lock storage was lost");
  Check(unlock() == d3d9::D3D_OK, "Final unlock failed");
  std::puts("allocation failure, unchanged outputs/state, retry, and storage reuse passed");
  std::fflush(stdout);
  std::_Exit(0);
}
