// Drive-relative opens must reach the same installed bytes as absolute paths.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <memory>
#include <string>
#include <unistd.h>

#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/runtime.h"

using namespace recomp;
namespace {
std::filesystem::path fixture;
void Check(bool ok, const char* message) {
  if (ok) return;
  std::fprintf(stderr, "%s\n", message);
  std::filesystem::remove_all(fixture);
  std::_Exit(1);
}
}

int main() {
  char directory[] = "/tmp/recomp-file-path-XXXXXX";
  Check(mkdtemp(directory) != nullptr, "Cannot create fixture");
  fixture = directory;
  const auto install = fixture / "install";
  std::filesystem::create_directories(install / "assets");
  std::ofstream(install / "assets" / "clip.dat") << "recorded data";
  setenv("XDG_DATA_HOME", directory, 1);
  auto process = std::make_unique<Process>();
  g_proc = process.get();
  process->game.id = "paths";
  process->game.guest_dir = "C:\\Program Files\\Fixture";
  process->game_dir = install.string();
  Check(process->mem.Reserve(), "Cannot reserve guest memory");
  process->vm.Init(&process->mem, 0x10000, 0x4000000);
  Check(process->mem.Commit(kPebVa, 0x1000, Prot::RW), "Cannot commit PEB");
  InitKernel32State();
  // File APIs set the current guest thread's TEB LastErrorValue.
  std::promise<void> finished;
  auto result = finished.get_future();
  StartServiceThread(CreateGuestThread(0x10000), [&] {
    const auto stack = static_cast<uint32_t>(HeapAllocProcess(4096, true));
    Check(stack != 0, "Cannot allocate call storage");
    const auto path = stack + 256, output = stack + 1024;
    Ctx context{};
    auto call = [&](const char* name, std::initializer_list<uint32_t> args) {
      context.esp = stack;
      Wr32(stack, kReturnToHost);
      uint32_t offset = 4;
      for (const auto arg : args) { Wr32(stack + offset, arg); offset += 4; }
      hle::CallThunk(context, hle::ThunkFor("kernel32.dll", name));
      Check(context.esp == stack + offset, "Incorrect stdcall argument count");
      return context.eax;
    };
    auto full = [&](const char* input, const char* expected) {
      std::strcpy(reinterpret_cast<char*>(GPtr(path)), input);
      const auto count = call("GetFullPathNameA", {path, 512, output, output + 512});
      if (GStr(output) != expected)
        std::fprintf(stderr, "%s resolved to %s, expected %s\n", input, GStr(output).c_str(), expected);
      Check(count == std::strlen(expected) && GStr(output) == expected,
            "Incorrect drive-relative full path");
    };
    full("C:.\\assets\\clip.dat", "C:\\Program Files\\Fixture\\assets\\clip.dat");
    full("c:assets/clip.dat", "C:\\Program Files\\Fixture\\assets\\clip.dat");
    full("C:", "C:\\Program Files\\Fixture");
    full("C:\\assets\\clip.dat", "C:\\assets\\clip.dat");
    full("D:clip.dat", "D:\\clip.dat");
    for (const char* input : {"C:.\\assets\\CLIP.DAT", "c:assets\\clip.dat", "assets\\clip.dat"}) {
      std::strcpy(reinterpret_cast<char*>(GPtr(path)), input);
      const auto handle = call("CreateFileA", {path, 0x80000000, 1, 0, 3, 0x80, 0});
      Check(handle != 0xFFFFFFFF, "Drive-relative CreateFileA missed installed data");
      Check(call("ReadFile", {handle, output, 13, output + 512, 0}) == 1 &&
                Rd32(output + 512) == 13 && std::memcmp(GPtr(output), "recorded data", 13) == 0,
            "Drive-relative ReadFile did not return the installed bytes");
      Check(call("CloseHandle", {handle}) == 1, "Cannot close fixture");
    }
    std::strcpy(reinterpret_cast<char*>(GPtr(path)), "C:assets");
    Check(call("SetCurrentDirectoryA", {path}) == 1, "Cannot set drive-relative cwd");
    full("C:clip.dat", "C:\\Program Files\\Fixture\\assets\\clip.dat");
    full("C:..\\assets\\clip.dat", "C:\\Program Files\\Fixture\\assets\\clip.dat");
    const auto other = std::filesystem::path(DataDir()) / "drive_c" / "drive_d" / "other";
    std::filesystem::create_directories(other);
    std::strcpy(reinterpret_cast<char*>(GPtr(path)), "D:\\other");
    Check(call("SetCurrentDirectoryA", {path}) == 1, "Cannot switch drive");
    full("\\root.dat", "D:\\root.dat");
    full("D:clip.dat", "D:\\other\\clip.dat");
    full("C:clip.dat", "C:\\Program Files\\Fixture\\assets\\clip.dat");
    std::strcpy(reinterpret_cast<char*>(GPtr(path)), "C:");
    Check(call("SetCurrentDirectoryA", {path}) == 1, "Cannot return to remembered drive cwd");
    Check(GuestCurrentDir() == "C:\\Program Files\\Fixture\\assets", "Drive cwd was not retained");
    Check(HostPath("C:clip.dat", PathAccess::kWrite) == DataDir() + "/install/assets/clip.dat",
          "Drive-relative writes escaped the install overlay");
    std::filesystem::remove_all(fixture);
    std::puts("Drive-relative paths, file reads, current directories and overlay passed");
    std::fflush(stdout);
    finished.set_value();
  });
  result.get();
  std::_Exit(0);
}
