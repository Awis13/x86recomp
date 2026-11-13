// Drive-relative file names must still reach a game's installed files.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <string>
#include <vector>
#include <unistd.h>

#include "runtime/runtime.h"
#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"

using namespace recomp;

int main() {
  namespace fs = std::filesystem;
  char temporary[] = "/tmp/recomp-file-paths-XXXXXX";
  if (!mkdtemp(temporary)) return 2;
  const fs::path root(temporary), install = root / "game";
  fs::create_directories(install / "pc/data");
  setenv("XDG_DATA_HOME", temporary, 1);
  g_proc = new Process;
  g_proc->game.id = "file_paths_test";
  g_proc->game.guest_dir = "C:\\Games\\Test";
  g_proc->game_dir = install.string();
  if (!g_proc->mem.Reserve(32)) return 2;
  g_proc->mem.Commit(kThunkBase, kThunkRegionSize, Prot::R);
  g_proc->mem.Commit(kPebVa, 0x1000, Prot::RW);
  g_proc->vm.Init(&g_proc->mem, 0x10000, kThunkBase);
  InitKernel32State();
  const uint32_t scratch = uint32_t(g_proc->vm.Reserve(0, 0x10000));
  g_proc->vm.Commit(scratch, 0x10000, Prot::RW);
  std::promise<int> done;
  auto result = done.get_future();
  StartServiceThread(CreateGuestThread(0x10000), [&] {
    auto call = [&](const char* name, const std::vector<uint32_t>& args) {
      Ctx c{};
      c.esp = scratch + 0xF000;
      Wr32(c.esp, 0x00401000);
      for (size_t i = 0; i < args.size(); ++i) Wr32(c.esp + 4 + 4*i, args[i]);
      hle::CallThunk(c, hle::ImportThunk("kernel32.dll", name));
      return c.eax;
    };
    int failures = 0;
    auto check = [&](const char* path, const fs::path& expected) {
      const std::string actual = HostPath(path);
      if (actual != expected.string()) {
        std::fprintf(stderr, "FAIL: %s resolved to %s, expected %s\n",
                     path, actual.c_str(), expected.c_str());
        ++failures;
      }
    };
    const fs::path recording = install / "pc/data/Benchmark.cli";
    if (FILE* f = std::fopen(recording.c_str(), "wb")) {
      std::fputs("replay fixture", f);
      std::fclose(f);
    } else { done.set_value(2); return; }
    check("C:.\\pc\\data\\Benchmark.cli", recording);
    check("c:pc/data/Benchmark.cli", recording);
    check("C:", install);
    check(".\\pc\\data\\Benchmark.cli", recording);
    check("C:\\Games\\Test\\pc\\data\\Benchmark.cli", recording);
    check("C:pc\\data\\..\\data\\Benchmark.cli", recording);
    // This call must open the existing recording, not just format a path.
    FILE* f = std::fopen(HostPath("C:.\\pc\\data\\Benchmark.cli").c_str(), "rb");
    if (!f) {
      std::fprintf(stderr, "FAIL: drive-relative benchmark recording cannot be opened\n");
      ++failures;
    } else {
      char text[32]{};
      std::fgets(text, sizeof text, f);
      std::fclose(f);
      if (std::string(text) != "replay fixture") ++failures;
    }
    const fs::path drive_d = root / "file_paths_test/drive_c/drive_d";
    fs::create_directories(drive_d / "Other");
    check("D:file.txt", drive_d / "file.txt");
    std::strcpy(reinterpret_cast<char*>(GPtr(scratch)), "C:.\\pc\\data\\Benchmark.cli");
    const uint32_t out = scratch + 0x100, filepart = scratch + 0x300;
    const std::string full = "C:\\Games\\Test\\pc\\data\\Benchmark.cli";
    if (call("GetFullPathNameA", {scratch, 256, out, filepart}) != full.size() ||
        GStr(out) != full || GStr(Rd32(filepart)) != "Benchmark.cli") {
      std::fprintf(stderr, "FAIL: GetFullPathNameA lost the drive-relative current directory\n");
      ++failures;
    }
    fs::remove_all(root);
    if (!failures) std::puts("file_paths_test: ok");
    done.set_value(failures ? 1 : 0);
  });
  return result.get();
}
