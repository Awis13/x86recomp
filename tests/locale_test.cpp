// locale_test: the user's language as kernel32 tells it to a 32-bit
// guest (GetUserDefaultLCID, GetUserDefaultLangID, GetUserDefaultUILanguage,
// GetSystemDefaultUILanguage, IsValidLocale, EnumSystemLocalesA), called
// through their import thunks, and the parts of hle/locale.h they are
// built from.
//
//   locale_test <port's user_lang_id> <expected language>   (hex)
//
// The language is read once per process, so each case is a run of its
// own (tests/CMakeLists.txt): the default, a port's language, and
// LOCALETEST_LANG_ID over the port's, valid and malformed.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/hle/locale.h"
#include "runtime/runtime.h"

using namespace recomp;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      g_failures++;                                                                 \
    }                                                                               \
  } while (0)

constexpr uint32_t kReturn = 0x00401000;  // never run: the handlers return here

// The guest thread: registers and a stack.
Ctx g_ctx{};
uint32_t g_stack_top = 0;

// A stdcall to `fn` with `args`, as guest code makes it.
uint32_t Call(gaddr fn, const std::vector<uint32_t>& args) {
  uint32_t sp = g_stack_top - 4 * uint32_t(args.size());
  for (size_t i = 0; i < args.size(); i++) Wr32(sp + 4 * uint32_t(i), args[i]);
  sp -= 4;
  Wr32(sp, kReturn);
  g_ctx.esp = sp;
  hle::CallThunk(g_ctx, fn);
  CHECK(g_ctx.esp == g_stack_top);  // all arguments popped
  return g_ctx.eax;
}

uint32_t Kernel32(const char* name, const std::vector<uint32_t>& args = {}) {
  return Call(hle::ImportThunk("kernel32.dll", name), args);
}

// The guest's EnumSystemLocalesA callback: records each name and returns
// TRUE until it has seen `g_enum_stop_after` of them (0: never stops).
std::vector<std::string> g_enum_names;
size_t g_enum_stop_after = 0;
HLE("locale_test.dll", EnumLocalesProc, 1) {
  g_enum_names.push_back(GStr(hle::Arg(c, 0)));
  return g_enum_stop_after && g_enum_names.size() >= g_enum_stop_after ? 0 : 1;
}

std::vector<std::string> Enumerate(uint32_t flags, size_t stop_after) {
  g_enum_names.clear();
  g_enum_stop_after = stop_after;
  const uint32_t proc = uint32_t(hle::ImportThunk("locale_test.dll", "EnumLocalesProc"));
  CHECK(Kernel32("EnumSystemLocalesA", {proc, flags}) == 1);
  return g_enum_names;
}

std::string LcidName(uint32_t lcid) {
  char s[16];
  std::snprintf(s, sizeof s, "%08x", unsigned(lcid));
  return s;
}

// hle/locale.h on its own.
void CheckParts() {
  uint16_t v = 0x1234;
  CHECK(ParseLangId("0x0419", &v) && v == 0x0419);
  CHECK(ParseLangId("0X419", &v) && v == 0x0419);
  CHECK(ParseLangId("0x0407", &v) && v == 0x0407);
  CHECK(ParseLangId("0x0c0a", &v) && v == 0x0C0A);  // Spanish (Spain), modern sort
  CHECK(ParseLangId("0xC0A", &v) && v == 0x0C0A);
  v = 0x1234;
  for (const char* bad : {"", "0x", "x0419", "0x00419", "10419", "0x0419 ", " 0x0419",
                          "0x04g9", "-419", "russian", "0x0000", "0x0400", "0xFC00",
                          "0419", "1049", "C0A",  // no 0x: a decimal habit is an error
                          "0x0019", "0x19", "0x9"})  // neutral: no sublanguage
    CHECK(!ParseLangId(bad, &v));
  CHECK(!ParseLangId(nullptr, &v));
  CHECK(v == 0x1234);  // unchanged by what failed

  CHECK(IsLangId(0x0409) && IsLangId(0x0419) && IsLangId(0x0401) && IsLangId(0xFC01));
  CHECK(!IsLangId(0) && !IsLangId(0x0400) && !IsLangId(0x10419));
  CHECK(!IsLangId(0x0019) && !IsLangId(0x0009) && !IsLangId(0x03FF));  // neutral
  CHECK(LcidOf(0x0419) == 0x00000419);

  CHECK(InstalledLocales(0x0409) == std::vector<uint32_t>{0x0409});
  CHECK((InstalledLocales(0x0419) == std::vector<uint32_t>{0x0409, 0x0419}));
  CHECK((InstalledLocales(0x0407) == std::vector<uint32_t>{0x0407, 0x0409}));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: locale_test <port's user_lang_id> <expected language>\n");
    return 2;
  }
  const uint16_t configured = uint16_t(std::strtoul(argv[1], nullptr, 16));
  const uint16_t lang = uint16_t(std::strtoul(argv[2], nullptr, 16));

  CheckParts();

  // A 32-bit process with its heap and a guest stack.
  g_proc = new Process;
  g_proc->game.id = "localetest";
  g_proc->game.guest_dir = "C:\\Test";
  g_proc->game.exe_name = "test.exe";
  g_proc->game.user_lang_id = configured;
  if (!g_proc->mem.Reserve(32)) return 1;
  g_proc->mem.Commit(kThunkBase, kThunkRegionSize, Prot::R);
  g_proc->mem.Commit(kPebVa, 0x1000, Prot::RW);
  g_proc->vm.Init(&g_proc->mem, 0x00010000, kThunkBase);
  InitKernel32State();
  const uint32_t stack = uint32_t(g_proc->vm.Reserve(0, 0x10000));
  g_proc->vm.Commit(stack, 0x10000, Prot::RW);
  g_stack_top = stack + 0x10000 - 0x100;

  // The four default languages are the one language.
  CHECK(Kernel32("GetUserDefaultLCID") == lang);
  CHECK(Kernel32("GetUserDefaultLangID") == lang);
  CHECK(Kernel32("GetUserDefaultUILanguage") == lang);
  CHECK(Kernel32("GetSystemDefaultUILanguage") == lang);

  // IsValidLocale: the user's locale and US English, whatever the flags
  // (LCID_INSTALLED 1, LCID_SUPPORTED 2); nothing else.
  for (uint32_t flags : {1u, 2u}) {
    CHECK(Kernel32("IsValidLocale", {lang, flags}) == 1);
    CHECK(Kernel32("IsValidLocale", {0x0409, flags}) == 1);
    CHECK(Kernel32("IsValidLocale", {0x0411, flags}) == 0);  // Japanese
    CHECK(Kernel32("IsValidLocale", {0x0400, flags}) == 0);  // LOCALE_USER_DEFAULT
    CHECK(Kernel32("IsValidLocale", {0x0800, flags}) == 0);  // LOCALE_SYSTEM_DEFAULT
    CHECK(Kernel32("IsValidLocale", {0, flags}) == 0);       // LOCALE_NEUTRAL
    CHECK(Kernel32("IsValidLocale", {0x00010000u | lang, flags}) == 0);  // another sort
  }

  // EnumSystemLocalesA: both, smallest first, each once; it stops when
  // the callback returns FALSE.
  std::vector<std::string> want{LcidName(0x0409)};
  if (lang < 0x0409) want.insert(want.begin(), LcidName(lang));
  if (lang > 0x0409) want.push_back(LcidName(lang));
  CHECK(Enumerate(1, 0) == want);  // LCID_INSTALLED
  CHECK(Enumerate(2, 0) == want);  // LCID_SUPPORTED
  CHECK(Enumerate(1, 1) == std::vector<std::string>{want[0]});
  if (lang == 0x0409) CHECK(want == std::vector<std::string>{"00000409"});

  std::printf("language %04X (port %04X): %zu locale(s):", unsigned(lang), unsigned(configured),
              want.size());
  for (const std::string& n : want) std::printf(" %s", n.c_str());
  std::printf("\n%s\n", g_failures ? "FAILED" : "ok");
  return g_failures ? 1 : 0;
}
