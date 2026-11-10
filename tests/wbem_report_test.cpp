// Replay stock report queries through guest COM, without game files or a GPU.
#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>

#include "runtime/hle/com.h"
#include "runtime/hle/hle.h"
#include "runtime/hle/kernel32_state.h"
#include "runtime/runtime.h"

using namespace recomp;
namespace {
void Check(bool ok, const char* message) {
  if (ok) return;
  std::fprintf(stderr, "%s\n", message);
  std::_Exit(1);
}
}
int main(int argc, char** argv) {
  Check(argc == 3, "Expected the recomp_xinput library path and pad count");
  const bool pad = std::string(argv[2]) == "1";
  setenv("WBEM_BACKGROUND", "1", 1);
  setenv("WBEM_TEST_PAD", pad ? "1" : "0", 1);
  Check(dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL) != nullptr, "Cannot load XInput HLE");
  // Process heaps and COM registrations live until _Exit, as in other HLE tests.
  auto process = std::make_unique<Process>();
  g_proc = process.get();
  process->game.id = "wbem";
  Check(process->mem.Reserve(), "Cannot reserve guest memory");
  process->vm.Init(&process->mem, 0x10000, 0x4000000);
  Check(process->mem.Commit(kPebVa, 0x1000, Prot::RW), "Cannot commit PEB");
  InitKernel32State();
  const auto stack = static_cast<uint32_t>(HeapAllocProcess(4096, true));
  Check(stack != 0, "Cannot allocate stack");
  const uint32_t out = stack + 256, returned = out + 4, objects = out + 8;
  Ctx context{};
  auto call = [&](const char* dll, const std::string& name, std::initializer_list<uint32_t> args) {
    context.esp = stack;
    Wr32(stack, kReturnToHost);
    uint32_t offset = 4;
    for (const auto arg : args) { Wr32(stack + offset, arg); offset += 4; }
    hle::CallThunk(context, hle::ThunkFor(dll, name));
    Check(context.esp == stack + offset, "Incorrect stdcall argument count");
    return context.eax;
  };
  auto wbem = [&](const std::string& name, std::initializer_list<uint32_t> args) {
    return call("wbemprox.dll", name, args);
  };
  // The report initializes a VARIANT before connecting/querying WMI and
  // clears it on both its success and unavailable-hardware paths.
  const uint32_t variant = out + 160;
  std::memset(GPtr(variant), 0xA5, 16);
  call("oleaut32.dll", "#8", {variant});
  Check(Rd16(variant) == 0 && Rd32(variant + 8) == 0xA5A5A5A5,
        "VariantInit must set VT_EMPTY without releasing uninitialized contents");
  Check(call("oleaut32.dll", "#9", {variant}) == 0 && Rd16(variant) == 0,
        "Cannot clear an empty report VARIANT");
  constexpr guest::GUID clsid{0x4590F811, 0x1D3A, 0x11D0, {0x89,0x1F,0,0xAA,0,0x4B,0x2E,0x24}};
  constexpr guest::GUID iid{0xDC12A687, 0x737F, 0x11CF, {0x88,0x4D,0,0xAA,0,0x4B,0x2E,0x24}};
  std::memcpy(GPtr(out + 64), &clsid, sizeof clsid);
  std::memcpy(GPtr(out + 80), &iid, sizeof iid);
  Check(call("ole32.dll", "CoCreateInstance", {out + 64, 0, 1, out + 80, out}) == 0, "Cannot create locator");
  const auto locator = Rd32(out);
  const auto owned_string = com::AllocBstr(u"hardware");
  Wr16(variant, 8); Wr32(variant + 8, owned_string);
  Check(call("oleaut32.dll", "#9", {variant}) == 0 && Rd16(variant) == 0 &&
            HeapSizeProcess(owned_string - 4) == UINT64_MAX, "VariantClear did not free its BSTR");
  const auto borrowed_string = com::AllocBstr(u"borrowed");
  Wr32(out + 144, borrowed_string);
  Wr16(variant, 0x4008); Wr32(variant + 8, out + 144);
  Check(call("oleaut32.dll", "#9", {variant}) == 0 &&
            GWStr(borrowed_string) == u"borrowed" && HeapSizeProcess(borrowed_string - 4) != UINT64_MAX,
        "VariantClear freed a borrowed BSTR");
  com::FreeBstr(borrowed_string);
  for (const uint16_t type : {9, 13}) {
    Check(wbem("IWbemLocator::AddRef", {locator}) == 2, "Cannot give VARIANT a reference");
    Wr16(variant, type); Wr32(variant + 8, locator);
    Check(call("oleaut32.dll", "#9", {variant}) == 0 && com::Lookup(locator)->refs == 1,
          "VariantClear did not release its interface");
    Wr32(out + 144, locator);
    Wr16(variant, type | 0x4000); Wr32(variant + 8, out + 144);
    Check(call("oleaut32.dll", "#9", {variant}) == 0 && com::Lookup(locator)->refs == 1,
          "VariantClear released a borrowed interface");
  }
  Wr16(variant, 3); Wr32(variant + 8, 42);
  Check(call("oleaut32.dll", "#9", {variant}) == 0 && Rd16(variant) == 0,
        "Cannot clear a scalar VARIANT");
  Wr16(variant, 0xFFFF);
  Check(call("oleaut32.dll", "#9", {variant}) == 0x80020008 && Rd16(variant) == 0xFFFF,
        "Invalid VARIANT type was silently discarded");
  Wr16(variant, 0x2003);
  Check(call("oleaut32.dll", "#9", {variant}) == 0x80004001 && Rd16(variant) == 0x2003,
        "Unsupported SAFEARRAY ownership was silently discarded");
  call("oleaut32.dll", "#8", {variant});
  const auto ns = com::AllocBstr(u"ROOT\\CIMV2");
  Check(wbem("IWbemLocator::ConnectServer", {locator, ns, 0, 0, 0, 0, 0, 0, out}) == 0, "Cannot connect WMI");
  com::FreeBstr(ns);
  const auto service = Rd32(out);
  const auto language = com::AllocBstr(u"WQL");
  for (const auto* text : {u"SELECT * FROM Win32_OperatingSystem WHERE Primary = TRUE", u"SELECT * FROM Win32_Processor"}) {
    const auto query = com::AllocBstr(text);
    Wr32(out, 0xDEADBEEF);
    Check(wbem("IWbemServices::ExecQuery", {service, language, query, 0x30, 0, out}) == 0x8004100C,
          "Unavailable hardware queries must return WBEM_E_NOT_SUPPORTED");
    Check(Rd32(out) == 0, "Failed query retained a stale enumerator");
    Check(wbem("IWbemServices::ExecQuery", {service, language, query, 0x30, 0, 0}) == 0x80004003,
          "Null query output must return E_POINTER");
    Check(call("oleaut32.dll", "#9", {variant}) == 0, "Failed report query cannot clean up its VARIANT");
    com::FreeBstr(query);
  }
  com::FreeBstr(language);
  // The native report also probes video hardware via class enumeration.
  const auto video = com::AllocBstr(u"Win32_VideoController");
  Wr32(out, 0xDEADBEEF);
  Check(wbem("IWbemServices::CreateInstanceEnum", {service, video, 0, 0, out}) == 0x80041010,
        "Unavailable video class must return WBEM_E_INVALID_CLASS");
  Check(Rd32(out) == 0, "Failed class enumeration retained an output");
  com::FreeBstr(video);
  // Preserve the existing DirectInput/XInput discovery path, including reset/end.
  const auto cls = com::AllocBstr(u"Win32_PNPEntity");
  Check(wbem("IWbemServices::CreateInstanceEnum", {service, cls, 0, 0, out}) == 0, "PnP enumeration regressed");
  com::FreeBstr(cls);
  const auto enumeration = Rd32(out);
  unsigned count = 0;
  while (true) {
    Wr32(returned, 123);
    const auto hr = wbem("IEnumWbemClassObject::Next", {enumeration, 0xFFFFFFFF, 1, objects, returned});
    if (hr == 1) { Check(Rd32(returned) == 0, "End must return zero objects"); break; }
    Check(hr == 0 && Rd32(returned) == 1 && ++count <= 4, "Invalid PnP object count");
    const auto object = Rd32(objects);
    const auto property = com::AllocBstr(u"DeviceID");
    Check(wbem("IWbemClassObject::Get", {object, property, 0, out + 128, 0, 0}) == 0, "Cannot read pad DeviceID");
    Check(Rd16(out + 128) == 8 && GWStr(Rd32(out + 136)).find(u"IG_") != std::u16string::npos,
          "Pad DeviceID lost XInput marker");
    com::FreeBstr(property);
    com::FreeBstr(Rd32(out + 136));
    Check(wbem("IWbemClassObject::Release", {object}) == 0, "Pad object leaked");
  }
  Check(count == (pad ? 1u : 0u), "Unexpected virtual pad count");
  Check(wbem("IEnumWbemClassObject::Reset", {enumeration}) == 0, "Cannot reset PnP enumeration");
  Check(wbem("IEnumWbemClassObject::Next", {enumeration, 0, 0, objects, returned}) == 0 && Rd32(returned) == 0,
        "Zero-count Next regressed");
  Check(wbem("IEnumWbemClassObject::Next", {enumeration, 0, 1, objects, returned}) == (pad ? 0u : 1u)
            && Rd32(returned) == count, "Reset did not rewind enumeration");
  if (pad) Check(wbem("IWbemClassObject::Release", {Rd32(objects)}) == 0, "Reset object leaked");
  Check(wbem("IEnumWbemClassObject::Release", {enumeration}) == 0, "Enumerator leaked");
  Check(wbem("IWbemServices::Release", {service}) == 0, "Service leaked");
  Check(wbem("IWbemLocator::Release", {locator}) == 0, "Locator leaked");
  std::printf("stock report queries, output/ABI contract, and %u PnP devices passed\n", count);
  std::fflush(stdout);
  std::_Exit(0);
}
