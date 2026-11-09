// The display mode list (runtime/hle/display_modes.h): what a desktop of a
// given size and orientation offers, with and without sizes added by the
// port, and the parser of the list a port passes.

#include <cstdio>
#include <string>
#include <vector>

#include "runtime/hle/display_modes.h"

namespace {

using recomp::ModeSize;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

std::string Text(const std::vector<ModeSize>& v) {
  std::string s;
  for (const ModeSize& m : v) {
    if (!s.empty()) s += ' ';
    s += std::to_string(m.w) + 'x' + std::to_string(m.h);
  }
  return s;
}

void Expect(const std::vector<ModeSize>& got, const char* want, const char* what) {
  const std::string text = Text(got);
  if (text != want) std::printf("  got:  %s\n  want: %s\n", text.c_str(), want);
  Check(text == want, what);
}

std::vector<ModeSize> Parsed(const char* text) {
  std::vector<ModeSize> v;
  Check(recomp::ParseModeSizes(text, &v), text);
  return v;
}

void Parser() {
  std::vector<ModeSize> v;
  Check(recomp::ParseModeSizes(nullptr, &v) && v.empty(), "unset is an empty list");
  Check(recomp::ParseModeSizes("", &v) && v.empty(), "empty text is an empty list");
  Expect(Parsed("1434x660"), "1434x660", "one size");
  Expect(Parsed("1434x660,1912x880,0x0"), "1434x660 1912x880 0x0", "sizes in the order given");
  for (const char* bad : {"1434", "1434x", "x660", "1434x660,", ",1434x660", "1434X660",
                          "1434x660;1912x880", "1434 x660", "-1434x660", "1434x660x2",
                          "12345678901x2", "wide"}) {
    v.clear();
    Check(!recomp::ParseModeSizes(bad, &v), bad);
  }
}

void Lists() {
  // A wide phone display in landscape, with native-aspect sizes from the port.
  const std::vector<ModeSize> phone = Parsed("1434x660,1912x880,2390x1100,2868x1320");
  Expect(recomp::BuildModeSizes(2868, 1320, {}),
         "640x480 800x600 1024x768 1152x864 1280x720 1280x960 1280x1024 1366x768 1600x900 "
         "1600x1200 1920x1080 1920x1200 2560x1080 2868x1320",
         "landscape desktop, no extra sizes");
  Expect(recomp::BuildModeSizes(2868, 1320, phone),
         "640x480 800x600 1024x768 1152x864 1280x720 1280x960 1280x1024 1366x768 1434x660 "
         "1600x900 1600x1200 1912x880 1920x1080 1920x1200 2390x1100 2560x1080 2868x1320",
         "landscape desktop: extra sizes sorted in, the desktop's own once");
  // The same display left in portrait: the wide sizes do not fit it.
  Expect(recomp::BuildModeSizes(1320, 2868, phone),
         "640x480 800x600 1024x768 1152x864 1280x720 1280x960 1280x1024 1320x2868",
         "portrait desktop: only what fits");
  // What is not positive or does not fit is left out; duplicates collapse.
  Expect(recomp::BuildModeSizes(1280, 800, Parsed("0x0,640x0,0x480,1281x800,1280x801,"
                                                   "960x600,960x600,800x600,1280x800")),
         "640x480 800x600 960x600 1024x768 1280x720 1280x800",
         "invalid, oversized and repeated extra sizes");
  Expect(recomp::BuildModeSizes(1280, 800, {{-640, 480}, {640, -480}}),
         "640x480 800x600 1024x768 1280x720 1280x800", "negative extra sizes");
  // A desktop below 1024x768 keeps the classic modes, and extra sizes up
  // to that floor.
  Expect(recomp::BuildModeSizes(800, 480, Parsed("1024x600,1025x600")),
         "640x480 800x480 800x600 1024x600 1024x768", "small desktop");
}

}  // namespace

int main() {
  Parser();
  Lists();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
