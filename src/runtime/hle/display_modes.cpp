#include "runtime/hle/display_modes.h"

#include <algorithm>

namespace recomp {

namespace {
// A decimal number of at most 9 digits at *p; advances p.
bool ParseSize(const char** p, int* out) {
  const char* s = *p;
  int v = 0, digits = 0;
  for (; *s >= '0' && *s <= '9'; s++, digits++) {
    if (digits == 9) return false;
    v = v * 10 + (*s - '0');
  }
  if (!digits) return false;
  *p = s;
  *out = v;
  return true;
}
}  // namespace

bool ParseModeSizes(const char* text, std::vector<ModeSize>* out) {
  if (!text || !*text) return true;
  for (const char* p = text;;) {
    ModeSize s;
    if (!ParseSize(&p, &s.w) || *p++ != 'x' || !ParseSize(&p, &s.h)) return false;
    out->push_back(s);
    if (!*p) return true;
    if (*p++ != ',') return false;
  }
}

std::vector<ModeSize> BuildModeSizes(int desktop_w, int desktop_h,
                                     const std::vector<ModeSize>& extra) {
  static const ModeSize kSizes[] = {
      {640, 480},   {800, 600},   {1024, 768},  {1152, 864},  {1280, 960},
      {1280, 1024}, {1600, 1200}, {1280, 720},  {1366, 768},  {1600, 900},
      {1920, 1080}, {1920, 1200}, {2560, 1080}, {2560, 1440}, {3440, 1440},
      {3840, 2160}};
  // Never smaller than a 1024x768 desktop, so the classic modes stay.
  const int mw = std::max(desktop_w, 1024), mh = std::max(desktop_h, 768);
  std::vector<ModeSize> v{{desktop_w, desktop_h}};
  for (const ModeSize& s : kSizes)
    if (s.w <= mw && s.h <= mh) v.push_back(s);
  for (const ModeSize& s : extra)
    if (s.w > 0 && s.h > 0 && s.w <= mw && s.h <= mh) v.push_back(s);
  std::sort(v.begin(), v.end(),
            [](const ModeSize& a, const ModeSize& b) { return a.w != b.w ? a.w < b.w : a.h < b.h; });
  v.erase(std::unique(v.begin(), v.end(),
                      [](const ModeSize& a, const ModeSize& b) { return a.w == b.w && a.h == b.h; }),
          v.end());
  return v;
}

}  // namespace recomp
