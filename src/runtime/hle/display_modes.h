#pragma once
// The list of display mode sizes games can switch to, built from the
// desktop's size (user32_window.cpp's DisplayModeSizes()). No host calls
// here, so tests can build lists for any desktop.

#include <vector>

namespace recomp {

struct ModeSize {
  int w, h;
};

// "WxH,WxH,..." (<ID>_DISPLAY_EXTRA_MODES) appended to *out. False if the
// text is not such a list; empty text is an empty list.
bool ParseModeSizes(const char* text, std::vector<ModeSize>* out);

// The modes of a desktop_w x desktop_h desktop, smallest first, each once:
// the desktop's own size, the classic 4:3 and 5:4 sizes, common wide ones,
// and `extra` (sizes the port asks for, such as the presets its launcher
// offers). Nothing larger than the desktop, which counts as at least
// 1024x768 so that the classic modes stay; extra sizes that are not
// positive are left out too.
std::vector<ModeSize> BuildModeSizes(int desktop_w, int desktop_h,
                                     const std::vector<ModeSize>& extra);

}  // namespace recomp
