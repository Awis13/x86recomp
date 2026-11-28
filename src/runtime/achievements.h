#pragma once
// Local achievement unlocks for the offline xlive stand-in, and a bitmap a
// backend can draw over the presented frame.
//
// Nothing is installed here. Until a port calls SetAchievementHandler,
// XUserWriteAchievements succeeds and the ids are discarded, and every
// achievement enumeration is empty. Until a port calls SetPresentOverlay,
// the present path draws no toast. A port's enabled() stays false until
// that port turns the option on.

#include <cstdint>
#include <string>
#include <vector>

namespace recomp {

struct AchievementItem {
  uint32_t id = 0;
  int points = 0;
  int64_t unlocked_at = 0;  // unix seconds
  std::string label;
  std::string description;
};

// Function pointers left null are the off state for that part.
struct AchievementHandler {
  // False, or null: writes are discarded and enumerations stay empty.
  bool (*enabled)() = nullptr;
  // One id from XUserWriteAchievements. The handler decides whether it is new.
  void (*record_unlock)(uint32_t id) = nullptr;
  // Unlocked rows, `start` skipped, at most `count`, in id order.
  std::vector<AchievementItem> (*current_unlocked)(uint32_t start, uint32_t count) = nullptr;
  // Bytes the enumerator should report so the wide strings fit after the rows.
  uint32_t (*enum_buffer_bytes)(uint32_t count) = nullptr;
  uint32_t (*enum_label_chars)() = nullptr;
  uint32_t (*enum_description_chars)() = nullptr;
};

void SetAchievementHandler(const AchievementHandler& handler);
// A copy of the installed handler. Safe to call from the guest's threads.
AchievementHandler AchievementHandlerSnapshot();

// Premultiplied RGBA, tightly packed, top row first.
struct PresentOverlayBitmap {
  const uint8_t* rgba = nullptr;
  int width = 0;
  int height = 0;
  int x = 0;  // top-left, in view pixels
  int y = 0;
  uint64_t serial = 0;  // changes when the pixels change
};

// False when there is nothing to draw. `rgba` stays valid until the next call.
// `scene` is true once the renderer has presented a first scene.
// `scale` is drawable pixels per window point.
using PresentOverlayFn = bool (*)(int view_px_w, int view_px_h, float scale, bool scene,
                                  PresentOverlayBitmap* out);

void SetPresentOverlay(PresentOverlayFn fn);
PresentOverlayFn PresentOverlay();

}  // namespace recomp
