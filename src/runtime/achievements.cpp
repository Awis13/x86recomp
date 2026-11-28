#include "runtime/achievements.h"

#include <mutex>

namespace recomp {
namespace {

std::mutex g_mu;
AchievementHandler g_handler;
PresentOverlayFn g_overlay = nullptr;

}  // namespace

void SetAchievementHandler(const AchievementHandler& handler) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_handler = handler;
}

AchievementHandler AchievementHandlerSnapshot() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_handler;
}

void SetPresentOverlay(PresentOverlayFn fn) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_overlay = fn;
}

PresentOverlayFn PresentOverlay() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_overlay;
}

}  // namespace recomp
