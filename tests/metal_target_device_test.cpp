// New render targets through the Metal device itself (CreateTexture, with
// a hidden window; gfx/metal_new_target.h): a new render target that is
// never drawn to reads back zeros; a fill or an upload into a new target
// in the frame it was made in is what reads back, and so is a fill a frame
// later; a target made while another is drawn to leaves that one's pass
// as it was.
//
//   metal_target_device_test   (skipped, exit 77, without a Metal device
//                               or a window)

#include <SDL3/SDL.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "runtime/gfx/device.h"

namespace {

using namespace recomp::gfx;

constexpr int kW = 256, kH = 256;
constexpr Rect kAll{0, 0, kW, kH};
int g_failed = 0;

// The colour (alpha left out: read back as 1) of every pixel of render
// target `t` must be `rgb`.
void Expect(const char* what, Device& d, TextureId t, uint32_t rgb) {
  std::vector<uint32_t> px(size_t(kW) * kH, 0x00ABCDEF);
  d.SetRenderTarget(t, 0);
  d.ReadBack(kAll, px.data(), uint32_t(kW) * 4, Format::A8R8G8B8);
  d.SetRenderTarget(0, 0);
  size_t bad = 0;
  uint32_t first_bad = 0;
  for (uint32_t p : px)
    if ((p & 0xFFFFFF) != rgb && !bad++) first_bad = p;
  if (bad) {
    std::printf("FAIL %s: %zu of %zu pixels are not %06X (the first: %08X)\n", what, bad, px.size(),
                unsigned(rgb), unsigned(first_bad));
    g_failed++;
  }
}

}  // namespace

int main() {
  SDL_SetHint(SDL_HINT_MAC_BACKGROUND_APP, "1");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::printf("no video: skipped (%s)\n", SDL_GetError());
    return 77;
  }
  SDL_Window* win = SDL_CreateWindow("metal_target_device_test", 64, 64, SDL_WINDOW_HIDDEN | SDL_WINDOW_METAL);
  if (!win) {
    std::printf("no window: skipped (%s)\n", SDL_GetError());
    return 77;
  }
  {
    std::unique_ptr<Device> d = CreateMetalDevice(win);
    if (!d) {
      std::printf("no Metal device: skipped\n");
      return 77;
    }
    d->ResetSwapChain(64, 64, false);
    // Target memory made white and freed: new targets may get it back.
    for (int i = 0; i < 4; i++) {
      const TextureId junk = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
      d->SetRenderTarget(junk, 0);
      d->Clear(0, nullptr, 1, 0xFFFFFFFF, 1.0f, 0);
      uint32_t one;
      d->ReadBack({0, 0, 1, 1}, &one, 4, Format::A8R8G8B8);
      d->SetRenderTarget(0, 0);
      d->DestroyTexture(junk);
    }
    const std::vector<uint32_t> green(size_t(kW) * kH, 0xFF00FF00);

    const TextureId a = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    Expect("a new target, never drawn to", *d, a, 0x000000);

    const TextureId b = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    d->FillTexture(b, kAll, 0xFFFFFFFF);
    Expect("a new target filled in the same frame", *d, b, 0xFFFFFF);

    const TextureId c = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    d->UpdateTexture(c, 0, kAll, green.data(), uint32_t(kW) * 4);
    Expect("a new target uploaded to in the same frame", *d, c, 0x00FF00);

    // (The read back ended the frame `a` was made in.)
    d->FillTexture(a, kAll, 0xFFFFFFFF);
    Expect("a target filled a frame later", *d, a, 0xFFFFFF);

    // A target made while another is being drawn to: both as drawn.
    const TextureId e = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    d->SetRenderTarget(e, 0);
    d->SetViewport({0, 0, uint32_t(kW), uint32_t(kH), 0.0f, 1.0f});
    d->Clear(0, nullptr, 1, 0xFFFF0000, 1.0f, 0);
    const TextureId f = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    Expect("a target cleared before another was made", *d, e, 0xFF0000);
    Expect("the target made meanwhile", *d, f, 0x000000);

    // Made and destroyed in one frame, then the frame ends.
    const TextureId gone = d->CreateTexture(kW, kH, 1, Format::A8R8G8B8, true);
    d->DestroyTexture(gone);
    Expect("after a target made and destroyed in one frame", *d, b, 0xFFFFFF);
  }
  SDL_DestroyWindow(win);
  SDL_Quit();
  if (g_failed) return 1;
  std::printf("ok\n");
  return 0;
}
