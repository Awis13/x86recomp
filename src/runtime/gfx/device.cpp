// Backend selection and presentation geometry for gfx::Device.

#include "runtime/gfx/device.h"

#include <atomic>
#include <mutex>

#include "runtime/gfx/rt_light_list.h"

namespace recomp::gfx {

namespace {
std::mutex g_present_mu;
Presentation g_present{};
std::atomic<bool> g_ray_shadows{false}, g_ray_reflections{false}, g_ray_forward{false}, g_ray_forward_shadows{false};
RtLightTaker<Device> g_light_device;  // NoteRtLightDevice's
}  // namespace

// The device is destroyed on the thread that makes its calls, the one
// SetRtLights is called on: it cannot go between the load and the call.
void SetRtLights(const RtLight* lights, size_t count) {
  if (Device* d = g_light_device.Get()) d->SetRtLights(lights, count);
}

// On: `d` takes the lights. Off: only if `d` is the one that takes them
// (RtLightTaker), so a device going after another took over leaves that one.
void NoteRtLightDevice(Device* d, bool on) { g_light_device.Note(d, on); }

void SetRtPlainPlaces(const float* places, size_t count) {
  if (Device* d = g_light_device.Get()) d->SetRtPlainPlaces(places, count);
}

bool RayTracedShadows() { return g_ray_shadows.load(std::memory_order_acquire); }

namespace {
// (Three floats as bits, and whether there is one: read and written whole
// under the lock, a few times a frame.)
std::mutex g_sun_mu;
float g_sun[3] = {0, 0, 0};
bool g_sun_known = false;
}  // namespace

bool RtSunToward(float out[3]) {
  std::lock_guard<std::mutex> l(g_sun_mu);
  for (int k = 0; k < 3; k++) out[k] = g_sun[k];
  return g_sun_known;
}

void NoteRtSunToward(const float toward[3]) {
  std::lock_guard<std::mutex> l(g_sun_mu);
  for (int k = 0; k < 3; k++) g_sun[k] = toward[k];
  g_sun_known = true;
}

void NoteRayTracedShadows(bool on) { g_ray_shadows.store(on, std::memory_order_release); }

bool RayTracedReflections() { return g_ray_reflections.load(std::memory_order_acquire); }

void NoteRayTracedReflections(bool on) { g_ray_reflections.store(on, std::memory_order_release); }

bool RayTracedForwardReflections() { return g_ray_forward.load(std::memory_order_acquire); }

void NoteRayTracedForwardReflections(bool on) { g_ray_forward.store(on, std::memory_order_release); }

bool RayTracedForwardShadows() { return g_ray_forward_shadows.load(std::memory_order_acquire); }

void NoteRayTracedForwardShadows(bool on) { g_ray_forward_shadows.store(on, std::memory_order_release); }

Rect FitRect(int src_w, int src_h, int dst_w, int dst_h) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return {0, 0, dst_w, dst_h};
  int64_t w = dst_w, h = dst_h;
  if (int64_t(dst_w) * src_h > int64_t(dst_h) * src_w)
    w = (int64_t(dst_h) * src_w + src_h / 2) / src_h;  // narrower: pillarbox
  else
    h = (int64_t(dst_w) * src_h + src_w / 2) / src_w;  // wider: letterbox
  const int x = int(dst_w - w) / 2, y = int(dst_h - h) / 2;
  return {x, y, x + int(w), y + int(h)};
}

void NotePresentation(const Presentation& p) {
  std::lock_guard<std::mutex> l(g_present_mu);
  g_present = p;
}

Presentation LastPresentation() {
  std::lock_guard<std::mutex> l(g_present_mu);
  return g_present;
}

std::unique_ptr<Device> CreateDevice(Backend b, SDL_Window* window) {
  switch (b) {
    case Backend::D3D9:
#ifdef RECOMP_HAVE_D3D9
      return CreateD3D9Device(window);
#else
      return nullptr;
#endif
    case Backend::Gpu:
      return CreateGpuDevice(window);
    case Backend::Metal:
#if defined(__APPLE__)
      return CreateMetalDevice(window);
#else
      return nullptr;
#endif
  }
  return nullptr;
}

}  // namespace recomp::gfx
