#pragma once
// Whether a GPU traces rays in hardware with what the ray-traced scene
// (gfx/rt_scene.h) uses: the Apple9 family (M3, A17 Pro and later), with
// ray tracing from compute pipelines. Header-only (Objective-C++, Metal
// only), so that a game's launcher can show its ray tracing choices by the
// same rule as the renderer before any device exists.

#import <Metal/Metal.h>

namespace recomp::gfx {

inline bool RtSupported(id<MTLDevice> dev) {
  if (!dev) return false;
  if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *))
    return [dev supportsFamily:MTLGPUFamilyApple9] && dev.supportsRaytracing;
  return false;
}

// The same for the system's default GPU.
inline bool RtSupportedHere() { return RtSupported(MTLCreateSystemDefaultDevice()); }

// And rays traced from render pipelines' stages too (a fragment function's
// ray queries: gfx/rt_forward_reflection.h).
inline bool RtSupportedFromRender(id<MTLDevice> dev) { return RtSupported(dev) && dev.supportsRaytracingFromRender; }

}  // namespace recomp::gfx
