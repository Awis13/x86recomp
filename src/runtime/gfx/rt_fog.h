#pragma once
// Ray-traced volumetric fog (Objective-C++, GPUs with hardware ray
// tracing): a grid of cells over the camera's view (kRtFogW x kRtFogH x
// kRtFogD, the slices spread exponentially in view depth from near to
// far), each cell lit by the sun as one ray toward it finds it (lit or in
// a caster's shadow: shafts between buildings) with a Henyey-Greenstein
// phase, plus the sky's colour; its density falls with height. Each frame
// the cells are traced at a depth jittered within their slice and blended
// with the last frame's, reprojected by the camera (the noise of one ray a
// cell averaged out); then integrated front to back (Frostbite's energy-
// conserving step) into a texture whose texel at (x, y, slice) holds the
// light scattered toward the camera up to the slice's far side and the
// transmittance there. The frame's output (gfx/metal_device.mm) multiplies
// each pixel by the transmittance at its depth and adds the light.
//
// As volumetric fog ships in Frostbite (Hillaire, "Physically Based and
// Unified Volumetric Rendering", 2015) and Unreal (froxels, temporal
// reprojection), with the shadow map's lookup replaced by a ray.

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

inline constexpr uint32_t kRtFogW = 128, kRtFogH = 72, kRtFogD = 48;

// What the fog looks like (RECOMP_FOG; ParseRtFog).
struct RtFogLook {
  float density = 0.0015f;   // per metre at height 0
  float height = 80.0f;      // metres over which the density falls by e
  float anisotropy = 0.6f;   // Henyey-Greenstein g: toward the sun brighter
  float sun = 0.6f;          // the sun's colour's share
  float sky = 0.5f;          // the sky's colour's share (the fog's own body)
  float near_z = 1.0f, far_z = 400.0f;  // the slices' view depths
};
// "density=d;height=m;anisotropy=g;sun=s;sky=s;far=m" (each optional);
// false if not understood.
bool ParseRtFog(const char* text, RtFogLook* out);

// A pass's parameters (MSL: RtFogParams).
struct RtFogParams {
  uint32_t dims[4];       // W, H, D, the frame's number
  float range[4];         // near_z, far_z, density, height
  float look[4];          // anisotropy, sun, sky, the slice's jitter (0..1)
  float sun[4];           // the sun's colour (rgb)
  float prev_vp[16];      // world to last frame's clip, as the camera's matrix (row vectors)
  float prev[4];          // w: 1 where there is a last frame to blend with
  float blend[4];         // x: this frame's weight
};
static_assert(sizeof(RtFogParams) == 160, "kRtFogMsl's RtFogParams");

extern const char* const kRtFogMsl;

class RtFog {
 public:
  static std::unique_ptr<RtFog> Make(id<MTLDevice> dev);
  // The frame's fog against f's TLAS from the camera of `p` (as the sun's
  // mask takes it: its eye, its light toward the sun, its size), the sun's
  // colour `sun`, the sky's colour at `sky` (a float4, w 0 until known),
  // last frame's camera `prev_vp` (world to clip, row vectors; null: none).
  void Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, const RtShadowParams& p, const float sun[3],
              id<MTLBuffer> sky, const double* prev_vp, uint32_t frame, const RtFogLook& look) const;
  // The integrated fog (RGBA16Float 3D: rgb the light up to the slice's
  // far side, a the transmittance there); nil before the first Encode.
  id<MTLTexture> Integrated() const { return integrated_; }

 private:
  id<MTLDevice> dev_ = nil;
  id<MTLComputePipelineState> trace_ = nil, integrate_ = nil;
  id<MTLIntersectionFunctionTable> table_ = nil;
  mutable id<MTLTexture> cells_[2] = {nil, nil};  // this frame's and the last's (ping-pong)
  mutable int cur_ = 0;
  mutable bool have_ = false;
  id<MTLTexture> integrated_ = nil;
  id<MTLBuffer> no_sky_ = nil;
};

}  // namespace recomp::gfx
