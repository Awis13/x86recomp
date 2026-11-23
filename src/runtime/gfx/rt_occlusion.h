#pragma once
// Ray-traced ambient occlusion over a deferred renderer's depth
// (Objective-C++, GPUs with hardware ray tracing; gfx::RayOcclusion,
// RayTracing::occlusion): how much of the sky above each pixel's point is
// not blocked within `radius`, as an R8Unorm texture of half the depth's
// size (1 open, 0 closed; read with linear filtering) that the lighting
// multiplies its flat ambient by.
//
// As ray-traced AO is done where it ships (Apple's "Accelerating ray
// tracing using Metal" sample's cosine-weighted short rays; Unreal's and
// FidelityFX's hybrid AO: a few rays a pixel, then a filter): at one pixel
// of each 4 x 4 block (turning each frame, RtLightSample's), kRtOcclusionRays
// ray in a cosine-weighted hemisphere around the surface's geometric
// normal (from the depth of its neighbours, as the shadow masks have it;
// the G-buffer's where the depth gives none), each blocked at a distance t
// taking 1 - t / radius of its share (a near wall darkens more than a far
// one; cut-out geometry, leaves and fences, taken as solid: no alpha test);
// then at every other pixel (the output's texels) the samples of the 3 x 3
// blocks around it whose
// planes it lies on (by how far its point lies off each sample's plane for
// its distance from it, as the many lights' reconstruction weighs theirs)
// averaged. Fresh rays every frame, nothing kept from one to the next (no
// lag behind what moves; the temporal anti-aliasing after it averages the
// noise). Sky pixels are open.

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

// (One ray at 1 pixel in 16; 2 at 1 in 4 before: incoherent rays in a
// city's scene cost several times a shadow ray, 4.6 ms a frame for 2.4 M
// at 3440 x 1440.)
inline constexpr uint32_t kRtOcclusionRays = 1;  // rays a traced sample
inline constexpr uint32_t kRtOcclusionBlock = 4;

// A pass's parameters (MSL: RtOcclusionParams).
struct RtOcclusionParams {
  uint32_t size[2];    // the depth's
  uint32_t blocks[2];  // size / kRtOcclusionBlock rounded up
  uint32_t frame;      // the sample's turn and the rays' directions
  uint32_t stride;     // RtLightStride(kRtOcclusionBlock), or 0: every block's top-left pixel
  float radius;        // world units
  uint32_t normals;    // 1: a normal texture bound (xyz as unorm)
};
static_assert(sizeof(RtOcclusionParams) == 32, "kRtOcclusionMsl's RtOcclusionParams");

extern const char* const kRtOcclusionMsl;

class RtOccluder {
 public:
  static std::unique_ptr<RtOccluder> Make(id<MTLDevice> dev);
  // The occlusion of `depth` (p as the shadow masks take it: RtShadowParamsFor)
  // traced against f's TLAS into `out` (NewOut: half the depth's size). Adds the
  // rays traced and those blocked to the two counters at `counts`.
  void Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> normals,
              const RtShadowParams& p, const RtOcclusionParams& op, id<MTLTexture> out, id<MTLBuffer> counts,
              size_t counts_offset) const;
  // An R8Unorm texture for a depth of w x h: half its size, rounded up
  // (private, written by Encode, sampled after).
  id<MTLTexture> NewOut(uint32_t w, uint32_t h) const;
  // 1 x 1, open everywhere: for draws whose occlusion cannot be traced.
  id<MTLTexture> Open() const { return open_; }

 private:
  id<MTLDevice> dev_ = nil;
  id<MTLComputePipelineState> trace_depth_ = nil, trace_color_ = nil, filter_depth_ = nil, filter_color_ = nil;
  id<MTLIntersectionFunctionTable> table_depth_ = nil, table_color_ = nil;
  mutable id<MTLTexture> samples_ = nil;   // RGBA32Float a block: occlusion (-1: sky), the point from the eye
  mutable id<MTLTexture> sample_n_ = nil;  // RGBA16Float a block: the point's normal
  id<MTLTexture> open_ = nil, white_ = nil;
};

}  // namespace recomp::gfx
