#pragma once
// A frame's many local lights lit by the device over a deferred renderer's
// G-buffer, every one shadowed by rays (Objective-C++, GPUs with hardware
// ray tracing): RayManyLights (device.h), the lights of the frame's list
// (gfx/rt_light_list.h), in place of (or, Add, besides) the renderer's
// own light volumes.
//
// As the many-light renderers that ship do it (Unreal's MegaLights, the
// stochastic tiled lighting of HypeHype's SIGGRAPH 2025 talk, ReSTIR's
// final visibility in NVIDIA's RTXDI): the lights binned into screen tiles
// first; at a few pixels a few lights picked by what they bring there, each
// with a shadow ray traced that frame; the lighting itself evaluated at
// every pixel; no visibility kept from one frame to the next (nothing that
// lags behind what moves: the temporal anti-aliasing that follows averages
// the noise).
//
// - Binning (CPU, RtManyLightsBin): each light's bounds on screen (gfx/
//   rt_shadow.h, RtLocalLightBounds) marked in kRtManyTile x kRtManyTile
//   pixel tiles; a tile's list in the order of the lights' power (colour
//   times radius^2, strongest first), at most kRtManyTileCap (the weakest
//   past it dropped, counted).
// - Each pass's threadgroup first keeps of its tile's list the lights whose
//   sphere reaches the box around its points, in threadgroup memory (the
//   tiled light culling of Apple's deferred lighting samples).
// - Trace (compute, a thread for one pixel of each 2 x 2 block, a group a
//   tile; the pixel turning each frame as RtLightSample's): every light
//   kept evaluated there unshadowed (the model's formula, device.h:
//   RayLightModel); the strongest picked, and of the rest one at random in
//   proportion to what each brings (weighted reservoir sampling) where the
//   rest bring at least kRtManyRestShare of the light there; a shadow ray
//   each (as the local lights' masks trace them: rt_shadow.h; a light the
//   surface's geometric normal faces away from, in shadow untraced). Kept a
//   block: both lights and whether each was seen, and the point's distance
//   from the camera.
// - Shade (compute, a thread a pixel, into a texture of its own, then added
//   into the lit scene by a full-screen draw): every light kept evaluated
//   at the pixel; its strongest times that light's visibility, the others
//   times the visibility of the rest, from the samples of the pixel's block
//   and its three nearest neighbours: all of them where they agree and lie
//   at the pixel's distance, else weighed by place (bilinear) and by how
//   far each one's point lies off the pixel's plane (its geometric normal)
//   for its distance from the pixel. A pixel with no sample on its surface
//   (at an object's edge) traces its own ray to its strongest light. A
//   light no ray saw adds nothing: no light through walls.
// - A group none of whose pixels comes out lit (most groups of a day's
//   frame, which cost about 0.8 ms at 3440 x 1440 all the same) does no
//   more than find that out: where its gather keeps no light its surfaces
//   are not read (four textures a pixel); nothing is written for it, and a
//   flag a group tells the addition's draw to leave its pixels alone (their
//   texels of the light's texture are an earlier frame's). The picture is
//   the same: such a group added zero. RECOMP_RT_MANY_EMPTY=0: every pixel
//   read, written and added, as before the flags.
//
// Per pixel, the rest's estimate is the sum of their light at the pixel
// times one of them's visibility, picked in proportion to its light at the
// block's sample: in expectation over frames the sum of each one's light
// times its own visibility (where the pixel and the sample are lit alike).
// Points whose depth is the far plane's (the sky) are not lit.

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "runtime/gfx/device.h"
#include "runtime/gfx/rt_light_list.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

inline constexpr uint32_t kRtManyTile = 32;       // a tile's side, pixels
inline constexpr uint32_t kRtManyTileCap = 128;   // lights a tile at most
inline constexpr uint32_t kRtManyBlock = 2;       // a traced sample's block's side, pixels
inline constexpr float kRtManyRestShare = 0.05f;  // the rest's share of the light that gets a ray
// Encode's retain_hits, uints: a slot each (a scene's slots, its large
// meshes in pieces, ran to some 30000 while driving through a city).
inline constexpr uint32_t kRtManyRetainSlots = 65536;
inline constexpr uint32_t kRtManyGroup = 16;      // the shading's threadgroup's side, pixels (a flag each)
// The fragment texture the addition's draw takes the groups' flags at: past
// those the device's own draws bind (their samplers at 0-15, the forward
// shaders' masks at 16-18: rt_forward_reflection.h), so a draw after it in
// the same pass finds its textures as it left them. (30: the last index
// every GPU's table has.)
inline constexpr uint32_t kRtManyFlagsTexture = 30;

// A light as the kernels take it (MSL: RtManyLightGpu).
struct RtManyLightGpu {
  float position[4];   // xyz relative to the camera (RtShadowParams::eye); w 1 / radius^2
  float direction[4];  // xyz the way a spot light shines (0 for a point); w its cone's offset (-2 for a point)
  float colour[4];     // rgb linear, the intensity included; w the cone's scale (1 for a point)
  uint32_t flags;      // the model (bits 0-1) and pixel class (bits 2-3)
  float lamp_gap;
  float reach;  // a shadow ray's length from the lit point at most (RECOMP_RT_LAMP_PROBE)
  float pad;
};
static_assert(sizeof(RtManyLightGpu) == 64, "kRtManyLightsMsl's RtManyLightGpu");

// A light model as the kernels take it (MSL: RtManyModel; RayLightModel's).
struct RtManyModel {
  float falloff_power, falloff_bias, falloff_scale, wrap;
  float wrap_scale, spec_scale, exponent_scale;
  uint32_t flags;  // 1 specular, 2 normal_low_bits
  uint32_t factor_classes;
  uint32_t pad[3];
};
static_assert(sizeof(RtManyModel) == 48, "kRtManyLightsMsl's RtManyModel");
RtManyModel RtManyModelFor(const RayLightModel& m);

// A pass's parameters (MSL: RtManyParams).
struct RtManyParams {
  uint32_t size[2];    // the depth's
  uint32_t blocks[2];  // the samples': size / kRtManyBlock rounded up
  uint32_t tiles[2];   // size / kRtManyTile rounded up
  uint32_t frame;      // the sample's turn (RtLightSample)
  uint32_t stride;     // RtLightStride(kRtManyBlock), or 0: every block's top-left pixel
  uint32_t count;      // lights
  uint32_t textures;   // 1: a surface texture bound, 2: a class texture bound
  float rest_share;    // kRtManyRestShare
  uint32_t empty;      // 1: a group nothing lights is skipped (RtManyLights::Encode sets it: RECOMP_RT_MANY_EMPTY)
  RtManyModel models[4];
  uint32_t classes[3][8];
};
static_assert(sizeof(RtManyParams) == 48 + 4 * 48 + 96, "kRtManyLightsMsl's RtManyParams");

// A frame's lights binned: the lights (as the kernels take them, strongest
// first), and per tile (row by row) where its list starts in `index` and
// how many it holds (heads[2 t], heads[2 t + 1]); lights left out of a full
// tile's list (`dropped`, entries), and those that reach no tile.
struct RtManyBins {
  uint32_t tiles[2] = {};
  std::vector<RtManyLightGpu> lights;
  std::vector<uint32_t> heads;
  std::vector<uint16_t> index;
  uint32_t dropped = 0, off_screen = 0;
};

// The bins of `lights` over a w x h buffer seen from frame `f`'s camera
// (its jitter, as RtLocalLightBounds takes it; eye as RtLightsEye gives it),
// their rays ending lamp_gap short of each light. Lights that cannot be
// placed (no finite position or radius) reach nothing.
extern uint32_t g_lamp_probe_masks;  // RECOMP_RT_LAMP_PROBE's 3rd number (to measure)
void RtManyLightsBin(const RtFrame& f, uint32_t w, uint32_t h, const float jitter[2], std::span<const RtLight> lights,
                     const double eye[3], float lamp_gap, RtManyBins* out);

// MSL after kRtMsl and kRtShadowMsl: the passes.
extern const char* const kRtManyLightsMsl;

// The G-buffer a pass lights (each texture the depth's size; surface and
// class optional: nil).
struct RtManyGBuffer {
  __unsafe_unretained id<MTLTexture> depth = nil, albedo = nil, normal = nil, surface = nil, cls = nil;
};

class RtManyLights {
 public:
  // Null if `dev` does not trace rays or the passes cannot be made. `empty`:
  // whether the groups nothing lights are skipped (1) or shaded, written
  // and added like the others (0); -1: as RECOMP_RT_MANY_EMPTY says (unset:
  // skipped), read here, once.
  static std::unique_ptr<RtManyLights> Make(id<MTLDevice> dev, int empty = -1);

  // Frame slot `slot`'s (0 to kSlots - 1) light buffers filled from `bins`,
  // the samples traced against `f`'s TLAS from `g` (p as the local lights'
  // masks take it: RtShadowParamsFor; mp the pass's, its count and tiles
  // from `bins`), then the light added into `target` (level `level`, a
  // colour format a render pipeline can blend) in a render pass of its own.
  // Adds to the four counters at `counts` (16 bytes): the rays traced, those
  // blocked, and, where the groups nothing lights are skipped, the shading's
  // groups (kRtManyGroup pixels a side) none of whose pixels came out lit
  // and, of those, the ones whose gather kept no light (else both stay).
  // Nothing where an input is missing or the sizes do not agree. Encoding
  // side; the slot's buffers must not be in use by the GPU. `parts`: 1 the
  // trace only, 2 the shading only (after a trace of the same inputs), 3
  // both (for timing them apart); with kNoAdd, the light is left for
  // EncodeAddInto or EncodeAddPass instead of its own render pass. With
  // `retain_hits` (kRtManyRetainSlots uints, zeroed): the trace also counts,
  // by slot, the retained instances lying within a few centimetres over the
  // points it lights (RtScene::DropRetained's evidence).
  static constexpr int kSlots = 4;
  static constexpr uint32_t kNoAdd = 4;
  void Encode(id<MTLCommandBuffer> cmd, int slot, const RtFrame* f, const RtManyGBuffer& g, const RtShadowParams& p,
              const RtManyParams& mp, const RtManyBins& bins, id<MTLTexture> target, uint32_t level,
              id<MTLBuffer> counts, size_t counts_offset, uint32_t parts = 3, id<MTLBuffer> retain_hits = nil) const;

  // The samples of the last Encode (RG32Uint, a texel a block: x the two
  // lights and their visibility, RtManySample; y the point's distance's
  // float bits, 0 for the sky): for tests.
  id<MTLTexture> Samples() const { return samples_; }

  // The last Encode's light added into `target` (level `level`) in a
  // render pass of its own (what Encode does without kNoAdd).
  void EncodeAddPass(id<MTLCommandBuffer> cmd, id<MTLTexture> target, uint32_t level) const;
  // Or drawn into the open render encoder `e` (its first draw: the pass's
  // own load of the target, no store and load of their own), whose
  // attachments are `colors` (`targets` of them; the light into the first,
  // the others untouched), `depth` and `stencil` (Invalid: none) at
  // `samples`. The encoder's pipeline and fragment texture 0 are changed
  // (the caller binds its own again), and, where the groups nothing lights
  // are skipped, fragment texture kRtManyFlagsTexture (their flags, left
  // bound: no other draw reads that index). False, nothing drawn, if no
  // pipeline can be made for those attachments.
  bool EncodeAddInto(id<MTLRenderCommandEncoder> e, const MTLPixelFormat colors[4], int targets, MTLPixelFormat depth,
                     MTLPixelFormat stencil, NSUInteger samples) const;

 private:
  // The addition's pipeline for those attachments (made once each).
  struct AddKey {
    MTLPixelFormat colors[4];
    int targets;
    MTLPixelFormat depth, stencil;
    NSUInteger samples;
    bool operator==(const AddKey& o) const {
      for (int i = 0; i < 4; i++)
        if (colors[i] != o.colors[i]) return false;
      return targets == o.targets && depth == o.depth && stencil == o.stencil && samples == o.samples;
    }
  };
  id<MTLRenderPipelineState> AddPipe(const AddKey& k) const;
  id<MTLRenderPipelineState> AddPipe(MTLPixelFormat pf) const {
    AddKey k{{pf, MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid},
             1,
             MTLPixelFormatInvalid,
             MTLPixelFormatInvalid,
             1};
    return AddPipe(k);
  }

  id<MTLDevice> dev_ = nil;
  id<MTLLibrary> lib_ = nil;
  id<MTLComputePipelineState> trace_depth_pipe_ = nil, trace_color_pipe_ = nil;
  id<MTLIntersectionFunctionTable> trace_depth_table_ = nil, trace_color_table_ = nil;
  id<MTLComputePipelineState> shade_depth_pipe_ = nil, shade_color_pipe_ = nil;
  id<MTLIntersectionFunctionTable> shade_depth_table_ = nil, shade_color_table_ = nil;
  mutable std::vector<std::pair<AddKey, id<MTLRenderPipelineState>>> add_pipes_;  // (encoding side)
  // The samples, and the light before it is added (RGBA16Float, the
  // depth's size).
  mutable id<MTLTexture> samples_ = nil, lit_ = nil;
  // Whether the groups nothing lights are skipped (Make), and the last
  // shading's flag a group (R8Uint, the depth's size / kRtManyGroup rounded
  // up: 1 some pixel of it lit; 0 none, its texels of lit_ not written, an
  // earlier frame's). With empty_, whatever reads lit_ reads the flag first.
  bool empty_ = true;
  // Whether the last Encode shaded (lit_ and flags_ are this frame's): the
  // additions add nothing otherwise (no lights this frame).
  mutable bool shaded_ = false;
  mutable id<MTLTexture> flags_ = nil;
  struct Slot {
    id<MTLBuffer> lights = nil, heads = nil, index = nil;
  };
  mutable Slot slots_[kSlots];
  id<MTLTexture> white_ = nil;
};

// A sample's x (RtManySamples): light index + 1 in bits 0-10 (0: none),
// seen in bit 11; the second light in bits 12-22, seen in bit 23.
struct RtManySample {
  uint32_t first = 0, second = 0;  // light index + 1, 0: none
  bool first_seen = false, second_seen = false;
};
inline RtManySample RtManySampleOf(uint32_t x) {
  RtManySample s;
  s.first = x & 0x7FFu;
  s.first_seen = (x >> 11) & 1u;
  s.second = (x >> 12) & 0x7FFu;
  s.second_seen = (x >> 23) & 1u;
  return s;
}
static_assert(kRtLightsMax < 0x7FF, "light indices + 1 fit 11 bits");

}  // namespace recomp::gfx
