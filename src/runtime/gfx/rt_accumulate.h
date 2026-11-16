#pragma once
// A term of the lighting traced sparsely (one ray for every step x step
// pixels: a shadow mask, later the light of many lights) made into a
// stable texture at the depth buffer's own resolution by accumulating it
// over frames (Objective-C++, Metal compute). No upscaler: every pixel of
// the output is the depth's, only the rays are fewer, reused across frames.
//
// Input, each frame: the term at one pixel of every step x step block, the
// pixel turning from frame to frame (RtLightSample with a stride and the
// full frame count: every pixel of a block once in step^2 frames, on every
// jitter phase over the cycles), and each sample's point and place in its
// block, as RtShadowMasker::EncodeSparse traces a far light's mask; the
// frame's depth (the scene's camera, its jitter removed where a temporal
// upscaler moved the draws: RtShadowParams' points, which the trace itself
// starts from) and, optionally, G-buffer normals (else the surfaces'
// normals come from the depth). Output: the term at every pixel.
//
// Two passes a frame (each timed by its name in the GPU pass report), as
// independent implementations of shadow and lighting denoisers do it (AMD's
// FidelityFX shadow denoiser, SVGF and its adaptive form A-SVGF, Apple's
// MPSSVGF); none of them takes sparse input as given, so the fill is this
// file's:
//  1. "rt accumulate temporal" (a group a tile of 8 x 8 blocks, the history
//     at unjittered pixel centres): the samples of the tile and a ring
//     around in threadgroup memory, each with its change against the
//     history (A-SVGF's temporal gradient, made robust to the jitter): how
//     far it lies outside the range of the history's 3 x 3 pixels around
//     it (a value none of them held: a moved shadow; judged where 5 of them
//     or more hold a sample of their own), and, where every sample around
//     agrees, how far from the history of its own pixel (what a thin or
//     partly covered caster left behind); each block's statistics of the 3
//     x 3 blocks around (range, mean and spread, the largest change). Each
//     pixel: its point; where every sample around agrees (most of the
//     screen) that value is the fill, else its normal (the normal
//     texture's, else the geometric one from the depth) and the samples on
//     its surface (plane distance within kRtAccumulatePlane of its
//     distance) of the blocks on its side (both sides for the middle row or
//     column of an odd step) give the fill, their tent-weighted mean (no
//     sample there: a hole, counted); the history reprojected by the camera
//     alone (this frame's and the history's, both the scene's, in double on
//     the CPU) and read bilinearly, each tap kept only on the same surface
//     (its distance from the camera within kRtAccumulateDepth relative,
//     more on surfaces seen at a slant; off the uniform path also its point
//     within a pixel's footprint of the pixel's plane: not across a
//     crease); where a change is seen around, the history clamped to the
//     samples' mean +- kRtAccumulateSigma sigma (within their range) and its
//     sample count cut by as much; the pixel's own sample (where this
//     frame's landed on it) added as a running mean of up to
//     kRtAccumulateMaxCount samples. A pixel with less than one sample of
//     its own shows the fill in proportion. Where every sample around agrees
//     and the history does too, the output is that value exactly (counted:
//     "uniform"). A tile with a pixel of less than one sample (where not
//     uniform) is listed.
//  2. "rt accumulate filter" (indirect: a group a listed tile): an
//     edge-aware a-trous step over its pixels with less than one sample of
//     their own (3 x 3 taps one pixel apart, 5 x 5 under half a sample,
//     stopping at another surface's plane and, with a normal texture,
//     normal; the pixel's own weight the smaller the fewer samples it
//     holds). Uniform regions and settled history pass through untouched.
// The history: an R32Uint a pixel (the distance from the camera, 16 bits
// logarithmic: 0.07% steps; the sample count, 6 bits; a one-channel term's
// value, 10 bits) and for three channels an RGBA16Float of the value, two
// of each traded every frame; dropped (the frame then the fill and the
// filter alone) on a reset asked for, a size, channel or step change, a
// first frame, a gap (the frame before not accumulated), the light turned
// by a degree or more, and a camera cut (RtAccumulateCut). It keeps no
// normal (a normal would double what the pass reads and writes a pixel; the
// plane check above stands in for it off the uniform path).
//
// Limits, measured (rt_accumulate_test): no per-object motion; a moving
// caster's shadow is found inside it within the frame it moves, but its
// edges clear only by the pixels' own samples (within step^2 frames), and
// under the jitter shadows' edges stay noisier than the term averaged over
// the phases. At 3440 x 1440 the passes cost about 0.9 ms. As the sun's mask
// (metal_device.mm) this costs more than the fewer rays save and leaves
// trails behind moving shadows: off there unless asked for.

#if !__has_feature(objc_arc)
#error "rt_accumulate.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

// The samples on a pixel's surface: within this fraction of its distance
// from the camera of its plane.
inline constexpr float kRtAccumulatePlane = 0.01f;
// A history tap on the pixel's surface: its distance from the camera within
// this fraction of the reprojected point's, plus 1.5 pixels' worth of the
// surface's slant (AMD's denoiser: 1% to 10%; where every sample around
// agrees no normal is made, and a slant of 4 is allowed).
inline constexpr float kRtAccumulateDepth = 0.01f;
// Samples a pixel's history holds at most (a running mean over that many:
// fewer follow changes the samples do not see sooner, more average more
// jitter phases). The kernels' RT_ACC_MOST (the count's encoding).
inline constexpr float kRtAccumulateMaxCount = 4.0f;
// The clamp where a change is seen: the samples' mean +- this many standard
// deviations, within their range.
inline constexpr float kRtAccumulateSigma = 1.25f;
// The sun's (a far light's) mask's default step with the accumulation
// (RECOMP_RT_ACCUMULATE_STEP: 2 to 4).
inline constexpr uint32_t kRtAccumulateStep = 3;
// The light's direction turned this much or more (cosine: about a degree)
// drops the history.
inline constexpr double kRtAccumulateLightCos = 0.99985;

// A frame's camera as the history keeps it: clip space to world (D3D's row
// vectors), in double (RtFrame::inv_view_proj_exact where the scene gives
// it, else the float one), and its position (RtLightsEye: the point the
// passes' points are relative to).
struct RtAccumulateCamera {
  double inv[16] = {};
  double eye[3] = {};
  bool valid = false;
};
// `f`'s; false (valid false) where it cannot be inverted or has no
// position.
bool RtAccumulateCameraOf(const RtFrame& f, RtAccumulateCamera* out);
// Whether `cur` after `prev` is a cut (the history is dropped): a point of
// the screen (its centre, and four towards the corners) moves by more than
// a third of the screen, or to behind the last camera, at either of two
// distances along its ray: the far plane's (a turn) and the geometric mean
// of the near and far planes' (a jump; ten times the near plane's distance
// where the far plane is at infinity). Nothing in world units. True where
// either is not valid.
bool RtAccumulateCut(const RtAccumulateCamera& prev, const RtAccumulateCamera& cur);
// The rows taking a point relative to `cur`'s position to `prev`'s clip
// space (c = q.x rows[0] + q.y rows[1] + q.z rows[2] + rows[3]), and
// cur's position less prev's; false where prev cannot be inverted or
// either is not valid.
bool RtAccumulateReproject(const RtAccumulateCamera& prev, const RtAccumulateCamera& cur, float rows[4][4],
                           float eye_delta[3]);

// What the passes are given (MSL: RtAccParams), made by Begin.
struct RtAccumulateParams {
  float rx[4], ry[4], rv[4], rc[4];  // a pixel's point relative to the camera, as RtShadowParams'
  // The point relative to this camera to the history's clip space: c = q.x
  // prev[0] + q.y prev[1] + q.z prev[2] + prev[3] (its x, y and w).
  float prev[4][4];
  float eye_delta[4];  // xyz: this camera's position less the history's
  float sky_value[4];  // the term where the depth is the sky's
  uint32_t size[2];
  float jitter[2];  // the scene's offset on screen, pixels (right, down)
  uint32_t low[2];  // the sparse term's texels
  uint32_t step, stride, frame;
  uint32_t depth_texture;  // 1: a depth texture (read as depth2d), 0: a float colour one (its red)
  float sky;               // the stored value of the far plane (0 or 1)
  float pixel;             // the angle between neighbouring pixels' rays (radians, at the centre)
  uint32_t history;        // 1: the history is read
  uint32_t normals;        // 1: the normal texture gives the pixels' normals (else the depth)
  float max_count;         // kRtAccumulateMaxCount
  float scale;             // the term's full value (a change is judged against it); 0: the larger of sample and history
  // A history pixel's ray relative to the history camera's position:
  // (x hdir[0] + y hdir[1] + hdir[2]).xyz / .w for its clip x, y.
  float hdir[3][4];
};
static_assert(sizeof(RtAccumulateParams) == 272, "kRtAccumulateMsl's RtAccParams");

// What a term is.
struct RtAccumulateTerm {
  uint32_t channels = 1;              // 1 (a mask: R8Unorm or R16Float in, out) or 3 (RGBA16Float)
  float sky[3] = {1.0f, 1.0f, 1.0f};  // its value where the depth is the sky's
  float scale = 1.0f;                 // RtAccumulateParams::scale
};

// The temporal pass's groups take a tile of kRtAccumulateGroup x
// kRtAccumulateGroup blocks (24 x 24 pixels at step 3), with as many
// threads; the tiles of w x h pixels at `step`.
inline constexpr uint32_t kRtAccumulateGroup = 8;
uint32_t RtAccumulateTiles(uint32_t w, uint32_t h, uint32_t step);

// One term's history (its owner keeps it from frame to frame; one a term).
struct RtAccumulateHistory {
  // Per pixel: distance, sample count and (one channel) value (R32Uint);
  // three channels' value (RGBA16Float; 1 x 1 for one).
  id<MTLTexture> hist[2] = {}, color[2] = {};
  // The tiles to filter: two lists (a frame's, the next's), each a count as
  // indirect dispatch arguments (at 0 and 16) and tile numbers (from 32,
  // tiles_most each: the tiles at step 1).
  id<MTLBuffer> tiles = nil;
  uint32_t tiles_most = 0;
  uint32_t w = 0, h = 0, channels = 0;
  int cur = 0;                      // which of hist/color holds the history
  bool have = false;                // a frame was accumulated
  RtAccumulateCamera camera;        // that frame's
  uint32_t frame = 0, step = 0;     // that frame's number and step
  float light[3] = {};              // its light (Begin's)
  uint64_t frames = 0, resets = 0;  // Begin's frames, those that dropped the history
};

// One frame's passes (Begin's): plain values and unretained textures (the
// history holds them; those it lets go are handed to Begin's caller).
struct RtAccumulateFrame {
  RtAccumulateParams p{};
  uint32_t channels = 1;
  bool reset = false;  // the history dropped this frame
  bool again = false;  // Begin refused: this frame's number accumulated already
  __unsafe_unretained id<MTLTexture> hist_in = nil, color_in = nil, hist_out = nil, color_out = nil;
  __unsafe_unretained id<MTLBuffer> tiles = nil;
  size_t tiles_args = 0, tiles_next = 0, tiles_list = 0;  // this frame's count, the next's, this frame's list
};

// What the passes count (Encode's `counts`, eight 32-bit counters, shared
// memory, added to): pixels that took the uniform shortcut, pixels without
// a history, pixels without a sample on their surface (holes), tiles
// filtered, pixels where a change was seen (its weight at least a half).
struct RtAccumulateCounts {
  uint32_t uniform = 0, fresh = 0, holes = 0, tiles = 0, changed = 0, pad[3] = {};
};
static_assert(sizeof(RtAccumulateCounts) == 32);

// MSL (standalone): the passes.
extern const char* const kRtAccumulateMsl;

class RtAccumulator {
 public:
  // Null if the kernels do not compile (said on stderr). Any Metal GPU.
  static std::unique_ptr<RtAccumulator> Make(id<MTLDevice> dev);
  // On the recording side: the frame of `h` for a sparse term at `step`
  // (its sample turning with `stride`: RtLightSample) over the depth `p`
  // describes (RtShadowParamsFor of the frame `f`, the same camera and
  // jitter the samples were traced with), `frame` the presented frame's
  // number (the samples' turn too), `light` the light's direction where
  // the term has one (else null). The history dropped where `reset`, at a
  // new size or channel count or step, without a camera before or at a cut
  // (RtAccumulateCut), where the frame before was not accumulated (a gap:
  // the history would be older than its camera says), and where the light
  // turned by a degree or more; textures (re)made as needed (those let go
  // added to `retired` where given: recorded work may still name them).
  // `h` then holds this frame. False where the camera cannot be used, a
  // texture cannot be made (the history then dropped next time), or this
  // frame's number was accumulated already (out->again: a second trace in
  // one frame; the caller makes it without the accumulation). Everything
  // that can fail before Encode must be checked before Begin: Encode is
  // then to be called with this frame (else the next reads a stale one).
  bool Begin(RtAccumulateHistory* h, const RtFrame& f, const RtShadowParams& p, const RtAccumulateTerm& term,
             uint32_t step, uint32_t stride, uint32_t frame, const float light[3], bool reset, RtAccumulateFrame* out,
             std::vector<id>* retired = nullptr) const;
  // The passes (Begin's frame): `sparse` (the term at the samples: R8Unorm
  // or R16Float, RGBA16Float for three channels; Begin's step) and
  // `points` (RGBA16Float, the same texels: each sample's point relative to
  // the camera, RtShadowParams::eye, and in w its place in its block, x + 8
  // y; w below 0: no sample, the sky's; as RtShadowMasker::EncodeSparse
  // writes them), `depth` (p's, ShaderRead), `normals` (xyz as unorm, the
  // depth's size: the surfaces' normals; nil: from the depth) into `out`
  // (the depth's size, ShaderWrite: R8Unorm or R16Float, or RGBA16Float for
  // three channels). Counts added at `counts` + offset (RtAccumulateCounts)
  // where given.
  void Encode(id<MTLCommandBuffer> cmd, const RtAccumulateFrame& fr, id<MTLTexture> sparse, id<MTLTexture> points,
              id<MTLTexture> depth, id<MTLTexture> normals, id<MTLTexture> out, id<MTLBuffer> counts = nil,
              size_t counts_offset = 0) const;

 private:
  id<MTLDevice> dev_ = nil;
  // [channels 1, 3][depth texture, float colour one]
  id<MTLComputePipelineState> temporal_[2][2] = {}, filter_[2][2] = {};
  id<MTLTexture> no_normals_ = nil;  // bound where none are given
  id<MTLBuffer> no_counts_ = nil;    // written where no counts are asked for
};

}  // namespace recomp::gfx
