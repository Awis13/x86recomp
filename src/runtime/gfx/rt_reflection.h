#pragma once
// Ray-traced reflections (Objective-C++), on GPUs with hardware ray
// tracing: for the depth buffer of a frame's main view and a deferred
// renderer's G-buffer (normals, specular intensity and gloss: gfx::
// RayReflection), what each shiny pixel's mirror reflection sees, traced
// against a TLAS of the frame's scene (gfx/rt_scene.h, kept with hit data:
// RtHitPrim, albedos). Written at the depth buffer's size as RGBA16Float
// (rgb the light reflected times RayReflection::output_scale, a 1 where a
// ray hit and 0 elsewhere) and R16Float (the hit's distance, for a
// denoiser later: kRtReflectionSkyDistance where a ray missed, 0 where none
// was traced). The Metal backend traces them before the first draw of a
// pixel shader the game names (gfx::Device::SetRayReflection) and binds the
// first in place of the texture the shader had at that sampler; the shader
// keeps its own reflection where a is 0.
//
// A pixel: its point from the depth as the shadow masks make it (gfx/
// rt_shadow.h: D3D9's pixel centres, the jitter, relative to the camera).
// Traced where it is not the sky, is shiny (a specular intensity above 0,
// or wet) and its gloss is sharp enough (the game's blur of its own
// reflection at most kRtReflectionMaxBlur). One ray along reflect(I, N):
// I from the camera to the point, N the G-buffer's normal (the surface's,
// bumps included, not a direction a game bends for an environment map),
// starting off the surface as a shadow ray does along the depth's normal
// (turned toward the camera: the side the surface is seen from), to a
// length that falls with the gloss's roughness (FidelityFX's hybrid
// reflections: RayReflection::max_distance exp(-5 roughness), the
// roughness of the Blinn-Phong exponent e, sqrt(2 / (e + 2))). A bumped or
// coarsely stored normal can send the ray below the surface: it is
// straightened to kRtReflectionGraze above it (else wet roads at grazing
// angles reflected what lies under them). Closest hit, alpha-cut instances
// by their materials, no face culled: a back face of an instance the draw
// culled (single-sided geometry seen from behind) stops the ray with a 0,
// the game's own reflection there. Not for another view's instances
// (RtView::Other, kRtInfoOtherView): a shadow map's pass culls and projects
// as it likes (front faces culled, a projection flipping y), so which side
// the main camera would see is not known; they are lit from either side. Static instances from the surface on,
// captured ones (people, vehicles) beyond kRtReflectionCapturedGapPixels
// pixel footprints: unlike a shadow ray, which starts on such a mesh as
// often as it is drawn, a reflection ray does only from a captured mesh's
// own pixels in the lighting pass (a game drawing those forward, after it,
// never does), and a gap of the shadows' 4% hid people and vehicles
// standing a metre from a puddle 25 m away. What it hits is lit as a deferred
// renderer's sun pass lights it: its albedo (RtHitAlbedo, at the level of
// the ray's cone: the pixel's angle times the camera's distance plus the
// ray's), the light (its colour, the wrapped diffuse term, a shadow ray
// from the hit, two-phase as above) and the ambient term; its normal turned
// toward the ray. A hit on an instance without an albedo counts as a miss
// (a 0: the shader's own reflection), apart in the counters.
//
// The checkerboard (RtReflectionParams::checker, where a temporal pass
// follows): half the listed pixels traced, x + y + the jitter phase's
// parity even; each of the others the mean of its four traced neighbours on
// the same surface (kRtReflectionFillPixels, kRtReflectionFillCos), or
// traced too where none is. The history of the pass after gathers both
// halves.
//
// How (RtReflector::Encode, one compute encoder): the pixels to trace are
// listed first (a light pass reading the surface texture of every pixel,
// the depth and normal only of the shiny ones, appending a SIMD-group's at
// once), then traced one thread a listed pixel through an indirect
// dispatch, SIMD-groups full of neighbouring rays, with the intersector
// object and the alpha test as an intersection function (Apple family 9:
// hardware traversal and the reorder stage; Apple tech talk 111375). The
// outputs hold 0 wherever nothing is reflected: the listing writes 0 only
// where its distance says the last frame left something (a fresh output
// is written whole once), so pixels without a ray cost a read. The alpha
// test samples with the sampler's filtering where the texture wraps or
// clamps (one sample, not four by hand). One traversal
// of static and captured instances stands for the two phases below unless
// it meets a captured instance within the gap.
//
// A pass is traced per depth contents and per what its shader traces with
// (RtReflectionKey: the RayReflection, the draw's constants and G-buffer
// textures); RtReflectionCache keeps kEntries of those a presented frame,
// each with its own textures, so shaders that alternate do not trace again.
//
// Limits: hits are lit by the one light and the ambient term only (no
// light volumes: at night they come out dark); hits are shaded with the
// triangle's flat normal (faceted); no rough reflections (no blur, no
// denoiser): pixels blurrier than kRtReflectionMaxBlur keep the game's;
// captured meshes are a frame behind where the snapshot carries them, and
// their UVs are gathered on the CPU every frame.

#if !__has_feature(objc_arc)
#error "rt_reflection.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

#include "runtime/gfx/device.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

// Traced where the game's own reflection is blurred by at most this (0 its
// sharpest, 1 its blurriest: RayReflection's blur): above it the game's
// blurred environment map is nearer what it means than a mirror's ray. A
// half is FidelityFX's roughness cutoff (0.22) on a game whose blur spans
// four mip levels of its map.
inline constexpr float kRtReflectionMaxBlur = 0.5f;
// The cosine a ray keeps at least with the surface it leaves (straightened
// to it): about a degree above it.
inline constexpr float kRtReflectionGraze = 0.02f;
// Captured instances are hit beyond this many pixel footprints at the
// point (its distance times the angle between neighbouring pixels).
inline constexpr float kRtReflectionCapturedGapPixels = 8.0f;
// The hit distance written where a ray missed: the largest half. Rays are
// at most kRtReflectionLongest long (RayReflection::max_distance clamped),
// so a hit's distance is always below half of it (what the debug view
// tells a miss by).
inline constexpr float kRtReflectionSkyDistance = 65504.0f;
// The checkerboard's fill (RtReflectionParams::checker): a neighbour lends
// its reflection where its distance from the camera is within this many
// pixel footprints of the pixel's (the pixels' angle times the distance,
// over the cosine of the surface's normal to the view) and 1%, and its
// G-buffer normal within this cosine (about 25 degrees).
inline constexpr float kRtReflectionFillPixels = 2.0f;
inline constexpr float kRtReflectionFillCos = 0.9f;
// With the checkerboard, wet ground (the stencil's, wetness above 0) whose
// own specular intensity is below kRtReflectionSparseSpec (GTA IV's
// lighting weights a wet surface's reflection at most 0.25, a shiny one's
// 2 spec: below a half, not a full mirror) is traced one pixel of 2 x 2 a
// frame, and one of
// 4 x 4 where the cosine between the view and its normal is at least this
// (seen from above: the Fresnel term near its least); the others filled
// from traced neighbours on the same surface (3 x 3, 5 x 5 around), the
// places taking turns frame by frame. Car paint, glass and every surface
// with a specular of its own keep the checkerboard.
inline constexpr float kRtReflectionSparseCos = 0.4f;
inline constexpr float kRtReflectionSparseSpec = 0.5f;
// Wet ground's rays (the stencil's, wetness above 0) are at most this long
// (metres): past it the ray counts as a miss, and the lighting keeps its own
// reflection (its environment map), as shipped hybrids do (rays near, the
// environment map far). Car paint, glass and dry surfaces keep
// RayReflection::max_distance's length.
inline constexpr float kRtReflectionWetDistance = 25.0f;
inline constexpr float kRtReflectionLongest = 16384.0f;

// What the kernel is given (MSL: RtReflectionParams), made on the CPU
// (RtReflectionParamsFor).
struct RtReflectionParams {
  RtShadowParams geo;  // the pixel's point, the light (toward it), the ray's start (diag 0)
  float colour[4];     // the light's colour (rgb times its w)
  float ambient[2][4]; // a, b: a + b sat(0.5 - 0.5 N.up)
  float up[4];
  float scale, wrap, exponent_scale, blur_k;  // blur = 1 - sat(e blur_k)
  float max_blur, max_distance, sky_distance, wet;  // wet: the wetness (RayReflection's register)
  float wet_slope, wet_gain, wet_max_exponent;
  uint32_t wet_on;  // a stencil texture bound, wetness asked for
  uint32_t spec_channel, gloss_channel;
  float graze;      // kRtReflectionGraze
  float gap_pixels; // kRtReflectionCapturedGapPixels
  // The checkerboard (0: every listed pixel traced): the pixels with x + y
  // + parity odd filled from their neighbours (the backend sets both: on
  // where the scene is jittered for a temporal pass, the parity the
  // jitter's phase's).
  uint32_t checker = 0, parity = 0;
  float fill_depth; // kRtReflectionFillPixels
  float fill_cos;   // kRtReflectionFillCos
  // A comparison (RECOMP_RT_REFL_DIAG): bit 0 the hits lit without their
  // shadow ray, bit 1 every instance taken as opaque (no alpha test).
  uint32_t diag = 0;
  uint32_t frame = 0;      // the sparse rates' place in their blocks (the backend's frame count)
  float sparse_cos = 0.4f; // kRtReflectionSparseCos
  float sparse_spec = 0.5f;  // kRtReflectionSparseSpec
  float wet_distance = 25.0f;  // kRtReflectionWetDistance
  // With the checkerboard, every reflecting pixel traced one of 2 x 2 a
  // frame (the backend: where the far light reaches nothing, under ground
  // and at night; a subway train's cars made 10 ms a frame of reflections
  // at a station's platform), the others filled as wet ground's.
  uint32_t sparse_all = 0;
  uint32_t pad2[2] = {};
};
static_assert(sizeof(RtReflectionParams) == 336, "kRtReflectionMsl's RtReflectionParams");

// The bounce's parameters (MSL: RtBounceParams).
inline constexpr uint32_t kRtBounceBlock = 4;
// The tiled filter's threadgroup, blocks a side (MSL: RT_BOUNCE_GROUP).
inline constexpr uint32_t kRtBounceGroup = 8;
struct RtBounceParams {
  uint32_t frame = 0;   // the sample's turn in its block and the ray's direction
  uint32_t stride = 0;  // RtLightStride(kRtBounceBlock), or 0: every block's top-left pixel
  float distance = 40.0f;   // a bounce ray's length, world units
  float strength = 1.0f;
  float blend = 0.15f;      // this frame's share against a block's last result (1: no history)
  // The angle (radians) between a block's ray and its last one's within
  // which it takes its last result (the backend: a block and a half's
  // width, 6 pixels' angle): a camera turned further takes fresh samples,
  // so no pattern stays on the screen while the world moves under it (a
  // check of the point's distance at 3% of it did keep one through turns:
  // it showed as a fine golden grain in the darks).
  float tolerance = 0.002f;
  uint32_t blocks[2] = {};  // (EncodeBounce's)
  // The sky light's occlusion from the same rays (the output's alpha): a
  // ray that stopped within ao_radius takes its share of the sky by how
  // far it got; ao_strength 0 none, 1 all of it.
  float ao_radius = 3.0f, ao_strength = 1.0f;
  // The last frame's result is read where this block's point was on its
  // screen (history 1): prev takes a point relative to this camera to the
  // last one's clip space, eye_delta this camera's position less the last
  // one's (gfx/rt_accumulate.h, RtAccumulateReproject); size the depth's.
  uint32_t history = 0, pad0 = 0;
  float prev[4][4] = {};
  float eye_delta[4] = {};
  uint32_t size[2] = {}, pad1[2] = {};
};
static_assert(sizeof(RtBounceParams) == 144, "kRtBounceMsl's RtBounceParams");

// What a draw's reflections are traced with, from its pixel shader
// constants `ps_float` (c0-c223) as `r` names them (RtReflectionInputsFor).
struct RtReflectionInputs {
  float light[3] = {};  // toward the light
  float colour[3] = {};
  float ambient[2][3] = {};
  float blur_k = 0, wet = 0;
};
static_assert(sizeof(RtReflectionInputs) == 56, "floats only: compared bitwise");
RtReflectionInputs RtReflectionInputsFor(const RayReflection& r, const float (*ps_float)[4]);

// What a pass is traced from besides the depth: its shader's
// RayReflection, the draw's inputs and G-buffer textures (normal,
// surface, stencil; null: none).
struct RtReflectionKey {
  RayReflection config;
  RtReflectionInputs inputs;
  const void* textures[3] = {};
};
// The same: every field, floats bit for bit (a NaN equals itself).
bool RtReflectionSameKey(const RtReflectionKey& a, const RtReflectionKey& b);
static_assert(sizeof(RayReflection) == 104, "a field added to RayReflection: compare it in RtReflectionSameKey");
// The passes of a presented frame (reset, {}, when it ends): kEntries keys,
// each with the choice RtShadowNext made last for it (memo).
struct RtReflectionCache {
  static constexpr int kEntries = 2;
  struct Entry {
    bool used = false;
    RtReflectionKey key;
    RtShadowMemo memo;
    uint64_t last_use = 0;
  };
  Entry entry[kEntries];
  uint64_t uses = 0;
};
// The entry for `key`: the one holding it, else the least recently used
// taken over (its memo reset: traced again).
int RtReflectionEntry(RtReflectionCache* c, const RtReflectionKey& key);

// The parameters for a w x h buffer of the frame `f`'s camera, depth as
// RtShadowParamsFor takes it, `r` and `in` (the draw's), `wet_on` where a
// stencil texture is bound. Without a light (no length) the hits get the
// ambient term only. False if the camera cannot be used.
bool RtReflectionParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                           const float jitter[2], const RayReflection& r, const RtReflectionInputs& in,
                           bool wet_on, RtReflectionParams* out);

// Screen-space reflections first (RtReflector::Encode's `ssr`): each pixel
// to reflect marches its ray through the depth buffer (McGuire and Mara,
// "Efficient GPU Screen-Space Ray Tracing", JCGT 2014: the ends projected,
// 1/w stepped across the screen, a crossing refined by halves); where it
// meets a surface facing it, its reflection is the previous frame's lit
// scene at that point (kept by EncodeKeep before the post-processing;
// reprojected with that frame's camera) times RayReflection::output_scale,
// and no ray is traced. A ray is traced where the march leaves the screen,
// finds nothing, meets a back face (what it meets is hidden), or the point
// was not in the previous frame; within kRtSsrEdge pixels of the screen's
// edges both, mixed. The lit scene holds the light the lighting computed
// (every light, the specular and the reflections), the trace's hits the
// sun and the ambient only: where both are known they differ at night.
inline constexpr float kRtSsrThickness = 0.3f;      // metres a surface is assumed deep, at least
inline constexpr float kRtSsrThicknessRel = 0.03f;  // and this fraction of its view depth
inline constexpr float kRtSsrEdge = 64.0f;          // pixels
inline constexpr uint32_t kRtSsrSteps = 64, kRtSsrRefine = 6;
// MSL: RtSsrParams. Matrices column after column (Metal's float4x4).
struct RtSsrParams {
  float to_clip[16];  // camera-relative point to (x, y, v, 1) times its view depth in metres
  float to_prev[16];  // camera-relative point to the previous frame's clip position (no jitter)
  float prev_size[2];
  float prev_jitter[2];  // the previous scene's offset, pixels (right, down)
  float thickness, thickness_rel, edge, scale;
  uint32_t steps, refine;
  float inv_depth[2];  // a stored depth v's 1 / view depth: [0] v + [1]
};
static_assert(sizeof(RtSsrParams) == 176, "kRtReflectionMsl's RtSsrParams");
// The parameters for reflections traced with `p` (RtReflectionParamsFor's:
// its camera) from a lit scene of prev_w x prev_h the camera `prev_vp`
// (world to clip, D3D's row vectors, without the jitter) drew `prev_jitter`
// pixels right and down; scale RayReflection::output_scale. False if the
// cameras cannot be used.
bool RtSsrParamsFor(const RtReflectionParams& p, const double prev_vp[16], const float prev_jitter[2],
                    uint32_t prev_w, uint32_t prev_h, float scale, RtSsrParams* out);

// The counters a trace adds to (shared memory, read once the GPU is done).
struct RtReflectionCounts {
  uint32_t traced;    // rays traced (pixels)
  uint32_t hits;      // rays that hit an instance with an albedo (a 1)
  uint32_t misses;    // rays that hit nothing
  uint32_t bare;      // rays that hit an instance without an albedo (a 0)
  float distance;     // the hits' distances, summed
  uint32_t back;      // rays stopped by a back face of an instance that culls (a 0)
  uint32_t back_captured;  // of those, on captured draws'
  uint32_t other_back;     // rays that hit a back face of another view's instance (lit, counted as hits)
  uint32_t filled;         // pixels of the checkerboard's untraced half filled from their neighbours
  uint32_t screen;         // pixels the screen-space march settled (no ray)
  uint32_t screen_mixed;   // pixels near the screen's edges it settled in part (traced too, mixed)
  uint32_t dense;          // rays of pixels traced at the checkerboard's rate (the rest sparser)
};
static_assert(sizeof(RtReflectionCounts) == 48, "kRtReflectionMsl's counters");

// MSL after kRtMsl: the reflection rays of this kernel and of forward
// pixel shaders tracing their own (gfx/rt_forward_reflection.h), shaded
// alike:
//   struct RtReflectLight  what the hits are lit with (the light toward
//     it and a shadow ray's length, its colour, the ambient's a and b, the
//     up axis, the diffuse wrap, the pixels' angle, the shadow rays'
//     captured gap and start offsets)
//   uint RtReflectRay(scene, materials, info, o, r, tmin, tmax, gap, dist,
//                     lt, rgb, hit_distance)
//     the closest hit of a ray (static instances from tmin, captured ones
//     beyond gap), its light (unscaled) and distance, and what it was (1 a
//     hit, 2 a miss, 3 no albedo, 4 and 5 a back face of a static and a
//     captured instance that culls, 6 another view's back face, lit).
extern const char* const kRtReflectRayMsl;

class RtReflector {
 public:
  // Null if `dev` does not trace rays (RtSupported) or the kernel does not
  // compile (said on stderr).
  static std::unique_ptr<RtReflector> Make(id<MTLDevice> dev);
  // The trace (one compute encoder): `depth` as RtShadowMasker::Encode
  // takes it; `normal`, `surface` (2D, ShaderRead; read at the pixel's
  // place scaled to their size) and `stencil` (nil: Miss() is read, dry)
  // the G-buffer's; into `out` (RGBA16Float) and `distance` (R16Float),
  // both ShaderWrite and the depth's size; against `f`'s TLAS (not nil),
  // traced with hit data. Adds to the RtReflectionCounts at `counts`.
  // (The output textures may be one set for every frame in flight: Metal
  // orders a frame's writes after the last one's reads.)
  // With `ssr` and `prev` (the previous frame's lit scene, as EncodeKeep
  // keeps it): screen space first (RtSsrParams).
  void Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> normal,
              id<MTLTexture> surface, id<MTLTexture> stencil, id<MTLTexture> out, id<MTLTexture> distance,
              const RtReflectionParams& p, id<MTLBuffer> counts, size_t counts_offset,
              const RtSsrParams* ssr = nullptr, id<MTLTexture> prev = nil) const;
  // The far light bounced once off what it lights (kRtBounceMsl), for the
  // lighting to add to its ambient: into `out` (NewBounceOut: RGBA16Float
  // of half the depth's size, rgb the light arriving, read with linear
  // filtering), from the depth and p (as Encode takes them, the same
  // frame's). Bounce(): whether its passes were made.
  bool Bounce() const { return bounce_filter_ != nil; }
  void EncodeBounce(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, const RtReflectionParams& p,
                    const RtBounceParams& bp, id<MTLTexture> out) const;
  id<MTLTexture> NewBounceOut(uint32_t w, uint32_t h) const;
  // 1 x 1 RGBA16Float (0, 0, 0, 1): no bounce, the sky open (for draws the
  // bounce was not traced for; Miss()'s zero alpha would black their
  // ambient out).
  id<MTLTexture> BounceNone() const { return bounce_none_; }
  // The lit scene `lit` (2D, ShaderRead, any float format) into `kept`
  // (NewKept's, the same size) for the next frame's march.
  void EncodeKeep(id<MTLCommandBuffer> cmd, id<MTLTexture> lit, id<MTLTexture> kept) const;
  id<MTLTexture> NewKept(uint32_t w, uint32_t h) const;
  // The reflections over `target` (a debug view: the colour over `scale`
  // where a ray hit, dark blue where one missed, purple where it stopped on
  // an instance without an albedo or a back face, black where none was
  // traced), scaled to its size. Encoding side.
  void EncodeView(id<MTLCommandBuffer> cmd, id<MTLTexture> out, id<MTLTexture> distance, float scale,
                  id<MTLTexture> target) const;
  // Textures of w x h (private, written by Encode, sampled after).
  id<MTLTexture> NewOutput(uint32_t w, uint32_t h) const;
  id<MTLTexture> NewDistance(uint32_t w, uint32_t h) const;
  // 1 x 1 RGBA16Float zeros: nothing hit (for draws whose reflections
  // cannot be traced).
  id<MTLTexture> Miss() const { return miss_; }
  // Whether the checkerboard may be used (RtReflectionParams::checker; off
  // with RECOMP_RT_REFL_CHECKER=0), and whether by turns (=ab: a
  // comparison within one run).
  bool Checker() const { return checker_; }
  bool CheckerAb() const { return checker_ab_; }
  // Whether screen space goes first (RtSsrParams; on with
  // RECOMP_RT_REFL_SSR=1, =ab by turns; off by default: in the game it
  // cost about what it saved).
  bool Ssr() const { return ssr_; }
  // RECOMP_RT_REFL_DIAG=1: RtReflectionParams::diag by turns (none, no hit
  // shadow rays, no alpha test), a comparison within one run.
  bool Diag() const { return diag_; }
  bool SsrAb() const { return ssr_ab_; }

 private:
  // The pixel list of an output (made with it, kept with it: an associated
  // object): kListHeader uints (the classification's count, the last
  // trace's length, whether the output is fresh, the trace's indirect
  // threadgroups at kListArgs), then a pixel per uint.
  static constexpr uint32_t kListHeader = 24, kListGroup = 64;
  // Where lists A, B and C start (uints), for an output of w x h.
  struct Layout {
    uint32_t a, b, c, pad;
  };
  static Layout LayoutFor(uint32_t w, uint32_t h);
  id<MTLBuffer> ListFor(id<MTLTexture> out) const;

  id<MTLDevice> dev_ = nil;
  // [0] a colour texture's depth, [1] a depth texture's.
  id<MTLComputePipelineState> classify_pipe_[2] = {}, trace_pipe_[2] = {}, fill_pipe_[2] = {}, ssr_pipe_[2] = {};
  id<MTLComputePipelineState> keep_pipe_ = nil;
  // The checkerboard's flags of an output (R8Uint, kept with it: 1 where
  // traced this frame, what the fill takes from); a 1 x 1 one bound
  // without.
  id<MTLTexture> FlagsFor(id<MTLTexture> out) const;
  id<MTLTexture> flags_none_ = nil;
  id<MTLIntersectionFunctionTable> table_[2] = {};  // trace_pipe_'s
  id<MTLComputePipelineState> args_pipe_ = nil, view_pipe_ = nil;
  id<MTLTexture> miss_ = nil, bounce_none_ = nil;
  bool checker_ = true, checker_ab_ = false, ssr_ = true, ssr_ab_ = false, diag_ = false;
  mutable id<MTLTexture> view_tex_ = nil;  // EncodeView's (encoding side)
  // The bounce (encoding side): its passes, a block's sample (colour,
  // point, normal) and the filtered blocks of this frame and the last.
  id<MTLComputePipelineState> bounce_trace_[2] = {}, bounce_up_[2] = {}, bounce_filter_ = nil;
  // The filter from a threadgroup's tile and the upsample by block, taken
  // in place of the two above where made (not with RECOMP_RT_BOUNCE_TILED=0).
  id<MTLComputePipelineState> bounce_filter_tile_ = nil, bounce_up_block_[2] = {};
  id<MTLIntersectionFunctionTable> bounce_table_[2] = {};
  mutable id<MTLTexture> b_rgb_ = nil, b_q_ = nil, b_n_ = nil, b_hist_[2] = {}, b_hist_q_[2] = {};
  mutable int b_turn_ = 0;
};

}  // namespace recomp::gfx
