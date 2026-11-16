#pragma once
// Ray-traced shadow masks (Objective-C++), on GPUs with hardware ray
// tracing: for the depth buffer of a frame's main view and a light far
// away (a sun or moon, given as a direction), whether each pixel's point
// sees the light, traced against a TLAS of the frame's scene (gfx/
// rt_scene.h, RtScene::Snapshot). One ray a pixel toward the light, alpha-
// cut instances tested by their materials (kRtMsl's RtTraceAny); written
// as R8Unorm at the depth buffer's size: 1 lit, 0 in shadow, 1 where the
// depth is the far plane's (the sky). The Metal backend makes one before
// the first draw of a pixel shader the game names (gfx::Device::RayShadow)
// and binds it in place of the shadow lookup the shader had.
//
// A pixel's point: its centre as D3D9 places it (pixel (i, j) of a w x h
// buffer at clip x = 2i / w - 1, y = 1 - 2j / h: the backend draws half a
// pixel right and down of Metal's centres, metal_prelude.h's r_position),
// less the frame's jitter where a temporal upscaler moved the scene's
// draws (RtFrame::inv_view_proj is the camera without it), and its depth,
// through the frame's camera.
// Computed relative to the camera's position (the rows below), so that
// points kilometres from the origin keep the precision near the camera;
// from a reversed buffer's stored value (1 - d) itself, not from d, which
// loses most of its bits far out.
//
// The ray starts off the surface by `off`: a share of the pixel's
// footprint at the point (its distance from the camera times the angle
// between neighbouring pixels' rays), of the distance (what drawing and
// tracing round differently) and of the point's distance from the origin
// (float rounding of the TLAS's coordinates). The surface's normal is the
// geometric one, made from the depth of the pixel's neighbours on the same
// surface (on each axis the side whose two neighbours' depths, affine
// across a plane, predict the pixel's: not the one across a crease or a
// depth edge); the G-buffer's (RayShadow::normal_sampler, xyz as unorm)
// only where the depth gives none: it is the shading normal, which a
// normal map tilts, at a grazing view or a low light away from the camera
// or the light where the surface itself is not (the start then below the
// surface, lit ground in shadow as the camera turns). Turned toward the
// camera. Seen from the light's side, the
// start is off / max(N.L, min_slope) along the light (clearing the surface
// by off however low the light: a step along the ray does not move the
// shadow's edges, one along the normal moves them by its length over the
// light's slope), plus along the normal what that leaves short of off
// where N.L is below min_slope. As AMD's FidelityFX hybrid shadows start a
// pixel's ray off the surface by a pixel's thickness, and Unreal's
// ray-traced shadows bias it along the ray more as N.L falls. Seen from the
// other side, off along the normal turned toward the light (the back of a
// leaf card still sees the light it is lit by; a solid's side away from
// the light starts inside it and stays dark).
// The point is the one the depth was drawn at: where the scene is
// jittered, the pixel's centre less the jitter (in the unjittered camera,
// the sample the rasterizer took), which lies on the surface. The centre
// itself would put it up to half a pixel's footprint off the surface, in
// and out of it as the jitter changes: the surface shadowing itself in a
// pattern that changes every frame.
// Nothing in it is in world units: it works whatever a game's scale.
//
// Hard shadows: one ray straight at the light (RtShadowParams::cone 0, as
// RtShadowParamsFor makes them, and as the Metal backend traces them).
// Soft ones need a denoiser of the mask over frames: with cone
// kRtShadowCone and `frame` the presented frame's number, each pixel's one
// ray goes in a direction of the cone that changes every frame
// (rt_cone_dir).
//
// Local lights (a point or spot light whose volume a deferred renderer
// draws, one light a draw: RayShadow::local): the same kernel with
// RtLightParams. A ray from each pixel's point to the light's position,
// ending lamp_gap short of it (the light sits inside its own lamp's mesh: a
// street lamp's housing, a car's headlight), captured instances beyond the
// same gap from the point as above. Pixels whose point lies outside the
// light's sphere, or outside its cone where it has one, are not traced
// (lit: the light adds nothing there); nor are those within lamp_gap of the
// light (lit). Only the pixels of the light's bounds on screen
// (RtLocalLightBounds) are dispatched; the mask's other texels keep what
// they had (the light adds nothing there either).
//
// Limits: a far light's rays have no end (a local light's end at it), so
// anything in the scene above the point blocks it: a sky dome drawn with
// depth writes into the main view shadows everything (the counters of rays that hit
// say so; a game leaves such draws out with its filter). Thin walls far
// out may let light through where the offset passes them; static meshes
// a vertex shader moves (wind in foliage) stand still for the rays.

#if !__has_feature(objc_arc)
#error "rt_shadow.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cstdint>
#include <memory>

#include "runtime/gfx/device.h"
#include "runtime/gfx/rt_scene.h"

namespace recomp::gfx {

// The half-angle of the cone a far light's shadow rays are traced in where
// a denoiser follows, radians: the sun's angular radius (0.0047: a disc of
// about 0.54 degrees seen from the earth), which a far light stands for.
// Without a denoiser the rays go straight at the light (cone 0: hard
// shadows), as one jittered ray a pixel flickers.
inline constexpr float kRtShadowCone = 0.0047f;
// The ray's start clears the surface (gfx/rt_shadow.h) by this many times
// the pixel's footprint at the point (its distance from the camera times
// the angle between neighbouring pixels' rays), plus this fraction of the
// distance itself (meshes drawn and traced differ by their rounding; a
// depth buffer not stored reversed loses bits far out), plus this
// fraction of the point's distance from the origin (float coordinates).
inline constexpr float kRtShadowOffsetPixels = 0.25f;
inline constexpr float kRtShadowOffsetView = 1e-4f;
inline constexpr float kRtShadowOffsetWorld = 1e-6f;
// The ray's step toward the light is the offset over N.L, N.L no less than
// this (at most ten times the offset, for light along the surface).
inline constexpr float kRtShadowMinSlope = 0.1f;
// Captured draws (skinned people, vehicles: kRtMaskCaptured*) block a ray
// only beyond this fraction of the point's distance from the camera:
// their triangles are not the ones the depth was drawn from (made again
// by the capture, one frame old for those drawn after the snapshot,
// moving by a frame's motion; their surfaces' normals from coarse
// triangles), so they shadowed themselves a few centimetres out. Static
// meshes still block from the surface on (contact shadows); a captured
// mesh's own shadow on itself and on what touches it is lost within the
// gap.
inline constexpr float kRtShadowCapturedGap = 0.04f;
// A dim far light's mask (RayShadow::dim_light: the moon) is traced at one
// pixel of every kRtShadowDimStep x kRtShadowDimStep, not every 2 x 2: its
// shadows are faint, their edges soft under the upsample.
inline constexpr uint32_t kRtShadowDimStep = 4;

// What the kernel is given (MSL: RtShadowParams), made on the CPU
// (RtShadowParamsFor).
struct RtShadowParams {
  // A pixel's point relative to `eye`: h = x rx + y ry + v rv + rc (x, y
  // the pixel's clip position, v the value the buffer stores), its xyz / w.
  float rx[4], ry[4], rv[4], rc[4];
  float eye[4];    // xyz: the camera's position (the reference of the above), world space
  float light[4];  // xyz: unit, toward the light; w: the ray's length
  uint32_t size[2];
  float jitter[2];         // the scene's offset on screen, pixels (right, down)
  uint32_t depth_texture;  // 1: a depth texture (read as depth2d), 0: a float colour one (its red)
  float sky;               // the stored value of the far plane (0 or 1): such pixels are lit
  float pixel;             // the angle between neighbouring pixels' rays (radians, at the centre)
  float offset_pixels, offset_view, offset_world;
  float cone;              // the cone's half-angle: 0 (hard), or kRtShadowCone where a denoiser follows
  uint32_t diag;           // 0, or the diagnostics' blocker counts (RtShadowDiag) kept
  float captured_gap;      // kRtShadowCapturedGap
  uint32_t frame;          // with a cone: the frame's number (each frame's rays go elsewhere in it)
  uint32_t normals;        // 1: the normals from Encode's normal texture (Encode sets it)
  float min_slope;         // kRtShadowMinSlope
};
static_assert(sizeof(RtShadowParams) == 160, "kRtShadowMsl's RtShadowParams");

// A game's cascaded shadow map (RayShadow::cascade_*), for the far
// light's hybrid trace, as AMD's FidelityFX Hybrid Shadows classifies by
// the shadow map: a ray only where the map cannot say for sure that the
// point is lit. Sure where the point lies within the map's range and its
// cascade's texels are no larger than max_texel, its surface faces the
// light (N.L of at least kRtCascadeMinSlope), and no texel within 1.5 of
// the point's in the map (a 2 x 2 maximum of the map read) holds a caster
// nearer the light than the point's own surface across that footprint
// (its slope over the footprint, plus a texel and the map's rounding), and
// the nearest is that surface (a map that does not see the point there,
// empty or another depth than the lookup expects, says nothing). Where the map has a caster
// there, the ray decides (leaf gaps, contact shadows), and beyond its range
// or in its coarse cascades the ray decides as before. Where it is sure,
// a short ray still looks for what the map cannot resolve: casters nearer
// the surface than the map's tolerance there (its slope across the
// footprint, a texel and the map's rounding, as a height h), out to 2 h /
// N.L along the light (farther, a caster that high would be in the map).
// And where every texel of the footprint holds a caster at least
// shadow_gap nearer the light (a building's: far above the fine gaps of a
// canopy near the surface, which the map's texels cannot resolve), the
// point is in shadow without a ray.
struct RtShadowCascade {
  float view[4];       // dot(view.xyz, world): the view depth (negated) the cascade is chosen by
  float splits[4];     // xyz: the cascades' bounds (s = -depth - splits >= 0)
  float select[4][4];  // (scale.x, scale.y, offset.x, offset.y) = (1, s) . select[k]
  float matrix[4][4];  // light space xyz = world.x m[0].xyw + world.y m[1].xyw + world.z m[2].xyw + m[3].xyw
  float range = 0.0f;  // points this far from the camera or farther: outside
  float max_texel = 0.0f;
  float atlas[2] = {};  // the map's texels (w, h): square cascades side by side
  uint32_t on = 0;
  // The map's values' rounding: relative (a float format's) and absolute
  // (a normalized one's); set by Encode from its format.
  float rounding_rel = 0.0f, rounding_abs = 0.0f;
  // Where every texel around the point holds a caster at least this far
  // nearer the light (world units), the point is in shadow without a ray;
  // 0: never.
  float shadow_gap = 0.0f;
};
static_assert(sizeof(RtShadowCascade) == 192, "kRtShadowMsl's RtShadowCascade");
inline constexpr float kRtCascadeMinSlope = 0.2f;
// From a draw's pixel shader constants `c` (RayShadow::cascade_*): `out`
// with on = 1 unless a register is out of range; atlas set by the caller.
bool RtShadowCascadeFor(const RayShadow& s, const float (*c)[4], RtShadowCascade* out);
// The world size of a texel of cascade `k` (0-3: k of the three splits
// passed) of `c` (atlas set): the larger of its two axes'.
double RtCascadeTexel(const RtShadowCascade& c, int k);

// What a trace with RtShadowParams::diag reports (Encode's `diag`, shared
// memory, read once the GPU is done; zeroed by its reader): the centre
// pixel's (w / 2, h / 2) trace, then `diag` 32-bit counts of the rays each
// instance blocked, by user ID (the last also counting the IDs beyond).
struct RtShadowDiag {
  float point[4];   // the pixel's point (world); w: the depth value stored
  float normal[4];  // the normal the ray starts along (turned toward the camera and the light)
  float origin[4];  // the ray's start; w: the offset
  float distance;   // to what blocked it
  uint32_t instance, primitive;  // what blocked it (user ID, triangle)
  uint32_t state;   // 0 not written, 1 sky (not traced), 2 lit, 3 blocked
};
static_assert(sizeof(RtShadowDiag) == 64, "kRtShadowMsl's RtShadowDiag; the counts follow");
inline constexpr uint32_t kRtShadowDiagCounts = 4096;

// The parameters for a w x h buffer of the frame `f`'s camera (its
// inv_view_proj_exact, else inv_view_proj, and depth_reversed), depth stored as 1 - d where
// `stored_reversed`, read from a depth texture where `depth_texture`, the
// scene drawn `jitter` pixels right and down (null: not moved), and the
// light the vector `to_light` points at. False if the camera cannot be
// used (a singular matrix) or `to_light` has no length.
bool RtShadowParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                       const float jitter[2], const float to_light[3], RtShadowParams* out);

// A pixel's world position as the kernel reconstructs it (in double, for
// tests and checks): pixel (i, j), stored value v, less the jitter.
void RtShadowPoint(const RtShadowParams& p, double i, double j, double v, double out[3]);

// ---- local lights

// A local light as a draw's pixel shader constants give it (RayShadow::
// local; RtLocalLightFor).
struct RtLocalLight {
  float position[3] = {};    // world space
  float inv_sq_radius = 0;   // 1 / radius^2: the light reaches points with d^2 inv_sq_radius < 1
  bool cone = false;         // a spot light: it reaches points where (dot(-l, direction) - cone_offset) cone_scale > 0
  float direction[3] = {};   // the way it shines, as the register holds it (l: unit, from the point to the light)
  float cone_offset = 0, cone_scale = 0;
  float lamp_gap = 0;        // world units: rays end this far short of the light
};
// From `s` and the draw's pixel shader constants `c` (null: none, a light
// without a radius).
RtLocalLight RtLocalLightFor(const RayShadow& s, const float (*c)[4]);
// Whether `a` and `b` are the same light (bit for bit).
bool RtSameLocalLight(const RtLocalLight& a, const RtLocalLight& b);
// Its radius: 1 / sqrt(inv_sq_radius); 0 where that is not a positive
// finite number (a light that reaches nothing).
double RtLocalLightRadius(const RtLocalLight& l);

// Pixels [x0, x1) x [y0, y1) of a buffer.
struct RtRect {
  uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  bool Empty() const { return x1 <= x0 || y1 <= y0; }
  uint64_t Area() const { return Empty() ? 0 : uint64_t(x1 - x0) * (y1 - y0); }
};
// The box around what light `l` reaches (RtLocalLightBox): its sphere's, or
// for a cone narrower than a half space the box around the cone's part of
// the sphere. False where it reaches nothing (no radius, a cone of no
// angle or a zero scale).
bool RtLocalLightBox(const RtLocalLight& l, double lo[3], double hi[3]);
// The pixels of a w x h buffer of `f`'s camera (drawn `jitter` pixels right
// and down; null: not moved) whose points `l` can reach: the projection of
// its box (RtLocalLightBox) clipped to the camera's side of its plane (the
// box's edges cut just in front of it), a pixel wider on each side (two
// past its far edges): the whole buffer with the camera inside the box;
// the whole buffer too where the camera cannot be inverted; empty where it
// is off screen, wholly behind the camera, or reaches nothing.
RtRect RtLocalLightBounds(const RtFrame& f, uint32_t w, uint32_t h, const float jitter[2], const RtLocalLight& l);

// What the kernel takes for a local light (MSL: RtLightParams; zeros and
// the whole buffer for a far light).
struct RtLightParams {
  float position[4];   // xyz: the light's position relative to RtShadowParams::eye; w: 1 / radius^2
  float direction[4];  // xyz: RtLocalLight::direction; w: cone_offset
  float cone_scale;
  uint32_t cone;       // 1: the cone cuts
  float lamp_gap;
  uint32_t local;      // 1: a local light (RtShadowParams::light unused), 0: a far one
  uint32_t origin[2], extent[2];  // the pixels traced: [origin, origin + extent)
};
static_assert(sizeof(RtLightParams) == 64, "kRtShadowMsl's RtLightParams");
// The parameters for light `l` over the pixels `r` of a w x h buffer
// (depth as RtShadowParamsFor takes it): `p` the frame's (RtShadowParamsFor,
// its light toward +z, unused), `lp` the light's. False where the camera
// cannot be used, the light has no radius or `r` is empty.
bool RtLightParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                      const float jitter[2], const RtLocalLight& l, const RtRect& r, RtShadowParams* p,
                      RtLightParams* lp);

// ---- the frame's local lights at once (RtShadowMasker::EncodeLights)
//
// Instead of one mask per light (above), or a ray per pixel of each light's
// volume drawn (gfx::RayForwardShadow::local, traced by the volumes' own
// pixels), a list of lights traced in one pass over the depth: a thread for
// one pixel of every `step` x `step`, its point and normal as a far light's
// mask has them; the lights reaching it (within the sphere, inside the
// cone) ranked by falloff (1 - d^2 / r^2)^2, cone and N.L (no less than
// 0.05), the strongest `cap` (at most kRtLightsCap) traced as a local
// light's mask traces its pixels (the ray's start, the lamp's gap, the
// captured instances' gap), the others lit; untraced, a light the surface
// faces away from (N.L < 0: in shadow, as a ray from there, inside a
// solid, would be) and one too faint there to matter (falloff times cone
// below 1/64: lit). Two passes, as the far light's mask: the rays' starts,
// then the rays (fewer registers live across them). Each light's results in its layer of an R8Unorm array
// (1 lit, 0 in shadow), the points' distances from the camera in an
// R32Float texture (0: the sky), both (w + step - 1) / step x (h + step -
// 1) / step: what the volumes' pixels read and upsample by depth
// (gfx/rt_forward_reflection.h, r_rt_light_mask).
inline constexpr uint32_t kRtLightSlots = 16;  // lights in one pass at most (layers)
inline constexpr uint32_t kRtLightsCap = 4;    // lights traced a pixel at most
// A light of the list (MSL: RtLightGpu), as RtLightParams has it.
struct RtLightGpu {
  float position[4];   // xyz relative to RtShadowParams::eye; w 1 / radius^2
  float direction[4];  // xyz RtLocalLight::direction; w cone_offset
  float cone_scale;
  uint32_t cone;
  float lamp_gap;
  float pad;
};
static_assert(sizeof(RtLightGpu) == 48, "kRtShadowMsl's RtLightGpu");
// The pass (MSL: RtLightsParams).
struct RtLightsParams {
  uint32_t count;  // lights (at most kRtLightSlots)
  uint32_t step;
  uint32_t cap;
  uint32_t frame;   // the sample's turn (RtLightSample)
  uint32_t low[2];  // the masks' size
  uint32_t stride;  // RtLightStride(step), or 0: every texel at its block's top-left pixel
  uint32_t pad;
};
static_assert(sizeof(RtLightsParams) == 32, "kRtShadowMsl's RtLightsParams");
// The pixel texel (tx, ty) of a pass at `step` traces: one of its step x
// step block's, a different one each frame of a jittered scene (the
// temporal upscaler then averages them: a caster thinner than the step,
// which a fixed pixel would miss everywhere or catch everywhere, shadows
// in proportion), every one of them once in step^2 frames, in an order
// each block starts at a place of its own (a hash of the texel: along a
// thin shadow some blocks catch it every frame, so it never vanishes all
// at once). `stride` (RtLightStride) 0: the block's top-left pixel.
// `frame` the full count of frames: each cycle of step^2 frames shifts the
// order (by c + c / 2 places in cycle c), so that a pixel's own sample
// falls on every phase of an 8-phase jitter within 32 cycles at any step
// from 2 to 8 (with step^2 frames a cycle alone, a step of 2 or 4 would
// keep each pixel on one phase); frames below step^2 as before.
// r_rt_light_mask (gfx/rt_forward_reflection.h) finds the texels' pixels
// the same way. The pixel may lie past the buffer's edge (the kernels
// clamp it).
void RtLightSample(uint32_t tx, uint32_t ty, uint32_t frame, uint32_t stride, uint32_t step, uint32_t* px,
                   uint32_t* py);
// The turn's stride through a block's step^2 pixels: coprime with step^2
// (every pixel once a cycle), about half way round (neighbouring turns far
// apart); 0 for step 1.
uint32_t RtLightStride(uint32_t step);
// Light `l` for the list of a pass whose camera `p` has (RtShadowParamsFor:
// its eye; `eye` that in double, RtLightsEye); false where it has no
// radius or no finite position.
bool RtLightGpuFor(const RtLocalLight& l, const double eye[3], RtLightGpu* out);
// The camera's position in double as RtShadowParamsFor finds it for `f`
// (what positions relative to RtShadowParams::eye are taken from).
bool RtLightsEye(const RtFrame& f, double eye[3]);
// The lights a pass traces are those drawn in the presented frame before
// (the lights' draws come after the pass, which must run before them): a
// light that moved since the frame before that (RtNearLocalLight: a
// vehicle's lamps) where it will be, as far again along the same step
// (RtLightPredict); a draw then reads the masks of a light of the pass
// that is the same (RtSameLocalLight), or, for a moved one, the same but
// for its position within kRtLightMatchOffset of the prediction and its
// direction within some 2 degrees (RtLightMatch); else it traces its own.
inline constexpr float kRtLightMatchOffset = 0.05f;  // world units
RtLocalLight RtLightPredict(const RtLocalLight& now, const RtLocalLight& before);
bool RtLightMatch(const RtLocalLight& traced, bool moved, const RtLocalLight& drawn);

// ---- what a draw of a mask-reading shader binds (the backend's choice, on
// the CPU)

// What a draw asks for: the texture at its depth sampler (null: none
// usable), the number of its contents (changed by every draw writing it and
// every clear), the camera of the last main-view draw that wrote it over
// its whole viewport (depth range 0-1; RtScene::TakeVote; 0: none, or one
// that did not), the scene's camera now (RtScene::CameraId), whether it is
// the main view's size, and the light's direction.
struct RtShadowWant {
  const void* depth = nullptr;
  uint64_t epoch = 0, depth_camera = 0, scene_camera = 0;
  bool main_size = false;
  float light[3] = {};
  // The light's colour is black (RayShadow::colour_register): the shader
  // multiplies the mask only into terms of that colour, so all lit draws
  // the same image untraced (a sun below the horizon, say).
  bool dark = false;
};
// The last choice within a presented frame (reset at its end: {}).
struct RtShadowMemo {
  bool valid = false;
  RtShadowWant last;
  bool white = false;
};
enum class RtShadowStep : uint8_t {
  Reuse,  // the mask chosen last (memo->white: all lit)
  Trace,  // trace one now (if that fails: memo->white = true, all lit)
  White,  // all lit: nothing that can be traced (no depth, light or camera; the depth is not the scene's view)
};
// The step for `w` after `memo` (updated). The depth is traced only from
// the scene's current camera: it wrote it last (not another view at that
// size, not last frame's camera), over the whole viewport, at the main
// view's size.
RtShadowStep RtShadowNext(RtShadowMemo* memo, const RtShadowWant& w);
// A target's camera (RtShadowWant::depth_camera) after a write into it: a
// clear forgets it; a main-view draw that voted (`vote`: its camera)
// names it where it drew over the whole viewport with depth range 0-1
// (`full_view`) and forgets it otherwise; other draws (other views,
// captured, not in the scene) keep it. Copies into a target are not
// writes of the game's (the backend copies only into its own textures).
uint64_t RtShadowWriter(uint64_t camera, bool clear, uint64_t vote, bool full_view);
// Why a draw gets the all-lit mask (RtShadowNext's White, or a trace that
// could not be made: NoTlas), for the counts.
// (Local lights also: Budget, past the frame's traced lights; NoMask, no
// mask texture or no parameters for the trace.) Dark: the light's colour
// black (RtShadowWant::dark).
enum class RtShadowWhite : uint8_t {
  None,
  NoDepth,
  WrongSize,
  NoLight,
  NoCamera,
  OtherCamera,
  NoTlas,
  Budget,
  NoMask,
  Dark,
  Count
};
RtShadowWhite RtShadowWhyWhite(const RtShadowWant& w);
const char* RtShadowWhiteName(RtShadowWhite w);
// Whether a whole-target clear still pending must be run (its own pass)
// before tracing: it clears depth or stencil (clear_flags bits 2 and 4) and
// the trace reads the pass's depth buffer.
bool RtShadowSettlesClear(uint32_t clear_flags, bool reads_pass_depth);
// The sampler a shader's draw binds a mask at: `mask_sampler` where the
// shader claims the mask (SetRayShadow) and there is one to bind, else -1.
int RtShadowSampler(bool claimed, uint32_t mask_sampler, bool have_mask);

// The draws of pixel shaders that still read a game's own shadow map
// (RayShadow::own_map): counted, and the first kRtOwnMapsSaid shaders
// (by hash) said, each once.
inline constexpr size_t kRtOwnMapsSaid = 16;
struct RtOwnMaps {
  uint64_t draws = 0;
  uint64_t said[kRtOwnMapsSaid] = {};
  size_t count = 0;  // of said
};
// Counts a draw of shader `hash`; true where it is to be said now (one of
// the first kRtOwnMapsSaid shaders, its first draw).
bool RtOwnMapDraw(RtOwnMaps* m, uint64_t hash);

// The masks of local lights: kEntries kept, one per light (RtLocalLight:
// bit for bit, else the nearest RtNearLocalLight not yet asked for this
// presented frame, a light that moved since: a vehicle's lamps, then
// `moved` this frame), the least recently used taken over past that; each entry's
// memo says whether its mask is still the one for the depth within a
// presented frame (RtShadowNext: traced again after the depth is written or
// the cameras change). Entries outlive presented frames (RtLightMaskEndFrame
// forgets only the memos): a light the frame's budget leaves out draws with
// its own mask of an earlier frame (`mask`, traced in presented frame
// `traced`) instead of all lit, so that which lights the budget leaves out,
// handed out in draw order (which changes from frame to frame), does not
// make shadows blink. A light that moved is traced every frame, ahead of
// the others and from a budget of its own (kMovingBudget): an earlier
// frame's mask would leave its shadows behind it. An entry is taken only for a light about to be traced
// or left out by the budget (not one all lit for want of a depth or camera).
struct RtLightMaskCache {
  static constexpr int kEntries = 16;
  // A mask older than this many presented frames is not drawn with (the
  // camera has moved on): the light is all lit until it is traced again.
  static constexpr uint64_t kMaxAge = 8;
  // Lights that moved traced a presented frame at most (beyond the budget
  // of the others, RtLightMaskChoose's): a street's vehicles' lamps in view.
  static constexpr uint32_t kMovingBudget = 16;
  struct Entry {
    bool used = false;
    RtLocalLight light;
    RtShadowMemo memo;
    uint64_t last_use = 0;
    bool mask = false;      // its mask holds a trace of this light
    uint64_t traced = 0;    // the presented frame of that trace
    uint64_t area = 0;      // its pixels (the light's coverage of the screen)
    uint64_t wanted = 0;    // the last presented frame a draw of it asked for a trace (RtLightMaskChoose)
    bool reserved = false;  // the frame's traces go to it first (RtLightMaskEndFrame)
    uint64_t born = 0;      // the presented frame it was taken
    uint64_t seen = 0;      // the last presented frame a draw found or took it
    uint64_t moved = 0;     // the last presented frame it was found moved (RtNearLocalLight)
  };
  Entry entry[kEntries];
  uint64_t uses = 0;
  uint64_t frame = 1;  // presented frames, counted from 1
  // Traces this presented frame (RtLightMaskTraced): of lights that moved,
  // of the others.
  uint32_t moving_traced = 0, still_traced = 0;
};
// Whether `b` may be `a` moved since the last presented frame: the same
// kind, radius and cone (to a relative 1e-3), its position within a tenth
// of the radius, a spot light's direction within some 10 degrees.
bool RtNearLocalLight(const RtLocalLight& a, const RtLocalLight& b);
// The entry holding `l` (marked used now): bit for bit, else the nearest
// RtNearLocalLight one no draw has found this presented frame (its light
// then `l`); or -1.
int RtLightMaskFind(RtLightMaskCache* c, const RtLocalLight& l);
// An entry for `l`, not held yet: a free one, else the least recently used
// taken over (its memo and mask reset).
int RtLightMaskTake(RtLightMaskCache* c, const RtLocalLight& l);
// What a light whose draw needs a trace gets (RtLightMaskChoose).
enum class RtLightStep : uint8_t {
  Trace,  // traced now
  Older,  // its mask of an earlier presented frame (no older than kMaxAge)
  Budget, // all lit: past the budget, and no mask of its own
};
// The step for a draw of the light in entry `e` (-1: none yet) that needs a
// trace: one that moved this frame is traced while kMovingBudget lasts,
// else all lit (never an earlier frame's mask); of the others, `budget` a
// frame, a reserved one (RtLightMaskEndFrame) is traced while the budget
// lasts, another only while that leaves room for the reserved ones not
// traced yet; past that its older mask, else all lit. Marks the entry
// wanted.
RtLightStep RtLightMaskChoose(RtLightMaskCache* c, int e, uint32_t budget);
// A trace of entry `e`'s light into its mask now, over `area` pixels
// (counted as a moving light's or another's).
void RtLightMaskTraced(RtLightMaskCache* c, int e, uint64_t area);
// A presented frame ends: the memos forgotten, and the next frame's traces
// reserved, `budget` of them, for the lights this frame wanted that the
// frame before had already (not one taken this frame: a light seen once
// would hold the budget for nothing; nor one that moved: traced anyway): those without a mask first, then the
// oldest masks, then the most pixels (so every light is traced again within
// a few frames, in turn).
void RtLightMaskEndFrame(RtLightMaskCache* c, uint32_t budget);

// One snapshot (RtScene::Snapshot, a TLAS each call) for the effects traced
// from the same depth: its texture, contents, camera and the scene's (as
// RtShadowWant has them) within one command buffer (`serial`: a snapshot's
// RtFrame is the work of the one it was taken in). The memo is reset ({})
// when a presented frame ends.
struct RtSnapshotMemo {
  bool valid = false;
  const void* depth = nullptr;
  uint64_t epoch = 0, depth_camera = 0, scene_camera = 0, serial = 0;
};
// True: trace against the snapshot taken last (the memo's); false: take one
// (the memo now names it).
bool RtSnapshotReuse(RtSnapshotMemo* memo, const RtShadowWant& w, uint64_t serial);

// MSL after kRtMsl: RtShadowParams and the shadow kernels, and what kernels
// reconstructing points from depth the same way use: rt_depth, rt_sky,
// rt_point (a pixel's point relative to the camera), rt_step (towards the
// neighbour on the pixel's surface). (Its rays: rt_shadow_blocked,
// kRtMsl's rt_shadowed's answer through Metal's intersector, the alpha test
// an intersection function; with diagnostics kRtMsl's rt_trace.) And
//   float3 rt_cone_dir(float3 l, float cone, uint2 c, uint frame)
// the direction pixel c's ray takes in frame `frame` within the cone of
// half-angle `cone` around the unit vector l: uniform over the cone's solid
// angle, a Halton (2, 3) point of the frame shifted per pixel by a hash (a
// Cranley-Patterson rotation), so that a pixel's directions over frames
// cover the cone evenly (their mean is l) and neighbours' differ; l itself
// where the cone is 0.
extern const char* const kRtShadowMsl;

class RtShadowMasker {
 public:
  // Null if `dev` does not trace rays (RtSupported) or the kernel does not
  // compile (said on stderr).
  static std::unique_ptr<RtShadowMasker> Make(id<MTLDevice> dev);
  // The trace (one compute encoder): `depth` (Depth32Float(_Stencil8) with
  // p.depth_texture, else a float colour texture holding D3D's depth, z /
  // w; ShaderRead) into `mask` (R8Unorm, ShaderWrite, the same size),
  // against `f`'s TLAS (not nil). Adds the rays traced, those that hit
  // something, the pixels left untraced outside a local light's reach and
  // those within its lamp gap to the four 32-bit counters at `counts`
  // (shared memory, read once the GPU is done; the last two only written
  // when not zero). With p.diag, reports into `diag` (RtShadowDiag, then
  // p.diag counts). With `light` (RtLightParamsFor), a local light's mask
  // over its pixels only; without, the far light's over the whole buffer.
  // With `normals` (the G-buffer's, xyz as unorm, the depth's size; ignored
  // at another size), the surfaces' normals from it where the depth gives
  // none. The far light's mask without diagnostics is traced at one pixel
  // of every `step` x `step` (2: a quarter of the rays; 4, a dim light's
  // (kRtShadowDimStep): a sixteenth) and upsampled by depth to every pixel;
  // with `cascade` (on) and `cascade_map` (its atlas: a depth or float
  // colour texture) its rays where the map is sure of the point (lit: a
  // short ray; in a building's shadow: none) are left out (RtShadowCascade;
  // counted at counts[3]).
  void Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> mask,
              const RtShadowParams& p, id<MTLBuffer> counts, size_t counts_offset, id<MTLBuffer> diag = nil,
              const RtLightParams* light = nullptr, id<MTLTexture> normals = nil, uint32_t step = 2,
              const RtShadowCascade* cascade = nullptr, id<MTLTexture> cascade_map = nil) const;
  // The far light's samples alone, for a temporal accumulation (gfx/
  // rt_accumulate.h): Encode's far light without diagnostics (its cascade
  // too), traced at one pixel of every step x step block (1 to 8), the
  // pixel turning from frame to frame (RtLightSample with `stride` and
  // `frame`; stride 0: the top-left one, as Encode's), into a texture of
  // (w + step - 1) / step x (h + step - 1) / step texels (R8Unorm, the
  // masker's: written by the next call again). With `mask` (R8Unorm, the
  // depth's size), the samples upsampled into it by depth as Encode does.
  // Nil where nothing could be traced (no TLAS, no pipelines).
  id<MTLTexture> EncodeSparse(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, const RtShadowParams& p,
                              id<MTLBuffer> counts, size_t counts_offset, id<MTLTexture> normals, uint32_t step,
                              uint32_t stride, uint32_t frame, const RtShadowCascade* cascade = nullptr,
                              id<MTLTexture> cascade_map = nil, id<MTLTexture> mask = nil) const;
  // The frame's local lights (`lights`, lp.count of them) traced from
  // `depth` (as Encode's; `normals` likewise) into `masks` (NewLightMasks)
  // and `dist` (NewLightDist), lp.low texels (above). Adds the rays traced,
  // those blocked, the texels reached by more lights than lp.cap and the
  // rays traced again next to a captured mesh to the four counters at
  // `counts` (the last two only when not zero).
  void EncodeLights(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> masks,
                    id<MTLTexture> dist, const RtShadowParams& p, const RtLightsParams& lp,
                    const RtLightGpu* lights, id<MTLBuffer> counts, size_t counts_offset,
                    id<MTLTexture> normals = nil) const;
  // The light masks' array (R8Unorm, kRtLightSlots layers) and their
  // distances (R32Float) for w x h texels: private, written by
  // EncodeLights, read after.
  id<MTLTexture> NewLightMasks(uint32_t w, uint32_t h) const;
  id<MTLTexture> NewLightDist(uint32_t w, uint32_t h) const;
  // 1 x 1 stand-ins for draws that read no light masks (bound all the
  // same: a patched shader declares them).
  id<MTLTexture> NoLightMasks() const { return no_masks_; }
  // Whether EncodeSparse can trace (its pipelines made).
  bool SparseReady() const { return cast_turn_color_pipe_ != nil; }
  // EncodeSparse's textures for a w x h depth at `step`, made now (on the
  // recording side: false where they cannot be, and EncodeSparse then would
  // not trace); its samples' points (RGBA16Float, the same texels: each
  // texel's sample point relative to the camera, RtShadowParams::eye, and
  // in w its place in its block, x + 8 y; -1: the sky, or past 60000 world
  // units, beyond half floats), written by EncodeSparse with the values.
  bool PrepareSparse(uint32_t w, uint32_t h, uint32_t step) const;
  id<MTLTexture> SparsePoints() const { return sparse_points_; }
  bool LightsReady() const { return lights_color_pipe_ != nil; }
  id<MTLTexture> NoLightDist() const { return no_dist_; }
  // The mask over `target` (a debug view: white lit, black in shadow),
  // scaled to its size. Encoding side.
  void EncodeView(id<MTLCommandBuffer> cmd, id<MTLTexture> mask, id<MTLTexture> target) const;
  // An R8Unorm mask of w x h (private, written by Encode, sampled after).
  id<MTLTexture> NewMask(uint32_t w, uint32_t h) const;
  // 1 x 1, every point lit: for draws whose mask cannot be traced.
  id<MTLTexture> White() const { return white_; }

 private:
  // The far light's low-resolution passes (Encode's, EncodeSparse's: with
  // `turn` the pixel turning within its block) into `low`, the timed trace
  // named `trace_name`; false where they could not run (no `low`, no starts'
  // texture). And the upsample of `low` into `mask`.
  bool EncodeLow(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, const RtShadowParams& q,
                 const RtLightParams& lp, id<MTLBuffer> counts, size_t counts_offset, id<MTLTexture> normals,
                 uint32_t step, bool turn, uint32_t stride, uint32_t frame, const RtShadowCascade* cascade,
                 id<MTLTexture> cascade_map, id<MTLTexture> low, const char* trace_name) const;
  // (stride and frame: where each low texel was traced, RtLightSample's;
  // stride 0, its block's top-left pixel.)
  void EncodeUp(id<MTLCommandBuffer> cmd, id<MTLTexture> depth, const RtShadowParams& q, id<MTLTexture> low,
                id<MTLTexture> mask, uint32_t step, uint32_t stride = 0, uint32_t frame = 0) const;
  // A private texture of w x h (shader read and write).
  id<MTLTexture> NewLow(MTLPixelFormat fmt, uint32_t w, uint32_t h) const;

  id<MTLDevice> dev_ = nil;
  id<MTLComputePipelineState> depth_pipe_ = nil, color_pipe_ = nil, view_pipe_ = nil;
  // The far light's half-resolution trace and its upsampling by depth.
  id<MTLComputePipelineState> up_depth_pipe_ = nil, up_color_pipe_ = nil;
  // The tracing pipelines' intersection function tables (the alpha test,
  // kRtShadowMsl's rt_shadow_alpha, at 0).
  id<MTLIntersectionFunctionTable> depth_table_ = nil, color_table_ = nil;
  mutable id<MTLTexture> half_ = nil;  // the half-size mask (encoding side)
  mutable id<MTLTexture> dim_ = nil;   // a dim light's (kRtShadowDimStep)
  // The two-pass trace: its rays' starts (RGBA16Float, the low-res mask's
  // size), the pipelines making them and tracing them.
  mutable id<MTLTexture> starts_ = nil;
  // The cascade map's 2 x 2 maxima (R32Float), and the pipelines making it.
  mutable id<MTLTexture> cascade_max_ = nil;
  id<MTLComputePipelineState> max_depth_pipe_ = nil, max_color_pipe_ = nil;
  id<MTLComputePipelineState> start_depth_pipe_ = nil, start_color_pipe_ = nil, cast_depth_pipe_ = nil,
                              cast_color_pipe_ = nil;
  id<MTLIntersectionFunctionTable> cast_depth_table_ = nil, cast_color_table_ = nil;
  // The same at the pixel turning within its block (EncodeSparse), and its
  // samples (encoding side).
  id<MTLComputePipelineState> start_turn_depth_pipe_ = nil, start_turn_color_pipe_ = nil, cast_turn_depth_pipe_ = nil,
                              cast_turn_color_pipe_ = nil;
  id<MTLIntersectionFunctionTable> cast_turn_depth_table_ = nil, cast_turn_color_table_ = nil;
  mutable id<MTLTexture> sparse_ = nil, sparse_starts_ = nil, sparse_points_ = nil;
  // The local lights' passes (EncodeLights: the rays' starts, then the
  // rays) and the tracing ones' tables; the starts (encoding side).
  id<MTLComputePipelineState> lights_start_depth_pipe_ = nil, lights_start_color_pipe_ = nil;
  id<MTLComputePipelineState> lights_depth_pipe_ = nil, lights_color_pipe_ = nil;
  mutable id<MTLTexture> lights_rays_ = nil;
  id<MTLIntersectionFunctionTable> lights_depth_table_ = nil, lights_color_table_ = nil;
  id<MTLTexture> no_masks_ = nil, no_dist_ = nil;
  id<MTLTexture> white_ = nil;
  mutable id<MTLTexture> view_tex_ = nil;  // EncodeView's (encoding side)
  id<MTLBuffer> no_diag_ = nil;  // bound where there are no diagnostics (never written)
};

}  // namespace recomp::gfx
