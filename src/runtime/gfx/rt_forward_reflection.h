#pragma once
// Ray-traced reflections in forward pixel shaders (Objective-C++), on GPUs
// that trace rays from render stages: a shader drawn after the G-buffer's
// lighting (blended glass, say), which the screen-space passes of
// gfx/rt_reflection.h cannot reach, traces its own reflection rays with
// ray queries in its fragment function.
//
// The translation (hlsl::RayReflectionCall) turns the shader's reads of
// one sampler into calls of r_rt_reflect(pixel, origin, dir, light,
// colour, ambient0, ambient1), defined in its GLSL as "nothing hit".
// RtForwardReflectionPatch rewrites the MSL SPIRV-Cross makes of it, at two
// fixed places:
// - before it: kRtMsl, kRtReflectRayMsl and r_rt_reflect_rt (below);
// - before the fragment entry point main0: a macro sending its calls of
//   r_rt_reflect to r_rt_reflect_rt with the scene's TLAS, materials,
//   instance info and RtForwardParams, which main0 takes as four more
//   parameters, fragment buffers kRtForwardSceneBuffer-
//   kRtForwardParamsBuffer (after the translation's 0-2).
// The stub itself stays (calls outside main0 still get "nothing hit").
// Likewise a translation's calls of r_rt_shadow (hlsl::RayShadowCall), the
// stub's "lit", by r_rt_shadow_rt, with the same four parameters; a
// shader may have either or both.
//
// Per pixel (r_rt_reflect_rt): none for helper lanes or with
// RtForwardParams::on 0; one ray from the origin (world space) along the
// direction (normalised), at most max_distance long; static instances
// from a start off the point (the shadows' offsets, gfx/rt_shadow.h: the
// pixel's footprint at the camera's distance and the point's distance from
// the origin), captured ones beyond captured_gap of the camera's distance
// (the shadows' rule: the shader's own mesh drawn here is not in the
// scene, but the body around it, made again by a capture, maybe a frame
// behind, is); closest hit, alpha-cut instances tested, no face culled;
// what it hits lit as gfx/rt_reflection.h lights its hits (RtReflectRay),
// times `scale`. Returns that and a 1 where a ray hit an instance with an
// albedo, 0 elsewhere (a miss, a back face of an instance that culls, an
// instance without an albedo: the shader's own reflection).
//
// Shadows (r_rt_shadow_rt): 1 (lit) for helper lanes, with
// RtForwardParams::shadow's bit 0 clear, or for a light direction of no
// length; else one ray from the point (world space) toward the far light
// (the register's direction, negated unless the bit 1 says it points
// toward the light), unbounded, any hit ending it: 0 where an instance
// blocks it. Its start is off the point along the light by the shadows'
// offset (gfx/rt_shadow.h: the pixel's footprint, of the distance, of the
// point's distance from the origin) over kRtShadowMinSlope (the surface's
// normal is not known here: as far as the shadow masks go at the lowest
// slope; a step toward the light does not move the shadows' edges), and
// captured instances block it only beyond kRtShadowCapturedGap of the
// camera's distance (or that start). Hard shadows: one ray straight at the
// light (no cone). A local light's (bit 2): none where the point is beyond
// its radius (RtForwardParams::reach, where known: the shader's falloff is
// 0 there); and where the device traced the frame's lights once per pixel
// before (RtForwardParams::light_slot: the light's layer of those masks,
// kRtForwardLightMasksTexture, their points' distances from the camera
// kRtForwardLightDistTexture, gfx/rt_shadow.h's RtShadowMasker::
// EncodeLights), no ray: the mask texels of the pixel's block and of the
// three blocks beside it toward the pixel, each weighted by a tent of the
// step's width around the pixel it traced (RtLightSample: turning frame to
// frame) and by how near its point's distance is to the pixel's own (no
// shadow bleeding across a depth edge); on a fixed grid, the bilinear
// weights the far light's mask is upsampled with. A patched shader calling r_rt_shadow takes those two textures
// too (fragment textures after D3D's sixteen samplers).
//
// Limits: the scene is the snapshot the frame's traced effects took (gfx::
// Device::SetRayForwardReflection): what is drawn after it is there from
// the last frame (a frame behind), and blended geometry is never in it
// (other glass is seen through); one ray per pixel and layer (overdraw
// traces again); no ray counters on the GPU (a fragment function writing
// memory would lose early depth testing).

#if !__has_feature(objc_arc)
#error "rt_forward_reflection.h needs ARC (-fobjc-arc)"
#endif

#import <Metal/Metal.h>

#include <cstdint>
#include <string>

#include "runtime/gfx/device.h"
#include "runtime/gfx/rt_scene.h"

namespace recomp::gfx {

// "This frame": the backend's, one command buffer, which a presented frame
// or a readback ends (a snapshot's RtFrame is that command buffer's work).
//
// The fragment buffers a patched shader takes.
inline constexpr uint32_t kRtForwardSceneBuffer = 3;     // instance_acceleration_structure
inline constexpr uint32_t kRtForwardMaterialBuffer = 4;  // RtMaterial per instance (RtFrame::materials)
inline constexpr uint32_t kRtForwardInfoBuffer = 5;      // RtInstanceInfo per instance (RtFrame::info)
inline constexpr uint32_t kRtForwardParamsBuffer = 6;    // RtForwardParams
// The fragment textures a patched shader calling r_rt_shadow takes: the
// frame's local lights' masks (R8Unorm, a layer per light) and their
// points' distances (R32Float, 0 for the sky), RtShadowMasker::
// EncodeLights's.
inline constexpr uint32_t kRtForwardLightMasksTexture = 16;
inline constexpr uint32_t kRtForwardLightDistTexture = 17;
// The fragment texture a patched shader calling r_rt_shadow_to takes: the
// frame's sun mask (R8Unorm, 1 lit), read with kRtForwardSunGate.
inline constexpr uint32_t kRtForwardSunMaskTexture = 18;
// RtForwardParams::light_slot for a draw whose light is in no mask.
inline constexpr uint32_t kRtForwardNoLightSlot = 0xFFFFFFFFu;
// The patch's version, in the patched text's first line. (A backend caches
// the plain MSL and patches it each time a shader is made: patched text
// never outlives the build that made it.)
inline constexpr uint32_t kRtForwardVersion = 11;
// RtForwardParams::shadow's bit: the far light (not a local one) is taken
// as reaching nothing (in shadow; reflections' hits lit by the ambient
// alone), its rays untraced.
inline constexpr uint32_t kRtForwardFarDark = 8;
// RtForwardParams::shadow's bit: by day (the far light reaching things), a
// local light's ray (r_rt_shadow_to) is not traced where the frame's sun
// mask holds the pixel lit (the light left as the game has it): beside the
// sun a lamp's shadow hardly shows, and under the El by day the lamps'
// rays cost ~20 ms a frame. The draw's target is the mask's size.
inline constexpr uint32_t kRtForwardSunGate = 16;
// RtForwardParams::shadow's bit: no local light's ray (r_rt_shadow_to) for
// this draw, its lamps left as the game lights them. The backend sets it
// for blended draws (a shop's interior behind glass, drawn in layers: 24 ms
// a frame of lamp rays at a car showroom by night, the picture the same
// without them).
inline constexpr uint32_t kRtForwardNoLamps = 32;
// The least weight (what the light brings the pixel: the shader's own
// falloff, facing and cone times its colour's greatest channel) at which a
// forward shader's local light gets a shadow ray; below it the light is left
// as the game has it. 0: every light that brings anything. (0.05: the
// owner's 20-minute play, 45 fps in the median against 36 the run before;
// whether a shadow he would miss is gone was not checked.)
inline constexpr float kRtForwardMinWeight = 0.05f;

// What a draw traces with (MSL: RtForwardParams), made on the CPU
// (RtForwardParamsFor; RtForwardParams{} traces nothing).
struct RtForwardParams {
  float eye[4];  // the camera's position (world)
  float up[4];   // xyz: world space, unit
  uint32_t on;   // 0: nothing traced (nothing hit)
  uint32_t toward_light;  // the light's register points toward it
  float scale, wrap;
  float max_distance;  // a ray's length
  float pixel;         // the angle between neighbouring pixels' rays (radians)
  float captured_gap;  // captured instances beyond this fraction of the camera's distance (kRtShadowCapturedGap)
  // The shadow calls (r_rt_shadow): bit 0 traced (else lit), bit 1 the
  // light's register points toward it, bit 2 the register is a local
  // light's position (RayForwardShadow::point).
  uint32_t shadow;  // (bit 3, kRtForwardFarDark: the far light reaches nothing here, untraced)
  // A ray's start off the point: kRtShadowOffsetPixels, kRtShadowOffsetView,
  // kRtShadowOffsetWorld; kRtShadowMinSlope (the shadow calls').
  float offsets[4];
  float lamp_gap;  // a local light's ray ends this far short of it (world units)
  float reach;     // a local light's 1 / radius^2: no ray beyond the radius (0: not known)
  // A local light's layer in the frame's masks (kRtForwardNoLightSlot:
  // none, the ray traced here), one pixel of every light_step x light_step
  // traced, and the target's pixel coordinates times light_scale are the
  // masks' depth's.
  uint32_t light_slot = kRtForwardNoLightSlot;
  uint32_t light_step = 1;
  float light_scale[2] = {1.0f, 1.0f};
  uint32_t light_frame = 0, light_stride = 0;  // the masks' samples' turn (gfx/rt_shadow.h, RtLightSample)
};
static_assert(sizeof(RtForwardParams) == 112, "kRtForwardReflectionMsl's RtForwardParams");

// The parameters for a draw of a shader configured `r` into a w x h target
// seen by frame `f`'s camera. False if the camera cannot be used.
bool RtForwardParamsFor(const RtFrame& f, uint32_t w, uint32_t h, const RayForwardReflection& r,
                        RtForwardParams* out);
// Likewise for a shader whose reflection calls trace as `r` says (null:
// nothing hit, on 0) and whose shadow calls as `s` says (null: lit, shadow
// 0).
bool RtForwardParamsFor(const RtFrame& f, uint32_t w, uint32_t h, const RayForwardReflection* r,
                        const RayForwardShadow* s, RtForwardParams* out);

// The MSL SPIRV-Cross made of a translation with r_rt_reflect or
// r_rt_shadow calls (hlsl::Program::ray_reflection_sampler,
// ray_shadow_sampler), given the ray queries (above). False, the text
// untouched, where it is not in the expected shape: no fragment function
// main0 whose parameter list closes before its body, neither stub defined
// before it and called in it, or fragment buffers kRtForwardSceneBuffer-
// kRtForwardParamsBuffer already taken. (A stub defined but not called in
// main0 keeps its calls elsewhere the stub's.)
bool RtForwardReflectionPatch(std::string* msl);
// Whether `msl` is one RtForwardReflectionPatch made.
bool RtForwardReflectionPatched(const std::string& msl);

// A pixel shader's library as the Metal backend makes it from its plain
// MSL (SPIRV-Cross's, what it caches: never the patched text, so a cache
// does not outlive the scene's MSL of the build that patched it): given
// the ray queries where `forward` (RtForwardReflectionPatch), the lean
// macro put in (MetalLeanMacros), compiled by `compile` (the text, whether
// it has the macro); the plain text where the patch is refused or its text
// does not compile.
using RtCompileMsl = id<MTLLibrary> (*)(void* ctx, const std::string& msl, bool lean, NSError** error);
struct RtForwardBuild {
  id<MTLLibrary> lib = nil;  // nil: the plain text did not compile either
  std::string msl;           // the text compiled last
  bool patched = false, lean = false;
  bool refused = false;          // the patch refused the text
  NSError* patch_error = nil;    // the patched text's compile error (then plain)
  NSError* error = nil;          // the last compile's
};
RtForwardBuild RtForwardCompile(const std::string& plain, bool forward, RtCompileMsl compile, void* ctx);

// What a draw that traces nothing binds in the scene's place (Metal's
// validation wants every argument of the function bound): a TLAS of one
// instance of a degenerate triangle, its BLAS (resident too), one zero
// RtMaterial and RtInstanceInfo.
struct RtForwardNullScene {
  id<MTLAccelerationStructure> tlas = nil, blas = nil;
  id<MTLBuffer> materials = nil, info = nil;
};
// Built now (a command buffer of `queue`, waited for); false if it cannot
// be.
bool RtForwardMakeNullScene(id<MTLDevice> dev, id<MTLCommandQueue> queue, RtForwardNullScene* out);

// Why a draw of such a shader traces nothing (RtForwardWhyOff).
enum class RtForwardOff : uint8_t {
  None,
  NoConfig,    // the shader was not taken (SetRayForwardReflection), or not given the ray queries
  NoSnapshot,  // no snapshot of the scene in this frame (no traced effect before the draw)
  NoCamera,    // the scene has no camera yet this frame, or the snapshot is not of it
  OtherView,   // drawn with another camera, or (camera not known) into another target and depth
  Count,
};
// What a draw has to trace with.
struct RtForwardWant {
  bool config = false;    // the shader taken and given the ray queries
  bool snapshot = false;  // a snapshot with a TLAS taken in this frame
  uint64_t snapshot_camera = 0, scene_camera = 0;
  // The draw's own camera (RtScene::CameraOf of its world and position
  // matrices), where its vertex shader reads both (`draw_known`); 0: none
  // of the frame's.
  bool draw_known = false;
  uint64_t draw_camera = 0;
  // Without it: whether the draw's colour target is the one bound when the
  // snapshot was taken (the pass whose traced effects took it), and the
  // camera that wrote the pass's depth buffer last (0: none, no depth).
  bool snapshot_target = false;
  uint64_t depth_camera = 0;
};
// None if it traces: the shader taken, a snapshot of the scene's current
// camera in this frame, and the draw in that camera's
// view: drawn with it where its matrices say so (with another: not), else
// drawn into the target the snapshot was taken for, or into a depth buffer
// that camera wrote last. (The depth's camera alone refused a game's glass
// in its main view too: any later draw of another of the frame's cameras
// names its own on the depth buffer.)
RtForwardOff RtForwardWhyOff(const RtForwardWant& w);
const char* RtForwardOffName(RtForwardOff o);

}  // namespace recomp::gfx
