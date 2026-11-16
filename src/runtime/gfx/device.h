#pragma once
// Backend-neutral rendering device with Direct3D 7 fixed-function
// semantics. The ddraw/d3d7 and OpenGL HLE talk only to this; backends
// translate:
//   - d3d9 via dxvk-native (gfx/d3d9_device.cpp): Linux bring-up and the
//     reference image;
//   - SDL_GPU with fixed-function uber-shaders (gfx/gpu_device.cpp):
//     Metal on macOS; Vulkan on Linux once its shaders exist;
//   - Metal directly (gfx/metal_device.mm, Apple platforms): Direct3D 9's
//     programmable pipeline only (HasFixedFunction false).
// Numeric enums (render states, texture stage states, transform types,
// primitive types, FVF codes) are D3D7's; render states D3D7 lacks use
// D3D9's numbers (COLORWRITEENABLE 168, SCISSORTESTENABLE 174,
// SLOPESCALEDEPTHBIAS 175, DEPTHBIAS 195), and FVFs may use D3D9's
// XYZW position (0x4002: homogeneous, untransformed). Plain data structs mirror the
// D3D7 ones (which D3D9 also kept unchanged).
//
// Direct3D 9's programmable pipeline (shaders translated from HLSL,
// runtime/hlsl) is a second way to draw: CreateShader / DrawShader, with
// up to four render targets (SetRenderTargets). Its render states are
// D3D9's (also COLORWRITEENABLE1-3 190-192, BLENDFACTOR 193,
// SEPARATEALPHABLENDENABLE 206 and the alpha blend 207-209). The d3d9
// backend does not have it.

#include <cstdint>
#include <memory>
#include <string>

#include "runtime/hlsl/hlsl.h"

struct SDL_Window;

namespace recomp::gfx {

struct RtLight;  // gfx/rt_light_list.h

enum class Format : uint32_t {
  Unknown, X8R8G8B8, A8R8G8B8, R8G8B8, R5G6B5, X1R5G5B5, A1R5G5B5, A4R4G4B4,
  X4R4G4B4, R3G3B2, A8, L8, A8L8, P8,
  D16, D24X8, D24S8, D32, D15S1,
  DXT1, DXT2, DXT3, DXT4, DXT5,
  // Floating point, in D3D9's memory layout (red first): uploaded and
  // stored as is.
  R16F, G16R16F, A16B16G16R16F, R32F, G32R32F, A32B32G32R32F,
};
uint32_t BitsPerPixel(Format f);
bool IsCompressed(Format f);
bool IsFloat(Format f);
// Bytes per row of `width` pixels (per row of 4x4 blocks if compressed).
uint32_t RowPitch(Format f, uint32_t width);
// Rows in the data for `height` (block rows if compressed).
uint32_t RowCount(Format f, uint32_t height);
// A DXT1-5 image (`pitch` bytes per row of 4x4 blocks) decoded into w x h
// A8R8G8B8, for GPUs that cannot sample BC formats (the A15 of an Apple TV).
void DecodeDxt(const uint8_t* src, uint32_t pitch, Format f, int w, int h, uint32_t* out);
// The least alpha (0-255) any texel of a w x h image (`pitch` bytes per row,
// of 4x4 blocks if compressed) in format `f` reads as: 255 where the format
// has none; 0 where it is not known (float, palette, depth formats).
uint8_t MinAlpha(const uint8_t* src, uint32_t pitch, Format f, int w, int h);

// Single-pixel conversion between uncompressed colour formats and 32-bit
// A8R8G8B8 (missing alpha reads as 0xFF). For blits and readbacks.
uint32_t DecodePixel(const uint8_t* p, Format f);
void EncodePixel(uint8_t* p, Format f, uint32_t argb);
// Raw pixel value (as stored) for colour keys and fills.
uint32_t RawPixel(const uint8_t* p, Format f);
void PutRawPixel(uint8_t* p, Format f, uint32_t raw);

struct Rect {
  int32_t left, top, right, bottom;
};

struct Viewport {
  uint32_t x, y, width, height;
  float min_z, max_z;
};

struct Color {
  float r, g, b, a;
};

struct Material {
  Color diffuse, ambient, specular, emissive;
  float power;
};

struct Light {
  uint32_t type;
  Color diffuse, specular, ambient;
  float position[3], direction[3];
  float range, falloff, attenuation0, attenuation1, attenuation2, theta, phi;
};

using TextureId = uint32_t;  // 0: none
using DepthId = uint32_t;    // 0: none
using ShaderId = uint32_t;   // 0: none

// A colour target: a mip level of a render-target texture, or the back
// buffer (tex 0, level 0). A level is max(1, size >> level) on each side.
struct ColorTarget {
  TextureId tex = 0;
  int level = 0;
};

// ---- the programmable pipeline (Direct3D 9 shaders translated from HLSL)

// A pixel shader that reads a ray-traced shadow mask (Device::SetRayShadow):
// where the mask goes, where the draw binds the depth it is traced from (the
// main view's depth buffer as a texture, or a float texture holding D3D's
// depth, z / w of the main view's projection, in its red), and the pixel
// shader float register holding the light's direction (xyz, world space;
// length ignored), with which way it points.
// Or (`local`) a local light's: a point or spot light a deferred renderer
// draws one light a draw (its volume), named by the draw's pixel shader
// float registers (c0-c223; components 0-3): its position (xyz, world
// space), 1 / radius^2 (it reaches points closer than the radius), and
// with `cone` a spot light's cone: it reaches points where
// (dot(-l, direction) - offset) scale > 0, l the unit vector from the
// point to the light and `direction` the register's xyz as it is (the way
// the light shines). Rays end `lamp_gap` world units short of the light,
// which sits inside its own lamp's mesh. light_register and toward_light
// are not read then. The shader must add nothing of the light where
// d^2 / radius^2 >= 1 or outside the cone (its attenuation exactly 0
// there): those pixels are not traced (lit), and the mask's texels outside
// the light's bounds on screen keep an earlier light's values.
// Optionally where the draw binds the G-buffer's normals (`normal_sampler`,
// world space, xyz as unorm: N = normalize(2 t - 1), at the depth's size;
// 16 or more: none): the rays start off the surface along them where the
// depth gives no normal of its own (gfx/rt_shadow.h).
// Optionally (a far light: `colour_register` c0-c223; 224 or more: none)
// the register holding the light's colour, rgb times w its intensity, as
// the shader multiplies it into every term the mask scales: where that is
// black (no component above 0) nothing is traced, the mask all lit (the
// same image); where its brightest component times w is below
// `dim_light` (a moon) the mask is traced at a sixteenth of the pixels
// (gfx/rt_shadow.h, kRtShadowDimStep), not a quarter.
// Optionally (a far light: `cascade_sampler` 0-15) the game's own cascaded
// shadow map the shader read before it read the mask, as its lookup made
// it: a point W (world) in the cascade the view depth dot(c[view].xyz, W)
// selects (s = 1 where -depth - c[split].xyz >= 0; s4 = (1, s)), at
// light-space (x, y, z) = W.x c[matrix].xyw + W.y c[matrix + 1].xyw +
// W.z c[matrix + 2].xyw + c[matrix + 3].xyw, texel (x, y) scale + offset
// with scale (s4 . c[select[0]], s4 . c[select[1]]) and offset
// (s4 . c[select[2]], s4 . c[select[3]]), lit where z >= the map's value;
// points c[range].w or farther from the camera are outside it. The mask's
// rays are left out where the map says for sure that the point is lit
// (gfx/rt_shadow.h, RtShadowCascade) and its texels are no larger than
// `cascade_max_texel` world units; and, where `cascade_shadow_gap` is above
// 0, where it is sure the point is in the shadow of a caster at least that
// far nearer the light.
// Or (`own_map`) a pixel shader that still reads a shadow map of the
// game's own at mask_sampler (one the game could not give the mask or ray
// queries of its own): nothing is traced for it, its draws are counted
// (perf stats: per presented frame; the first 16 such shaders' hashes
// logged once each), and with RayTracing::own_maps_lit a 1x1 texture whose
// red is own_map_lit (R32Float) is bound there in place of what the draw
// has, the value the game's comparison takes as lit: a debug view (what
// still changes in the frame then is a shadow of the game's own). Its other
// fields are not read.
struct RayShadow {
  uint32_t mask_sampler = 0;    // 0-15
  uint32_t depth_sampler = 0;   // 0-15
  uint32_t normal_sampler = 16;  // 0-15, or none
  uint32_t light_register = 0;  // c0-c223
  bool toward_light = false;    // the vector points toward the light (else away from it)
  uint32_t colour_register = 224;  // c0-c223, or none
  float dim_light = 0.0f;          // 0: none dim
  uint32_t cascade_sampler = 16;   // 0-15, or none
  uint32_t cascade_view_register = 0, cascade_split_register = 0, cascade_matrix_register = 0;
  uint32_t cascade_select_registers[4] = {};
  uint32_t cascade_range_register = 0;
  float cascade_max_texel = 0.0f;
  float cascade_shadow_gap = 0.0f;  // world units; 0: the map never decides a shadow
  bool local = false;
  uint32_t position_register = 0;
  uint32_t radius_register = 0, radius_component = 0;  // 1 / radius^2
  bool cone = false;
  uint32_t direction_register = 0;
  uint32_t cone_offset_register = 0, cone_offset_component = 0;
  uint32_t cone_scale_register = 0, cone_scale_component = 0;
  float lamp_gap = 0.0f;
  bool own_map = false;
  float own_map_lit = 0.0f;
};

// A pixel shader that reads ray-traced reflections (Device::SetRayReflection)
// over a deferred renderer's G-buffer, in place of an environment map's
// sample of its own: where the reflections go and what the device traces
// them from, all read at the pixel being drawn, and the pixel shader float
// registers (c0-c223) the hits are lit with. Per pixel:
// - its point from the depth (as RayShadow's), its world-space normal N
//   from `normal_sampler` (xyz as unorm: N = normalize(2 t - 1)), its specular
//   intensity and gloss g from `surface_sampler` (channels spec_channel and
//   gloss_channel);
// - the Blinn-Phong exponent e = exponent_scale g^2, and how blurred the
//   game shows its own reflection there: 1 - sat(e c[blur_register]
//   [blur_component]) (0 sharp; with blur_register out of range, 1 -
//   sat(e / exponent_scale));
// - wet (wet_slope > 0, wet_register and stencil_sampler in range): w =
//   sat(wet_slope N.up) c[wet_register][wet_component] where the stencil
//   texture's red is 0, the exponent then max(e, mix(e, min(wet_gain e,
//   wet_max_exponent), w)) and the pixel shiny where w > 0;
// - traced where it is not the sky, is shiny (spec > 0, or wet) and is
//   sharp enough (gfx/rt_reflection.h): one ray along reflect(I, N), I from
//   the camera to the point, at most max_distance long (world units, at
//   the smoothest gloss); what it hits lit by the light (direction in
//   light_register, as RayShadow's; colour c[colour_register].rgb times its
//   w; diffuse sat((N.L - diffuse_wrap) / (1 - diffuse_wrap)), a shadow ray
//   toward it) and the ambient a + b sat(0.5 - 0.5 N.up) (a, b
//   c[ambient_registers]), times the hit's albedo and output_scale.
// The output's rgb is that, a 1 where a ray hit something and 0 elsewhere
// (the shader keeps its own reflection there).
struct RayReflection {
  uint32_t output_sampler = 0;   // 0-15: RGBA16F, the pixel's
  uint32_t depth_sampler = 0;    // 0-15
  uint32_t normal_sampler = 0;   // 0-15
  uint32_t surface_sampler = 0;  // 0-15
  uint32_t spec_channel = 0, gloss_channel = 1;  // 0-3
  float exponent_scale = 512.0f;
  uint32_t blur_register = 224, blur_component = 0;
  uint32_t light_register = 0;
  bool toward_light = false;
  uint32_t colour_register = 0;
  uint32_t ambient_registers[2] = {0, 0};
  float up[3] = {0.0f, 0.0f, 1.0f};  // world space
  float diffuse_wrap = 0.0f;
  float output_scale = 1.0f;
  // A ray's length at roughness 0, world units (FidelityFX's hybrid
  // reflections: 100, metres there); exp(-5 roughness) of it at the pixel's.
  float max_distance = 100.0f;
  // Wetness (off with wet_slope 0).
  float wet_slope = 0.0f, wet_gain = 1.0f, wet_max_exponent = 0.0f;
  uint32_t wet_register = 224, wet_component = 0;
  uint32_t stencil_sampler = 16;
};

// A pixel shader that takes part in denoising the lit scene (Device::
// SetRayDenoise: MetalFX's denoiser, gfx/rt_denoise.h), while shadows are
// traced. Either:
// - a deferred renderer's lighting pass (`post_input` false): the colour
//   target its draws write is the lit scene, and its G-buffer what the
//   denoiser is guided by, read at the pixel being drawn: the diffuse
//   albedo (rgb at albedo_sampler), the world-space normal (xyz as unorm
//   at normal_sampler: N = normalize(2 t - 1)), the specular intensity and
//   gloss g (channels spec_channel and gloss_channel at surface_sampler).
//   Its roughness is that of the Blinn-Phong exponent e = exponent_scale
//   g^2, sqrt(2 / (e + 2)) (as RayReflection's); its specular albedo
//   (grey) sat(spec_scale spec F), with the Fresnel term F = fresnel_min +
//   (1 - fresnel_min) (1 - |I.N|)^fresnel_power (I the view direction).
// - or a post-process pass (`post_input`): its draws read the lit scene at
//   input_sampler. The scene is denoised once a frame, before the first
//   such draw that binds the lit scene's texture there (or a whole copy of
//   it made since the lighting: StretchRect), into that texture.
struct RayDenoise {
  bool post_input = false;
  uint32_t input_sampler = 16;  // 0-15
  uint32_t albedo_sampler = 16, normal_sampler = 16, surface_sampler = 16;  // 0-15
  uint32_t spec_channel = 0, gloss_channel = 1;  // 0-3
  float exponent_scale = 512.0f;
  float spec_scale = 1.0f;
  float fresnel_min = 0.04f, fresnel_power = 5.0f;
};

// A pixel shader drawn forward (blended glass, say: not in a G-buffer)
// whose translation reads ray-traced reflections through ray queries of its
// own (Device::SetRayForwardReflection): the reads of `sampler` became
// calls of r_rt_reflect (hlsl::RayReflectionCall, which names the origin's
// input and the registers below to the translation). Per pixel, one ray
// from the point in input register v<origin_input> (xyz, world space)
// along the read's coordinate (xyz: the reflected direction the shader
// computed), at most max_distance long (world units); what it hits lit as
// RayReflection's hits are (the light's direction in light_register, which
// way it points, its colour c[colour_register].rgb times its w, the wrapped
// diffuse term and a shadow ray; the ambient a + b sat(0.5 - 0.5 N.up)),
// times its albedo and output_scale. The call returns that (rgb) and a 1
// where a ray hit something, 0 elsewhere (the shader keeps its own
// reflection there).
struct RayForwardReflection {
  uint32_t sampler = 0;       // 0-15
  uint32_t origin_input = 0;  // v0-v9
  uint32_t light_register = 0;  // c0-c223
  bool toward_light = false;
  uint32_t colour_register = 0;
  uint32_t ambient_registers[2] = {0, 0};
  float up[3] = {0.0f, 0.0f, 1.0f};  // world space
  float diffuse_wrap = 0.0f;
  float output_scale = 1.0f;
  float max_distance = 100.0f;
};

// A pixel shader drawn forward (decals, blended or cut-out geometry: not
// all in the main view's depth) whose translation traces its far light's
// shadow itself through ray queries (Device::SetRayForwardShadow): the
// reads of `sampler` became calls of r_rt_shadow (hlsl::RayShadowCall,
// which names the light's register to the translation), each one ray from
// the point the read's coordinate gives (xyz, world space) toward the light
// (the register's direction, with which way it points, as RayShadow's) in
// place of a shadow lookup of its own (gfx/rt_forward_reflection.h): 1
// where it sees the light, 0 where not; 1 (lit) where nothing is traced.
// With `point`, the register holds a local light's position (world space)
// instead: the ray goes from the point to it and ends `lamp_gap` short of
// it (the lamp's own mesh around the light).
// With `local` too, the shader is a deferred renderer's local light volume
// drawn over the depth at depth_sampler (its pixel's point that depth's,
// its draws the light's registers, as RayShadow's local light): the
// device may trace the frame's lights once per pixel of that depth before
// they draw (one compute pass for all of them, the lights of the frame
// before; RayTracing::local_light_set), and a draw of a light in that set
// then reads its pixel's result instead of tracing (a light not in it, as
// one that just came into view, traces its own as above).
struct RayForwardShadow {
  uint32_t sampler = 0;         // 0-15
  uint32_t light_register = 0;  // c0-c223
  bool toward_light = false;
  bool point = false;
  float lamp_gap = 0;
  bool local = false;
  uint32_t depth_sampler = 16;  // 0-15
  uint32_t radius_register = 224, radius_component = 0;  // 1 / radius^2
  bool cone = false;
  uint32_t direction_register = 224;
  uint32_t cone_offset_register = 224, cone_offset_component = 0;
  uint32_t cone_scale_register = 224, cone_scale_component = 0;
};

// A pixel shader of a deferred renderer's lighting (Device::
// SetRayLightVolume): where its G-buffer is, and whether its draws are
// local lights' volumes (`volume`: one light a draw, added into the lit
// scene over the G-buffer). The device lights the frame's list of local
// lights itself with ray-traced shadows (gfx/rt_many_lights.h,
// RayTracing::many_lights) from these textures, read at the pixel being
// drawn: the depth (as RayShadow's), the diffuse albedo (rgb), the world-
// space normal (RayLightModel::normal_low_bits says how it is stored), the
// surface's specular intensity, gloss and a factor (x, y, z: RayLightModel)
// and a pixel class (red as unorm times 255: RayManyLights::classes). A
// shader may name some of them (16: none): the device takes each from the
// last draw that named it, the volumes' from where it is wanted. Before the
// first draw of a volume in a frame it lights the frame from them, into the
// colour target that draw writes.
struct RayLightVolume {
  bool volume = false;
  uint32_t depth_sampler = 16, albedo_sampler = 16, normal_sampler = 16, surface_sampler = 16, class_sampler = 16;
};

// A pixel shader of a deferred renderer's lighting whose flat ambient
// light is multiplied by a ray-traced ambient occlusion (Device::
// SetRayOcclusion, gfx/rt_occlusion.h): the occlusion (R8Unorm, 1 open) at
// occlusion_sampler, at the pixel being drawn, traced before its first
// draw in a frame from the depth it binds at depth_sampler (the main
// view's, as RayShadow's) and its normals at normal_sampler (xyz as unorm;
// 16: none).
struct RayOcclusion {
  uint32_t occlusion_sampler = 16, depth_sampler = 16, normal_sampler = 16;
};

// How a deferred renderer lights a pixel with a local light, for the device
// lighting its list (RayManyLights; a light's model, RtLight's flags). At a
// point p, d its distance from the light, L the unit vector from p to the
// light (normalize(light - p + 1e-5)), N the pixel's normal, I the unit
// vector from the camera to p (normalize(p - camera + 1e-5)), s its
// surface (x, y, z), and the light's colour c (intensity included) and cone
// k (saturate((dot(L, -direction) - cone_offset) cone_scale); 1 for a point
// light):
//   a = saturate(1 - d^2 / r^2), f = a^falloff_power - falloff_bias
//   (nothing where f < 0), light l = c f falloff_scale k;
//   diffuse: l saturate((N.L - wrap) wrap_scale) times the albedo;
//   with `specular`: plus l spec_scale s.x saturate(dot(L, reflect(I,
//   N)))^(exponent_scale s.y^2);
//   all times saturate(s.z^2 + 0.5) where the light's pixel class (0-3) is
//   in factor_classes (bit k).
// The normal: normalize(2 t - 1) of the texel t's xyz, or with
// normal_low_bits more bits from its w: e = frac(t.w (0.99609375, 7.96875,
// 63.75)), N = normalize(256 t.xyz - 127.99999 + (2 e.x - e.y / 4, 2 e.y -
// e.z / 4, 2 e.z)).
struct RayLightModel {
  float falloff_power = 4.0f, falloff_bias = 0.1f, falloff_scale = 1.0f / 0.9f;
  float wrap = 0.0f, wrap_scale = 1.0f;
  bool specular = true;
  float spec_scale = 2.0f, exponent_scale = 512.0f;
  uint32_t factor_classes = 0;
  bool normal_low_bits = false;
};

// The device lighting a deferred renderer's local lights itself (gfx/
// rt_many_lights.h): the lights of the frame's list (gfx::SetRtLights),
// each lit as its model says, shadowed by rays, over the G-buffer the
// renderer's light volumes read (RayLightVolume). Add: added to the lit
// scene before the volumes draw (they still draw); Only: the volumes'
// draws left out (what lights the scene is the device's alone).
struct RayManyLights {
  enum class Mode { Off, Add, Only } mode = Mode::Off;
  RayLightModel models[4];
  // A pixel's class value v (its class texel's red times 255, rounded):
  // pixel class k (1-3) holds the pixels whose v has bit v % 32 set in
  // classes[k - 1][v / 32]; class 0, every pixel. A light of class k lights
  // only its pixels.
  uint32_t classes[3][8] = {};
  float lamp_gap = 0.0f;  // world units: rays end this far short of the light (its lamp's mesh around it)
};

// Where a vertex shader input (hlsl::Program::inputs slot) comes from: a
// stream, the byte offset in its vertices and the D3DDECLTYPE; type 17
// (D3DDECLTYPE_UNUSED) if the vertex declaration does not supply it.
struct VertexInput {
  uint32_t stream = 0, offset = 0, type = 17;
};

struct VertexStream {
  const uint8_t* data = nullptr;  // vertex 0 (the buffer at its stream offset)
  uint32_t size = 0;              // bytes readable from data
  uint32_t stride = 0;
  // Only for a device that asks (TracksBufferContents), else 0: the
  // buffer's contents, a number that changes whenever they may have (a
  // lock that can write, and its unlock), never reused; 0 for data not in
  // a buffer (a draw from the application's memory). The buffer itself (a
  // number of the same series, kept for its life, renamed or not) and
  // `data`'s offset in it. `dynamic`: rewritten often (D3DUSAGE_DYNAMIC,
  // or locked frame after frame).
  uint64_t version = 0;
  uint64_t buffer = 0;
  uint32_t buffer_offset = 0;
  bool dynamic = false;
};

// A Direct3D 9 draw with shaders. Vertex v of a stream is at data +
// v * stride; v is index + base_vertex (indexed) or base_vertex + i, i <
// vertex_count (not indexed). Indexed draws read only vertices
// base_vertex + min_index .. + vertex_count - 1.
struct ShaderDraw {
  ShaderId vs = 0, ps = 0;
  uint32_t prim = 0;        // D3DPRIMITIVETYPE
  uint32_t primitives = 0;  // primitive count
  const VertexInput* inputs = nullptr;  // one per vertex shader input slot
  uint32_t input_count = 0;
  VertexStream streams[16];
  int32_t base_vertex = 0;
  uint32_t min_index = 0, vertex_count = 0;
  const void* indices = nullptr;  // null: not indexed
  uint32_t index_size = 2;        // bytes
  // The index buffer's, as VertexStream's version, buffer, buffer_offset
  // (of the first index) and dynamic.
  uint64_t index_version = 0, index_buffer = 0;
  uint32_t index_offset = 0;
  bool index_dynamic = false;
  // Drawn once per instance (instancing); streams of stride 0 then hold
  // the instance's data.
  bool instanced = false;
  // Constant registers: float c (vs 256, ps 224), int i (16 x 4), bool b
  // (16).
  const float (*vs_float)[4] = nullptr;
  const int32_t (*vs_int)[4] = nullptr;
  const uint32_t* vs_bool = nullptr;
  const float (*ps_float)[4] = nullptr;
  const int32_t (*ps_int)[4] = nullptr;
  const uint32_t* ps_bool = nullptr;
  // Change counters of vs_float and ps_float (0: unknown): a backend need
  // not compare registers whose counter has not moved.
  uint64_t vs_float_version = 0, ps_float_version = 0;
  // The same for the int and bool registers together.
  uint64_t vs_ib_version = 0, ps_ib_version = 0;
  // Samplers 0-15, then the vertex samplers 0-3: texture (0: none) and
  // D3DSAMPLERSTATETYPE values (index 1-13).
  TextureId textures[20] = {};
  const uint32_t (*sampler_states)[14] = nullptr;
  // Change counters of the sampler_states rows (0: unknown): a backend
  // need not convert a row whose counter has not moved.
  const uint64_t* sampler_versions = nullptr;
  // State changed since the last draw, applied first as SetRenderState,
  // SetViewport and SetScissor would (one call and one lock per draw):
  // `state_count` (state, value) pairs, and the viewport and scissor
  // rectangle where not null.
  const uint32_t (*states)[2] = nullptr;
  uint32_t state_count = 0;
  const Viewport* viewport = nullptr;
  const Rect* scissor = nullptr;
};

class Device {
 public:
  virtual ~Device() = default;

  // Whether Draw / MapDraw (the fixed-function pipeline) and the 2D blits
  // work. Without them, front ends draw fixed-function state with shaders
  // of their own.
  virtual bool HasFixedFunction() const { return true; }

  // ---- memory the CPU and the GPU share (Apple's unified memory)
  // Whether the device can read host memory in place (MapSharedMemory).
  virtual bool SharesMemory() const { return false; }
  // Lets the device read [host, host + size) in place: draws whose vertex
  // and index data lie there do not copy it. Page-aligned; the memory must
  // stay mapped for the device's lifetime. False if it cannot.
  virtual bool MapSharedMemory(void* host, size_t size) {
    (void)host, (void)size;
    return false;
  }
  // Frames, for memory read in place: what draws recorded now belong to,
  // and the last one the GPU has finished (everything up to it). Data a
  // draw reads in place must not change until its frame has finished.
  virtual uint64_t FrameSerial() const { return 0; }
  virtual uint64_t CompletedSerial() const { return 0; }

  // GPU time per presented frame, in ms, averaged over the frames the GPU
  // finished since the last call; negative if the backend cannot tell.
  virtual double TakeGpuFrameMs() { return -1; }
  // Memory the GPU allocations take, in bytes (-1: unknown).
  virtual int64_t GpuAllocatedBytes() const { return -1; }
  // What the GPU allocations are, for the FPS report (empty: not known).
  virtual std::string GpuMemoryText() const { return {}; }
  // Intervals between frames as the display showed them, in ms, since the
  // last call: median, 99th percentile and maximum; false if the backend
  // cannot tell.
  // Milliseconds from Present to the display, since the last call: median
  // and 99th percentile; false if the backend cannot tell.
  virtual bool TakePresentLatency(double* median, double* p99) {
    (void)median, (void)p99;
    return false;
  }
  virtual bool TakeShownIntervals(double* median, double* p99, double* max) {
    (void)median, (void)p99, (void)max;
    return false;
  }
  // Counters for tuning (the runtime's <ID>_PERF): kept once enabled;
  // TakePerfStats says what they show since the last call, per presented
  // frame ("" from a backend without them).
  virtual void EnablePerfStats() {}
  virtual std::string TakePerfStats() { return {}; }

  // (Re)creates the swap chain: a w x h back buffer presented to the
  // window. Called on SetDisplayMode / device creation.
  virtual void ResetSwapChain(int w, int h, bool vsync) = 0;
  // Whether the back buffer keeps alpha (A8R8G8B8, A2R10G10B10): blending
  // with its destination alpha then reads it, as a game masking with it
  // expects (GTA IV's round radar); without, alpha reads 1.
  virtual void SetBackBufferAlpha(bool) {}

  // ---- resources
  // A render target or depth buffer may be made at another size than
  // asked for, the game none the wiser (the Metal backend's internal
  // resolution and RECOMP_RT_BOOST); gfx/target_size.h has a port's way to
  // keep the ones it names exact (BeginExactTargets).
  virtual TextureId CreateTexture(int w, int h, int levels, Format fmt,
                                  bool render_target) = 0;
  virtual void DestroyTexture(TextureId t) = 0;
  // Fills `r` of a render-target texture's top level with an A8R8G8B8
  // colour (DirectDraw colour-fill Blt; its memory is GPU-only).
  virtual void FillTexture(TextureId t, const Rect& r, uint32_t argb) = 0;
  // Uploads `r` of mip `level` from linear memory (pitch in bytes).
  virtual void UpdateTexture(TextureId t, int level, const Rect& r,
                             const void* src, uint32_t pitch) = 0;
  // A cube texture: six square faces in D3D's order (+X, -X, +Y, -Y, +Z,
  // -Z), filled with UpdateCubeTexture and freed with DestroyTexture.
  virtual TextureId CreateCubeTexture(int size, int levels, Format fmt) = 0;
  virtual void UpdateCubeTexture(TextureId t, int face, int level, const Rect& r,
                                 const void* src, uint32_t pitch) = 0;
  virtual DepthId CreateDepth(int w, int h, Format fmt) = 0;
  // A depth-stencil buffer (D24S8) that shaders can also sample, Direct3D 9's
  // INTZ: the TextureId SetTexture binds (depth in every channel read) and,
  // in *depth, the DepthId render targets use. DestroyTexture releases it.
  virtual TextureId CreateDepthTexture(int w, int h, DepthId* depth) {
    *depth = 0;
    (void)w, (void)h;
    return 0;
  }
  virtual void DestroyDepth(DepthId d) = 0;

  // ---- render target (color 0 = back buffer; Present copies it to the
  // front buffer and shows that)
  virtual void SetRenderTarget(TextureId color, DepthId depth) = 0;
  // Up to four colour targets drawn at once (shaders' COLOR0-3): color[0]
  // as SetRenderTarget's (or a level of it), the others texture levels or
  // tex 0 (none, and none after the first one).
  virtual void SetRenderTargets(const ColorTarget color[4], DepthId depth) = 0;
  // Copies `sr` of a colour target to `dr` of another, scaled with linear
  // or nearest filtering (Direct3D 9's StretchRect).
  virtual void StretchRect(ColorTarget src, const Rect& sr, ColorTarget dst, const Rect& dr,
                           bool linear) = 0;

  // ---- 2D (DirectDraw Blt onto the back buffer / render target)
  // Copies `src_rect` of a linear image into `dst_rect` of the current
  // render target, stretching if the sizes differ. If `colorkey`, source
  // pixels equal to `key` (in the source format) are skipped.
  virtual void Blit(const void* src, uint32_t pitch, Format fmt,
                    const Rect& src_rect, const Rect& dst_rect, bool colorkey,
                    uint32_t key) = 0;
  // Draws `src_rect` of an A8R8G8B8 image over `dst_rect` of the current
  // render target, blended by the image's alpha (source over, straight
  // alpha). For what a host hook draws in place of a game's own CPU
  // blending onto read-back pixels (a software cursor), which would make
  // the CPU wait for the frame's GPU work.
  virtual void BlitBlend(const void* src, uint32_t pitch,
                         const Rect& src_rect, const Rect& dst_rect) = 0;
  // Reads `r` of the current render target into linear memory.
  virtual void ReadBack(const Rect& r, void* dst, uint32_t pitch,
                        Format fmt) = 0;
  // ReadBack in two steps, so the CPU need not wait for the GPU:
  // StartReadBack takes `r` of the current render target as the draws so
  // far leave it and returns a ticket (never 0); FinishReadBack waits for
  // that copy and converts it as ReadBack does, or only drops it if `dst`
  // is null. Every ticket is finished exactly once.
  virtual uint32_t StartReadBack(const Rect& r) = 0;
  virtual void FinishReadBack(uint32_t ticket, void* dst, uint32_t pitch,
                              Format fmt) = 0;

  // ---- front buffer (the DirectDraw primary): the last presented frame,
  // which the game may also read and draw on between frames (its cursor).
  virtual void BlitFront(const void* src, uint32_t pitch, Format fmt,
                         const Rect& src_rect, const Rect& dst_rect) = 0;
  virtual void ReadBackFront(const Rect& r, void* dst, uint32_t pitch,
                             Format fmt) = 0;
  // Shows the front buffer as it is now.
  virtual void PresentFront() = 0;

  // ---- fixed-function state (D3D7 enums)
  virtual void SetRenderState(uint32_t state, uint32_t value) = 0;
  virtual void SetTextureStageState(uint32_t stage, uint32_t type,
                                    uint32_t value) = 0;
  virtual void SetTexture(uint32_t stage, TextureId t) = 0;
  virtual void SetTransform(uint32_t type, const float m[16]) = 0;
  virtual void SetViewport(const Viewport& v) = 0;
  // In render-target pixels; used while SCISSORTESTENABLE (174) is set.
  virtual void SetScissor(const Rect& r) = 0;
  virtual void SetMaterial(const Material& m) = 0;
  virtual void SetLight(uint32_t index, const Light& l) = 0;
  virtual void LightEnable(uint32_t index, bool enable) = 0;
  virtual void SetClipPlane(uint32_t index, const float plane[4]) = 0;

  // ---- drawing
  virtual void Clear(uint32_t count, const Rect* rects, uint32_t flags,
                     uint32_t color, float z, uint32_t stencil) = 0;
  virtual void BeginScene() = 0;
  virtual void EndScene() = 0;
  // Non-indexed when `indices` is null. `stride` from the FVF.
  virtual void Draw(uint32_t prim, uint32_t fvf, const void* vertices,
                    uint32_t vertex_count, uint32_t stride,
                    const uint16_t* indices, uint32_t index_count) = 0;
  // Indexed draw built in place: MapDraw returns room for `vertex_count`
  // vertices (`stride` bytes each) and `index_count` indices in memory the
  // device reads them from, for the caller to fill and then draw with
  // DrawMapped before the next MapDraw. Saves Draw's copy for vertices
  // made per draw.
  struct Mapped {
    uint8_t* vertices;
    uint16_t* indices;
  };
  virtual Mapped MapDraw(uint32_t vertex_count, uint32_t stride,
                         uint32_t index_count) = 0;
  virtual void DrawMapped(uint32_t prim, uint32_t fvf) = 0;
  virtual void Present() = 0;
  // Post-process anti-aliasing (SMAA 1x) of each presented frame, where the
  // backend has it (SDL_GPU).
  virtual void SetPostAntialiasing(bool on) { (void)on; }
  // Anisotropic filtering of at least `level` (2..16; 0: as the game sets
  // it) for mipmapped textures sampled with linear filtering.
  virtual void SetMinAnisotropy(int level) { (void)level; }
  // Depth from w (Metal: gfx/reversed_z.h): the vertex shader constants
  // c[wvp_register]..+3 where the game keeps its world-view-projection
  // (rows, D3D's row vectors; -1: none), perspective draws whose position
  // is made from them getting their depth from w alone; and, optional,
  // the near and far planes of the projection the game draws with at the
  // time of the call (false if it cannot tell), exact where the matrix's
  // are rounded. Called from the drawing thread, at a draw.
  using DepthPlanes = bool (*)(float* near_plane, float* far_plane);
  virtual void SetDepthFromW(int wvp_register, DepthPlanes planes) { (void)wvp_register, (void)planes; }
  // Ray tracing of the scene (Metal on GPUs with hardware ray tracing;
  // others ignore it): acceleration structures of the geometry each frame
  // draws (gfx/rt_scene.h), for effects that trace rays: static meshes,
  // placed by their world matrix, and draws whose positions only the
  // vertex shader makes (skinned, displaced by a vertex texture, drawn per
  // instance), captured from it; opaque, or cut out by their alpha where
  // they alpha test or texkill. Before the first frame.
  struct RayTracing {
    // What rays are traced for; any but Off keeps the scene.
    enum class Mode { Off, On, Shadows, Reflections } mode = Mode::Off;
    // Drawn over each presented frame: the scene's first hits (normals,
    // instances), or the frame's last traced shadow mask (white lit).
    enum class Debug { Off, Normals, Instances, ShadowMask } debug = Debug::Off;
    // Ray-traced shadows' diagnostics on stderr every 120 presented frames
    // (also with perf stats on): the light, the camera, the centre pixel's
    // trace and what blocked it, the instances that blocked most rays.
    bool shadow_log = false;
    // Pixel shaders that still read a shadow map of the game's own
    // (RayShadow::own_map) read one that is lit everywhere instead (a
    // debug view).
    bool own_maps_lit = false;
    // Drawn over each presented frame with Reflections: the frame's last
    // traced reflections (their colour where a ray hit, dark blue where one
    // missed, black where none was traced).
    bool reflection_view = false;
    // The vertex shader constants c[n]..c[n + 3] where the game keeps the
    // object-to-world matrix of the draw (rows, D3D's row vectors); -1:
    // not known, no draw is taken.
    int world_register = -1;
    // Optional: called on the drawing thread for each draw that would be
    // taken, with its vertex shader constants and primitive count; false
    // leaves it out (geometry that moves with the camera and writes depth,
    // a sky dome, say). `captured`: a draw whose positions its vertex
    // shader makes (its shader need not read the world registers, so
    // vs_float's may be another draw's); `other_view`: one of another pass
    // (other_passes).
    bool (*filter)(const float (*vs_float)[4], uint32_t primitives, bool captured, bool other_view) = nullptr;
    // Optional: where the alpha of draws that discard by it (the alpha
    // test, texkill) comes from, for the pixel shader `ps_hash` (its
    // translation's, as RECOMP_SHADER_DUMP names its files): true with the
    // sampler register whose texture's alpha, at the vertex input
    // TEXCOORD<texcoord>'s x and y, times COLOR0's alpha where the vertex
    // has one, is what the pixel shader compares; false (or none): found
    // from the shaders (gfx/metal_device.mm, RtAlphaFor).
    bool (*alpha_source)(uint64_t ps_hash, uint32_t* sampler, uint32_t* texcoord) = nullptr;
    // Optional (Reflections): where draws' albedo comes from, for the pixel
    // shader `ps_hash`: true with the sampler register and the vertex input
    // TEXCOORD<n> its UVs are; false (or none): an alpha-cut draw's alpha
    // texture, else sampler 0 at TEXCOORD0 (gfx/rt_alpha.h,
    // RtResolveAlbedo).
    bool (*albedo_source)(uint64_t ps_hash, uint32_t* sampler, uint32_t* texcoord) = nullptr;
    // Static draws of other passes that test and write depth but no colour,
    // or colour only into single-channel float targets (a shadow map's
    // casters, its depth kept in colour too), in the scene too, placed by
    // their world matrix alone (gfx/rt_scene.h, RtView::Other), so that what the main
    // camera does not see still casts shadows; with other_color_passes
    // also those of passes that write colour (reflections, environment
    // maps, models drawn into textures: the filter then picks). Off for a
    // game whose other passes keep something else in the world registers.
    bool other_passes = true, other_color_passes = false;
    // Local lights' shadow masks (RayShadow::local) traced at most per
    // presented frame: each ends the render pass open (a store and a load
    // of its targets, ~0.5 ms at 2560 x 1440 by estimate); the lights past
    // it are all lit. 0: none traced.
    uint32_t local_light_masks = 8;
    // Local lights whose volumes' pixels trace their own shadow
    // (RayForwardShadow::local): the lights drawn in the presented frame
    // before traced at once, in one compute pass over the depth before the
    // lights draw (gfx/rt_shadow.h, RtShadowMasker::EncodeLights), at one
    // pixel of every local_light_step x local_light_step, at most
    // local_light_cap of them a pixel (the strongest there; 1-4); their
    // draws read that, upsampled by depth, instead of a ray per pixel of
    // each volume (which overlapping volumes at a high resolution make
    // millions a frame). 0: off, every volume's pixels trace their own.
    // RECOMP_RT_LIGHT_STEP and RECOMP_RT_LIGHT_CAP set them too. (At 3440
    // x 1440 in a dense street at night, 15 shadowed lights: some 3 ms of
    // GPU time at step 3, 5 at step 2; each volume's pixels tracing their
    // own, 35-40.)
    uint32_t local_light_step = 3, local_light_cap = 4;
    // MetalFX's denoiser over the lit scene before post-processing (gfx/
    // rt_denoise.h), where shadows are traced, the GPU and system have it
    // (macOS 26 / iOS 26, Apple9) and a game names its shaders
    // (SetRayDenoise): the scene then jittered and denoised at the back
    // buffer's size, in place of a post-process anti-aliasing or upscaler
    // (RECOMP_UPSCALER, RECOMP_RENDER_SCALE: not scaled down). Off unless
    // asked for (<ID>_RT_DENOISE=1): at the back buffer's size it costs
    // what the upscaler saves, and the shadows it was for are hard.
    bool denoise = false;
    // The local lights lit by the device itself (RayManyLights), where
    // shadows are traced; Off unless asked for.
    RayManyLights many_lights;
    // Ray-traced ambient occlusion (RayOcclusion) where shadows are traced,
    // its rays this long (world units); off with 0.
    float occlusion_radius = 0.0f;
    // The far light bounced once off what it lights (gfx/rt_reflection.h,
    // EncodeBounce), where reflections are traced: the shaders that read
    // the occlusion (RayOcclusion) get that light at the occlusion's
    // sampler instead (rgb, to add to their ambient). Its rays this long
    // (world units); off with 0.
    float bounce_distance = 0.0f;
  };
  virtual void SetRayTracing(const RayTracing& rt) { (void)rt; }
  // Whether draws should carry their buffers' versions (VertexStream):
  // true once ray tracing is on. Front ends ask when the device is made.
  virtual bool TracksBufferContents() const { return false; }
  // Whether ray tracing is on (SetRayTracing took): front ends then
  // translate shaders with what the scene needs of them
  // (hlsl::TranslateOptions::coverage). SetRayTracing comes before the
  // first shader.
  virtual bool RayTracingOn() const { return false; }
  // Decode DXT textures on upload even where the GPU samples BC formats
  // (to check the path GPUs without them take). Before any texture exists.
  virtual void DecodeDxtOnUpload() {}

  // ---- programmable pipeline. A shader from the HLSL translation (null
  // on failure); render states (blend, depth, stencil, cull, colour
  // write, alpha test, scissor), the viewport and the targets as set
  // above, the rest per draw.
  virtual ShaderId CreateShader(const hlsl::Program& p) = 0;
  virtual void DestroyShader(ShaderId s) = 0;
  virtual void DrawShader(const ShaderDraw& d) = 0;
  // Ray-traced shadows (RayTracedShadows): pixel shader `ps` reads a
  // shadow mask (gfx/rt_shadow.h: .x 1 lit, 0 in shadow) at sampler
  // `s.mask_sampler`, at the pixel being drawn, in place of a shadow lookup
  // of its own. Before its first draw in a frame (and again once the depth
  // it reads has been drawn to) the device traces the mask from the depth
  // texture the draw binds at `s.depth_sampler` and the light direction in
  // its constants, and binds it there; where nothing can be traced, a mask
  // with everything lit (also for a shader whose other slots are out of
  // range). Traced only from the main view's depth: the back buffer's
  // size, written last by the frame's camera over its whole viewport
  // (depth range 0-1). A depth texture gets its camera only from the
  // main view's draws into it (the scene's static draws, which vote):
  // depth copied into it by the game's own draws without that camera
  // (a full-screen copy, a resolve) leaves it all lit. A local light's
  // (s.local): a mask per light, traced before the first draw naming that
  // light (its registers' values) in a frame and kept for its other draws
  // while the depth is unchanged, over the light's bounds on screen
  // (gfx/rt_shadow.h: RtLocalLightBounds); each one traced ends the render
  // pass open (a store and a load), so at most RayTracing::
  // local_light_masks lights a frame are traced, the rest all lit. Ignored
  // by devices that do not trace shadows.
  virtual void SetRayShadow(ShaderId ps, const RayShadow& s) { (void)ps, (void)s; }
  // Ray-traced reflections (RayTracedReflections): pixel shader `ps` reads
  // them (gfx/rt_reflection.h: rgb the light reflected, a 1 where a ray hit)
  // at sampler `r.output_sampler`, at the pixel being drawn. Traced before
  // its first draw in a frame (and again once the depth it reads has been
  // drawn to, or its constants changed) from what the draw binds at
  // r's samplers and holds in r's registers; the same main-view depth only
  // as SetRayShadow's (else, and for a shader whose other slots are out of
  // range, nothing hit: the shader keeps its own reflection). Ignored by
  // devices that do not trace reflections.
  virtual void SetRayReflection(ShaderId ps, const RayReflection& r) { (void)ps, (void)r; }
  // Ray-traced reflections in forward pixel shaders
  // (RayTracedForwardReflections): pixel shader `ps`, translated with
  // hlsl::TranslateOptions::ray_reflection from `r` (its sampler, origin
  // and registers), traces its reflections itself (gfx/
  // rt_forward_reflection.h). Its draws trace against the scene's snapshot
  // the frame's traced effects took last (the TLAS must exist before the
  // forward pass: none is built for them), and only in the view of that
  // snapshot's camera (the draw's world and position matrices give that
  // camera where its vertex shader reads both; else it draws into the
  // target the snapshot was taken for, or a depth buffer that camera wrote
  // last); elsewhere, and for a shader whose translation could not be given
  // the ray queries, nothing is hit (the shader keeps its own reflection).
  // Ignored by devices that do not trace them.
  virtual void SetRayForwardReflection(ShaderId ps, const RayForwardReflection& r) { (void)ps, (void)r; }
  // Ray-traced shadows in forward pixel shaders: pixel shader `ps`,
  // translated with hlsl::TranslateOptions::ray_shadow from `s` (its sampler
  // and the light's register), traces its far light's shadow itself (gfx/
  // rt_forward_reflection.h), on GPUs that trace rays from render pipelines
  // while shadows are traced; its draws trace against the scene's
  // snapshot and only in the view SetRayForwardReflection's do (the same
  // rules); elsewhere, on other GPUs, and for a shader whose translation
  // could not be given the ray queries, everything is lit (the
  // translation's stub). Ignored by devices that do not trace them.
  virtual void SetRayForwardShadow(ShaderId ps, const RayForwardShadow& s) { (void)ps, (void)s; }
  // Denoising of the lit scene (RayTracing::denoise): pixel shader `ps` is
  // the lighting pass or a post-process pass that reads it, as `d` says
  // (RayDenoise). Before the first post-process draw of a frame that reads
  // the colour target the lighting drew into this frame (or a whole copy of
  // it), the device denoises that texture (gfx/rt_denoise.h). Ignored by
  // devices that do not denoise.
  virtual void SetRayDenoise(ShaderId ps, const RayDenoise& d) { (void)ps, (void)d; }
  // The local lights lit by the device (RayTracing::many_lights): pixel
  // shader `ps` names where the G-buffer is, or draws local lights' volumes
  // (RayLightVolume). Ignored by devices that do not.
  virtual void SetRayLightVolume(ShaderId ps, const RayLightVolume& v) { (void)ps, (void)v; }
  // Ray-traced ambient occlusion (RayTracing::occlusion_radius): pixel
  // shader `ps` reads it (RayOcclusion). Ignored by devices that do not.
  virtual void SetRayOcclusion(ShaderId ps, const RayOcclusion& o) { (void)ps, (void)o; }
  // The frame's local lights (gfx/rt_light_list.h), what gfx::SetRtLights
  // hands to the device that takes them (NoteRtLightDevice). Ignored by
  // devices that do not.
  virtual void SetRtLights(const RtLight* lights, size_t count) { (void)lights, (void)count; }
  // Places whose forward draws trace no lamp ray
  // (gfx::SetRtPlainPlaces).
  virtual void SetRtPlainPlaces(const float* places, size_t count) { (void)places, (void)count; }
};

// Whether the device made last traces shadows (RayShadow): ray tracing on,
// on a GPU that traces rays in hardware, in a mode that traces shadows
// (Shadows; Reflections includes them). False until SetRayTracing. What a
// game asks before it makes its shaders, to give the ones that read the
// mask (they read nothing else in their shadow's place).
bool RayTracedShadows();
// Backends say so (in SetRayTracing).
void NoteRayTracedShadows(bool on);
// Likewise for reflections (RayReflection): the mode Reflections, on such a
// GPU, its kernel made.
bool RayTracedReflections();
void NoteRayTracedReflections(bool on);
// Likewise for reflections traced by forward pixel shaders themselves
// (RayForwardReflection): reflections traced, on a GPU that traces rays
// from render stages (Metal's supportsRaytracingFromRender). A game then
// translates the shaders its hook names with hlsl::TranslateOptions::
// ray_reflection.
bool RayTracedForwardReflections();
void NoteRayTracedForwardReflections(bool on);
// Likewise for shadows traced by pixel shaders themselves
// (RayForwardShadow): shadows traced, on such a GPU. A game then gives the
// shaders its hook names hlsl::TranslateOptions::ray_shadow.
bool RayTracedForwardShadows();
void NoteRayTracedForwardShadows(bool on);
// The direction toward the light of the last ray-traced shadow mask of the
// whole frame (a game's sun; RtShadowParams::light, unit, world space):
// false until one was traced. For a game's culling: what lies toward the
// sun from the view casts shadows into it.
bool RtSunToward(float out[3]);
// Backends say so as they trace such a mask.
void NoteRtSunToward(const float toward[3]);

// The local lights the game lights the frame being recorded with
// (gfx/rt_light_list.h: RtLight), for effects that light the scene by ray
// tracing: copied (up to kRtLightsMax valid ones) for that frame's work,
// the last call of a frame winning (none: an empty list). From the thread
// that makes the device's calls, as it records the frame (a game's hook on
// its own frame's light list). Nothing without a device that takes them.
void SetRtLights(const RtLight* lights, size_t count);
// Places (`count` of them, five floats each: x, y, z, a radius across and a
// half height) where a forward draw, its world matrix's origin within one,
// traces no lamp shadows (its reflection and its sun's shadow still): what
// is drawn there costs more than such rays show (a subway train's glass and
// body filling the view, lit inside by its own lamps). The last call of a frame wins,
// from the thread SetRtLights is called on.
void SetRtPlainPlaces(const float* places, size_t count);
// Backends that take them say so (in SetRayTracing: true once ray tracing
// is on) and say false before they go; false unregisters `d` only while it
// is the registered one (another device that took over stays). Devices are
// made and destroyed on the thread that makes their calls, SetRtLights's.
void NoteRtLightDevice(Device* d, bool on);

// A ShaderDraw's state changes through the setters, for backends that
// take them no faster way.
inline void ApplyDrawState(Device& dev, const ShaderDraw& d) {
  for (uint32_t i = 0; i < d.state_count; i++) dev.SetRenderState(d.states[i][0], d.states[i][1]);
  if (d.viewport) dev.SetViewport(*d.viewport);
  if (d.scissor) dev.SetScissor(*d.scissor);
}

// Where a src_w x src_h frame is shown in a dst_w x dst_h output: scaled
// to fit with its aspect kept and centred, black around it (letterbox or
// pillarbox). Backends present the front buffer through it; the window
// system maps the pointer back through it.
Rect FitRect(int src_w, int src_h, int dst_w, int dst_h);

// The last frame any device showed, in pixels: the frame's size, the
// output's (swap chain) and FitRect's result. Zeros before the first.
struct Presentation {
  int src_w, src_h, out_w, out_h;
  Rect rect;
};
// Backends call this when they show the front buffer.
void NotePresentation(const Presentation& p);
Presentation LastPresentation();

// Vertex size in bytes for a D3D7 FVF code.
uint32_t FvfStride(uint32_t fvf);
uint32_t PrimitiveCountFromVertices(uint32_t prim_type, uint32_t verts);

enum class Backend {
  D3D9,  // dxvk-native; only in builds with RECOMP_HAVE_D3D9
  Gpu,   // SDL_GPU
  Metal,  // Metal directly; only on Apple platforms
};
// Null if `b` is not in this build or cannot start on this host.
std::unique_ptr<Device> CreateDevice(Backend b, SDL_Window* window);

std::unique_ptr<Device> CreateD3D9Device(SDL_Window* window);
std::unique_ptr<Device> CreateGpuDevice(SDL_Window* window);
std::unique_ptr<Device> CreateMetalDevice(SDL_Window* window);

}  // namespace recomp::gfx
