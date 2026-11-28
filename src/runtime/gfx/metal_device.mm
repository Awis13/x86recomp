// gfx::Device on Metal (Apple GPUs: macOS and iPadOS), for Direct3D 9.
//
// Only the programmable pipeline: shaders translated from HLSL
// (runtime/hlsl), then SPIR-V and MSL as for SDL_GPU, with the same
// prelude and Metal binding layout as gpu_device.cpp. Fixed-function draws
// are not provided (HasFixedFunction false: the d3d9 front end draws them
// with its stand-in shaders), nor are DirectDraw blits.
//
// The game's thread (its render thread) records; an encoding thread of the
// device's own does every Metal command-buffer and encoder call, from
// closures recorded in order (Rec), so encoding, pipeline compiles and
// waiting for a drawable leave the game's thread. The frame is one command
// buffer, from Present to Present (or until a readback needs the GPU's
// result). A render target change ends one render
// encoder and the next draw starts another; a clear of a whole target
// before anything is drawn is the encoder's load action. Per-draw data
// (vertices, indices, constants) is written once, from guest memory, into
// a bump arena of shared buffers per command buffer, reused when that
// command buffer has finished (the buffers made ahead, off the game's
// thread); vertex and index data in the memory the front end shares with
// the GPU (MapSharedMemory) is not copied at all. Texture uploads go to a blit encoder in a
// command buffer queued before the frame's, so they never split a render
// pass; only an upload to a texture the frame has already drawn with or
// into is encoded in order in the frame itself.
//
// Formats as in gpu_device.cpp: colour textures and targets B8G8R8A8 (other
// D3D colour formats converted on upload), DXT as BC1-3, float formats as
// is. Depth: D16 as Depth16Unorm, 24-bit depth as 32-bit float (Apple GPUs
// have no 24-bit depth), with stencil as Depth32Float_Stencil8; float depth
// stored reversed (1 - d, gfx/reversed_z.h) unless RECOMP_REVERSED_Z=0.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>
#import <MetalFX/MetalFX.h>
#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include <TargetConditionals.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <thread>
#include <deque>
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/gfx/camera_vote.h"
#include "runtime/gfx/device.h"
#include "runtime/gfx/flight_recorder.h"
#include "runtime/gfx/frame_log.h"
#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/metal_archive.h"
#include "runtime/gfx/metal_cache.h"
// kSmaaHlsl: SMAA.hlsl (smaa/, MIT), and its lookup textures.
#include "runtime/gfx/smaa/AreaTex.h"
#include "runtime/gfx/smaa/SearchTex.h"
#include "smaa_hlsl.h"
#include "runtime/gfx/metal_new_target.h"
#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/output_curve.h"
#include "runtime/gfx/rcas.h"
#include "runtime/gfx/reversed_z.h"
#include "runtime/gfx/rt_accumulate.h"
#include "runtime/gfx/rt_alpha.h"
#include "runtime/gfx/rt_denoise.h"
#include "runtime/gfx/rt_forward_reflection.h"
#include "runtime/gfx/rt_light_list.h"
#include "runtime/gfx/rt_many_lights.h"
#include "runtime/gfx/rt_fog.h"
#include "runtime/gfx/rt_occlusion.h"
#include "runtime/gfx/rt_reflection.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"
#include "runtime/gfx/shader_compile.h"
#include "runtime/hlsl/capture_position.h"
#include "runtime/gfx/shader_jobs.h"
#include "runtime/gfx/target_size.h"
#include "runtime/achievements.h"
#include "runtime/log.h"

// RECOMP_DISPLAY_LINK=1 (macOS 14+): the display hands out each drawable
// in time for its next refresh (CAMetalDisplayLink, Apple's frame pacing
// for games, variable refresh included); Show takes the latest one instead
// of asking the layer (nextDrawable) whenever a frame is done. A frame is
// then shown on the refresh it was made for: no uneven gaps without vsync,
// no queue of finished frames waiting (input lag) with it.
struct RecompLinkSlot {
  std::mutex mu;
  std::condition_variable cv;
  id<CAMetalDrawable> drawable = nil;
};

API_AVAILABLE(macos(14.0))
@interface RecompDisplayLinkTarget : NSObject <CAMetalDisplayLinkDelegate>
@property(nonatomic, assign) RecompLinkSlot* slot;
@end

@implementation RecompDisplayLinkTarget
- (void)metalDisplayLink:(CAMetalDisplayLink*)link needsUpdate:(CAMetalDisplayLinkUpdate*)update {
  (void)link;
  {
    std::lock_guard<std::mutex> l(_slot->mu);
    _slot->drawable = update.drawable;  // the one before, unused, goes back
  }
  _slot->cv.notify_one();
}
@end

namespace recomp::gfx {

namespace {

float AsFloat(uint32_t v) {
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

void Argb(float* out, uint32_t c) {
  out[0] = float((c >> 16) & 0xFF) / 255.0f;
  out[1] = float((c >> 8) & 0xFF) / 255.0f;
  out[2] = float(c & 0xFF) / 255.0f;
  out[3] = float(c >> 24) / 255.0f;
}

// ---- D3D enums -> Metal

MTLBlendFactor BlendFactor(uint32_t d3d, bool target_alpha) {
  switch (d3d) {
    case 1: return MTLBlendFactorZero;
    case 2: return MTLBlendFactorOne;
    case 3: return MTLBlendFactorSourceColor;
    case 4: return MTLBlendFactorOneMinusSourceColor;
    case 5: return MTLBlendFactorSourceAlpha;
    case 6: return MTLBlendFactorOneMinusSourceAlpha;
    // A target without alpha (X8R8G8B8, R5G6B5) reads alpha as 1.
    case 7: return target_alpha ? MTLBlendFactorDestinationAlpha : MTLBlendFactorOne;
    case 8: return target_alpha ? MTLBlendFactorOneMinusDestinationAlpha : MTLBlendFactorZero;
    case 9: return MTLBlendFactorDestinationColor;
    case 10: return MTLBlendFactorOneMinusDestinationColor;
    case 11: return MTLBlendFactorSourceAlphaSaturated;
    case 14: return MTLBlendFactorBlendColor;
    case 15: return MTLBlendFactorOneMinusBlendColor;
    default: return MTLBlendFactorOne;
  }
}

MTLBlendOperation BlendOp(uint32_t d3d) {
  switch (d3d) {
    case 2: return MTLBlendOperationSubtract;
    case 3: return MTLBlendOperationReverseSubtract;
    case 4: return MTLBlendOperationMin;
    case 5: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
  }
}

// D3DCMPFUNC (NEVER 1 .. ALWAYS 8) and D3DSTENCILOP (KEEP 1 .. DECR 8) are
// Metal's numbering plus one.
MTLCompareFunction CompareOp(uint32_t d3d) {
  return d3d >= 1 && d3d <= 8 ? MTLCompareFunction(d3d - 1) : MTLCompareFunctionAlways;
}
MTLStencilOperation StencilOp(uint32_t d3d) {
  return d3d >= 1 && d3d <= 8 ? MTLStencilOperation(d3d - 1) : MTLStencilOperationKeep;
}

MTLSamplerAddressMode AddressMode(uint32_t d3d) {
  switch (d3d) {
    case 2: return MTLSamplerAddressModeMirrorRepeat;
    case 3: return MTLSamplerAddressModeClampToEdge;
    case 4: return MTLSamplerAddressModeClampToBorderColor;
    case 5: return MTLSamplerAddressModeMirrorClampToEdge;  // MIRRORONCE
    default: return MTLSamplerAddressModeRepeat;
  }
}

// COLORWRITEENABLE (red 1, green 2, blue 4, alpha 8) in Metal's bit order
// (red 8, green 4, blue 2, alpha 1).
uint8_t WriteMask(uint32_t d3d) {
  return uint8_t(((d3d & 1) << 3) | ((d3d & 2) << 1) | ((d3d & 4) >> 1) | ((d3d & 8) >> 3));
}

bool HasAlpha(Format f) {
  switch (f) {
    case Format::A8R8G8B8: case Format::A1R5G5B5: case Format::A4R4G4B4:
    case Format::A8: case Format::A8L8: case Format::DXT1: case Format::DXT2:
    case Format::DXT3: case Format::DXT4: case Format::DXT5:
    case Format::A16B16G16R16F: case Format::A32B32G32R32F:
      return true;
    default:
      return false;
  }
}

MTLPixelFormat PixelFormat(Format f) {
  switch (f) {
    case Format::DXT1: return MTLPixelFormatBC1_RGBA;
    case Format::DXT2: case Format::DXT3: return MTLPixelFormatBC2_RGBA;
    case Format::DXT4: case Format::DXT5: return MTLPixelFormatBC3_RGBA;
    case Format::R16F: return MTLPixelFormatR16Float;
    case Format::G16R16F: return MTLPixelFormatRG16Float;
    case Format::A16B16G16R16F: return MTLPixelFormatRGBA16Float;
    case Format::R32F: return MTLPixelFormatR32Float;
    case Format::G32R32F: return MTLPixelFormatRG32Float;
    case Format::A32B32G32R32F: return MTLPixelFormatRGBA32Float;
    default: return MTLPixelFormatBGRA8Unorm;
  }
}

void ConvertToArgb(const uint8_t* src, uint32_t pitch, Format f, int w, int h, uint32_t* out) {
  uint32_t bpp = BitsPerPixel(f) / 8;
  for (int y = 0; y < h; y++) {
    const uint8_t* s = src + size_t(y) * pitch;
    uint32_t* d = out + size_t(y) * uint32_t(w);
    if (f == Format::A8R8G8B8) {
      std::memcpy(d, s, size_t(w) * 4);
    } else if (f == Format::X8R8G8B8) {
      const auto* p = reinterpret_cast<const uint32_t*>(s);
      for (int x = 0; x < w; x++) d[x] = p[x] | 0xFF000000u;
    } else {
      for (int x = 0; x < w; x++) d[x] = DecodePixel(s + size_t(x) * bpp, f);
    }
  }
}

Rect Intersect(const Rect& a, const Rect& b) {
  return {std::max(a.left, b.left), std::max(a.top, b.top), std::min(a.right, b.right),
          std::min(a.bottom, b.bottom)};
}
Rect Clip(const Rect& r, int w, int h) { return Intersect(r, {0, 0, w, h}); }
bool Empty(const Rect& r) { return r.right <= r.left || r.bottom <= r.top; }

// ---- cache keys (compared and hashed as bytes: no padding; PipeKey in
// metal_cache.h)

struct DepthKey {
  uint8_t test, write, func, stencil;
  uint8_t fail, zfail, pass, sfunc;     // front faces (D3D's clockwise ones)
  uint8_t bfail, bzfail, bpass, bfunc;  // back faces (TWOSIDEDSTENCILMODE's CCW ops)
  uint8_t read, wmask, pad[2];
  bool operator==(const DepthKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};
static_assert(sizeof(DepthKey) == 16, "no padding");

struct SamplerKey {
  uint8_t min, mag, mip, u, v, w, aniso, border;  // mip 0xFF: MIPFILTER NONE
  float min_lod;
  bool operator==(const SamplerKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};
static_assert(sizeof(SamplerKey) == 12, "no padding");

template <typename K>
struct BytesHash {
  size_t operator()(const K& k) const {
    uint64_t h = 0xCBF29CE484222325ull;
    const auto* b = reinterpret_cast<const uint8_t*>(&k);
    for (size_t i = 0; i < sizeof k; i++) h = (h ^ b[i]) * 0x100000001B3ull;
    return size_t(h);
  }
};

// The device's own shaders: clears and textured quads (StretchRect, the
// window).
constexpr const char* kInternalMsl = R"(
#include <metal_stdlib>
using namespace metal;
struct ClearOut { float4 pos [[position]]; };
// A triangle over the whole target at depth p.x (the scissor bounds it).
vertex ClearOut clear_vs(uint vid [[vertex_id]], constant float4& p [[buffer(0)]]) {
  float2 t = float2((vid << 1) & 2, vid & 2);
  ClearOut o;
  o.pos = float4(t * 2.0 - 1.0, p.x, 1.0);
  return o;
}
fragment float4 clear_fs(constant float4& c [[buffer(0)]]) { return c; }
struct QuadOut { float4 pos [[position]]; float2 uv; };
// A strip of four: r[0] the rectangle in clip space (x0, y0, x1, y1),
// r[1] the source in texture coordinates, r[2].x the source level.
vertex QuadOut quad_vs(uint vid [[vertex_id]], constant float4* r [[buffer(0)]]) {
  float2 t = float2(vid & 1, vid >> 1);
  QuadOut o;
  o.pos = float4(mix(r[0].xy, r[0].zw, t), 0.0, 1.0);
  o.uv = mix(r[1].xy, r[1].zw, t);
  return o;
}
fragment float4 quad_fs(QuadOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                        sampler s [[sampler(0)]], constant float4* r [[buffer(0)]]) {
  return tex.sample(s, in.uv, level(r[2].x));
}
// quad_fs through output_curve (output_curve.h): r[2].y its mode, r[2].z
// its exponent.
fragment float4 quad_curve_fs(QuadOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                              sampler s [[sampler(0)]], constant float4* r [[buffer(0)]]) {
  const float4 c = tex.sample(s, in.uv, level(r[2].x));
  return float4(output_curve(c.rgb, r[2].y, r[2].z), c.a);
}
// A triangle over the whole target; uv 0..1 from the top left.
struct TriOut { float4 pos [[position]]; float2 uv; };
vertex TriOut tri_vs(uint vid [[vertex_id]]) {
  float2 t = float2((vid << 1) & 2, vid & 2);
  TriOut o;
  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  o.uv = float2(t.x, 1.0 - t.y);
  return o;
}
// A depth buffer copied to one of another size (nearest).
struct DepthOut { float d [[depth(any)]]; };
fragment DepthOut depth_copy_fs(TriOut in [[stage_in]], depth2d<float> src [[texture(0)]],
                                sampler s [[sampler(0)]]) {
  DepthOut o;
  o.d = src.sample(s, in.uv);
  return o;
}
// Motion from the camera alone: each pixel's point (from its depth) where
// the last frame's camera saw it, in texture coordinates (previous - now).
// m: this frame's clip space (the depth as stored) to the last frame's.
// The depth is copied in the same pass (MetalFX's depth input: the
// scene's own may have stencil).
struct MotionOut { float2 mv [[color(0)]]; float d [[depth(any)]]; };
fragment MotionOut motion_fs(TriOut in [[stage_in]], depth2d<float> depth [[texture(0)]],
                             constant float4x4& m [[buffer(0)]]) {
  float d = depth.read(uint2(in.pos.xy));
  float4 prev = m * float4(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, d, 1.0);
  float2 p = prev.xy / prev.w;
  MotionOut o;
  o.mv = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5) - in.uv;
  o.d = d;
  return o;
}

// Temporal antialiasing (RECOMP_UPSCALER=taa), one pass at the back
// buffer's size. The history is reprojected by the camera (motion from the
// nearest depth around the pixel, so edges move with their foreground),
// read through a Catmull-Rom filter (five bilinear taps, it stays sharp
// when it moves), and clipped toward the mean of this frame's 3x3
// neighbourhood in YCoCg (variance clipping: what moves on its own, cars,
// people, rain, swaying trees, has no motion here and must not ghost).
// The more the content moves, the more of this frame goes in. This
// frame's colour is reconstructed at the pixel's centre from its 3x3
// neighbourhood (each sample where the jitter put it, Blackman-Harris
// weights as Karis's UE4 TAA), not taken where the jitter left it: that
// offset, accumulated, is what blurs a plain TAA.
struct TaaParams {
  float4x4 m;      // this frame's clip space to the last frame's
  float2 inv_size;
  float2 jitter;   // the scene's shift this frame, pixels (x right, y down)
  float w_min, w_max;  // this frame's weight, still and moving
  float w_per_px;      // the weight's rise per pixel of motion
  float gamma;         // the clip box's half size, in standard deviations
  uint reset;
  uint direct;     // write the result to `out` too (no sharpening after)
  uint reversed;   // depth stored as 1 - d (m takes it so): the nearest is the largest
};
half3 rgb_to_ycocg(half3 c) {
  return half3(0.25h * (c.r + 2.0h * c.g + c.b), 0.5h * (c.r - c.b), 0.25h * (2.0h * c.g - c.r - c.b));
}
half3 ycocg_to_rgb(half3 c) { return half3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
float3 taa_history(texture2d<half> h, sampler s, float2 uv, float2 size, float2 inv_size) {
  const float2 t = floor(uv * size - 0.5) + 0.5, f = uv * size - t;
  const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  const float2 w12 = w1 + w2;
  const float2 p0 = (t - 1.0) * inv_size, p3 = (t + 2.0) * inv_size, p12 = (t + w2 / w12) * inv_size;
  const float a = w12.x * w0.y, b = w0.x * w12.y, c = w12.x * w12.y, d = w3.x * w12.y, e = w12.x * w3.y;
  const float3 r = float3(h.sample(s, float2(p12.x, p0.y)).rgb) * a + float3(h.sample(s, float2(p0.x, p12.y)).rgb) * b +
                   float3(h.sample(s, p12).rgb) * c + float3(h.sample(s, float2(p3.x, p12.y)).rgb) * d +
                   float3(h.sample(s, float2(p12.x, p3.y)).rgb) * e;
  return max(r / (a + b + c + d + e), 0.0);
}
kernel void taa_cs(texture2d<half, access::read> color [[texture(0)]],
                   depth2d<float, access::read> depth [[texture(1)]],
                   texture2d<half> history [[texture(2)]],
                   texture2d<half, access::write> history_out [[texture(3)]],
                   texture2d<half, access::write> out [[texture(4)]],
                   constant TaaParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const uint2 size = uint2(color.get_width(), color.get_height());
  if (gid.x >= size.x || gid.y >= size.y) return;
  const int2 last = int2(size) - 1;
  half3 mean = 0.0h, sq = 0.0h;
  float3 filtered = 0.0;
  float wsum = 0.0;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
      const half3 c = rgb_to_ycocg(color.read(uint2(clamp(int2(gid) + int2(x, y), 0, last))).rgb);
      // The neighbour's sample lies at its centre less the jitter.
      const float2 d = float2(x, y) - p.jitter;
      const float w = exp(-2.29 * dot(d, d));
      filtered += float3(c) * w;
      wsum += w;
      mean += c;
      sq += c * c;
    }
  const half3 cur = half3(filtered / wsum);
  half3 res = cur;
  if (p.reset == 0) {
    // The nearest of the pixel and its diagonal neighbours.
    int2 q = int2(gid);
    float d = depth.read(gid);
    for (int k = 0; k < 4; k++) {
      const int2 n = clamp(int2(gid) + int2(k & 1 ? 1 : -1, k & 2 ? 1 : -1), 0, last);
      const float dn = depth.read(uint2(n));
      if (p.reversed ? dn > d : dn < d) { d = dn; q = n; }
    }
    const float2 uq = (float2(q) + 0.5) * p.inv_size;
    const float4 prev = p.m * float4(uq.x * 2.0 - 1.0, 1.0 - uq.y * 2.0, d, 1.0);
    const float2 mv = float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5) - uq;
    const float2 uv = (float2(gid) + 0.5) * p.inv_size + mv;
    if (prev.w > 0.0 && all(uv >= 0.0) && all(uv <= 1.0)) {
      constexpr sampler s(filter::linear, address::clamp_to_edge);
      mean /= 9.0h;
      const half3 sigma = sqrt(max(sq / 9.0h - mean * mean, 0.0h)) * half(p.gamma) + 1.0e-3h;
      half3 h = rgb_to_ycocg(half3(taa_history(history, s, uv, float2(size), p.inv_size)));
      const half3 off = h - mean, ext = abs(off) / sigma;
      h = mean + off / max(1.0h, max(ext.x, max(ext.y, ext.z)));
      const float moved = length(mv * float2(size));
      res = mix(h, cur, half(mix(p.w_min, p.w_max, saturate(moved * p.w_per_px))));
    }
  }
  const half4 o = half4(max(ycocg_to_rgb(res), 0.0h), 1.0h);
  history_out.write(o, gid);
  if (p.direct != 0) out.write(o, gid);
}
// The volumetric fog (gfx/rt_fog.h) over the finished scene, a draw
// blended into the back buffer (the colour there times the transmittance up
// to the pixel's depth, plus the light scattered on the way: the
// integrated texture's texel at the pixel's slice; the sky's: the last
// slice's). rgb the light, a the transmittance (blend: one, source alpha).
struct FogOver {
  float near_plane, far_plane;  // the camera's
  uint reversed, pad;           // depth stored as 1 - d
  float near_z, far_z, slices, pad1;  // the fog's slices (RtFogLook)
};
fragment float4 fog_over_fs(TriOut in [[stage_in]], depth2d<float> depth [[texture(0)]],
                            texture3d<half> vol [[texture(1)]], constant FogOver& fg [[buffer(0)]]) {
  constexpr sampler ns(filter::nearest, address::clamp_to_edge);
  constexpr sampler ls(filter::linear, address::clamp_to_edge);
  const float d = depth.sample(ns, in.uv);
  const float ds = fg.reversed != 0u ? 1.0 - d : d;
  float w = 1.0 - 0.5 / fg.slices;
  if (ds < 0.99999) {  // (not the sky)
    const float n = fg.near_plane, f = fg.far_plane;
    const float z = n * f / (f - ds * (f - n));
    const float sl = log(max(z, 1e-3) / fg.near_z) / log(fg.far_z / fg.near_z) * fg.slices;
    if (!(sl > 0.0)) return float4(0.0, 0.0, 0.0, 1.0);
    w = clamp((sl - 0.5) / fg.slices, 0.5 / fg.slices, 1.0 - 0.5 / fg.slices);
  }
  const float4 v = float4(vol.sample(ls, float3(in.uv, w)));
  return float4(v.rgb, v.a);
}
)";

constexpr uint32_t kClearVs = 0x80000001, kClearPs = 0x80000002;
constexpr uint32_t kQuadVs = 0x80000003, kQuadPs = 0x80000004;
constexpr uint32_t kQuadCurvePs = 0x80000005;

class MetalDevice final : public Device {
 public:
  explicit MetalDevice(SDL_Window* w) : window_(w) {}

  bool Init() {
    @autoreleasepool {
      device_made_s_ = ProcessSeconds();
      dev_ = MTLCreateSystemDefaultDevice();
      if (!dev_) {
        WarnF("gfx: no Metal device\n");
        return false;
      }
      queue_ = [dev_ newCommandQueue];
      view_ = SDL_Metal_CreateView(window_);
      if (!view_) {
        WarnF("gfx: SDL_Metal_CreateView: %s\n", SDL_GetError());
        return false;
      }
      layer_ = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(view_);
      layer_.device = dev_;
      layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
      layer_.framebufferOnly = YES;
#if TARGET_OS_IOS || TARGET_OS_TV
      // The launcher may have initialized Metal for cache inspection
      // before its final options were applied. Use the documented layer
      // control so the overlay follows the choice made before Play.
      if (const char* hud = std::getenv("RECOMP_METAL_HUD")) {
        layer_.developerHUDProperties = @{@"mode": *hud == '1' ? @"main" : @"disabled"};
        WarnF("gfx: Metal HUD %s for the game layer\n", *hud == '1' ? "on" : "off");
      }
#endif
      // RECOMP_DRAWABLES=2|3: the layer's queue of drawables (3 by
      // default). With vsync, every queued drawable is a refresh of input
      // lag; 2 keeps it short.
      if (const char* e = std::getenv("RECOMP_DRAWABLES"))
        layer_.maximumDrawableCount = std::clamp(std::atoi(e), 2, 3);
      else
        layer_.maximumDrawableCount = 3;
      // Constant buffers' offset alignment: 256 bytes on Mac GPUs, less on
      // Apple's.
      uniform_align_ = [dev_ supportsFamily:MTLGPUFamilyApple1] ? 16 : 256;
#if TARGET_OS_IOS || TARGET_OS_TV
      if (@available(iOS 16.4, tvOS 16.4, *)) bc_ = dev_.supportsBCTextureCompression;
      if (!bc_) WarnF("gfx: no BC sampling on this GPU: DXT decoded on upload\n");
#endif
      // RECOMP_DISPLAY_LINK=1 (macOS 14+); off by default: with variable
      // refresh, frames presented as soon as they are done (nextDrawable)
      // played smoother than snapped to the link's refreshes, which showed
      // as constant spikes at 70-100 fps. Not for hidden windows.
      {
        const char* e = std::getenv("RECOMP_DISPLAY_LINK");
        const char* hidden = std::getenv("GTAIV_HIDDEN");
        if (e && *e == '1' && !(hidden && *hidden == '1')) StartDisplayLink();
      }
      // RECOMP_FRAMES_IN_FLIGHT=1..kSlots: how many frames the CPU may
      // record ahead of the GPU (kSlots by default). Fewer: less latency
      // between input and picture, more: more overlap.
      const char* pp = std::getenv("RECOMP_PACE_PRESENT");
      pace_present_ = pp && *pp == '1';
      int in_flight = kSlots;
      if (const char* e = std::getenv("RECOMP_FRAMES_IN_FLIGHT"))
        in_flight = std::clamp(std::atoi(e), 1, int(kSlots));
      frames_free_ = dispatch_semaphore_create(in_flight);
      // One arena chunk per frame slot, made ahead: a slot's first frame
      // takes one, and EndFrame makes another for a slot that fills past 3/4
      // (a chunk a device makes is resident for its lifetime).
      MakeSpareChunks(kSlots);

      NSError* err = nil;
      id<MTLLibrary> lib = [dev_ newLibraryWithSource:@((std::string(kOutputCurveMsl) + kInternalMsl + kRcasMsl).c_str())
                                                options:nil
                                                  error:&err];
      if (!lib) FatalF("gfx: the device's own shaders: %s", err.localizedDescription.UTF8String);
      internal_lib_ = lib;
      SetUpAheadLanes();
      LoadRecipes();
      LoadMslCache();
      SetUpShaderLanes();
      SetUpArchive();
      MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
      dd.depthCompareFunction = MTLCompareFunctionAlways;
      dd.depthWriteEnabled = YES;
      depth_copy_ds_ = [dev_ newDepthStencilStateWithDescriptor:dd];
      // RECOMP_UPSCALER=metalfx: MetalFX's temporal antialiasing of the
      // scene (before the HUD); metalfx-spatial: SMAA at the internal
      // resolution, then MetalFX's spatial upscale (no history, a fraction
      // of the temporal one's cost); taa: this device's own (taa_cs), at the
      // back buffer's size only. RECOMP_RENDER_SCALE (0.25-1, default 1)
      // draws the scene smaller and upscales it (metalfx, metalfx-spatial
      // or bilinear).
      if (const char* u = std::getenv("RECOMP_UPSCALER")) {
        if (!std::strcmp(u, "metalfx")) upscaler_ = Upscaler::MetalFX;
        else if (!std::strcmp(u, "taa")) upscaler_ = Upscaler::Taa;
        else if (!std::strcmp(u, "smaa")) upscaler_ = Upscaler::Smaa;
        else if (!std::strcmp(u, "bilinear")) upscaler_ = Upscaler::Bilinear;
        else if (!std::strcmp(u, "metalfx-spatial")) upscaler_ = Upscaler::MetalFXSpatial;
      }
      // RECOMP_SHARPNESS=0..1 (default 0.5): sharpening (RCAS, rcas.h)
      // after TAA and after either MetalFX upscale, before the HUD (0:
      // none). RECOMP_TAA_SHARPNESS, its older name, if it is not set: its
      // values (once for a different filter) read on the same scale.
      const char* sh = std::getenv("RECOMP_SHARPNESS");
      if (!sh) sh = std::getenv("RECOMP_TAA_SHARPNESS");
      if (sh) sharpness_ = std::clamp(float(std::atof(sh)), 0.0f, 1.0f);
      // RECOMP_OUTPUT_CURVE=bt709|<power>: the curve the finished frame,
      // the HUD with it, goes through to the window (output_curve.h).
      output_curve_ = ParseOutputCurve(std::getenv("RECOMP_OUTPUT_CURVE"));
      if (output_curve_.kind == OutputCurve::kBt709)
        std::fprintf(stderr, "gfx: output curve: sRGB decoded, BT.709 encoded\n");
      else if (output_curve_.kind == OutputCurve::kPower)
        std::fprintf(stderr, "gfx: output curve: power %.2f\n", output_curve_.exponent);
      // RECOMP_MFX_SIGNS=jx,jy,mv: the jitter's and motion's signs as given
      // to MetalFX (for checking conventions).
      if (const char* g = std::getenv("RECOMP_MFX_SIGNS"))
        std::sscanf(g, "%f,%f,%f", &jitter_sign_[0], &jitter_sign_[1], &mv_sign_);
      // RECOMP_RT_BOOST=WxH:k,...: render targets (and depth buffers) of
      // those sizes made k times larger in each direction, but for those a
      // port creates as exact (gfx::BeginExactTargets, target_size.h).
      if (const char* b = std::getenv("RECOMP_RT_BOOST")) {
        for (const char* q = b; *q;) {
          RtBoost rb{};
          if (std::sscanf(q, "%dx%d:%d", &rb.w, &rb.h, &rb.k) == 3 && rb.w > 0 && rb.h > 0 && rb.k > 1)
            rt_boost_.push_back(rb);
          const char* comma = std::strchr(q, ',');
          if (!comma) break;
          q = comma + 1;
        }
        for (const RtBoost& rb : rt_boost_)
          std::fprintf(stderr, "gfx: %dx%d render targets made %dx%d\n", rb.w, rb.h, rb.w * rb.k, rb.h * rb.k);
      }
      if (const char* r = std::getenv("RECOMP_RENDER_SCALE"))
        render_scale_ = std::clamp(float(std::atof(r)), 0.25f, 1.0f);
      if (upscaler_ == Upscaler::Smaa || upscaler_ == Upscaler::Taa) render_scale_ = 1.0f;  // antialiasing only
      if (upscaler_ != Upscaler::Off)
        std::fprintf(stderr, "gfx: scene at %.2f of the back buffer, then %s\n", render_scale_,
                     upscaler_ == Upscaler::MetalFX ? "MetalFX (temporal)"
                     : upscaler_ == Upscaler::MetalFXSpatial ? "SMAA and MetalFX (spatial)"
                     : upscaler_ == Upscaler::Taa   ? "TAA"
                     : upscaler_ == Upscaler::Smaa  ? "SMAA 1x"
                                                    : "bilinear filtering");
      for (auto [sid, name] : {std::pair{kClearVs, "clear_vs"}, {kClearPs, "clear_fs"},
                               {kQuadVs, "quad_vs"}, {kQuadPs, "quad_fs"},
                               {kQuadCurvePs, "quad_curve_fs"}}) {
        Shader s;
        s.fn = [lib newFunctionWithName:@(name)];
        shaders_[sid] = s;
        enc_shaders_[sid] = s.fn;
      }
      for (int linear = 0; linear < 2; linear++) {
        MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter =
            linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        quad_sampler_[linear] = [dev_ newSamplerStateWithDescriptor:sd];
      }

      // Shaders' samplers without a texture read (0, 0, 0, 1), as in D3D9.
      auto dummy = [&](MTLTextureType type) {
        const uint32_t black = 0xFF000000u;
        MTLTextureDescriptor* td = [MTLTextureDescriptor new];
        td.textureType = type;
        td.pixelFormat = MTLPixelFormatBGRA8Unorm;
        td.width = td.height = td.depth = 1;
        td.storageMode = MTLStorageModeShared;
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> t = [dev_ newTextureWithDescriptor:td];
        for (int s = 0; s < (type == MTLTextureTypeCube ? 6 : 1); s++)
          [t replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
               mipmapLevel:0
                     slice:NSUInteger(s)
                 withBytes:&black
               bytesPerRow:4
             bytesPerImage:4];
        return t;
      };
      dummy_2d_ = dummy(MTLTextureType2D);
      dummy_cube_ = dummy(MTLTextureTypeCube);
      dummy_3d_ = dummy(MTLTextureType3D);
      // D3D9's defaults (the front ends send only what changes).
      rs_.fill(0);
      rs_[8] = 3;
      rs_[14] = 1;
      rs_[19] = 2;
      rs_[20] = 1;
      rs_[22] = 3;
      rs_[23] = 4;
      rs_[25] = 8;
      rs_[53] = rs_[54] = rs_[55] = 1;
      rs_[56] = 8;
      rs_[58] = rs_[59] = 0xFFFFFFFFu;
      rs_[168] = 0xF;
      rs_[171] = 1;
      // RECOMP_METAL_SYNC=1: no encoding thread (for comparisons).
      // RECOMP_METAL_LABELS=1: render passes named in GPU traces.
      const char* labels = std::getenv("RECOMP_METAL_LABELS");
      labels_ = labels && *labels == '1';
      // RECOMP_GPU_PASSES=1: GPU time of every render pass, from Metal's
      // stage-boundary timestamps, reported every 120 frames. The frame log
      // (RECOMP_FRAME_LOG, frame_log.h) writes the same times a frame, so
      // with it they are sampled without the variable too; the report
      // still comes only where it was asked for.
      const char* gp = std::getenv("RECOMP_GPU_PASSES");
      gp_report_ = gp && *gp == '1';
      if (gp_report_ || frame_log_) SetUpPassTimes();
      if (frame_log_) {
        // (Its g, s and sg lines are on the host's clock: where that is on
        // the log's.)
        const double host_s = CACurrentMediaTime();
        FrameLogLine("# host clock %.6f at t_s %.6f", host_s, FrameLogSeconds());
        if (!gp_on_) FrameLogLine("# no q lines: this GPU gives no stage-boundary timestamps");
      }
      // RECOMP_PASS_LOADS=0: every render pass attaches the depth buffer
      // bound, loaded and stored, as before (BareDraw). =2, for measuring
      // only: also left out of passes on the game's textures (BareTargets).
      const char* pass_loads = std::getenv("RECOMP_PASS_LOADS");
      pass_loads_ = !pass_loads ? 1 : *pass_loads == '0' ? 0 : *pass_loads == '2' ? 2 : 1;
      if (pass_loads_ != 1)
        std::fprintf(stderr, "gfx: passes without their unused depth buffer %s\n",
                     pass_loads_ ? "on the game's textures too (RECOMP_PASS_LOADS=2)" : "off");
      const char* sync = std::getenv("RECOMP_METAL_SYNC");
      sync_ = sync && *sync && *sync != '0';
      // RECOMP_REVERSED_Z=0: float depth buffers hold D3D9's values, not
      // 1 - d (reversed_z.h).
      const char* rz = std::getenv("RECOMP_REVERSED_Z");
      reversed_z_ = !(rz && *rz == '0');
      if (!reversed_z_) std::fprintf(stderr, "gfx: reversed depth off\n");
      // RECOMP_DEPTH_FROM_W=0: perspective draws keep z (reversed_z.h).
      const char* dw = std::getenv("RECOMP_DEPTH_FROM_W");
      depth_from_w_ = reversed_z_ && !(dw && *dw == '0');
      if (reversed_z_ && !depth_from_w_) std::fprintf(stderr, "gfx: depth from w off\n");
      if (!sync_) encoder_ = std::thread([this] { EncoderMain(); });
      // A game: no App Nap, timers not coalesced. Without it, in a window
      // not in front, the benchmark street had twice the hitches.
      activity_ = [NSProcessInfo.processInfo
          beginActivityWithOptions:NSActivityUserInitiated | NSActivityLatencyCritical
                            reason:@"game"];
      std::fprintf(stderr, "gfx: Metal (%s)\n", dev_.name.UTF8String);
    }
    return true;
  }

  ~MetalDevice() override {
    if (!dev_) return;
    NoteRtLightDevice(this, false);
    if (shadow_masker_) NoteRayTracedShadows(false);
    if (reflector_) NoteRayTracedReflections(false);
    if (rt_forward_) {
      NoteRayTracedForwardReflections(false);
      NoteRayTracedForwardShadows(false);
    }
    // Lean shader and pipeline prebuild blocks, and the shader lanes',
    // capture this device's state: they are finished below, before its
    // maps, mutexes and Metal objects are destroyed. Those still queued see
    // closing_ (the lanes' a closed shader_jobs_) and do nothing, so only
    // the one running on each queue is waited for (quitting while loading,
    // hundreds are queued).
    closing_ = true;
    shader_jobs_.Close();
    // No more pipelines kept for the archive, its saves stopped but for the
    // last one below (on iOS and tvOS they run on AheadQueue, drained
    // below).
    if (archive_) archive_->BeginClose();
    // Each step that takes long is told, to find what holds a quit up.
    uint64_t t = NowNs();
    auto step = [&t](const char* what) {
      const uint64_t now = NowNs();
      if (now - t > 500000000ull) std::fprintf(stderr, "gfx: closing: %s took %.1f s\n", what, double(now - t) / 1e9);
      t = now;
    };
    std::lock_guard<std::recursive_mutex> l(mu_);
    step("waiting for the device's lock");
    if (frame_open_) EndFrame();
    WaitIdle();
    step("the last frame");
    {
      std::lock_guard<std::mutex> q(q_mu_);
      stop_ = true;
    }
    q_cv_.notify_all();
    if (encoder_.joinable()) encoder_.join();
    step("the encoding thread");
    if (last_cb_) [last_cb_ waitUntilCompleted];
    step("the GPU");
    // Twice: work queued again at background priority while the first pass
    // ran (QueueAhead) is done by the second.
    for (int pass = 0; pass < 2; pass++)
      for (dispatch_queue_t q : ahead_lanes_) dispatch_sync(q, ^{});
    dispatch_sync(NowQueue(), ^{});
    step("the build ahead");
    // The shader lanes likewise: a compile running is waited for, those
    // queued do nothing (shader_jobs_ closed).
    for (dispatch_queue_t q : lanes_) dispatch_sync(q, ^{});
    step("the shader lanes");
    // What was added to the pipeline archive saved, waited for 2 s at most.
    if (archive_) archive_->Close(2.0);
    step("the pipeline archive");
    StopDisplayLink();
    if (view_) SDL_Metal_DestroyView(view_);
  }

  bool HasFixedFunction() const override { return false; }

  // ---- memory shared with the GPU

  bool SharesMemory() const override { return true; }

  bool MapSharedMemory(void* host, size_t size) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (shared_buf_) return false;  // one region
    shared_buf_ = [dev_ newBufferWithBytesNoCopy:host
                                          length:size
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
    if (!shared_buf_) {
      WarnF("gfx: cannot map %zu bytes of shared memory\n", size);
      return false;
    }
    shared_lo_ = static_cast<const uint8_t*>(host);
    shared_hi_ = shared_lo_ + size;
    return true;
  }

  // Without mu_: front ends ask for every buffer a draw reads.
  uint64_t FrameSerial() const override { return draw_serial_.load(std::memory_order_acquire); }
  uint64_t CompletedSerial() const override { return completed_.load(std::memory_order_acquire); }

  bool TakePresentLatency(double* median, double* p99) override {
    std::vector<double> d;
    {
      std::lock_guard<std::mutex> l(shown_mu_);
      d.swap(latency_);
    }
    if (d.empty()) return false;
    std::sort(d.begin(), d.end());
    *median = d[d.size() / 2];
    *p99 = d[size_t(0.99 * double(d.size() - 1) + 0.5)];
    return true;
  }

  bool TakeShownIntervals(double* median, double* p99, double* max) override {
    std::vector<double> d;
    {
      std::lock_guard<std::mutex> l(shown_mu_);
      for (size_t i = 1; i < shown_.size(); i++)
        if (shown_[i] > shown_[i - 1]) d.push_back((shown_[i] - shown_[i - 1]) * 1e3);
      if (!shown_.empty()) shown_.erase(shown_.begin(), shown_.end() - 1);
    }
    if (d.empty()) return false;
    std::sort(d.begin(), d.end());
    *median = d[d.size() / 2];
    *p99 = d[size_t(0.99 * double(d.size() - 1) + 0.5)];
    *max = d.back();
    return true;
  }

  void EnablePerfStats() override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    perf_t0_ = NowNs();
    perf_ = true;
  }

  std::string TakePerfStats() override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (!perf_ || !pc_.frames) return {};
    const PerfCounts c = pc_;
    pc_ = {};
    const uint64_t now = NowNs();
    const double wall = double(now - perf_t0_);
    perf_t0_ = now;
    const double f = double(c.frames);
    const double busy = double(enc_busy_ns_.exchange(0));
    const uint64_t pipes = pipe_compiles_.exchange(0), pipe_ns = pipe_ns_.exchange(0),
                   pipe_max = pipe_max_ns_.exchange(0), prebuilt = pipe_prebuilt_.exchange(0);
    const uint64_t ahead_done = ahead_done_.load(std::memory_order_relaxed);
    const uint64_t ahead_queued = ahead_queued_.load(std::memory_order_relaxed);
    char b[1024];
    int len = std::snprintf(b, sizeof b,
                  "per frame %.0f draws (%.0f of them depth from w, %.0f not for another projection; %.2f ms "
                  "recording them, %.2f us each), "
                  "%.0f recorded ops, "
                  "%.1f render passes, %.1f uploads ahead + %.1f in order (%.2f MB, %.2f MB in "
                  "order); encoder busy %.0f%% (%.2f ms a frame); in %.0f frames: %llu textures + "
                  "%llu depth buffers created, %llu arena buffers (%.0f MB) + %llu made ahead, %llu shaders (%.1f ms, "
                  "%llu MSL compiles), %llu pipelines compiled while drawing (%.1f ms, max %.1f), "
                  "%llu built ahead used; of the shaders' time %.1f ms translating, %.1f ms compiling MSL; "
                  "%llu waiting to be built ahead; shaders %llu compiled when first drawn, %llu waited for on a "
                  "lane (%.1f ms), %llu compiled on the lanes (%.1f ms there), %zu translations not compiled yet",
                  double(c.draws) / f, double(c.w_draws) / f, double(c.w_other) / f, double(c.draw_ns) / f / 1e6,
                  c.draws ? double(c.draw_ns) / double(c.draws) / 1e3 : 0.0, double(c.ops) / f,
                  double(c.passes) / f, double(c.uploads_ahead) / f, double(c.uploads_in_order) / f,
                  double(c.upload_bytes) / f / 1048576.0, double(c.in_order_bytes) / f / 1048576.0,
                  wall > 0 ? 100.0 * busy / wall : 0.0, busy / f / 1e6, f,
                  (unsigned long long)c.textures, (unsigned long long)c.depths,
                  (unsigned long long)c.arena_buffers, double(c.arena_bytes) / 1048576.0,
                  (unsigned long long)c.arena_ahead,
                  (unsigned long long)c.shaders, double(c.shader_ns) / 1e6,
                  (unsigned long long)c.msl_compiles, (unsigned long long)pipes, double(pipe_ns) / 1e6,
                  double(pipe_max) / 1e6, (unsigned long long)prebuilt, double(c.translate_ns) / 1e6,
                  double(c.library_ns) / 1e6,
                  (unsigned long long)(ahead_queued > ahead_done ? ahead_queued - ahead_done : 0),
                  (unsigned long long)c.first_drawn, (unsigned long long)c.joins, double(c.join_ns) / 1e6,
                  (unsigned long long)c.ahead_compiles, double(c.ahead_ns) / 1e6, shader_jobs_.Pending());
    std::string out(b, size_t(std::clamp(len, 0, int(sizeof b) - 1)));
    {
      // Where the render thread waits and what the ray scene costs it, per
      // frame: blocked on a frame slot (BeginFrame), noting draws for the
      // scene and recording their captures (1 draw in 32 timed, in mach
      // ticks, scaled by the calls), its snapshots and its frame's end; and
      // the encoding thread's wait for a drawable (Show; part of its busy
      // time above).
      mach_timebase_info_data_t tb{};
      mach_timebase_info(&tb);
      const double tick_ns = tb.denom ? double(tb.numer) / double(tb.denom) : 1.0;
      const double note_ns =
          c.rt_note_timed ? double(c.rt_note_ticks) * tick_ns * double(c.rt_note_calls) / double(c.rt_note_timed) : 0.0;
      const double capture_ns = c.rt_capture_timed ? double(c.rt_capture_ticks) * tick_ns *
                                                         double(c.rt_capture_calls) / double(c.rt_capture_timed)
                                                   : 0.0;
      const double show_wait = double(show_wait_ns_.exchange(0));
      std::snprintf(b, sizeof b,
                    "; render thread per frame: %.2f ms waiting for a frame slot, %.2f ms noting %.0f draws for the "
                    "ray scene, %.2f ms recording %.0f captures (1 draw in 32 timed), %.2f ms in its snapshots, "
                    "%.2f ms ending its frame; encoding thread: %.2f ms waiting for a drawable",
                    double(c.slot_wait_ns) / f / 1e6, note_ns / f / 1e6, double(c.rt_note_calls) / f,
                    capture_ns / f / 1e6, double(c.rt_capture_calls) / f, double(c.snapshot_ns) / f / 1e6,
                    double(c.rt_end_ns) / f / 1e6, show_wait / f / 1e6);
      out += b;
      // RECOMP_PASS_LOADS (BareDraw): passes with a depth buffer bound, those
      // opened without it, the ones of those ended because it was needed
      // after all, and those that had it attached and never used it (seen
      // fewer than kBareAfter times, on a texture of the game's, not opened
      // by a draw, or the switch off). And the passes whose first draw may
      // cover the target (ColorLoadCandidate): counted only, their colour
      // is loaded.
      const char* loads = pass_loads_ == 1 ? "" : pass_loads_ ? " (RECOMP_PASS_LOADS=2)" : " (RECOMP_PASS_LOADS=0)";
      std::snprintf(b, sizeof b,
                    "; render passes per frame with a depth buffer bound %.1f: %.2f opened without it%s, %.2f of "
                    "those ended for a draw or clear that needed it, %.2f more had it and never used it; %.2f passes "
                    "not cleared whose first draw may cover the target unblended (colour loaded all the same)",
                    double(c.depth_passes) / f, double(c.bare_passes) / f, loads, double(c.bare_splits) / f,
                    double(c.idle_passes) / f, double(c.color_candidates) / f);
      out += b;
    }
    if (archive_) {
      // Lookups in the pipeline archive (found or not, and the time the
      // misses cost), new pipelines saved in all and still to be saved, and
      // its last save.
      const PipelineArchive::Stats a = archive_->TakeStats();
      std::snprintf(b, sizeof b,
                    "; pipeline archive: %zu shards, %llu found, %llu missed (%.1f ms), %llu added, %llu to add, "
                    "last saved in %.0f ms (%.1f MB), %.0f ms of it adding",
                    a.shards, (unsigned long long)a.hits, (unsigned long long)a.misses, double(a.miss_ns) / 1e6,
                    (unsigned long long)a.saved, (unsigned long long)a.pending, a.save_ms,
                    double(a.save_bytes) / 1048576.0, a.add_ms);
      out += b;
    }
    if (Temporal()) {
      std::snprintf(b, sizeof b,
                    "; camera: %llu frames from two draws that agree, %llu from one unconfirmed, %llu none; %llu "
                    "candidate draws rejected",
                    (unsigned long long)c.camera_confirmed, (unsigned long long)c.camera_unconfirmed,
                    (unsigned long long)c.camera_none, (unsigned long long)c.camera_rejected);
      out += b;
    }
    if (rt_) {
      out += "; " + rt_->TakeStats();
      frame_log_rt_seen_ = {};  // (the scene's counters start from 0 again: FrameLogPresent's deltas)
      std::snprintf(b, sizeof b,
                    "; local lights per frame %.2f lists, %.1f lights given, %.1f kept (%.1f point, %.1f spot)",
                    double(c.rtl_lists) / f, double(c.rtl_given) / f, double(c.rtl_kept) / f, double(c.rtl_point) / f,
                    double(c.rtl_spot) / f);
      out += b;
    }
    if (shadow_masker_) {
      std::snprintf(b, sizeof b,
                    "; ray-traced shadows per frame %.2f masks (%.0f pixels, %.0f rays, %.0f of them blocked, %.0f "
                    "traced again next to a captured mesh; %.0f texels decided by the game's shadow map: lit but "
                    "for a short ray, or in a building's shadow)",
                    double(c.shadow_masks) / f, double(c.shadow_pixels) / f, double(c.shadow_rays) / f,
                    double(c.shadow_hits) / f, double(c.shadow_retraced) / f, double(c.shadow_cascade_lit) / f);
      out += b;
      if (c.cascade_size[0]) {
        std::snprintf(b, sizeof b, " (its map %ux%u format %u, range %.4g, texels %.3g %.3g %.3g %.3g)",
                      c.cascade_size[0], c.cascade_size[1], c.cascade_format, double(c.cascade_range),
                      double(c.cascade_texel[0]), double(c.cascade_texel[1]), double(c.cascade_texel[2]),
                      double(c.cascade_texel[3]));
        out += b;
      }
      if (c.shadow_colour_seen) {
        std::snprintf(b, sizeof b, " (the light's colour (%.3g %.3g %.3g) times %.3g; %.2f masks dim)",
                      double(c.shadow_colour[0]), double(c.shadow_colour[1]), double(c.shadow_colour[2]),
                      double(c.shadow_colour[3]), double(c.shadow_dim) / f);
        out += b;
      }
      auto jitter_counts = [&](const uint64_t* n) {
        std::snprintf(b, sizeof b,
                      " (%.2f with the jitter; without: %.2f not temporal, %.2f no camera yet, %.2f from another "
                      "depth than the jittered scene's)",
                      double(n[0]) / f, double(n[1]) / f, double(n[2]) / f, double(n[3]) / f);
        out += b;
      };
      jitter_counts(c.shadow_jitter);
      out += "; all lit:";
      bool any = false;
      for (size_t k = 1; k < size_t(RtShadowWhite::Count); k++) {
        if (!c.shadow_white[k]) continue;
        std::snprintf(b, sizeof b, "%s %.2f %s", any ? "," : "", double(c.shadow_white[k]) / f,
                      RtShadowWhiteName(RtShadowWhite(k)));
        out += b;
        any = true;
      }
      if (!any) out += " none";
      std::snprintf(b, sizeof b,
                    "; local lights' shadows (%llu shaders flagged, %llu of them not traceable) per frame %.2f draws: "
                    "%.2f masks traced (%.0f pixels dispatched, %.0f rays, %.0f of them blocked, %.0f pixels outside "
                    "the lights, %.0f within their lamp gaps), %.2f draws reusing one, %.2f render passes ended for "
                    "them, %.2f lights off screen; masks",
                    (unsigned long long)light_shaders_, (unsigned long long)light_shaders_refused_,
                    double(c.light_draws) / f, double(c.light_masks) / f, double(c.light_pixels) / f,
                    double(c.light_rays) / f, double(c.light_hits) / f, double(c.light_outside) / f,
                    double(c.light_in_gap) / f, double(c.light_reused) / f, double(c.light_splits) / f,
                    double(c.light_offscreen) / f);
      out += b;
      jitter_counts(c.light_jitter);
      out += "; all lit:";
      any = false;
      for (size_t k = 1; k < size_t(RtShadowWhite::Count); k++) {
        if (!c.light_white[k]) continue;
        std::snprintf(b, sizeof b, "%s %.2f %s", any ? "," : "", double(c.light_white[k]) / f,
                      RtShadowWhiteName(RtShadowWhite(k)));
        out += b;
        any = true;
      }
      if (!any) out += " none";
      std::snprintf(b, sizeof b,
                    "; masks traced %.2f of moving lights, %.2f of still ones; %.2f draws of still lights past the "
                    "budget on an earlier frame's mask (at most %llu frames old)",
                    double(c.light_moving) / f, double(c.light_masks - c.light_moving) / f, double(c.light_older) / f,
                    (unsigned long long)c.light_older_age);
      out += b;
      std::snprintf(b, sizeof b, "; %.2f snapshots of the scene per frame", double(c.snapshots) / f);
      out += b;
      std::snprintf(b, sizeof b, "; game shadow-map draws per frame %.2f (shaders still reading their own%s)",
                    double(c.own_map_draws) / f, own_maps_lit_ ? ", bound lit" : "");
      out += b;
    }
    if (reflector_) {
      std::snprintf(b, sizeof b,
                    "; ray-traced reflections per frame %.2f passes (%.0f pixels, %.0f rays, %.0f hits, %.0f misses, "
                    "%.0f on instances without an albedo, %.0f stopped on back faces (%.0f of them captured "
                    "draws'), %.0f back faces of other views' instances lit; mean hit distance %.1f; %.0f pixels "
                    "filled from their neighbours; %.0f rays at the checkerboard's rate; %.0f%% of passes on the "
                    "checkerboard; %.0f from the screen, "
                    "%.0f of them mixed with a ray, %.0f%% of passes in screen space first, %.2f lit scenes kept), "
                    "%.2f draws with none traced (diag passes: %.0f as traced, %.0f no hit shadows, %.0f opaque), "
                    "%.2f snapshots shared with the shadows",
                    double(c.refl_passes) / f, double(c.refl_pixels) / f, double(c.refl_rays) / f,
                    double(c.refl_hits) / f, double(c.refl_misses) / f, double(c.refl_bare) / f,
                    double(c.refl_back) / f, double(c.refl_back_captured) / f, double(c.refl_other_back) / f,
                    c.refl_hits ? c.refl_distance / double(c.refl_hits) : 0.0, double(c.refl_filled) / f,
                    double(c.refl_dense) / f, c.refl_passes ? 100.0 * double(c.refl_checker) / double(c.refl_passes) : 0.0,
                    double(c.refl_screen + c.refl_screen_mixed) / f, double(c.refl_screen_mixed) / f,
                    c.refl_passes ? 100.0 * double(c.refl_screen_passes) / double(c.refl_passes) : 0.0,
                    double(c.refl_kept) / f, double(c.refl_none) / f, double(c.refl_diag[0]),
                    double(c.refl_diag[1]), double(c.refl_diag[2]),
                    double(c.snapshots_shared) / f);
      out += b;
    }
    if (c.bounce_skipped) {
      std::snprintf(b, sizeof b, "; the bounce's light left out on %.0f%% of frames (the far light reached too little)",
                    100.0 * double(c.bounce_skipped) / f);
      out += b;
    }
    if (rt_forward_) {
      std::snprintf(b, sizeof b,
                    "; local lights traced at once per frame %.2f passes (%.2f lights, %.0f rays, %.0f of them "
                    "blocked, %.0f traced again next to a captured mesh, %.0f points reached by more than %u), "
                    "%.2f render passes ended for them; their draws %.2f reading them, %.2f tracing their own",
                    double(c.lset_passes) / f, double(c.lset_lights) / f, double(c.lset_rays) / f,
                    double(c.lset_hits) / f, double(c.lset_again) / f, double(c.lset_capped) / f, lights_cap_,
                    double(c.lset_splits) / f, double(c.lset_draws) / f, double(c.lset_own) / f);
      out += b;
      if (occluder_) {
        std::snprintf(b, sizeof b, "; ambient occlusion per frame %.2f passes (%.0f rays, %.0f of them blocked)",
                      double(c.ao_passes) / f, double(c.ao_rays) / f, double(c.ao_blocked) / f);
        out += b;
      }
      if (ml_) {
        // (The lights binned on the encoding side, since the last report.)
        const uint64_t lights = ml_enc_lights_.exchange(0), dropped = ml_enc_dropped_.exchange(0),
                       entries = ml_enc_entries_.exchange(0);
        std::snprintf(b, sizeof b,
                      "; many lights per frame %.2f passes (%.1f lights on screen, %.0f tile entries, %.0f dropped "
                      "from full tiles, %.0f rays, %.0f of them blocked; %.2f added in a pass of their own, the rest "
                      "in the pass after them), %.2f light volumes' draws left out",
                      double(c.ml_passes) / f, double(lights) / f, double(entries) / f, double(dropped) / f,
                      double(c.ml_rays) / f, double(c.ml_blocked) / f, double(c.ml_add_passes) / f,
                      double(c.ml_skipped) / f);
        out += b;
      }
      std::snprintf(b, sizeof b, "; forward shaders' ray queries (reflections, shadows) per frame %.2f draws traced",
                    double(c.fwd[0]) / f);
      out += b;
      for (size_t k = 1; k < size_t(RtForwardOff::Count); k++) {
        if (!c.fwd[k]) continue;
        std::snprintf(b, sizeof b, ", %.2f not (%s)", double(c.fwd[k]) / f, RtForwardOffName(RtForwardOff(k)));
        out += b;
      }
      std::snprintf(b, sizeof b,
                    ", %llu such shaders without the ray queries, %llu reads in subroutines left without, %llu "
                    "pipelines made without",
                    (unsigned long long)fwd_plain_, (unsigned long long)fwd_stub_calls_,
                    (unsigned long long)fwd_pipe_plain_.load(std::memory_order_relaxed));
      out += b;
    }
    if (denoiser_ || !dn_why_.empty()) {
      if (dn_on_)
        std::snprintf(b, sizeof b,
                      "; MetalFX's denoiser per frame %.2f lit scenes denoised (%.2f history resets, %.0f reactive "
                      "pixels), %.2f lit scenes no post-process draw read, %.2f render passes ended for its guides",
                      double(c.dn_frames) / f, double(c.dn_resets) / f, double(c.dn_reactive) / f,
                      double(c.dn_missed) / f, double(c.dn_splits) / f);
      else
        std::snprintf(b, sizeof b, "; MetalFX's denoiser off: %s", dn_why_.c_str());
      out += b;
    }
    return out;
  }

  double TakeGpuFrameMs() override {
    const uint64_t ns = gpu_ns_.exchange(0), frames = gpu_frames_.exchange(0);
    return frames ? double(ns) / 1e6 / double(frames) : -1;
  }

  // ---- swap chain

  void ResetSwapChain(int w, int h, bool vsync) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    @autoreleasepool {
      Settle();
      pass_ = {};
      Keep(backbuffer_);
      Keep(front_);
      // The temporal antialiasing and MetalFX write their output straight
      // into the back buffer; the two trade places at Present, so alike.
      const MTLTextureUsage extra =
          Temporal() || upscaler_ == Upscaler::MetalFXSpatial ? MTLTextureUsageShaderWrite : 0;
      backbuffer_ = NewTexture(MTLTextureType2D, MTLPixelFormatBGRA8Unorm, w, h, 1, true, false, extra);
      front_ = NewTexture(MTLTextureType2D, MTLPixelFormatBGRA8Unorm, w, h, 1, true, false, extra);
      bb_w_ = w;
      bb_h_ = h;
#if TARGET_OS_OSX
      Rec([this, vsync] { layer_.displaySyncEnabled = vsync; });
#endif
      vsync_ = vsync;
      // Both start black, as a fresh D3D swap chain does.
      for (id<MTLTexture> t : {backbuffer_, front_}) {
        Pass p{};
        p.color[0] = t;
        p.fmt[0] = MTLPixelFormatBGRA8Unorm;
        p.targets = 1;
        p.w = w;
        p.h = h;
        SetPass(p);
        clear_flags_ = 1;
        clear_color_ = MTLClearColorMake(0, 0, 0, 1);
        Settle();
      }
      BindTarget();
      std::fprintf(stderr, "gfx: back buffer %dx%d%s\n", w, h, vsync ? " (vsync)" : "");
    }
  }

  // ---- resources

  TextureId CreateTexture(int w, int h, int levels, Format fmt, bool rt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (w <= 0 || h <= 0) return 0;
    if (levels <= 0) {
      levels = 1;
      for (int s = std::max(w, h); s > 1; s >>= 1) levels++;
    }
    Tex t;
    t.w = w;
    t.h = h;
    t.levels = levels;
    t.fmt = fmt;
    t.rt = rt;
    // X8R8G8B8 is stored as is, alpha read as 1 (uploads copy it untouched).
    t.opaque = !rt && fmt == Format::X8R8G8B8;
    int rw = w, rh = h;
    t.scaled = rt && ScaledSize(w, h, &rw, &rh, &t.scene);
    t.tex = NewTexture(MTLTextureType2D, TexFormat(fmt), rw, rh, levels, rt, t.opaque);
    if (perf_) pc_.textures++;
    if (!t.tex) {
      WarnF("gfx: CreateTexture %dx%d x%d fmt %u failed\n", w, h, levels, unsigned(fmt));
      return 0;
    }
    if (fmt == Format::P8) WarnF("gfx: P8 texture (no palette support)\n");
    TextureId tid = NewTexId();
    AddTex(tid, t);
    if (rt) {
      // Zeros before anything can touch it (metal_new_target.h): cleared
      // in the frame's upload command buffer, which the GPU runs before
      // the frame's own (BeginFrame), so before every draw into it, every
      // read of it and every upload made in order; an upload made ahead of
      // the frame goes into the same buffer, later in it, as recorded.
      // The frame's render pass stays open. These passes are not the
      // frame's: neither counted nor timed with them.
      EnsureFrame();
      __unsafe_unretained id<MTLTexture> fresh = t.tex;  // kept by textures_, then by Keep
      Rec([this, fresh] {
        if (upload_blit_) [upload_blit_ endEncoding];
        upload_blit_ = nil;
        ClearNewTarget(upload_cb_, fresh);
      });
    }
    return tid;
  }

  void DestroyTexture(TextureId tid) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = textures_.find(tid);
    if (it == textures_.end()) return;
    bool bound = current_color_ == tid;
    if (current_color_ == tid) current_color_ = 0, current_level_ = 0;
    for (ColorTarget& e : current_extra_)
      if (e.tex == tid) e = {}, bound = true;
    if (const DepthId z = it->second.depth) {  // a depth texture: its depth entry too
      if (current_depth_ == z) current_depth_ = 0, bound = true;
      depths_.erase(z);
    }
    if (bound) BindTarget();
    // Command buffers keep what they use alive; until recorded work has
    // been encoded, Keep does.
    Keep(it->second.tex);
    tex_at_[tid] = nullptr;
    textures_.erase(it);
    free_tex_ids_.push_back(tid);
  }

  void FillTexture(TextureId tid, const Rect& r, uint32_t argb) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = textures_.find(tid);
    if (it == textures_.end()) return;
    Rect c = Clip(r, it->second.w, it->second.h);
    int w = c.right - c.left, h = c.bottom - c.top;
    if (w <= 0 || h <= 0) return;
    std::vector<uint32_t> px(size_t(w) * h, argb);
    Upload(it->second, 0, 0, c, px.data(), uint32_t(w) * 4, Format::A8R8G8B8);
  }

  void UpdateTexture(TextureId tid, int level, const Rect& r, const void* src,
                     uint32_t pitch) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    Update(tid, 0, level, r, src, pitch);
  }

  TextureId CreateCubeTexture(int size, int levels, Format fmt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (size <= 0) return 0;
    if (levels <= 0) {
      levels = 1;
      for (int s = size; s > 1; s >>= 1) levels++;
    }
    Tex t;
    t.w = t.h = size;
    t.levels = levels;
    t.fmt = fmt;
    t.cube = true;
    t.opaque = fmt == Format::X8R8G8B8;
    t.tex = NewTexture(MTLTextureTypeCube, TexFormat(fmt), size, size, levels, false, t.opaque);
    if (perf_) pc_.textures++;
    if (!t.tex) {
      WarnF("gfx: CreateCubeTexture %d x%d fmt %u failed\n", size, levels, unsigned(fmt));
      return 0;
    }
    TextureId tid = NewTexId();
    AddTex(tid, t);
    return tid;
  }

  void UpdateCubeTexture(TextureId tid, int face, int level, const Rect& r, const void* src,
                         uint32_t pitch) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (face < 0 || face > 5) return;
    Update(tid, face, level, r, src, pitch);
  }

  DepthId CreateDepth(int w, int h, Format fmt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    int rw = w, rh = h;
    bool scene = false;
    const bool scaled = ScaledSize(w, h, &rw, &rh, &scene);
    // Sampled when copied to a full-size depth buffer (DepthProxy), or read
    // by TAA.
    Depth d = NewDepth(rw, rh, fmt, scaled || scene);
    d.scaled = scaled;
    if (!d.tex) return 0;
    if (perf_) pc_.depths++;
    DepthId did = next_id_++;
    depths_[did] = d;
    return did;
  }

  TextureId CreateDepthTexture(int w, int h, DepthId* depth) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    *depth = 0;
    int rw = w, rh = h;
    bool scene = false;
    const bool scaled = ScaledSize(w, h, &rw, &rh, &scene);
    Depth d = NewDepth(rw, rh, Format::D24S8, true);
    if (!d.tex) return 0;
    if (perf_) pc_.depths++;
    d.scaled = scaled;
    Tex t;
    t.scaled = scaled;
    t.scene = scene;
    t.tex = d.tex;
    t.w = w;
    t.h = h;
    t.fmt = Format::D24S8;
    t.depth = next_id_++;
    depths_[t.depth] = d;
    TextureId tid = NewTexId();
    AddTex(tid, t);
    *depth = t.depth;
    return tid;
  }

  void DestroyDepth(DepthId did) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = depths_.find(did);
    if (it == depths_.end()) return;
    if (current_depth_ == did) {
      current_depth_ = 0;
      BindTarget();
    }
    Keep(it->second.tex);
    depths_.erase(it);
  }

  void SetRenderTarget(TextureId color, DepthId depth) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    current_color_ = color;
    current_level_ = 0;
    current_extra_.fill({});
    current_depth_ = depth;
    BindTarget();
  }

  void SetRenderTargets(const ColorTarget color[4], DepthId depth) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    current_color_ = color[0].tex;
    current_level_ = color[0].tex ? color[0].level : 0;
    for (int i = 0; i < 3; i++) current_extra_[i] = color[i + 1];
    current_depth_ = depth;
    BindTarget();
  }

  void StretchRect(ColorTarget src, const Rect& sr, ColorTarget dst, const Rect& dr,
                   bool linear) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    @autoreleasepool {
      int sw, sh, dw, dh;
      Tex* st = nullptr;
      Tex* dt = nullptr;
      id<MTLTexture> s = Target(src, &sw, &sh, &st);
      id<MTLTexture> d = Target(dst, &dw, &dh, &dt);
      if (!s || !d) return;
      // (The many lights' light into their target first: a copy reads it.)
      if (ml_add_.on && (s == ml_add_.target || d == ml_add_.target)) {
        Settle();
        MlAddFlush();
      }
      Rect a = Clip(sr, sw, sh), b = Clip(dr, dw, dh);
      if (Empty(a) || Empty(b)) return;
      // Targets at the internal resolution: rectangles in their pixels.
      auto scale = [](Rect r, id<MTLTexture> t, int level, int gw, int gh) {
        const float fx = float(std::max<NSUInteger>(1, t.width >> level)) / float(gw);
        const float fy = float(std::max<NSUInteger>(1, t.height >> level)) / float(gh);
        if (fx == 1.0f && fy == 1.0f) return r;
        return Rect{int32_t(std::lround(r.left * fx)), int32_t(std::lround(r.top * fy)),
                    int32_t(std::lround(r.right * fx)), int32_t(std::lround(r.bottom * fy))};
      };
      a = scale(a, s, src.tex ? src.level : 0, sw, sh);
      b = scale(b, d, dst.tex ? dst.level : 0, dw, dh);
      if (Empty(a) || Empty(b)) return;
      for (Tex* t : {st, dt})
        if (t) t->used = serial_;
      end_why_ = PassWhy::Blit;
      Settle();
      EnsureFrame();
      const int sl = src.tex ? src.level : 0, dl = dst.tex ? dst.level : 0;
      const bool copy = a.right - a.left == b.right - b.left &&
                        a.bottom - a.top == b.bottom - b.top && s.pixelFormat == d.pixelFormat;
      // A whole copy of the lit scene (before its denoise): the post-
      // processing may read that (DenoiseAt).
      if (dn_on_ && dn_lit_ && !dn_done_ && s == dn_lit_ && copy && dt && sl == 0 && dl == 0 && a.left == 0 &&
          a.top == 0 && NSUInteger(a.right) == s.width && NSUInteger(a.bottom) == s.height && b.left == 0 &&
          b.top == 0 && d.width == s.width && d.height == s.height)
        dn_copy_id_ = dst.tex;
      // Likewise for the reflections' keep (ReflectionKeepAt).
      if (refl_lit_ && !refl_keep_done_ && s == refl_lit_ && copy && dt && sl == 0 && dl == 0 && a.left == 0 &&
          a.top == 0 && NSUInteger(a.right) == s.width && NSUInteger(a.bottom) == s.height && b.left == 0 &&
          b.top == 0 && d.width == s.width && d.height == s.height)
        refl_lit_copy_ = dst.tex;
      __unsafe_unretained id<MTLTexture> su = s, du = d;
      Rec([=, this] {
        if (copy) {
          id<MTLBlitCommandEncoder> bl = TimedBlit(cmd_, "stretch copy");
          [bl copyFromTexture:su
                   sourceSlice:0
                   sourceLevel:NSUInteger(sl)
                  sourceOrigin:MTLOriginMake(NSUInteger(a.left), NSUInteger(a.top), 0)
                    sourceSize:MTLSizeMake(NSUInteger(a.right - a.left),
                                           NSUInteger(a.bottom - a.top), 1)
                     toTexture:du
              destinationSlice:0
              destinationLevel:NSUInteger(dl)
             destinationOrigin:MTLOriginMake(NSUInteger(b.left), NSUInteger(b.top), 0)];
          [bl endEncoding];
          return;
        }
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = du;
        rp.colorAttachments[0].level = NSUInteger(dl);
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "stretch quad");
        DrawQuad(e, du.pixelFormat, su, sl, float(sw), float(sh), a, float(dw), float(dh), b, linear);
        [e endEncoding];
      });
    }
  }

  // ---- 2D: DirectDraw's, not provided

  void Blit(const void*, uint32_t, Format, const Rect&, const Rect&, bool, uint32_t) override {
    WarnOnce(1, "Blit");
  }
  void BlitBlend(const void*, uint32_t, const Rect&, const Rect&) override {
    WarnOnce(2, "BlitBlend");
  }
  void BlitFront(const void*, uint32_t, Format, const Rect&, const Rect&) override {
    WarnOnce(4, "BlitFront");
  }

  void ReadBack(const Rect& r, void* dst, uint32_t pitch, Format fmt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    ReadTexture(pass_.color[0], pass_.w, pass_.h, pass_.level[0], r, dst, pitch, fmt);
  }

  uint32_t StartReadBack(const Rect& r) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    uint32_t t = next_read_++;
    if (!t) t = next_read_++;
    Read& rd = reads_[t];
    rd.r = r;
    rd.px.resize(size_t(std::max(r.right - r.left, 0)) * size_t(std::max(r.bottom - r.top, 0)));
    ReadTexture(pass_.color[0], pass_.w, pass_.h, pass_.level[0], r, rd.px.data(),
                uint32_t(r.right - r.left) * 4, Format::A8R8G8B8);
    return t;
  }

  void FinishReadBack(uint32_t ticket, void* dst, uint32_t pitch, Format fmt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = reads_.find(ticket);
    if (it == reads_.end()) return;
    if (dst) {
      const Read& rd = it->second;
      int w = rd.r.right - rd.r.left, h = rd.r.bottom - rd.r.top;
      uint32_t bpp = BitsPerPixel(fmt) / 8;
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
          EncodePixel(static_cast<uint8_t*>(dst) + size_t(y) * pitch + size_t(x) * bpp, fmt,
                      rd.px[size_t(y) * w + x]);
    }
    reads_.erase(it);
  }

  void ReadBackFront(const Rect& r, void* dst, uint32_t pitch, Format fmt) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    ReadTexture(front_, bb_w_, bb_h_, 0, r, dst, pitch, fmt);
  }

  void PresentFront() override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    Settle();
    EnsureFrame();
    __unsafe_unretained id<MTLTexture> front = front_;
    const int bw = bb_w_, bh = bb_h_;
    const bool scene = first_scene_;
    Rec([=, this] { Show(front, bw, bh, 0, scene); });
    EndFrame();
  }

  // ---- state. Only what the shader path uses: the fixed-function state
  // (texture stages, transforms, material, lights) has no effect here.

  void SetRenderState(uint32_t s, uint32_t v) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (s < rs_.size()) rs_[s] = v;
  }
  void SetTextureStageState(uint32_t, uint32_t, uint32_t) override {}
  void SetTexture(uint32_t, TextureId) override {}
  void SetTransform(uint32_t, const float*) override {}
  void SetViewport(const Viewport& v) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    viewport_ = v;
  }
  void SetScissor(const Rect& r) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    scissor_ = r;
  }
  void SetMaterial(const Material&) override {}
  void SetLight(uint32_t, const Light&) override {}
  void LightEnable(uint32_t, bool) override {}
  void SetClipPlane(uint32_t i, const float* p) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (i < 6) std::memcpy(clip_planes_[i], p, sizeof clip_planes_[i]);
  }

  // ---- drawing

  void Clear(uint32_t count, const Rect* rects, uint32_t flags, uint32_t color, float z,
             uint32_t stencil) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    EnsureTarget();
    if (!pass_.color[0]) return;
    if (!pass_.depth) flags &= ~6u;
    if (!pass_.stencil) flags &= ~4u;
    if (!(flags & 7)) return;
    Rect area = ViewportRect();
    if (rs_[174]) area = Intersect(area, scissor_);
    std::vector<Rect> todo;
    if (!count || !rects) {
      todo.push_back(area);
    } else {
      for (uint32_t i = 0; i < count; i++) {
        Rect c = Intersect(rects[i], area);
        if (!Empty(c)) todo.push_back(c);
      }
    }
    if (todo.empty()) return;
    if (pass_.reversed) z = 1.0f - z;
    // A new content of the whole buffer: the next draw from w sets its
    // projection.
    // (A traced shadow mask of what was there is out of date.)
    if (shadow_masker_) RtNoteWrites(flags & 2, flags & 1, true);
    if ((flags & 2) && todo.size() == 1 && todo[0].left == 0 && todo[0].top == 0 &&
        todo[0].right == pass_.w && todo[0].bottom == pass_.h)
      if (auto dit = depths_.find(pass_.depth_id); dit != depths_.end()) dit->second.wd.fresh = true;
    // A pass opened without the depth buffer (BareDraw) ends before a clear
    // of depth or stencil: the clear is then the next pass's, as below.
    if (enc_open_ && pass_bare_ && (flags & 6)) BareSplit();
    // A clear of the whole target before anything was drawn is the render
    // pass's load action.
    if (!enc_open_ && todo.size() == 1 && todo[0].left == 0 && todo[0].top == 0 &&
        todo[0].right == pass_.w && todo[0].bottom == pass_.h) {
      if (flags & 1) Argb(&clear_rgba_[0], color);
      clear_color_ = MTLClearColorMake(clear_rgba_[0], clear_rgba_[1], clear_rgba_[2],
                                       clear_rgba_[3]);
      if (flags & 2) clear_depth_ = z;
      if (flags & 4) clear_stencil_ = uint8_t(stencil);
      clear_flags_ |= flags & 7;
      return;
    }
    // Otherwise a triangle per rectangle, scissored to it, writing exactly
    // what the flags say.
    if ((flags & 1) && pass_.targets > 1)
      FatalF("gfx: partial Clear of %d render targets is not supported", pass_.targets);
    if (!BeginDraw()) return;
    if (flags & 6) pass_depth_used_ = true;  // (BareDraw: this pass needs its depth buffer)
    PipeKey pk = PassKey();
    pk.vs = kClearVs;
    pk.ps = kClearPs;
    pk.mask[0] = (flags & 1) ? MTLColorWriteMaskAll : 0;
    for (int i = 1; i < 4; i++) pk.mask[i] = 0;
    BindPipeline(pk);
    DepthKey dk{};
    dk.func = uint8_t(MTLCompareFunctionAlways);
    if (flags & 2) dk.test = dk.write = 1;
    dk.sfunc = dk.bfunc = uint8_t(MTLCompareFunctionAlways);
    if (flags & 4) {
      dk.stencil = 1;
      dk.fail = dk.zfail = dk.pass = uint8_t(MTLStencilOperationReplace);
      dk.bfail = dk.bzfail = dk.bpass = uint8_t(MTLStencilOperationReplace);
      dk.read = dk.wmask = 0xFF;
    }
    BindDepth(dk);
    Raster(MTLCullModeNone, MTLTriangleFillModeFill, 0, 0);
    ViewportOf({0, 0, double(pass_.w), double(pass_.h), 0, 1});
    StencilRef(uint8_t(stencil));
    struct Vec {
      float v[4];
    } c, zp{{z, 0, 0, 0}};
    Argb(c.v, color);
    Rec([this, c, zp] {
      [enc_ setVertexBytes:zp.v length:sizeof zp.v atIndex:0];
      [enc_ setFragmentBytes:c.v length:sizeof c.v atIndex:0];
    });
    bound_.vbuf[0] = bound_.fbuf[0] = nullptr;
    for (const Rect& r : todo) {
      ScissorOf({NSUInteger(r.left), NSUInteger(r.top), NSUInteger(r.right - r.left),
                 NSUInteger(r.bottom - r.top)});
      Rec([this] { [enc_ drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; });
    }
  }

  void BeginScene() override {}
  void EndScene() override {}

  void Draw(uint32_t, uint32_t, const void*, uint32_t, uint32_t, const uint16_t*,
            uint32_t) override {
    FatalF("gfx: fixed-function draw on the Metal backend");
  }
  Mapped MapDraw(uint32_t, uint32_t, uint32_t) override {
    FatalF("gfx: fixed-function draw on the Metal backend");
  }
  void DrawMapped(uint32_t, uint32_t) override {
    FatalF("gfx: fixed-function draw on the Metal backend");
  }

  void Present() override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    {
      FirstScene();
      EndBoostLate();
      // The flight recorder (flight_recorder.h): the frame as shown and the
      // sun's traced shadow mask, with the frame's numbers, taken before
      // the frame's end clears its camera and snapshot.
      RecorderBegin();
      FlightRecorder* rec = recorder_.get();
      const uint64_t rec_frame = rec ? ++rec_frame_ : 0;
      if (rec) {
        std::lock_guard<std::mutex> nl(rec_note_mu_);
        rec_notes_[rec_frame % kRecNotes] = RecorderNote(CACurrentMediaTime());
      }
      __unsafe_unretained id<MTLTexture> rec_mask = rec && shadow_masker_ ? shadow_mask_[slot_] : nil;
      if (rec) RecorderPad();
      AdoptFinished();
      EndUpscalerFrame();
      Settle();
      EnsureFrame();
      RtEndFrame(true);
      const bool smaa = upscaler_ == Upscaler::Smaa;
      if (smaa) SmaaFrame();
      __unsafe_unretained id<MTLTexture> back = backbuffer_, front = front_;
      const int bw = bb_w_, bh = bb_h_;
      const double t_present = CACurrentMediaTime();  // the game's Present call
      // SMAA filtered the back buffer into the front one; else the back
      // buffer becomes the front one and the old front the next back
      // buffer, D3D9's flip (a discarded back buffer's contents after
      // Present are undefined) instead of a full-size copy.
      __unsafe_unretained id<MTLTexture> shown = smaa ? front : back;
      // Captured after FirstScene(), for the toast: the encode thread runs
      // later, when a later frame may already have set first_scene_.
      const bool scene = first_scene_;
      Rec([=, this] {
        Show(shown, bw, bh, t_present, scene);
        if (rec) RecordFrame(rec, shown, bw, bh, rec_mask, rec_frame, t_present);
      });
      // The frame's local lights are over with it (SetRtLights).
      RtLightsPresent(rt_lights_, rt_ != nullptr, [this](auto f) { Rec(f); });
      if (!smaa) std::swap(backbuffer_, front_);
      presented_ = true;
      if (frame_log_) FrameLogPresent();
      if (perf_) {
        pc_.frames++;
        pc_.upload_bytes += frame_upload_bytes_;
      }
      if (frame_upload_bytes_ > (8u << 20))
        std::fprintf(stderr, "gfx: frame uploaded %.1f MB of textures\n", frame_upload_bytes_ / 1048576.0);
      frame_upload_bytes_ = 0;
      EndFrame();
    }
  }

  // The frame log's lines of a presented frame (frame_log.h), at Present on
  // the recording thread: "p", and "c" from the frame's own counters
  // (FrameLogCounts: all of the frame, also where a readback cut it into
  // several serials). The ray scene's counts ("r") are taken here and
  // written by the encoding thread at the frame's end (EncodeEndFrame),
  // which knows how many lights were binned on screen.
  void FrameLogPresent() {
    const uint64_t now = NowNs();
    FrameLogCounts& c = frame_log_counts_;
    // (Ops: the batches handed over since the last line and the one being
    // filled; the op that ends the frame is counted with the next.)
    const uint64_t ops = frame_log_ops_ + rec_.ops.size();
    FrameLogLine("p %llu %.4f %.3f %llu %llu %llu", (unsigned long long)serial_, FrameLogSeconds(),
                 frame_log_last_ ? double(now - frame_log_last_) * 1e-6 : 0.0, (unsigned long long)c.draws,
                 (unsigned long long)c.passes, (unsigned long long)(ops - frame_log_ops_seen_));
    frame_log_ops_seen_ = ops;
    frame_log_last_ = now;
    // The 1 draw in 32 that RecordDraw timed, in mach ticks, scaled by the
    // calls (as TakePerfStats does): an estimate, some 100 samples a frame.
    static const double tick_ns = [] {
      mach_timebase_info_data_t tb{};
      mach_timebase_info(&tb);
      return tb.denom ? double(tb.numer) / double(tb.denom) : 1.0;
    }();
    const double note_ns =
        c.note_timed ? double(c.note_ticks) * tick_ns * double(c.note_calls) / double(c.note_timed) : 0.0;
    // Recording: from the frame's first slot to here, less its later waits
    // for one (a frame a readback cut waits again).
    const uint64_t span = c.begun_ns && now > c.begun_ns ? now - c.begun_ns : 0;
    const uint64_t record_ns = span > c.later_wait_ns ? span - c.later_wait_ns : 0;
    FrameLogLine("c %llu %.3f %.3f %.3f %.3f %.3f", (unsigned long long)serial_, double(c.slot_wait_ns) * 1e-6,
                 double(record_ns) * 1e-6, note_ns * 1e-6, double(c.snapshot_ns) * 1e-6, double(c.rt_end_ns) * 1e-6);
    FrameLogRay r;
    r.fwd_draws = c.fwd_draws;
    if (rt_) {
      // The scene's counters since the last Present (rt_scene.h's RtStats):
      // the TLAS's instances and the captured draws among them are added at
      // the presented frame's end, which has run (RtEndFrame). TakePerfStats
      // zeroes all but total_builds between frames, and what was seen of
      // them here with them; `since` covers a scene made anew as well.
      const RtStats st = rt_->Stats();
      FrameLogRtSeen& seen = frame_log_rt_seen_;
      auto since = [](uint64_t cur, uint64_t* last) {
        const uint64_t d = cur >= *last ? cur - *last : cur;
        *last = cur;
        return uint32_t(d);
      };
      r.instances = since(st.instances, &seen.instances);
      r.captured = since(st.captured, &seen.captured);
      // BLASes built anew: the static meshes', and the captured draws' that
      // did not refit an earlier frame's (those run every frame).
      const uint32_t capture_builds = since(st.capture_builds, &seen.capture_builds);
      const uint32_t capture_refits = since(st.capture_refits, &seen.capture_refits);
      r.builds = since(st.total_builds, &frame_log_rt_builds_) +
                 (capture_builds > capture_refits ? capture_builds - capture_refits : 0);
    }
    frame_log_ray_ = r;
    c = {};
  }

  // ---- the flight recorder (flight_recorder.h)

  // Made at the first Present with a back buffer, where RECOMP_RECORD_DIR
  // names a folder (RECOMP_RECORD=0: not): two streams, the frame as shown
  // and the sun's traced shadow mask, at most 1720 pixels wide.
  void RecorderBegin() {
    if (recorder_tried_ || bb_w_ <= 0 || bb_h_ <= 0) return;
    recorder_tried_ = true;
    const char* dir = std::getenv("RECOMP_RECORD_DIR");
    const char* on = std::getenv("RECOMP_RECORD");
    if (!dir || !*dir || (on && *on == '0')) return;
    const int w = std::min(bb_w_, 1720);
    const int h = int(std::lround(double(bb_h_) * w / bb_w_));
    recorder_ = FlightRecorder::Make(dev_, dir, 2, w, h);
  }

  // A presented frame's numbers for the recorder: the camera's position
  // and the way it looks (from the frame's view-projection, D3D's clip
  // space), the jitter and the time since the last Present.
  std::string RecorderNote(double t) {
    char buf[256];
    std::string out;
    double inv[16];
    if (have_vp_ && Invert4(vp_cur_, inv)) {
      auto at = [&](double z, double o[3]) {
        double p[4] = {0, 0, z, 1}, r[4] = {};
        for (int j = 0; j < 4; j++)
          for (int i = 0; i < 4; i++) r[j] += p[i] * inv[i * 4 + j];
        for (int k = 0; k < 3; k++) o[k] = r[3] != 0 ? r[k] / r[3] : 0;
      };
      double n[3], f[3];
      at(0.0, n);
      at(1.0, f);
      double d[3] = {f[0] - n[0], f[1] - n[1], f[2] - n[2]};
      const double l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (l > 0)
        for (double& v : d) v /= l;
      std::snprintf(buf, sizeof buf, "\"eye\":[%.2f,%.2f,%.2f],\"fwd\":[%.3f,%.3f,%.3f],", n[0], n[1], n[2], d[0],
                    d[1], d[2]);
      out += buf;
    }
    std::snprintf(buf, sizeof buf, "\"jit\":[%.3f,%.3f],\"dt_ms\":%.2f,\"tlas\":%u", double(jitter_[0]),
                  double(jitter_[1]), rec_last_t_ > 0 ? (t - rec_last_t_) * 1e3 : 0.0,
                  snap_frame_ ? snap_frame_->instances : 0u);
    out += buf;
    // Draws left out of the ray-traced scene this frame, by why: blended,
    // its BLAS waiting for a later frame, past the capture budget (deltas
    // of the scene's counters, which the perf report resets).
    if (rt_) {
      const RtStats st = rt_->Stats();
      const RtSkip kinds[] = {RtSkip::Blended, RtSkip::BuildBudget, RtSkip::CaptureBudget};
      const char* names[] = {"sk_blend", "sk_budget", "sk_capture"};
      for (int k = 0; k < 3; k++) {
        const uint64_t now = st.skipped[size_t(kinds[k])];
        const uint64_t d = now >= rec_skip_[k] ? now - rec_skip_[k] : now;
        rec_skip_[k] = now;
        std::snprintf(buf, sizeof buf, ",\"%s\":%llu", names[k], (unsigned long long)d);
        out += buf;
      }
    }
    rec_last_t_ = t;
    return out;
  }

  // A gamepad's two sticks pressed together: a mark (on its own thread:
  // finishing the segments takes a moment).
  void RecorderPad() {
    int n = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&n);
    bool both = false;
    if (ids && n > 0)
      if (SDL_Gamepad* g = SDL_GetGamepadFromID(ids[0]))
        both = SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_LEFT_STICK) &&
               SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_RIGHT_STICK);
    SDL_free(ids);
    if (both && !rec_pad_down_) std::thread([] { FlightMark("pad"); }).detach();
    rec_pad_down_ = both;
  }

  // On the encoder thread, after Show: the shown frame (and the mask) drawn
  // into the recorder's buffers, appended once the GPU is done.
  void RecordFrame(FlightRecorder* rec, __unsafe_unretained id<MTLTexture> shown, int bw, int bh,
                   __unsafe_unretained id<MTLTexture> mask, uint64_t frame, double t) {
    std::string note;
    {
      std::lock_guard<std::mutex> nl(rec_note_mu_);
      note.swap(rec_notes_[frame % kRecNotes]);
    }
    CVPixelBufferRef pb[2] = {};
    CVMetalTextureRef ref[2] = {};
    const int rw = rec->Width(), rh = rec->Height();
    auto draw = [&](int k, __unsafe_unretained id<MTLTexture> src, int sw, int sh) {
      id<MTLTexture> tex = nil;
      if (!src || sw <= 0 || sh <= 0 || !rec->NewFrame(&pb[k], &ref[k], &tex)) return;
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = tex;
      rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "flight recorder");
      DrawQuad(e, MTLPixelFormatBGRA8Unorm, src, 0, float(sw), float(sh), {0, 0, sw, sh}, float(rw), float(rh),
               {0, 0, rw, rh}, true);
      [e endEncoding];
    };
    draw(0, shown, bw, bh);
    if (mask) draw(1, mask, int(mask.width), int(mask.height));
    CVPixelBufferRef p0 = pb[0], p1 = pb[1];
    CVMetalTextureRef r0 = ref[0], r1 = ref[1];
    std::string n = note;
    [cmd_ addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      rec->Append(0, p0, r0, t);
      rec->Append(1, p1, r1, t);
      char g[48];
      std::snprintf(g, sizeof g, "\"gpu_ms\":%.2f", (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
      rec->Note(frame, t, n.empty() ? std::string(g) : n + "," + g);
    }];
  }

  void SetDepthFromW(int reg, DepthPlanes planes) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    wvp_reg_ = depth_from_w_ && reg >= 0 && reg <= 252 ? reg : -1;
    depth_planes_ = planes;
  }

  void SetRayTracing(const RayTracing& c) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    if (rt_) return;
    NoteRayTracedShadows(false);  // (until the kernels below are made)
    NoteRayTracedReflections(false);
    NoteRayTracedForwardReflections(false);
    NoteRayTracedForwardShadows(false);
    if (c.mode == RayTracing::Mode::Off) return;
    if (!RtSupported(dev_)) {
      WarnF("gfx: ray tracing needs a GPU that traces rays in hardware (Apple9 family): off\n");
      return;
    }
    RtSceneOptions o;
    o.debug = c.debug == RayTracing::Debug::Normals     ? RtDebug::Normals
              : c.debug == RayTracing::Debug::Instances ? RtDebug::Instances
                                                        : RtDebug::Off;  // (ShadowMask: the device's)
    // The end of a frame's TLAS only for the scene's debug view: the traced
    // effects use the frame's snapshots.
    o.end_tlas = o.debug != RtDebug::Off;
    // (To measure: RECOMP_RT_RETAIN_FRAMES=n, retained instances' frames;
    // 0: none retained.)
    if (const char* e = std::getenv("RECOMP_RT_RETAIN_FRAMES"); e && *e)
      o.retain_frames = uint32_t(std::strtoul(e, nullptr, 10));
    // New meshes' gathering on the drawing thread bounded to 4 ms a
    // presented frame (RtSceneOptions::build_ms_per_frame); to measure:
    // RECOMP_RT_BUILD_MS=n, 0 not bounded.
    o.build_ms_per_frame = 4.0f;
    if (const char* e = std::getenv("RECOMP_RT_BUILD_MS"); e && *e)
      o.build_ms_per_frame = std::max(0.0f, float(std::atof(e)));
    // Reflections shade what rays hit: meshes keep their hit data once the
    // kernel is made (the other modes, or without it, do without that
    // memory). Each frame slot's counters (RtReflectionCounts).
    if (c.mode == RayTracing::Mode::Reflections) {
      reflector_ = RtReflector::Make(dev_);
      for (int k = 0; k < kSlots && reflector_; k++)
        if ((refl_counts_[k] = [dev_ newBufferWithLength:sizeof(RtReflectionCounts)
                                                 options:MTLResourceStorageModeShared]))
          std::memset(refl_counts_[k].contents, 0, sizeof(RtReflectionCounts));
        else
          reflector_.reset();
      refl_view_ = c.reflection_view;
      if (!reflector_) WarnF("gfx: ray-traced reflections could not start: the game's own reflections\n");
    } else if (c.reflection_view) {
      WarnF("gfx: the reflections' debug view needs RT=reflections\n");
    }
    o.hit_data = reflector_ != nullptr;
    rt_ = MakeRtScene(dev_, o);
    // Its BLASes resident through a residency set the queue has, not
    // declared in every encoder that traces them (RtScene::UseResidency).
    if (rt_ && !rt_->UseResidency(queue_)) WarnF("gfx: ray tracing declares its BLASes in each encoder\n");
    if (!rt_) {
      reflector_.reset();
      rt_forward_ = false;
      WarnF("gfx: ray tracing could not start: off\n");
      return;
    }
    // The frame's local lights from here on (SetRtLights).
    NoteRtLightDevice(this, true);
    rt_->SetAllocator(
        [](void* ctx, size_t bytes, size_t align) -> RtUpload {
          const Slice sl = static_cast<MetalDevice*>(ctx)->Alloc(bytes, align, false);
          return {sl.buf, sl.off, sl.cpu};
        },
        this);
    // Shaders made before have none of what the scene reads from them
    // (Shader::analyzed): their draws that alpha test or discard are left
    // out (RtSkip::AlphaSource).
    for (const auto& [sid, sh] : shaders_)
      if (sid < kClearVs) {
        WarnF("gfx: ray tracing turned on after shaders were made: their draws that alpha test or discard are left "
              "out\n");
        break;
      }
    rt_world_ = c.world_register >= 0 && c.world_register <= 252 ? c.world_register : -1;
    light_budget_ = c.local_light_masks;
    lights_step_ = std::min(c.local_light_step, 8u);
    lights_cap_ = std::clamp(c.local_light_cap, 1u, kRtLightsCap);
    // RECOMP_RT_LIGHT_STEP, RECOMP_RT_LIGHT_CAP: those (0: off); and
    // RECOMP_RT_LIGHT_ALT=1 (to measure, LightAlt): in turn for 240
    // presented frames each, on; off (each volume's pixels tracing its own,
    // as before); no local light's shadow traced at all; on at a step one
    // larger.
    if (const char* e = std::getenv("RECOMP_RT_LIGHT_STEP")) lights_step_ = std::min(uint32_t(std::atoi(e)), 8u);
    if (const char* e = std::getenv("RECOMP_RT_LIGHT_CAP"))
      lights_cap_ = std::clamp(uint32_t(std::atoi(e)), 1u, kRtLightsCap);
    if (const char* e = std::getenv("RECOMP_RT_LIGHT_ALT")) lights_alt_ = *e == '1';
    rt_filter_ = c.filter;
    rt_alpha_source_ = c.alpha_source;
    rt_albedo_source_ = c.albedo_source;
    rt_hit_data_ = o.hit_data;
    rt_other_passes_ = c.other_passes;
    rt_other_color_ = c.other_color_passes;
    if (frame_open_) rt_->BeginFrame(serial_, CompletedSerial());
    // Shadows (Reflections includes them): the masks' kernel; without it
    // games keep their own shadows (RayTracedShadows false).
    if (c.mode == RayTracing::Mode::Shadows || c.mode == RayTracing::Mode::Reflections) {
      shadow_masker_ = RtShadowMasker::Make(dev_);
      // Each frame slot's counters of rays traced and rays that hit (read
      // when the slot comes round again: its frame is done).
      for (int k = 0; k < kSlots && shadow_masker_; k++)
        if ((shadow_counts_[k] = [dev_ newBufferWithLength:kShadowCountsSize options:MTLResourceStorageModeShared]))
          std::memset(shadow_counts_[k].contents, 0, kShadowCountsSize);
        else
          shadow_masker_.reset();
      shadow_log_ = c.shadow_log;
      // RECOMP_RT_SHADOW_LOG_EVERY=n: the diagnostics every n presented
      // frames (120): a camera turned step by step, say, each step logged.
      if (const char* e = std::getenv("RECOMP_RT_SHADOW_LOG_EVERY"))
        shadow_log_every_ = std::max(1u, uint32_t(std::strtoul(e, nullptr, 10)));
      // Ray-traced ambient occlusion (RayOcclusion; with the masks'
      // counters).
      // The volumetric fog (RtFog; RECOMP_FOG, off unless set).
      if (const char* e = std::getenv("RECOMP_FOG"); shadow_masker_ && e && *e) {
        if (ParseRtFog(e, &fog_look_)) fog_ = RtFog::Make(dev_);
        WarnF("gfx: ray-traced volumetric fog: %s (%s)\n", fog_ ? "on" : "not available or not understood", e);
      }
      if (reflector_ && reflector_->Bounce() && c.bounce_distance > 0.0f) {
        bounce_distance_ = c.bounce_distance;
        if (const char* e = std::getenv("RECOMP_RT_BOUNCE_STRENGTH"); e && *e) bounce_strength_ = float(std::atof(e));
        if (const char* e = std::getenv("RECOMP_RT_AO_STRENGTH"); e && *e)
          bounce_ao_strength_ = std::clamp(float(std::atof(e)), 0.0f, 1.0f);
        WarnF("gfx: the far light bounced once by rays: on (%.0f units, strength %.2f)\n", double(bounce_distance_),
              double(bounce_strength_));
      }
      if (shadow_masker_ && c.occlusion_radius > 0.0f) {
        occluder_ = RtOccluder::Make(dev_);
        ao_radius_ = c.occlusion_radius;
        WarnF("gfx: ray-traced ambient occlusion: %s (%.1f units)\n", occluder_ ? "on" : "not available",
              double(ao_radius_));
      }
      // The local lights lit by the device (RayManyLights; with the
      // masks' counters).
      ml_cfg_ = c.many_lights;
      if (shadow_masker_ && ml_cfg_.mode != RayManyLights::Mode::Off) {
        ml_ = RtManyLights::Make(dev_);
        ml_only_ = ml_ && ml_cfg_.mode == RayManyLights::Mode::Only;
        WarnF("gfx: many lights lit with ray-traced shadows: %s\n", !ml_       ? "not available"
                                                                    : ml_only_ ? "on, the light volumes left out"
                                                                               : "on, added to the light volumes");
      }
      shadow_view_ = c.debug == RayTracing::Debug::ShadowMask;
      if (!shadow_masker_) WarnF("gfx: ray-traced shadows could not start: the game's own shadows\n");
      NoteRayTracedShadows(shadow_masker_ != nullptr);
      lights_ready_ = shadow_masker_ && shadow_masker_->LightsReady();
      // The sun's mask accumulated over frames: off unless asked for
      // (RECOMP_RT_ACCUMULATE=1; RECOMP_RT_ACCUMULATE_STEP: its step, 2 to
      // 4): measured, it costs more than the fewer rays save and leaves
      // trails behind moving shadows (rt_accumulate_test); never where
      // every pixel is traced (RECOMP_RT_SHADOW_FULL=1).
      const char* acc = std::getenv("RECOMP_RT_ACCUMULATE");
      const char* full = std::getenv("RECOMP_RT_SHADOW_FULL");
      if (shadow_masker_ && shadow_masker_->SparseReady() && acc && *acc == '1' && !(full && *full == '1'))
        accumulator_ = RtAccumulator::Make(dev_);
      if (const char* e = std::getenv("RECOMP_RT_ACCUMULATE_STEP"))
        sun_step_ = std::clamp(uint32_t(std::atoi(e)), 2u, 4u);
      // (0: every 2 x 2's top-left pixel, not turning, as before.)
      if (const char* e = std::getenv("RECOMP_RT_SUN_TURN")) sun_turn_ = std::min(uint32_t(std::atoi(e)), 4u);
      if (sun_turn_ == 1) sun_turn_ = 2;
      if (accumulator_)
        std::fprintf(stderr, "gfx: the sun's mask accumulated over frames, traced at 1 of %u x %u\n", sun_step_,
                     sun_step_);
      own_maps_lit_ = shadow_masker_ && c.own_maps_lit;
    }
    // Forward pixel shaders tracing their own reflections or shadows
    // (rt_forward_reflection.h): ray queries in fragment functions.
    // (RECOMP_RT_FORWARD=0: none, for comparisons.)
    const char* fwd_env = std::getenv("RECOMP_RT_FORWARD");
    if ((reflector_ || shadow_masker_) && !(fwd_env && *fwd_env == '0')) {
      rt_forward_ = RtSupportedFromRender(dev_);
      if (rt_forward_ && !RtForwardMakeNullScene(dev_, queue_, &rt_fwd_null_)) {
        rt_forward_ = false;
        WarnF("gfx: ray tracing in forward shaders could not start: they keep their own reflections and are lit "
              "where they read shadows\n");
      } else if (!rt_forward_)
        WarnF("gfx: the GPU does not trace rays from render pipelines: forward shaders keep their own "
              "reflections and are lit where they read shadows\n");
    }
    if (shadow_masker_ && c.denoise) StartDenoiser();
    NoteRayTracedReflections(reflector_ != nullptr);
    NoteRayTracedForwardReflections(rt_forward_ && reflector_);
    NoteRayTracedForwardShadows(rt_forward_ && shadow_masker_);
    std::fprintf(stderr, "gfx: ray tracing on (%s)%s\n",
                 c.mode == RayTracing::Mode::Shadows       ? "shadows"
                 : c.mode == RayTracing::Mode::Reflections ? "reflections"
                                                           : "the scene only",
                 o.debug == RtDebug::Normals ? " (debug view: normals)"
                 : o.debug == RtDebug::Instances ? " (debug view: instances)"
                 : c.debug == RayTracing::Debug::ShadowMask ? " (debug view: shadow mask)"
                 : c.reflection_view                        ? " (debug view: reflections)"
                                                            : "");
    if (rt_world_ < 0) WarnF("gfx: ray tracing: the game's world matrix register is not known: no draw is taken\n");
  }

  bool TracksBufferContents() const override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    return rt_ != nullptr;
  }

  bool RayTracingOn() const override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    return rt_ != nullptr;
  }

  void SetRayShadow(ShaderId sid, const RayShadow& s) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (!shadow_masker_ || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel) return;
    Shader& sh = it->second;
    const hlsl::ParamType t = s.mask_sampler < 16 ? sh.sampler_type[s.mask_sampler] : hlsl::ParamType::Void;
    // One still reading the game's own shadow map: counted; with
    // own_maps_lit, a lit one bound there (a 2D sampler).
    if (s.own_map) {
      sh.own_map = true;
      sh.shadow = s;
      sh.shadow_claimed = own_maps_lit_ && (t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler) &&
                          OwnMapLit(s.own_map_lit);
      return;
    }
    // (A 2D sampler the game says reads the mask: never what the draw has
    // there, whatever else is wrong; all lit then.)
    sh.shadow_claimed = t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler;
    sh.shadow = s;
    // A local light's registers (RayShadow::local), else the far light's.
    const bool light_ok =
        s.local ? s.position_register < 224 && s.radius_register < 224 && s.radius_component < 4 &&
                      (!s.cone || (s.direction_register < 224 && s.cone_offset_register < 224 &&
                                   s.cone_scale_register < 224 && s.cone_offset_component < 4 &&
                                   s.cone_scale_component < 4)) &&
                      std::isfinite(s.lamp_gap)
                : s.light_register < 224;
    if (s.local) (sh.shadow_claimed && s.depth_sampler < 16 && light_ok ? light_shaders_ : light_shaders_refused_)++;
    if (!sh.shadow_claimed || s.depth_sampler >= 16 || !light_ok) {
      WarnF("gfx: ray-traced shadows: pixel shader %016llx: mask sampler %u (%s), depth sampler %u, %s c%u%s: "
            "not traced%s\n",
            (unsigned long long)sh.hash, s.mask_sampler,
            t == hlsl::ParamType::Void ? "not read" : sh.shadow_claimed ? "2D" : "not 2D", s.depth_sampler,
            s.local ? "local light at" : "light", s.local ? s.position_register : s.light_register,
            s.local && !light_ok ? " (its registers out of range)" : "", sh.shadow_claimed ? " (all lit)" : "");
      return;
    }
    sh.ray_shadow = true;
  }

  void SetRayReflection(ShaderId sid, const RayReflection& r) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (!reflector_ || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel) return;
    Shader& sh = it->second;
    const hlsl::ParamType t = r.output_sampler < 16 ? sh.sampler_type[r.output_sampler] : hlsl::ParamType::Void;
    // (A 2D sampler the game says reads the reflections: never what the
    // draw has there; nothing hit where they cannot be traced.)
    sh.reflection_claimed = t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler;
    sh.reflection = r;
    const bool slots = r.depth_sampler < 16 && r.normal_sampler < 16 && r.surface_sampler < 16 &&
                       r.light_register < 224 && r.colour_register < 224 && r.ambient_registers[0] < 224 &&
                       r.ambient_registers[1] < 224 && r.spec_channel < 4 && r.gloss_channel < 4;
    if (!sh.reflection_claimed || !slots) {
      WarnF("gfx: ray-traced reflections: pixel shader %016llx: output sampler %u (%s), depth %u, normal %u, "
            "surface %u, light c%u, colour c%u, ambient c%u c%u: not traced%s\n",
            (unsigned long long)sh.hash, r.output_sampler,
            t == hlsl::ParamType::Void ? "not read" : sh.reflection_claimed ? "2D" : "not 2D", r.depth_sampler,
            r.normal_sampler, r.surface_sampler, r.light_register, r.colour_register, r.ambient_registers[0],
            r.ambient_registers[1], sh.reflection_claimed ? " (nothing hit)" : "");
      return;
    }
    sh.ray_reflection = true;
  }

  void SetRayForwardReflection(ShaderId sid, const RayForwardReflection& r) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (!rt_forward_ || !reflector_ || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel) return;
    Shader& sh = it->second;
    sh.forward = r;
    const bool slots = r.light_register < 224 && r.colour_register < 224 && r.ambient_registers[0] < 224 &&
                       r.ambient_registers[1] < 224;
    // (Its variants are made from the same translation: given the ray
    // queries, or not, alike.)
    if (!sh.rt_forward || !slots) {
      WarnF("gfx: ray-traced forward reflections: pixel shader %016llx: sampler %u, origin v%u, light c%u, colour "
            "c%u, ambient c%u c%u%s: nothing hit\n",
            (unsigned long long)sh.hash, r.sampler, r.origin_input, r.light_register, r.colour_register,
            r.ambient_registers[0], r.ambient_registers[1],
            sh.rt_forward ? "" : " (its translation not given the ray queries)");
      return;
    }
    sh.ray_forward = true;
  }

  void SetRayForwardShadow(ShaderId sid, const RayForwardShadow& s) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (!rt_forward_ || !shadow_masker_ || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel) return;
    Shader& sh = it->second;
    sh.forward_shadow = s;
    if (!sh.rt_forward || sh.ray_shadow_sampler < 0 || s.light_register >= 224) {
      WarnF("gfx: ray-traced forward shadows: pixel shader %016llx: sampler %u, light c%u%s: all lit\n",
            (unsigned long long)sh.hash, s.sampler, s.light_register,
            sh.rt_forward && sh.ray_shadow_sampler >= 0 ? "" : " (its translation not given the ray queries)");
      return;
    }
    sh.ray_forward_shadow = true;
  }

  void SetRayDenoise(ShaderId sid, const RayDenoise& d) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    // (Without the denoiser, the post-processing's inputs only: where the
    // reflections keep the lit scene for their screen-space march.)
    if ((!denoiser_ && !(reflector_ && d.post_input)) || it == shaders_.end() ||
        it->second.stage != hlsl::Stage::Pixel)
      return;
    Shader& sh = it->second;
    auto flat = [&](uint32_t r) {
      const hlsl::ParamType t = r < 16 ? sh.sampler_type[r] : hlsl::ParamType::Void;
      return t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler;
    };
    const bool ok = d.post_input ? flat(d.input_sampler)
                                 : flat(d.albedo_sampler) && flat(d.normal_sampler) && flat(d.surface_sampler) &&
                                       d.spec_channel < 4 && d.gloss_channel < 4;
    if (!ok) {
      WarnF("gfx: denoising: pixel shader %016llx: %s sampler(s) not read as 2D: not taken\n",
            (unsigned long long)sh.hash, d.post_input ? "its input" : "its G-buffer");
      return;
    }
    sh.denoise = d;
    (d.post_input ? sh.denoise_input : sh.denoise_guides) = true;
  }

  void SetRayOcclusion(ShaderId sid, const RayOcclusion& o) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if ((!occluder_ && bounce_distance_ <= 0.0f) || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel)
      return;
    Shader& sh = it->second;
    auto flat = [&](uint32_t r, bool need) {
      const hlsl::ParamType t = r < 16 ? sh.sampler_type[r] : hlsl::ParamType::Void;
      return (!need && r >= 16) || t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler;
    };
    if (!flat(o.occlusion_sampler, true) || !flat(o.depth_sampler, true) || !flat(o.normal_sampler, false)) {
      WarnF("gfx: ray-traced occlusion: pixel shader %016llx: a sampler not read as 2D: not taken\n",
            (unsigned long long)sh.hash);
      return;
    }
    sh.occlusion = o;
    sh.ray_occlusion = true;
  }

  void SetRayLightVolume(ShaderId sid, const RayLightVolume& v) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (!ml_ || it == shaders_.end() || it->second.stage != hlsl::Stage::Pixel) return;
    Shader& sh = it->second;
    auto flat = [&](uint32_t r) {
      const hlsl::ParamType t = r < 16 ? sh.sampler_type[r] : hlsl::ParamType::Void;
      return r >= 16 || t == hlsl::ParamType::Sampler2D || t == hlsl::ParamType::Sampler;
    };
    if (!flat(v.depth_sampler) || !flat(v.albedo_sampler) || !flat(v.normal_sampler) || !flat(v.surface_sampler) ||
        !flat(v.class_sampler)) {
      WarnF("gfx: many lights: pixel shader %016llx: a G-buffer sampler not read as 2D: not taken\n",
            (unsigned long long)sh.hash);
      return;
    }
    sh.light_volume = v;
    sh.light_volume_on = true;
  }

  // The frame's local lights (gfx::SetRtLights): copied now, handed to the
  // encoding side in the recorded order (rt_light_list.h), where the work
  // recorded after this reads them (rt_lights_.Current()) until the next
  // call or the presented frame's end.
  void SetRtPlainPlaces(const float* places, size_t count) override {
    plain_places_.assign(places, places + (places ? count * 5 : 0));
  }

  void SetRtLights(const RtLight* lights, size_t count) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    RtLightCounts n;
    if (!RtLightsGive(rt_lights_, rt_ != nullptr, lights, count, [this](auto f) { Rec(f); }, &n)) return;
    if (perf_) {
      pc_.rtl_lists++;
      pc_.rtl_given += n.given;
      pc_.rtl_kept += n.kept;
      pc_.rtl_point += n.point;
      pc_.rtl_spot += n.spot;
    }
  }

  void SetMinAnisotropy(int level) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    min_aniso_ = std::clamp(level, 0, 16);
    for (auto& [k, smp] : samplers_) Keep(smp);
    samplers_.clear();
    for (SlotSampler& c : slot_sampler_) c = {};
    if (min_aniso_ > 1) std::fprintf(stderr, "gfx: anisotropic filtering %dx\n", min_aniso_);
  }

  void SetPostAntialiasing(bool on) override {
    if (on) WarnF("gfx: post-process anti-aliasing is not in the Metal backend yet\n");
  }

  // ---- translated shaders (runtime/hlsl). The prelude completes the
  // translation (hlsl.h) as gpu_device.cpp's Metal build does: buffers
  // 0 float registers, 1 int and bool registers, 2 the prelude's state,
  // vertex shaders' 3 the vertices; textures and samplers at their
  // register.

  struct VsState {
    int32_t in[16][4];  // per input slot: byte offset of vertex 0, stride, D3DDECLTYPE (-1: none), 0
    float pixel[4];     // 1 / viewport width, 1 / viewport height; jitter (clip space)
    float clip[6][4];   // SetClipPlane's, clip space
    int32_t clip_mask[4];  // CLIPPLANEENABLE; depth (r_position: 0 D3D9's, 1 reversed, 2 from w)
    float depth[4];        // depth from w: k0, k1 (reversed_z.h)
  };
  struct PsState {
    int32_t alpha[4];  // D3DCMPFUNC (ALWAYS when the test is off), ALPHAREF,
                       // bit n: sampler n reads a reversed depth texture (R_DEPTH_TEX),
                       // 1 if the depth buffer is reversed (R_DEPTH_OUT)
    float bias[16];    // per sampler register: MIPMAPLODBIAS (R_BIAS)
    float vscale[4];   // game pixels per pixel of the target (r_vpos)
  };

  static std::string Prelude(hlsl::Stage stage) { return MetalShaderPrelude(stage); }

  ShaderId CreateShader(const hlsl::Program& prog) override { return MakeShader(prog, true, false); }

  // A translated shader, or (`capture`) the capture variant of a vertex
  // shader, prog.glsl then hlsl::VertexWithCapturedPosition's. A capture
  // variant is optional: one that does not compile is 0 (CaptureFailed),
  // where a shader the game made ends the process; it is compiled here (its
  // pipeline is made with it, CaptureShaderFor).
  //
  // CreateShader's: the shader registered and, unless it is compiled now,
  // compiled when first drawn (EnsureShader) or ahead on the shader lanes
  // (`ahead`: once they have started, Prefetch). One an earlier run's
  // recipe names goes to the lanes at once, before the first frame and
  // ahead of the others (its pipelines are built ahead, PrebuildFor, once
  // its shaders are both there: Finish). Compiled now: with
  // RECOMP_LAZY_SHADERS=0, or a recipe's with no lanes
  // (RECOMP_SHADER_LANES=0). (Compiled here, recipes' shaders held the
  // game's thread for seconds while loading when only the OS's cache of
  // compiled shaders was empty.)
  //
  // A pixel shader to be given ray queries (`forward`): its MSL (from
  // earlier runs, else translated here) is tried with the patch here, so
  // that Shader::rt_forward is known when the game configures it
  // (SetRayForwardReflection, SetRayForwardShadow) before it is compiled;
  // the compile, wherever it runs, makes the patched MSL again from the
  // plain one and settles it (Give: without them if they do not compile).
  ShaderId MakeShader(const hlsl::Program& prog, bool ahead, bool capture) {
    std::lock_guard<std::recursive_mutex> l(mu_);
    TakeOverFinished();
    const uint64_t perf_t0 = perf_ ? NowNs() : 0;
    @autoreleasepool {
      Shader sh;
      sh.stage = prog.stage;
      sh.inputs = prog.inputs.size();
      sh.float_regs = prog.float_regs;
      sh.position_matrix = prog.position_matrix;
      sh.position_note = prog.position_note;
      sh.float_reads = prog.float_reads;
      sh.rel_const_base = prog.rel_const_base;
      for (size_t i = 0; i < prog.inputs.size(); i++) {
        const hlsl::Semantic in = prog.inputs[i];
        if (in.usage == hlsl::kPosition && in.index == 0 && sh.pos_slot < 0) sh.pos_slot = int(i);
        if (in.usage == hlsl::kBlendWeight || in.usage == hlsl::kBlendIndices) sh.blend = true;
        if (in.usage == hlsl::kTexcoord && in.index < 8 && sh.tex_slot[in.index] < 0) sh.tex_slot[in.index] = int8_t(i);
        if (in.usage == hlsl::kColor && in.index == 0 && sh.color0_slot < 0) sh.color0_slot = int(i);
      }
      // (The bytecode's texkill and HLSL's clip and discard all become
      // GLSL discards; the alpha test is the prelude's.)
      sh.discards = prog.stage == hlsl::Stage::Pixel && prog.glsl.find("discard") != std::string::npos;
      if (rt_) {
        sh.moved = prog.moved_lanes;
        sh.analyzed = !prog.coverage_skipped;
      }
      sh.sampler_type.fill(hlsl::ParamType::Void);
      for (const hlsl::Sampler& smp : prog.samplers) {
        if (smp.reg >= 16) FatalF("gfx: shader sampler register %u", smp.reg);
        sh.sampler_type[smp.reg] = smp.type;
        sh.samplers = std::max(sh.samplers, smp.reg + 1);
      }
      bool vertex = prog.stage == hlsl::Stage::Vertex;
      std::string glsl = "#version 450\n" + Prelude(prog.stage) + (capture ? MetalCaptureDefines() : "") + prog.glsl;
      // A translation with calls of r_rt_reflect or r_rt_shadow (hlsl::
      // RayReflectionCall, RayShadowCall):
      // its MSL given ray queries (rt_forward_reflection.h) each time it is
      // made, from the plain MSL, which is what is cached (a cache never
      // holds the scene's MSL of another build).
      const bool forward =
          !vertex && (prog.ray_reflection_sampler >= 0 || prog.ray_shadow_sampler >= 0) && rt_forward_;
      // (A capture variant's MSL is made with other options: its own hash.)
      const uint64_t hash =
          Fnv64(glsl.data(), glsl.size()) ^ uint64_t(prog.stage) ^ (capture ? 0x6361707475726531ull : 0);
      // RECOMP_SHADER_DUMP=dir: each translated shader's GLSL, by hash.
      if (const char* dir = std::getenv("RECOMP_SHADER_DUMP")) {
        char path[1024];
        std::snprintf(path, sizeof path, "%s/%016llx.%s.glsl", dir, (unsigned long long)hash,
                      capture ? "capture" : vertex ? "vs" : "ps");
        if (FILE* f = std::fopen(path, "wx")) {
          std::fwrite(glsl.data(), 1, glsl.size(), f);
          std::fclose(f);
        }
      }
      sh.hash = hash;
      sh.ray_shadow_sampler = prog.ray_shadow_sampler;
      sh.varyings = hlsl::VaryingMask(prog);
      // Pixel shaders' for their variants; vertex shaders' (for their
      // capture variants) while the ray-traced scene is on.
      if (!vertex || (rt_ && !capture)) sh.program = std::make_shared<const hlsl::Program>(prog);
      sh.capture_none = capture;  // (no variant of a variant)
      // The MSL comes from earlier runs when it can (CompileShader); a
      // shader to be given ray queries has it now (translated here if not),
      // unless another of its translation is alive (whose answer it shares).
      std::shared_ptr<const std::string> msl = CachedMsl(hash);
      bool translated = false;
      // A shader of the same translation alive (and, if compiled, its function).
      ShaderId twin = 0;
      if (auto t = sid_by_hash_.find(hash); t != sid_by_hash_.end()) twin = t->second;
      if (forward && twin) {
        sh.rt_forward = shaders_.at(twin).rt_forward;
      } else if (forward) {
        if (!msl) {
          Compiled tr = TranslateShader(glsl, vertex, capture);
          if (!tr.error.empty()) Fail(tr);
          msl = tr.plain;
          translated = true;
          if (perf_) pc_.translate_ns += tr.translate_ns;  // (its MSL compile counted when taken over)
        }
        std::string probe = *msl;
        sh.rt_forward = RtForwardReflectionPatch(&probe);
        if (!sh.rt_forward) {
          fwd_plain_++;
          WarnF("gfx: ray-traced forward shaders: pixel shader %016llx: its MSL not in the expected shape\n",
                (unsigned long long)hash);
        }
      }
      // Calls in subroutines (functions before main0) stay the stub's.
      if (sh.rt_forward && prog.ray_reflection_stub_calls + prog.ray_shadow_stub_calls) {
        fwd_stub_calls_ += prog.ray_reflection_stub_calls + prog.ray_shadow_stub_calls;
        WarnF("gfx: ray-traced forward shaders: pixel shader %016llx: %u reads in subroutines: nothing hit (or lit) "
              "there\n",
              (unsigned long long)hash, prog.ray_reflection_stub_calls + prog.ray_shadow_stub_calls);
      }
      const bool shared = lazy_ && twin && shaders_.at(twin).fn;
      // Compiled now: a capture variant before it is registered (one that
      // does not compile is not).
      const bool now = !shared && (capture || !lazy_ || (recipe_by_shader_.count(hash) && lanes_.empty()));
      Compiled c;
      if (now) {
        c = CompileShader(dev_, glsl, vertex, capture, forward, msl, translated, false);
        if (capture && !c.error.empty())
          return CaptureFailed(hash, c.error_stage, c.error_log.c_str());
      }
      ShaderId sid = next_id_++;
      shaders_[sid] = sh;
      sid_by_hash_[sh.hash] = sid;
      if (shared) {
        const Shader& t = shaders_.at(twin);
        Give(sid, t.fn, t.has_lean, t.rt_forward, t.fwd_plain, t.fwd_plain_fn);
        PrebuildFor(sh.hash);
      } else if (now) {
        TakeOver(hash, c, true);
        Give(sid, c.fn, c.has_lean, c.patched, c.plain, c.plain_fn);
        PrebuildFor(sh.hash);
      } else if (waiters_.Add(hash, sid)) {
        // The first of its translation: a job (the others wait for it).
        id<MTLDevice> dev = dev_;
        shader_jobs_.Add(hash, [dev, glsl = std::move(glsl), vertex, forward, msl = std::move(msl),
                                translated](bool on_lane) {
          return CompileShader(dev, glsl, vertex, false, forward, msl, translated, on_lane);
        });
        if ((ahead && prefetching_) || recipe_by_shader_.count(hash)) Prefetch(hash);
      }
      if (perf_t0) {
        pc_.shaders++;
        pc_.shader_ns += NowNs() - perf_t0;
      }
      return sid;
    }
  }

  // ---- shaders compiled when first drawn (RECOMP_LAZY_SHADERS, on by
  // default). Games create most of their shaders long before drawing with
  // them (at loading, by the hundred), and not all are drawn with in a
  // session: compiling each in CreateShader (translation then Metal's
  // compile, tens of ms each when not cached) held the game's thread for
  // the whole of it. MakeShader registers a job per translation instead
  // (shader_jobs_, OnceJobs: run once, by whoever needs it first, by
  // Shader::hash; the shaders made from it, its waiters_); the draw that
  // first uses one runs it there and then on the game's thread
  // (EnsureShader), or waits for it if a shader lane is running it. From
  // the first presented frame on, the lanes compile the rest in the
  // background, in the order the game created them; what they finished
  // is taken over at each Present (AdoptFinished). A shader made while
  // one of its translation is compiled shares its function.
  //
  // Threads and locks: jobs are made, taken and dropped on the game's
  // thread under mu_ (and waiters_ is mu_'s); a lane takes only
  // shader_jobs_'s own lock (never mu_, so the game's thread may wait for
  // a lane while holding it), and a job touches nothing of the device's
  // but the MTLDevice. Everything a compile changes in the device
  // (Shader::fn, has_lean and rt_forward, the MSL cache, the lean variant,
  // the plain MSL of one with ray queries (fwd_src_), the encoding
  // thread's maps) is done on the game's thread (TakeOver, Give), before
  // any draw with the shader is recorded: the encoding thread sees a
  // shader only once it is whole.
  //
  // A job keeps its GLSL (and the MSL from earlier runs it shares with the
  // cache) until it runs: with RECOMP_SHADER_LANES=0, for the session if
  // the shader is never drawn.

  // A translated shader compiled: its function (and its lean variant's,
  // if a lane compiled it), the MSL compiled (with the ray queries and the
  // lean macros: for the lean variant) and the plain MSL (for the cache,
  // and for PlainPipeline), and what it took; or an error, reported where
  // it is taken over (a capture variant's: at `error_stage`,
  // CaptureFailed). Made on any thread.
  struct Compiled {
    id<MTLFunction> fn = nil, lean = nil;
    // With the ray queries (`patched`): the function of its plain MSL too,
    // which a draw that cannot wait for its pipeline draws with (StandIn);
    // nil if that could not be made.
    id<MTLFunction> plain_fn = nil;
    std::string msl;
    std::shared_ptr<const std::string> plain;
    std::shared_ptr<const std::string> from;  // the MSL from earlier runs it started from, if any
    bool has_lean = false;
    bool patched = false;      // given the ray queries (Shader::rt_forward)
    std::string patch_error;   // why the patched MSL did not compile (then plain)
    bool fresh = false;  // translated (the MSL not from an earlier run)
    uint64_t translate_ns = 0, library_ns = 0;
    std::string error, error_source;  // FatalF's message; the source printed before it
    const char* error_stage = "";     // GLSL, SPIR-V to MSL, MSL
    std::string error_log;
  };
  using ShaderJobs = OnceJobs<uint64_t, Compiled>;

  // GLSL to SPIR-V to MSL (`plain` of the result, fresh); any thread.
  static Compiled TranslateShader(const std::string& glsl, bool vertex, bool capture) {
    Compiled c;
    const uint64_t t = NowNs();
    std::vector<uint32_t> spirv;
    std::string log;
    if (!GlslToSpirv(glsl, vertex ? ShaderStage::Vertex : ShaderStage::Fragment, &spirv, &log)) {
      c.error_stage = "GLSL";
      c.error_log = log;
      c.error_source = glsl;
      c.error = "gfx: compiling a translated shader (source above): " + log;
      return c;
    }
    SpirvMslOptions mo;
    mo.disable_rasterization = capture;
    std::string msl;
    if (!SpirvToMsl(spirv, &msl, &log, mo)) {
      c.error_stage = "SPIR-V to MSL";
      c.error_log = log;
      c.error = "gfx: SPIR-V to MSL: " + log;
      return c;
    }
    c.plain = std::make_shared<const std::string>(std::move(msl));
    c.fresh = true;
    c.translate_ns = NowNs() - t;
    return c;
  }

  // GLSL to SPIR-V to MSL takes most of the time (Metal's own compile of
  // the MSL is cached by the system): `cached` is the plain MSL from
  // earlier runs, if any (`fresh`: translated in this run, not yet kept),
  // made again where it does not compile. A pixel shader given ray queries
  // (`forward`) has them put in its MSL here (RtForwardCompile: without
  // them where they cannot be put in or do not compile, the stub's
  // "nothing hit"). `lean`: the lean variant too (on a lane: else it is
  // made on AheadQueue, and the pipeline's first compile misses it). Any
  // thread: only `dev` of the device is used.
  static Compiled CompileShader(id<MTLDevice> dev, const std::string& glsl, bool vertex, bool capture,
                                bool forward, std::shared_ptr<const std::string> cached, bool fresh, bool lean) {
    @autoreleasepool {
      Compiled c;
      if (cached && !fresh) c.from = cached;
      c.plain = std::move(cached);
      c.fresh = c.plain && fresh;
      NSError* err = nil;
      id<MTLLibrary> lib = nil;
      std::string msl;
      uint64_t t = NowNs();
      auto lap = [&t](uint64_t* ns) {
        const uint64_t now = NowNs();
        *ns += now - t;
        t = now;
      };
      for (int pass = 0; pass < 2 && !lib; pass++) {
        if (pass == 1) {
          if (c.fresh) break;
          c.plain.reset();
        }
        if (!c.plain) {
          Compiled tr = TranslateShader(glsl, vertex, capture);
          if (!tr.error.empty()) return tr;
          c.plain = std::move(tr.plain);
          c.fresh = true;
          c.translate_ns += tr.translate_ns;
          t = NowNs();
        }
        if (vertex) {
          msl = *c.plain;
          c.has_lean = false;
          err = nil;
          lib = [dev newLibraryWithSource:@(msl.c_str()) options:nil error:&err];
        } else {
          RtForwardBuild b = RtForwardCompile(*c.plain, forward, CompileMsl, (__bridge void*)dev);
          c.patch_error = b.patch_error ? b.patch_error.localizedDescription.UTF8String : "";
          lib = b.lib;
          msl = std::move(b.msl);
          c.patched = b.patched;
          c.has_lean = b.lean;
          err = b.error;
        }
        lap(&c.library_ns);
      }
      if (!lib) {
        c.error_stage = "MSL";
        c.error_log = err.localizedDescription.UTF8String ?: "";
        c.error_source = msl;
        c.error = "gfx: compiling MSL (above): " + c.error_log;
        return c;
      }
      c.fn = [lib newFunctionWithName:@"main0"];
      if (!c.fn) {
        c.error_stage = "MSL";
        c.error_log = "no main0";
        c.error_source = msl;
        c.error = "gfx: the MSL above has no main0";
        return c;
      }
      if (!vertex && c.patched) {
        // (The alpha test's function constant made a macro, as every MSL
        // compiled here, if the plain text still has it: a function with
        // the constant left cannot make a pipeline.)
        std::string src = *c.plain;
        const bool macros = MetalLeanMacros(&src, false) || c.has_lean;
        NSError* perr = nil;
        id<MTLLibrary> pl = [dev newLibraryWithSource:@(src.c_str()) options:macros ? MslOptions(true) : nil error:&perr];
        c.plain_fn = [pl newFunctionWithName:@"main0"];
        lap(&c.library_ns);
      }
      if (lean && c.has_lean) {
        // As Lean does (of the MSL compiled: with the ray queries if given).
        id<MTLLibrary> ll = [dev newLibraryWithSource:@(msl.c_str()) options:MslOptions(false) error:&err];
        c.lean = [ll newFunctionWithName:@"main0"];
        lap(&c.library_ns);
      }
      c.msl = std::move(msl);
      return c;
    }
  }

  // A compile that failed (`c`), for a shader the game made: the end.
  [[noreturn]] static void Fail(const Compiled& c) {
    if (!c.error_source.empty()) std::fprintf(stderr, "%s\n", c.error_source.c_str());
    FatalF("%s", c.error.c_str());
  }

  // On the game's thread: what compiling the translation of `hash` (`c`;
  // `here`: on this thread) changes besides its shaders: the MSL cache
  // and the lean variant. A fatal error if it failed.
  void TakeOver(uint64_t hash, Compiled& c, bool here) {
    if (!c.error.empty()) Fail(c);
    if (!c.patch_error.empty())
      WarnF("gfx: ray-traced forward shaders: pixel shader %016llx: %s\n", (unsigned long long)hash,
            c.patch_error.c_str());
    if (perf_) {
      if (c.fresh) pc_.msl_compiles++;
      if (here) {
        pc_.translate_ns += c.translate_ns;
        pc_.library_ns += c.library_ns;
      } else {
        pc_.ahead_compiles++;
        pc_.ahead_ns += c.translate_ns + c.library_ns;
      }
    }
    // Translated afresh: kept for later runs (the plain MSL: the ray
    // queries are put in each time it is compiled), also over the earlier
    // runs' MSL it started from (which then no longer compiled: the last
    // record read wins), but not over one another compile has put there
    // since.
    if (c.fresh) {
      auto it = msl_cache_.find(hash);
      if (it == msl_cache_.end() || it->second == c.from) CacheMsl(hash, *c.plain);
    }
    if (c.lean) {
      std::lock_guard<std::mutex> l(lean_mu_);
      lean_.emplace(hash, c.lean);
    } else if (c.has_lean) {
      NSString* src = @(c.msl.c_str());
      std::atomic<bool>* closing = &closing_;
      Ahead(hash, ^{
        if (closing->load()) return;
        Lean(hash, src);
      });
    }
    c.msl = std::string();
  }

  // On the game's thread: shader `sid` given its function (`patched`: with
  // the ray queries, `plain` its MSL without), and handed to the encoding
  // thread. One expected to have the ray queries whose patched MSL did not
  // compile is drawn without them (and configured for none).
  void Give(ShaderId sid, id<MTLFunction> fn, bool has_lean, bool patched,
            const std::shared_ptr<const std::string>& plain, id<MTLFunction> plain_fn) {
    Shader& sh = shaders_.at(sid);
    sh.fn = fn;
    sh.has_lean = has_lean;
    if (sh.rt_forward && !patched) {
      sh.ray_forward = sh.ray_forward_shadow = false;
      fwd_plain_++;
    }
    sh.rt_forward = patched;
    if (patched) {
      // (Its plain MSL, for a pipeline the patched function cannot make:
      // PlainPipeline.)
      sh.fwd_plain = plain;
      sh.fwd_plain_fn = plain_fn;
      std::lock_guard<std::mutex> pl(fwd_src_mu_);
      fwd_src_[sid] = {*plain, has_lean};
    }
    // (Kept for the device's life, as the shader's own: Keep.)
    if (patched && plain_fn) Keep(plain_fn);
    __unsafe_unretained id<MTLFunction> pf = patched ? plain_fn : nil;
    Keep(sh.fn);
    __unsafe_unretained id<MTLFunction> f = sh.fn;
    const uint64_t hash = sh.hash;
    Rec([this, sid, f, hash, pf] {
      enc_shaders_[sid] = f;
      enc_shader_hash_[sid] = hash;
      // (StandIn's function, made with the shader: nothing compiled for it
      // by the draw that first needs it but its pipeline.)
      if (pf) standin_fn_[sid] = pf;
    });
  }

  // The job of translation `hash` done (`c`): every shader waiting for it
  // given its function, the job dropped.
  void Finish(uint64_t hash, Compiled& c, bool here) {
    TakeOver(hash, c, here);
    for (ShaderId sid : waiters_.Take(hash)) Give(sid, c.fn, c.has_lean, c.patched, c.plain, c.plain_fn);
    shader_jobs_.Remove(hash);
    // The recipes it completes (one an earlier run's recipe names).
    PrebuildFor(hash);
  }

  // Before a draw with shader `sid` is recorded: it compiled, if it is
  // not yet (its job run here, or waited for on the lane running it).
  void EnsureShader(ShaderId sid) {
    if (shaders_.at(sid).fn) return;
    TakeOverFinished();  // (this one's, if a lane has just done it)
    const Shader& sh = shaders_.at(sid);
    if (sh.fn) return;
    const uint64_t hash = sh.hash;
    const uint64_t t0 = NowNs();
    ShaderJobs::How how;
    std::shared_ptr<Compiled> c = shader_jobs_.Get(hash, &how);
    if (!c) FatalF("gfx: shader %u has neither a function nor a job", sid);
    Finish(hash, *c, how == ShaderJobs::How::RanHere);
    if (perf_) {
      const uint64_t ns = NowNs() - t0;
      pc_.shader_ns += ns;
      if (how == ShaderJobs::How::RanHere) pc_.first_drawn++;
      if (how == ShaderJobs::How::Joined) {
        pc_.joins++;
        pc_.join_ns += ns;
      }
    }
  }

  // At Present: the shaders the lanes have compiled taken over (so their
  // first draws find them ready); the first time, the lanes started on
  // all the shaders not compiled yet, in the order they were made.
  void AdoptFinished() {
    if (!lazy_) return;
    TakeOverFinished();
    if (prefetching_ || lanes_.empty()) return;
    prefetching_ = true;
    for (uint64_t hash : waiters_.KeysInOrder()) Prefetch(hash);
  }

  // The shaders the lanes have compiled taken over: at Present, and when a
  // shader is made or first drawn (a load presents seldom, and its
  // recipes' pipelines are built ahead as each pair is done, not all in
  // the first frame). A lock and an empty list when there are none.
  void TakeOverFinished() {
    if (!lazy_) return;
    const uint64_t t0 = perf_ ? NowNs() : 0;
    for (uint64_t hash : shader_jobs_.TakeFinished()) {
      if (!waiters_.Has(hash)) {
        shader_jobs_.Remove(hash);
        continue;
      }
      if (std::shared_ptr<Compiled> c = shader_jobs_.Get(hash)) Finish(hash, *c, false);
    }
    if (t0) pc_.shader_ns += NowNs() - t0;
  }

  // The job of translation `hash` queued on the next shader lane.
  void Prefetch(uint64_t hash) {
    dispatch_queue_t q = lanes_[next_lane_++ % lanes_.size()];
    shader_jobs_.Prefetch(hash, [q](std::function<void()> task) { dispatch_async(q, ^{ task(); }); });
  }

  // RECOMP_LAZY_SHADERS=0: every shader compiled in CreateShader, as
  // before. RECOMP_SHADER_LANES=0..8: the lanes compiling shaders ahead (0:
  // none, only when first drawn); by default 3 on macOS, at user-initiated
  // priority, and 1 on iOS and tvOS, which runs on AheadQueue: one Metal
  // compile at a time there, at background priority (an iPad's GPU
  // compiler and cores busy with several slowed the game for minutes).
  void SetUpShaderLanes() {
    const char* lz = std::getenv("RECOMP_LAZY_SHADERS");
    lazy_ = !(lz && *lz == '0');
    if (!lazy_) {
      std::fprintf(stderr, "gfx: shaders compiled when created (RECOMP_LAZY_SHADERS=0)\n");
      return;
    }
#if TARGET_OS_IOS || TARGET_OS_TV
    int n = 1;
#else
    int n = 3;
#endif
    if (const char* e = std::getenv("RECOMP_SHADER_LANES")) n = std::clamp(std::atoi(e), 0, 8);
    for (int i = 0; i < n; i++) {
#if TARGET_OS_IOS || TARGET_OS_TV
      // (Several still one at a time: all on AheadQueue.)
      lanes_.push_back(dispatch_queue_create_with_target("recomp.gfx.shaders", DISPATCH_QUEUE_SERIAL, AheadQueue()));
#else
      lanes_.push_back(dispatch_queue_create(
          "recomp.gfx.shaders",
          dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0)));
#endif
    }
  }

  void DestroyShader(ShaderId sid) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    auto it = shaders_.find(sid);
    if (it == shaders_.end() || sid >= kClearVs) return;
    const auto variants = std::move(it->second.variants);
    for (const auto& [mask, vid] : variants) DestroyShader(vid);  // `it` stays valid
    if (const ShaderId cap = it->second.capture) {
      Keep(it->second.capture_pipe);  // (recorded captures may still use it)
      DestroyShader(cap);
    }
    Keep(it->second.fn);
    // Not compiled yet: the last waiting for its translation drops its job.
    if (!it->second.fn && waiters_.Remove(it->second.hash, sid)) shader_jobs_.Remove(it->second.hash);
    auto h = sid_by_hash_.find(it->second.hash);
    if (h != sid_by_hash_.end() && h->second == sid) sid_by_hash_.erase(h);
    shaders_.erase(it);
    bound_.pipe_set = false;
    {
      std::lock_guard<std::mutex> pl(prebuilt_mu_);
      for (auto p = prebuilt_.begin(); p != prebuilt_.end();)
        p = p->first.vs == sid || p->first.ps == sid ? prebuilt_.erase(p) : std::next(p);
    }
    Rec([this, sid] {
      last_pipe_ = nil;
      for (auto p = pipelines_.begin(); p != pipelines_.end();) {
        // Work encoded so far holds what it uses: erased at once here.
        if (p->first.vs == sid || p->first.ps == sid)
          p = pipelines_.erase(p);
        else
          ++p;
      }
      for (auto p = standin_.begin(); p != standin_.end();) {
        if (p->first.vs == sid || p->first.ps == sid)
          p = standin_.erase(p);
        else
          ++p;
      }
      for (auto p = standin_started_.begin(); p != standin_started_.end();) {
        if (p->vs == sid || p->ps == sid)
          p = standin_started_.erase(p);
        else
          ++p;
      }
      standin_fn_.erase(sid);
      enc_shaders_.erase(sid);
      enc_shader_hash_.erase(sid);
    });
    std::lock_guard<std::mutex> fl(fwd_src_mu_);
    fwd_src_.erase(sid);
  }

  void DrawShader(const ShaderDraw& d) override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    for (uint32_t i = 0; i < d.state_count; i++)
      if (d.states[i][0] < rs_.size()) rs_[d.states[i][0]] = d.states[i][1];
    if (d.viewport) viewport_ = *d.viewport;
    if (d.scissor) scissor_ = *d.scissor;
    frame_log_counts_.draws++;
    if (!perf_) return RecordDraw(d);
    const uint64_t t0 = NowNs();
    RecordDraw(d);
    pc_.draws++;
    pc_.draw_ns += NowNs() - t0;
  }

  void RecordDraw(const ShaderDraw& d) {
    EnsureTarget();
    auto vit = shaders_.find(d.vs), pit = shaders_.find(d.ps);
    if (vit == shaders_.end() || pit == shaders_.end()) return;
    const Shader& vs = vit->second;
    const Shader& ps = pit->second;
    // The device's own lighting of the local lights (MlAt): the G-buffer
    // noted; before the first volume, lit; the volumes left out with Only.
    if (ps.light_volume_on) {
      MlNote(d, ps.light_volume);
      if (ps.light_volume.volume) {
        MlAt();
        if (ml_only_) {
          if (perf_) pc_.ml_skipped++;
          return;
        }
      }
    }
    // A pixel shader reading the traced shadow mask: the frame's for what
    // it reads, traced now if there is none yet (before its render pass
    // opens: the trace is compute work).
    rt_mask_ = ps.own_map         ? OwnMapDraw(ps)
               : !ps.ray_shadow   ? (ps.shadow_claimed ? shadow_masker_->White() : nil)
               : ps.shadow.local ? RtLightShadowFor(d, ps)
                                 : RtShadowFor(d, ps);
    // Likewise one reading the traced reflections.
    rt_refl_ = ps.ray_reflection ? RtReflectionFor(d, ps) : ps.reflection_claimed ? reflector_->Miss() : nil;
    // And one reading the ray-traced ambient occlusion.
    // (Or, in its place, the far light's bounce RtReflectionFor just traced.)
    rt_ao_ = !ps.ray_occlusion        ? nil
             : bounce_distance_ > 0.0f ? BounceFor(d, ps)
                                       : RtOcclusionFor(d, ps);
    // The local lights traced at once where those traces just ended the
    // render pass (the far light's mask or the reflections, over the
    // G-buffer's depth before the lights draw): no pass ended for them.
    if (!enc_open_ && (ps.ray_shadow && !ps.shadow.local))
      LightSetAt(FindTex(d.textures[ps.shadow.depth_sampler]));
    else if (!enc_open_ && ps.ray_reflection)
      LightSetAt(FindTex(d.textures[ps.reflection.depth_sampler]));
    // The lit scene kept for the next frame's reflections, before the
    // post-processing reads it.
    if (reflector_ && ps.denoise_input) ReflectionKeepAt(d, ps);
    // The lit scene's guides at the lighting, its denoise before the
    // post-processing reads it (compute work, as the traces).
    if (dn_on_ && (ps.denoise_guides || ps.denoise_input)) DenoiseAt(d, ps);
    ShaderId ps_id = (ps.varyings & ~vs.varyings) ? PixelShaderFor(d.ps, pit->second, vs.varyings) : d.ps;
    // Whether the draw needs the pass's depth buffer attached (BareDraw): it
    // tests or writes depth or stencil, or its pixel shader may have ray
    // queries. One that does not goes into a pass opened without the
    // buffer where there is one, with the pixel shader made for such a pass.
    const bool depth_draw = pass_.depth && (rs_[7] || (pass_.stencil && rs_[52]));
    const bool keeps_depth = depth_draw || (pass_.depth && !BareShader(ps_id));
    const bool bare = BareDraw(d.ps, keeps_depth);
    if (!pass_.depth || bare) ps_id = PixelShaderWithoutDepth(ps_id);
    // Compiled when first drawn (EnsureShader); of the pixel shader, only
    // the variant drawn (whose compile settles whether it has the ray
    // queries).
    const Shader& drawn_ps = ps_id == d.ps ? ps : shaders_.at(ps_id);
    if (!vs.fn) EnsureShader(d.vs);
    if (!drawn_ps.fn) EnsureShader(ps_id);
    // One tracing its reflections itself (the shader drawn has the ray
    // queries; the game took the one it named): what it traces with.
    rt_fwd_bind_ = rt_forward_ && drawn_ps.rt_forward;
    if (rt_fwd_bind_) RtForwardFor(d, vs, ps);
    const uint32_t count = d.primitives;
    MTLPrimitiveType type;
    uint32_t n;  // vertices or indices used
    bool tri = true;
    switch (d.prim) {
      case 1: type = MTLPrimitiveTypePoint; n = count; tri = false; break;
      case 2: type = MTLPrimitiveTypeLine; n = count * 2; tri = false; break;
      case 3: type = MTLPrimitiveTypeLineStrip; n = count + 1; tri = false; break;
      case 4: type = MTLPrimitiveTypeTriangle; n = count * 3; break;
      case 5: type = MTLPrimitiveTypeTriangleStrip; n = count + 2; break;
      case 6: type = MTLPrimitiveTypeTriangle; n = count + 2; break;  // fan, as a list
      default: return;
    }
    if (!count) return;
    WarnUnsupportedShaderState();
    MTLScissorRect sc;
    if (!DrawScissor(&sc)) return;
    // (BareDraw) A pass open without the depth buffer ends before a draw
    // that needs it; a pass this draw opens is opened as the draw goes, and
    // remembered by the pixel shader the game named.
    if (enc_open_ && pass_bare_ && !bare) BareSplit();
    const bool opens = !enc_open_;
    const bool cleared = (clear_flags_ & 1) != 0;
    bare_next_ = bare;
    open_ps_ = d.ps;
    const bool begun = BeginDraw();
    bare_next_ = false;
    open_ps_ = 0;
    if (!begun) return;
    if (keeps_depth) pass_depth_used_ = true;
    if (opens && !cleared && perf_ && ColorLoadCandidate(ps, tri, count, depth_draw)) pc_.color_candidates++;
    if (pass_.scene && !pass_.low) frame_scene_ = true;
    // Jittered for the temporal upscaler: the scene's geometry, with the
    // decals and posters on it (MetalJitteredDraw), not full-target quads.
    // The camera from those that write depth, once two with different
    // world matrices agree on it (CameraVote); the scene's depth that of
    // the first.
    const int pm = vs.position_matrix;
    const float(*matrix)[4] = pm >= 0 && pm <= 252 && d.vs_float ? d.vs_float + pm : nullptr;
    jitter_draw_ = Temporal() && pass_.scene && !pass_.low && MetalJitteredDraw(rs_[7], rs_[14], count, matrix);
    if (jitter_draw_ && rs_[14] && count > 4 && !have_vp_ && d.vs_float) {
      if (cam_vote_.Offer(d.vs_float + (rt_world_ >= 0 ? rt_world_ : 0), matrix ? matrix : d.vs_float + 8)) TakeCamera();
      if (!scene_depth_) scene_depth_ = pass_.depth;
    }

    // Pipeline: shaders, blending, write masks.
    PipeKey pk = PassKey();
    pk.vs = d.vs;
    pk.ps = ps_id;
    if (rs_[27]) {  // ALPHABLENDENABLE
      uint32_t s = rs_[19], dst = rs_[20];
      if (s == 12) { s = 5; dst = 6; }        // BOTHSRCALPHA
      else if (s == 13) { s = 6; dst = 5; }   // BOTHINVSRCALPHA
      pk.blend = 1;
      pk.src = uint8_t(BlendFactor(s, pass_.alpha));
      pk.dst = uint8_t(BlendFactor(dst, pass_.alpha));
      pk.op = uint8_t(BlendOp(rs_[171]));
      if (rs_[206]) {  // SEPARATEALPHABLENDENABLE
        pk.src_a = uint8_t(BlendFactor(rs_[207], pass_.alpha));
        pk.dst_a = uint8_t(BlendFactor(rs_[208], pass_.alpha));
        pk.op_a = uint8_t(BlendOp(rs_[209]));
      } else {
        pk.src_a = pk.src;
        pk.dst_a = pk.dst;
        pk.op_a = pk.op;
      }
    }
    pk.mask[0] = WriteMask(rs_[168]);
    for (int i = 1; i < 4; i++) pk.mask[i] = i < pass_.targets ? WriteMask(rs_[189 + i]) : 0;
    // ALPHATESTENABLE off, or ALPHAFUNC ALWAYS: the shader without it.
    // (Of the pixel shader drawn: a variant for unwritten varyings has its own.)
    const bool ps_lean = drawn_ps.has_lean;
    if (ps_lean && (!rs_[15] || rs_[25] == 8)) pk.lean |= kLeanPs;
    if (drawn_ps.rt_forward) pk.lean |= kRtForwardPs;
    BindPipeline(pk);

    // Depth and stencil.
    DepthKey dk{};
    dk.func = dk.sfunc = dk.bfunc = uint8_t(MTLCompareFunctionAlways);
    float bias = 0, slope = 0;
    clip_bias_ = 0;
    if (pass_.depth) {
      dk.test = rs_[7] ? 1 : 0;
      dk.write = (rs_[7] && rs_[14]) ? 1 : 0;
      dk.func = uint8_t(CompareOp(pass_.reversed ? MirrorDepthCompare(rs_[23]) : rs_[23]));
      if (pass_.stencil && rs_[52]) {
        dk.stencil = 1;
        dk.fail = uint8_t(StencilOp(rs_[53]));
        dk.zfail = uint8_t(StencilOp(rs_[54]));
        dk.pass = uint8_t(StencilOp(rs_[55]));
        dk.sfunc = uint8_t(CompareOp(rs_[56]));
        const bool two = rs_[185] != 0;  // TWOSIDEDSTENCILMODE: CCW_STENCIL* for back faces
        dk.bfail = two ? uint8_t(StencilOp(rs_[186])) : dk.fail;
        dk.bzfail = two ? uint8_t(StencilOp(rs_[187])) : dk.zfail;
        dk.bpass = two ? uint8_t(StencilOp(rs_[188])) : dk.pass;
        dk.bfunc = two ? uint8_t(CompareOp(rs_[189])) : dk.sfunc;
        dk.read = uint8_t(rs_[58]);
        dk.wmask = uint8_t(rs_[59]);
      }
      // DEPTHBIAS (less the legacy ZBIAS) and SLOPESCALEDEPTHBIAS; on a
      // reversed buffer the constant part from the vertex shader
      // (reversed_z.h).
      const DepthBias db = DepthBiasOf(AsFloat(rs_[195]) - float(rs_[47]) * (1.0f / 65535.0f), AsFloat(rs_[175]),
                                       pass_.reversed, viewport_.min_z, viewport_.max_z, pass_.bias_scale);
      bias = db.constant;
      slope = db.slope;
      clip_bias_ = db.clip;
    }
    depth_used_ = dk.test || dk.write;
    BindDepth(dk);
    if (dk.write && tri && pass_.w == bb_w_ && pass_.h == bb_h_) frame_depth_draws_++;

    // Rasterizer: D3D culls by screen winding, CW (2) or CCW (3); front
    // faces are D3D's (clockwise) whatever is culled, for VFACE and
    // two-sided stencil.
    MTLCullMode cull = MTLCullModeNone;
    if (tri && rs_[22] == 2) cull = MTLCullModeFront;
    else if (tri && rs_[22] == 3) cull = MTLCullModeBack;
    Raster(cull, rs_[8] != 3 ? MTLTriangleFillModeLines : MTLTriangleFillModeFill, bias, slope);
    const DepthRange zr = pass_.reversed ? ReversedViewportDepth(viewport_.min_z, viewport_.max_z)
                                         : DepthRange{viewport_.min_z, viewport_.max_z};
    ViewportOf({double(viewport_.x), double(viewport_.y), double(std::max(viewport_.width, 1u)),
                double(std::max(viewport_.height, 1u)), double(zr.min_z), double(zr.max_z)});
    ScissorOf(sc);
    StencilRef(uint8_t(rs_[57]));
    float blend[4];
    Argb(blend, rs_[193]);
    BlendColor(blend);
    // Vertices: read in place if every stream used is in shared memory
    // (the draw's base vertex then Metal's), else each used stream's range
    // copied once, all into one chunk.
    const bool indexed = d.indices != nullptr;
    bool in_place = shared_buf_ != nil;
    for (size_t i = 0; i < vs.inputs && in_place; i++) {
      VertexInput in = i < d.input_count ? d.inputs[i] : VertexInput{};
      if (in.type >= 17 || in.stream >= 16 || !d.streams[in.stream].data) continue;
      const VertexStream& vsm = d.streams[in.stream];
      in_place = vsm.data >= shared_lo_ && vsm.data + vsm.size <= shared_hi_;
    }
    // (For the perf line and the frame log's: 1 draw in 32 timed, in mach
    // ticks, the report scaling them by the calls.)
    const bool perf = perf_;
    const bool timed = rt_ && (perf || frame_log_) && (++rt_timed_tick_ & 31) == 0;
    const uint64_t note_t0 = timed ? mach_absolute_time() : 0;
    const RtCapture cap = rt_ ? RtNoteDraw(d, vs, ps, tri, in_place) : RtCapture{};
    if (rt_ && (perf || frame_log_)) {
      const uint64_t ticks = timed ? mach_absolute_time() - note_t0 : 0;
      if (perf) {
        pc_.rt_note_calls++;
        pc_.rt_note_ticks += ticks;
        pc_.rt_note_timed += timed;
      }
      if (frame_log_) {
        frame_log_counts_.note_calls++;
        frame_log_counts_.note_ticks += ticks;
        frame_log_counts_.note_timed += timed;
      }
    }
    if (shadow_masker_) RtNoteWrites(dk.write, pk.mask[0] || pk.mask[1] || pk.mask[2] || pk.mask[3], false);
    if (in_place) {
      DrawInPlace(d, vs, type, n, count);
      if (cap.vs) RtCaptureTimed(cap, true, d.base_vertex, timed);
      AfterDraw();
      return;
    }
    const int64_t first = int64_t(d.base_vertex) + (indexed ? d.min_index : 0);
    const uint32_t nverts = indexed ? d.vertex_count : n;
    VsState st{};
    struct Copy {
      int64_t start, end;
      size_t pos;
    };
    Copy copies[16];
    bool used[16] = {};
    size_t total = 0;
    for (size_t i = 0; i < vs.inputs; i++) {
      VertexInput in = i < d.input_count ? d.inputs[i] : VertexInput{};
      if (in.type >= 17 || in.stream >= 16 || !d.streams[in.stream].data || used[in.stream])
        continue;
      const VertexStream& vsm = d.streams[in.stream];
      Copy& c = copies[in.stream];
      c.start = first * vsm.stride;
      // (A stride of 0, an instance's data: one element, of unknown size.)
      const size_t span = vsm.stride ? size_t(nverts) * vsm.stride : std::min<size_t>(vsm.size, 256);
      c.end = std::min<int64_t>(c.start + int64_t(span), vsm.size);
      c.pos = total;
      // Room for a whole last vertex and the shader's unaligned reads.
      total += (span + 8 + 3) & ~size_t(3);
      used[in.stream] = true;
    }
    Slice vb = Alloc(std::max<size_t>(total, 16), 4);
    for (int s = 0; s < 16; s++) {
      if (!used[s]) continue;
      const Copy& c = copies[s];
      if (c.start >= 0 && c.end > c.start)
        std::memcpy(vb.cpu + c.pos, d.streams[s].data + c.start, size_t(c.end - c.start));
    }
    for (size_t i = 0; i < vs.inputs; i++) {
      VertexInput in = i < d.input_count ? d.inputs[i] : VertexInput{};
      if (in.type >= 17 || in.stream >= 16 || !used[in.stream]) {
        st.in[i][2] = -1;
        continue;
      }
      const VertexStream& vsm = d.streams[in.stream];
      st.in[i][0] = int32_t(vb.off + copies[in.stream].pos) -
                    int32_t(indexed ? d.min_index : 0) * int32_t(vsm.stride) + int32_t(in.offset);
      st.in[i][1] = int32_t(vsm.stride);
      st.in[i][2] = int32_t(in.type);
    }
    PixelState(st.pixel);
    ClipState(st, d, vs);
    VertexBuffer(3, vb.buf, 0);

    ShaderResources(d, vs, ps, st);

    // The draw.
    if (indexed && d.prim != 6) {
      const size_t bytes = size_t(n) * d.index_size;
      Slice ib = Alloc(bytes, 4);
      std::memcpy(ib.cpu, d.indices, bytes);
      const MTLIndexType it = d.index_size == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
      Rec([this, type, n, it, ib] {
        GpDraw(n);
        [enc_ drawIndexedPrimitives:type
                         indexCount:n
                          indexType:it
                        indexBuffer:ib.buf
                  indexBufferOffset:ib.off];
      });
    } else if (d.prim == 6) {
      Slice ib = FanIndices(d, count);
      Rec([this, type, count, ib] {
        GpDraw(count * 3);
        [enc_ drawIndexedPrimitives:type
                         indexCount:count * 3
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:ib.buf
                  indexBufferOffset:ib.off];
      });
    } else {
      Rec([this, type, n] {
        GpDraw(n);
        [enc_ drawPrimitives:type vertexStart:0 vertexCount:n];
      });
    }
    if (cap.vs) RtCaptureTimed(cap, false, d.base_vertex, timed);
    AfterDraw();
  }

 private:
  struct Shader;  // (below)
  struct Tex;     // (below)

  // ---- the scene at an internal resolution, upscaled (RECOMP_UPSCALER)

  // Denoise: MetalFX's denoiser on the lit scene before post-processing
  // (rt_denoise.h, SetRayTracing's StartDenoiser): the scene jittered at
  // the back buffer's size, as TAA's, and nothing done after the
  // composite.
  enum class Upscaler { Off, Bilinear, MetalFX, MetalFXSpatial, Taa, Smaa, Denoise };
  // Temporal: the scene is jittered and resolved with its history.
  bool Temporal() const {
    return upscaler_ == Upscaler::MetalFX || upscaler_ == Upscaler::Taa || upscaler_ == Upscaler::Denoise;
  }
  // The scene at the back buffer's size, its composite the back buffer
  // itself (TAA's, the denoiser's).
  bool FullSizeScene() const { return upscaler_ == Upscaler::Taa || upscaler_ == Upscaler::Denoise; }

  // A depth buffer of format `pf` stores 1 - d (reversed_z.h): float ones,
  // unless RECOMP_REVERSED_Z=0.
  bool ReversedDepth(MTLPixelFormat pf) const {
    return reversed_z_ && (pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth32Float_Stencil8);
  }


  // The size a game's render target or depth buffer of w x h is made at
  // (target_size.h): the back buffer's and its halves and quarters (the
  // scene's buffers and their downsamples) at the internal resolution,
  // RECOMP_RT_BOOST's sizes larger unless the port wants this one exact.
  // False for the others. It is called on the thread that creates the
  // target, whose scope decides.
  bool ScaledSize(int w, int h, int* rw, int* rh, bool* scene = nullptr) const {
    TargetSizing s;
    // SMAA works on the finished frame (Present): the scene is drawn as
    // with no upscaler.
    s.internal = upscaler_ != Upscaler::Off && upscaler_ != Upscaler::Smaa;
    s.taa = upscaler_ == Upscaler::Taa;
    s.bb_w = bb_w_;
    s.bb_h = bb_h_;
    s.render_scale = render_scale_;
    s.boost = rt_boost_;
    return TargetSize(s, w, h, rw, rh, scene);
  }

  // The composite's target: the back buffer at the internal resolution.
  void EnsureLowColor() {
    const NSUInteger w = NSUInteger(std::max(1L, std::lround(bb_w_ * render_scale_)));
    const NSUInteger h = NSUInteger(std::max(1L, std::lround(bb_h_ * render_scale_)));
    if (low_color_ && low_color_.width == w && low_color_.height == h) return;
    if (low_color_) Keep(low_color_);
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    low_color_ = [dev_ newTextureWithDescriptor:td];
    fx_reset_ = true;
  }

  // Pixel size and jitter for the vertex shaders' r_pixel.
  void PixelState(float* px) const {
    const float w = float(std::max(viewport_.width, 1u)) * pass_.sx;
    const float h = float(std::max(viewport_.height, 1u)) * pass_.sy;
    px[0] = 1.0f / w;
    px[1] = 1.0f / h;
    px[2] = jitter_draw_ ? 2.0f * jitter_[0] / w : 0.0f;
    px[3] = jitter_draw_ ? 2.0f * jitter_[1] / h : 0.0f;
  }

  // The vertex state's user clip planes (D3D9 with shaders: clip space),
  // how depth is made (r_position): reversed, its bias, and from w where
  // the draw qualifies (reversed_z.h); and the vertex samplers that read a
  // reversed depth texture (R_DEPTH_TEX).
  void ClipState(VsState& st, const ShaderDraw& d, const Shader& vs) {
    st.clip_mask[0] = int32_t(rs_[152] & 0x3F);
    if (st.clip_mask[0]) std::memcpy(st.clip, clip_planes_, sizeof st.clip);
    for (uint32_t r = 0; r < vs.samplers && r < 4; r++)
      if (const Tex* t = FindTex(d.textures[16 + r]); t && t->depth && ReversedDepth(t->tex.pixelFormat))
        st.clip_mask[2] |= int32_t(1u << r);
    st.clip_mask[1] = pass_.reversed ? 1 : 0;
    st.depth[2] = clip_bias_;
    if (!pass_.reversed || wvp_reg_ < 0 || !d.vs_float) return;
    if (vs.position_matrix != wvp_reg_) {
      NoteNotFromW(vs);
      return;
    }
    WProjection p;
    if (!PerspectiveOf(d.vs_float + wvp_reg_, &p)) return;
    auto dit = depths_.find(pass_.depth_id);
    if (dit == depths_.end()) return;
    WDepth& wd = dit->second.wd;
    // The first after a clear: the game's planes, where it knows them.
    WProjection exact;
    float n, f;
    const bool fresh = wd.fresh && depth_used_;
    const bool planes = fresh && depth_planes_ && depth_planes_(&n, &f) && n > 0 && f > n;
    if (planes) exact = PerspectiveOfPlanes(n, f);
    if (!wd.Use(p, planes ? &exact : nullptr, depth_used_)) {
      // Tests or writes depth, with another projection than the content's.
      if (perf_ && depth_used_) pc_.w_other++;
      return;
    }
    // The first few contents' projections, for the log.
    if (fresh && w_logged_ < 4) {
      w_logged_++;
      std::fprintf(stderr, "gfx: depth from w: a %.9g, n %.6g (%s)\n", wd.a, wd.n,
                   !planes ? "the matrix's" : wd.n == exact.n ? "the game's" : "the matrix's: the game's disagree");
    }
    st.clip_mask[1] = 2;
    st.depth[0] = wd.K0();
    st.depth[1] = wd.K1();
    if (perf_) pc_.w_draws++;
  }

  // A vertex shader drawn into a reversed buffer, testing or writing
  // depth, that reads the world-view-projection's registers but does not
  // qualify for depth from w: once per shader, why (the first 256).
  void NoteNotFromW(const Shader& vs) {
    if (!depth_used_ || !vs.hash || w_noted_.size() > 256) return;
    bool reads = false;
    for (int k = 0; k < 4; k++) reads = reads || vs.float_reads.test(size_t(wvp_reg_ + k));
    if (!reads || !w_noted_.insert(vs.hash).second) return;
    if (w_noted_.size() > 256) {
      std::fprintf(stderr, "gfx: depth from w: more vertex shaders kept w - z, not logged\n");
      return;
    }
    char other[64] = "";
    if (vs.position_matrix >= 0)
      std::snprintf(other, sizeof other, "its position is from c%d-c%d", vs.position_matrix, vs.position_matrix + 3);
    std::fprintf(stderr, "gfx: depth from w: vertex shader %016llx reads c%d-c%d, keeps w - z: %s\n",
                 (unsigned long long)vs.hash, wvp_reg_, wvp_reg_ + 3,
                 *other ? other : vs.position_note.empty() ? "(no analysis)" : vs.position_note.c_str());
  }


  // The frame's camera (world to clip space, D3D's row vectors) from the
  // vote (cam_vote_: scene draws' world register, GameConfig::
  // world_register, else c0, and the matrix their position goes through,
  // else c8), once it has one.
  void TakeCamera() {
    std::memcpy(vp_cur_, cam_vote_.camera(), sizeof vp_cur_);
    have_vp_ = true;
  }
  // Whether the frame has a camera now: the vote's, else (needed before
  // two draws agreed) its likeliest candidate, unconfirmed.
  bool HaveCamera() {
    if (!have_vp_ && Temporal() && cam_vote_.Settle() != CameraVote::State::None) TakeCamera();
    return have_vp_;
  }

  // Near and far planes of a perspective world-to-clip matrix (row vectors,
  // depth 0..1 at near..far, w = view depth). Column 2 (z) is a * column 3
  // (w) + b * (0, 0, 0, 1) with a = f / (f - n), b = -n a. False if it is
  // not one.
  static bool NearFar(const double* vp, double* near_plane, double* far_plane) {
    auto at = [&](int r, int c) { return vp[r * 4 + c]; };
    int r = 0;
    for (int i = 1; i < 3; i++)
      if (std::fabs(at(i, 3)) > std::fabs(at(r, 3))) r = i;
    if (!(std::fabs(at(r, 3)) > 1e-9)) return false;
    const double a = at(r, 2) / at(r, 3), b = at(3, 2) - a * at(3, 3);
    if (!(std::fabs(a) > 1e-9) || !(std::fabs(1.0 - a) > 1e-12)) return false;
    *near_plane = -b / a;
    *far_plane = b / (1.0 - a);
    return *near_plane > 0 && *far_plane > *near_plane;
  }

  // A cut: m (this frame's clip space to the last frame's, row vectors)
  // moves points of the screen at depth value z by more than a third of
  // it, or behind the last camera.
  static bool CameraCut(const double* m, double z) {
    static constexpr double kPts[5][2] = {{0, 0}, {-0.8, -0.8}, {0.8, -0.8}, {-0.8, 0.8}, {0.8, 0.8}};
    for (const auto& q : kPts) {
      const double v[4] = {q[0], q[1], z, 1.0};
      double o[4] = {};
      for (int k = 0; k < 4; k++)
        for (int i = 0; i < 4; i++) o[k] += v[i] * m[i * 4 + k];
      if (!(o[3] > 1e-9)) return true;
      const double dx = o[0] / o[3] - q[0], dy = o[1] / o[3] - q[1];
      if (dx * dx + dy * dy > (2.0 / 3.0) * (2.0 / 3.0)) return true;
    }
    return false;
  }

  static float Halton(int i, int b) {
    float f = 1, r = 0;
    for (; i > 0; i /= b) {
      f /= float(b);
      r += f * float(i % b);
    }
    return r;
  }

  // After each draw: the composite's done once it has covered the whole
  // back buffer.
  void AfterDraw() {
    if (pass_.low && viewport_.x == 0 && viewport_.y == 0 && int(viewport_.width) >= bb_w_ &&
        int(viewport_.height) >= bb_h_)
      Composite();
  }

  // ---- ray tracing (SetRayTracing, gfx/rt_scene.h)

  // A draw to capture (RtNoteDraw): its vertex shader's capture variant,
  // where it writes, and the vertices (base vertex included).
  struct RtCapture {
    ShaderId vs = 0;  // 0: none
    __unsafe_unretained id<MTLRenderPipelineState> pipe = nil;  // the variant's (Shader::capture_pipe)
    RtCaptureSlot slot;
    uint32_t first = 0, count = 0;
  };

  // What makes a draw alpha-cut in the scene (RtAlphaFor: `on`): its
  // material and texture, the vertex inputs read for its UVs and alpha
  // factor, and their stream (the UVs', or without UVs the alpha's). With
  // hit data, its albedo (RtAlbedoFor: albedo_on), read at those UVs (the
  // albedo's own where the draw is not cut, or cut without UVs).
  struct RtCutDraw {
    bool on = false;
    RtCut cut;
    RtAttribute uv, alpha;
    VertexInput uv_in, alpha_in;
    const VertexStream* stream = nullptr;
    bool albedo_on = false;
    RtAlbedo albedo;
  };

  // A draw being recorded: into the frame's ray-traced scene if it is
  // geometry of the main view (a target of the back buffer's size, depth
  // tested and written) or a static mesh of another pass that tests and
  // writes depth (RtView::Other, rt_other_passes_), not blended, else
  // counted under why not;
  // discarding pixels by alpha (the alpha test, the pixel shader's
  // texkill), alpha-cut where that is resolved (RtAlphaFor). Static meshes are added as they are
  // (RtNoteStatic); draws whose positions only the vertex shader makes are
  // captured (RtCaptureFor): RecordDraw then runs the capture after the
  // draw.
  // `in_place`: its vertices are read where they are (not copied).
  RtCapture RtNoteDraw(const ShaderDraw& d, const Shader& vs, const Shader& ps, bool tri, bool in_place) {
    RtScene* rt = rt_.get();
    RtSkip why = RtSkip::Count;
    const bool tested = rs_[15] && rs_[25] != 8;  // ALPHATESTENABLE, not ALWAYS
    RtCutDraw cut;
    // Positions only the vertex shader makes: blend inputs (skinning), a
    // vertex texture (displacement), a draw per instance (placed by the
    // instance's data). Such shaders need not read the world registers.
    bool vertex_texture = false;
    for (uint32_t r = 0; r < vs.samplers && r < 4; r++) vertex_texture = vertex_texture || d.textures[16 + r] != 0;
    const bool captured = vs.blend || vertex_texture || d.instanced;
    RtDrawPass dp;
    dp.depth = pass_.depth != nil;
    dp.test = rs_[7] != 0;    // ZENABLE
    dp.write = rs_[14] != 0;  // ZWRITEENABLE
    dp.color = (rs_[168] & 0xF) != 0;
    dp.main_size = RtMainSize(pass_.w, pass_.h, bb_w_, bb_h_);  // (the game's size: scaled targets keep it)
    dp.depth_color = pass_.targets > 0;
    for (int i = 0; i < pass_.targets; i++)
      dp.depth_color =
          dp.depth_color && (pass_.fmt[i] == MTLPixelFormatR32Float || pass_.fmt[i] == MTLPixelFormatR16Float);
    const RtView view = RtViewOf(dp, captured, rt_other_passes_, rt_other_color_);
    RtAlphaResult alpha;
    const bool asked = tested || ps.discards;
    // What the alpha and the albedo read of the shaders and textures, made
    // once for both (where the shaders were analyzed: made before ray
    // tracing was on, neither is resolved).
    const bool analyzed = vs.analyzed && ps.analyzed;
    RtAlphaShaders sh;
    RtAlphaTexture tex[16];
    if (analyzed && tri && view != RtView::None && !rs_[27] && (asked || rt_hit_data_))
      RtShadersOf(d, vs, ps, &sh, tex);
    if (!tri)
      why = RtSkip::NotTriangles;
    else if (view == RtView::None)
      why = RtSkip::NotScene;
    else if (rs_[27])
      why = RtSkip::Blended;
    else if (asked && !(analyzed && RtAlphaFor(d, ps, sh, tex, &cut, &alpha)))
      why = RtSkip::AlphaSource;  // ALPHAFUNC NEVER, or not resolved
    if (why != RtSkip::Count) {
      rt->Skip(why);
      return {};
    }
    if (rt_hit_data_ && analyzed) RtAlbedoFor(d, ps, sh, tex, asked ? &alpha : nullptr, &cut);
    const RtCutDraw* c = cut.on || cut.albedo_on ? &cut : nullptr;
    // (Decided before the matrices; captured draws are the main view's.)
    if (captured) return RtCaptureFor(d, vs, in_place, c);
    RtNoteStatic(d, vs, c, view);
    return {};
  }

  // Where an alpha-tested (ALPHATESTENABLE, `tested`) or discarding draw's
  // alpha comes from (gfx/rt_alpha.h, RtResolveAlpha), into `out` (on:
  // alpha-cut; off: added opaque). False if it is left out.
  // What gfx/rt_alpha.h reads of a draw's shaders and pixel textures.
  void RtShadersOf(const ShaderDraw& d, const Shader& vs, const Shader& ps, RtAlphaShaders* sh,
                   RtAlphaTexture tex[16]) {
    sh->pixel = ps.program ? &ps.program->pixel_alpha : nullptr;
    sh->discards = ps.discards;
    sh->sampler_type = ps.sampler_type;
    sh->moved = &vs.moved;
    sh->tex_slot = vs.tex_slot;
    sh->color0_slot = vs.color0_slot;
    for (uint32_t r = 0; r < ps.samplers && r < 16; r++)
      if (const Tex* t = FindTex(d.textures[r])) {
        tex[r].present = true;
        tex[r].flat = !t->cube && !t->depth && t->tex.textureType == MTLTextureType2D;
        tex[r].opaque = t->opaque;
        tex[r].unit = !IsFloat(t->fmt);
        tex[r].alpha_min = t->opaque ? 1.0f : !t->rt && t->alpha_whole ? float(t->alpha_lo) / 255.0f : 0.0f;
      }
  }

  // `sh` and `tex`: RtShadersOf's (the shaders analyzed).
  bool RtAlphaFor(const ShaderDraw& d, const Shader& ps, RtAlphaShaders& sh, const RtAlphaTexture tex[16],
                  RtCutDraw* out, RtAlphaResult* res) {
    *out = RtCutDraw{};
    sh.hook = rt_alpha_source_ && rt_alpha_source_(ps.hash, &sh.hook_sampler, &sh.hook_texcoord);
    const RtAlphaResult a = RtResolveAlpha(d, sh, tex, rs_[15] != 0, rs_[25], rs_[24]);  // ALPHATESTENABLE, FUNC, REF
    *res = a;
    switch (a.kind) {
      case RtAlphaKind::Skip: return false;
      case RtAlphaKind::Uncut: rt_->NoteUncut(); return true;
      case RtAlphaKind::Opaque: return true;
      case RtAlphaKind::Cut: break;
    }
    out->on = true;
    out->cut.material = a.material;
    out->cut.texture = a.sampler >= 0 ? FindTex(d.textures[a.sampler])->tex : nil;
    // The least value its texel takes: the texture's least alpha where the
    // alpha is the channel read (in 0..1: not a float format); none read, 1.
    if (a.sampler < 0) {
      out->cut.texel_min = 1.0f;
    } else if (((a.material.flags >> 24) & 3u) == 3u && tex[a.sampler].unit) {
      out->cut.texel_min = tex[a.sampler].alpha_min;
    }
    out->uv = a.uv;
    out->alpha = a.alpha;
    out->uv_in = a.uv_in;
    out->alpha_in = a.alpha_in;
    out->stream = a.stream >= 0 ? &d.streams[a.stream] : nullptr;
    return true;
  }

  // The albedo of a draw in a scene with hit data (gfx/rt_alpha.h,
  // RtResolveAlbedo; `alpha`: RtAlphaFor's result, null if not asked), into
  // `out` (albedo_on; the UVs where it has none). None where its UVs are
  // in another stream than an alpha-cut draw's alpha (a mesh's key names
  // one), or in a stream that is not a buffer's that stays (as the
  // positions must be).
  // `sh` and `tex`: RtShadersOf's (the shaders analyzed).
  void RtAlbedoFor(const ShaderDraw& d, const Shader& ps, RtAlphaShaders& sh, const RtAlphaTexture tex[16],
                   const RtAlphaResult* alpha, RtCutDraw* out) {
    sh.albedo_hook = rt_albedo_source_ && rt_albedo_source_(ps.hash, &sh.albedo_sampler, &sh.albedo_texcoord);
    const RtAlbedoResult a = RtResolveAlbedo(d, sh, tex, alpha);
    if (!a.on || a.stream < 0) return;
    // (UVs from the application's memory or a buffer rewritten: no albedo,
    // the draw stays in the scene without one.)
    const VertexStream& as = d.streams[a.stream];
    if (!as.version || as.dynamic) return;
    if (!out->on || !out->uv.data) {
      if (out->on && out->alpha.data && out->alpha_in.stream != a.uv_in.stream) return;
      out->uv = a.uv;
      out->uv_in = a.uv_in;
      out->stream = &d.streams[a.stream];
    }
    out->albedo_on = true;
    out->albedo = a.albedo;
    out->albedo.texture = FindTex(d.textures[a.sampler])->tex;
  }

  // RtIndexRange of a captured draw, kept by where its indices are
  // (RtIndexRangeMemo): a captured mesh (a skinned model, a moving part) is
  // drawn every frame from the same indices. They are a buffer's that stays
  // (RtCaptureFor asks before this), so its version names their contents.
  RtIndexRangeMemo rt_range_memo_;

  // A draw to capture: added to the scene (RtScene::AddCaptured) unless
  // its data is not from buffers (draws from the application's memory),
  // or from buffers the CPU rewrites (left out as static draws of them
  // are), or the game's filter leaves it out.
  RtCapture RtCaptureFor(const ShaderDraw& d, const Shader& vs, bool in_place, const RtCutDraw* cut) {
    RtScene* rt = rt_.get();
    RtSkip why = RtSkip::Count;
    const VertexInput in = vs.pos_slot >= 0 && uint32_t(vs.pos_slot) < d.input_count ? d.inputs[vs.pos_slot]
                                                                                      : VertexInput{};
    if (vs.pos_slot < 0 || uint32_t(vs.pos_slot) >= d.input_count || in.type >= 17 || in.stream >= 16 ||
        !d.streams[in.stream].data)
      why = RtSkip::NoPosition;
    else if (!d.streams[in.stream].version || (d.indices && !d.index_version))
      why = RtSkip::UserMemory;
    else if (d.streams[in.stream].dynamic || (d.indices && d.index_dynamic))
      why = RtSkip::Dynamic;
    else if (rt_filter_ && !rt_filter_(d.vs_float, d.primitives, true, false))
      why = RtSkip::Filtered;
    RtGeometry g;
    g.prim = d.prim;
    g.primitives = d.primitives;
    g.base_vertex = d.base_vertex;
    g.indices = d.indices;
    g.index_size = d.index_size;
    uint32_t first = 0, count = 0;
    if (why == RtSkip::Count) {
      // The vertices its triangles use; copied, within those the draw
      // declares (MinVertexIndex, NumVertices: all the copy has).
      const int64_t lo = int64_t(d.base_vertex) + d.min_index;
      if (!rt_range_memo_.Range(g, d.index_buffer, d.index_version, d.index_offset, &first, &count) ||
          (!in_place && d.indices && (int64_t(first) < lo || int64_t(first) + count > lo + d.vertex_count)))
        why = RtSkip::BadIndices;
    }
    ShaderId cvs = 0;
    Shader& mvs = shaders_.find(d.vs)->second;
    if (why == RtSkip::Count && !(cvs = CaptureShaderFor(mvs))) why = RtSkip::NoCapture;
    if (why != RtSkip::Count) {
      rt->SkipCapture(why);
      return {};
    }
    // What its triangles are made from: its positions' stream (not its
    // contents' version: another mesh may share the index buffer, its
    // triangles elsewhere) and indices from buffers that stay, as the
    // filter above asks (the scene refits the BLAS an earlier frame built
    // of the same); and its UVs where they are in a buffer that stays too
    // (the scene keeps what it makes of them on the CPU for the next
    // frames' draws of the same).
    RtMeshKey k;
    const VertexStream& pos = d.streams[in.stream];
    k.vertex_buffer = pos.buffer;
    k.vertex_offset = pos.buffer_offset;
    k.offset = in.offset;
    k.stride = pos.stride;
    k.type = uint16_t(in.type);
    k.index_version = d.indices ? d.index_version : 0;
    k.index_buffer = d.indices ? d.index_buffer : 0;
    k.index_offset = d.indices ? d.index_offset : 0;
    k.index_size = d.indices ? uint16_t(d.index_size) : 0;
    k.base_vertex = d.base_vertex;
    k.min_index = first;
    k.vertex_count = count;
    k.prim = d.prim;
    k.primitives = d.primitives;
    const RtMeshKey* key = !d.indices || (d.index_version && !d.index_dynamic) ? &k : nullptr;
    if (cut) {
      g.uv = cut->uv;
      g.alpha = cut->alpha;
      const VertexStream* as = cut->stream;
      if (key && as && as->version && !as->dynamic) RtAttrKey(*cut, &k);
    }
    const uint64_t actor_id = RtActorOf(d, vs);
    const RtCaptureSlot slot = rt->AddCaptured(g, first, count, rs_[22], cut && cut->on ? &cut->cut : nullptr,
                                               cut && cut->albedo_on ? &cut->albedo : nullptr, key,
                                               actor_id);  // CULLMODE
    if (!slot.buf) return {};
    return {cvs, mvs.capture_pipe, slot, first, count};
  }

  // The actor a captured draw is a part of (RtScene::AddCaptured), for a
  // skinned draw (not an instanced one: its instances differ by their
  // stream): the draws in a row with the same world matrix registers
  // (RayTracing::world_register) and vertex textures whose first bone
  // stays near the first's. A game sets the world matrix once for each
  // whole it draws (a person, a vehicle) even where its skinned shaders
  // place the vertices by the bones alone, and each part's palette holds
  // only that part's bones (GTA IV: the same c0-c3 over a person's three
  // or four parts and a vehicle's, other values from one whole to the
  // next; the bones differing between parts). Bone 0's translation (the
  // palette's first matrix, D3D's 4x3 columns: the .w of its three
  // registers) within kActorReach of the actor's first part keeps a whole
  // left with another's matrix apart. A wrong guess either way costs only
  // speed: parts split stay a BLAS each, as before; wholes joined are
  // still where they are. 0 (none: a BLAS of its own) for the others.
  uint64_t RtActorOf(const ShaderDraw& d, const Shader& vs) {
    const int wr = rt_world_, base = vs.rel_const_base;
    if (!vs.blend || d.instanced || !d.vs_float || wr < 0 || wr > 252 || base < 0 || base > 253) return 0;
    auto mix = [](uint64_t h, uint64_t v) {
      h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
      h *= 0xff51afd7ed558ccdull;
      return h ^ (h >> 33);
    };
    uint64_t h = 0x84222325cbf29ce4ull;
    for (int r = wr; r < wr + 4; r++) {
      uint64_t w[2];
      std::memcpy(w, d.vs_float[r], sizeof w);
      h = mix(mix(h, w[0]), w[1]);
    }
    for (int k = 0; k < 4; k++) h = mix(h, uint64_t(d.textures[16 + k]));
    const float t[3] = {d.vs_float[base][3], d.vs_float[base + 1][3], d.vs_float[base + 2][3]};
    float d2 = 0;
    for (int k = 0; k < 3; k++) d2 += (t[k] - actor_anchor_[k]) * (t[k] - actor_anchor_[k]);
    constexpr float kActorReach = 16.0f;  // world units: a bus's length and more
    if (h == actor_state_ && actor_id_ && d2 <= kActorReach * kActorReach) return actor_id_;
    actor_state_ = h;
    std::memcpy(actor_anchor_, t, sizeof t);
    actor_id_ = mix(h, ++actor_seq_) | 1;
    return actor_id_;
  }

  // The capture of the draw just recorded, in its render pass with its
  // bindings: its vertices once each (points, rasterization off), with
  // the draw's vertex numbering (its base vertex included where it is read
  // in place; a copy starts at it).
  void RtCaptureDraw(const RtCapture& c, bool in_place, int32_t base_vertex) {
    const uint32_t start = RtCaptureStart(c.first, base_vertex, in_place);
    const uint32_t count = c.count;
    __unsafe_unretained id<MTLBuffer> buf = c.slot.buf;
    __unsafe_unretained id<MTLRenderPipelineState> pipe = c.pipe;
    const size_t off = c.slot.capture_off;
    bound_.pipe_set = false;  // the next draw sets its own again
    Rec([this, pipe, buf, off, start, count] {
      const int32_t range[4] = {int32_t(start), int32_t(count), 0, 0};
      [enc_ setRenderPipelineState:pipe];
      [enc_ setVertexBuffer:buf offset:off atIndex:kMetalCaptureBuffer];
      [enc_ setVertexBytes:range length:sizeof range atIndex:kMetalCaptureRangeBuffer];
      [enc_ drawPrimitives:MTLPrimitiveTypePoint vertexStart:start vertexCount:count];
    });
  }

  // RtCaptureDraw, counted for the perf line and, for a draw RecordDraw
  // times (`timed`: 1 in 32), timed in mach ticks.
  void RtCaptureTimed(const RtCapture& c, bool in_place, int32_t base_vertex, bool timed) {
    const uint64_t t0 = timed ? mach_absolute_time() : 0;
    RtCaptureDraw(c, in_place, base_vertex);
    if (!perf_) return;  // (`timed` is set for the frame log alone too)
    pc_.rt_capture_calls++;
    if (timed) {
      pc_.rt_capture_ticks += mach_absolute_time() - t0;
      pc_.rt_capture_timed++;
    }
  }

  // The key fields of a draw's UVs and alpha (RtMeshKey's attr_*, uv_*,
  // alpha_*, cut), from `cut` (its stream set).
  static void RtAttrKey(const RtCutDraw& cut, RtMeshKey* k) {
    const VertexStream* as = cut.stream;
    k->cut = cut.on ? 1 : 0;
    k->attr_version = as->version;
    k->attr_buffer = as->buffer;
    k->attr_stream_offset = as->buffer_offset;
    k->attr_stride = as->stride;
    k->uv_type = cut.uv.data ? uint8_t(cut.uv_in.type) : 17;
    k->uv_offset = cut.uv.data ? cut.uv_in.offset : 0;
    k->uv_lanes = uint8_t(cut.uv.lane[0] | cut.uv.lane[1] << 2);
    k->alpha_type = cut.alpha.data ? uint8_t(cut.alpha_in.type) : 17;
    k->alpha_offset = cut.alpha.data ? cut.alpha_in.offset : 0;
    k->alpha_lane = cut.alpha.lane[0];
  }

  // A static mesh's draw (RtNoteDraw) of view `view`; alpha-cut, its UVs
  // and albedo with `cut`.
  void RtNoteStatic(const ShaderDraw& d, const Shader& vs, const RtCutDraw* cut, RtView view) {
    RtScene* rt = rt_.get();
    // The world matrix where the game keeps it, read by this shader (so set
    // for this draw), and the world-view-projection its position is made
    // with (found in its bytecode). Another view's draw is placed by its
    // world matrix alone: one whose position is made otherwise (GTA IV's
    // cascades: the world point times the light's matrix, its depth in
    // colour) is taken without it, its triangles not culled and their
    // winding the world matrix's.
    const bool by_world = view == RtView::Other && vs.position_matrix < 0;
    if (rt_world_ < 0 || (vs.position_matrix < 0 && !by_world) || !d.vs_float)
      return rt->Skip(RtSkip::NoMatrices);
    for (int k = 0; k < 4; k++)
      if (!vs.float_reads.test(size_t(rt_world_ + k))) return rt->Skip(RtSkip::NoMatrices);
    if (vs.pos_slot < 0 || uint32_t(vs.pos_slot) >= d.input_count) return rt->Skip(RtSkip::NoPosition);
    const VertexInput in = d.inputs[vs.pos_slot];
    if (in.type >= 17 || in.stream >= 16 || !d.streams[in.stream].data) return rt->Skip(RtSkip::NoPosition);
    const VertexStream& s = d.streams[in.stream];
    // (A stride of 0 elsewhere is a value every vertex shares.)
    if (!s.stride) return rt->Skip(RtSkip::ConstantStream);
    if (!(rt_types_seen_ & (1u << in.type))) {  // once per type: what the game's meshes use
      rt_types_seen_ |= 1u << in.type;
      std::fprintf(stderr, "gfx: ray tracing: positions of D3DDECLTYPE %u (stride %u)%s\n", in.type, s.stride,
                   RtPositionType(in.type) ? "" : ", not taken");
    }
    if (!RtPositionType(in.type)) return rt->Skip(RtSkip::PositionType);
    const VertexStream* as = cut ? cut->stream : nullptr;  // the UVs' and alpha's
    if (!s.version || (d.indices && !d.index_version) || (as && !as->version)) return rt->Skip(RtSkip::UserMemory);
    if (s.dynamic || (d.indices && d.index_dynamic) || (as && as->dynamic)) return rt->Skip(RtSkip::Dynamic);
    if (rt_filter_ && !rt_filter_(d.vs_float, d.primitives, false, view == RtView::Other))
      return rt->Skip(RtSkip::Filtered);
    RtMeshKey k;
    k.vertex_version = s.version;
    k.index_version = d.indices ? d.index_version : 0;
    k.vertex_buffer = s.buffer;
    k.index_buffer = d.indices ? d.index_buffer : 0;
    k.vertex_offset = s.buffer_offset;
    k.index_offset = d.indices ? d.index_offset : 0;
    k.base_vertex = d.base_vertex;
    k.min_index = d.min_index;
    k.vertex_count = d.vertex_count;
    k.prim = d.prim;
    k.primitives = d.primitives;
    k.offset = in.offset;
    k.stride = s.stride;
    k.type = uint16_t(in.type);
    k.index_size = d.indices ? uint16_t(d.index_size) : 0;
    if (cut) RtAttrKey(*cut, &k);
    RtGeometry g;
    g.positions = s.data + in.offset;
    g.size = s.size > in.offset ? s.size - in.offset : 0;
    g.stride = s.stride;
    g.type = in.type;
    g.prim = d.prim;
    g.primitives = d.primitives;
    g.base_vertex = d.base_vertex;
    g.indices = d.indices;
    g.index_size = d.index_size;
    if (cut) {
      g.uv = cut->uv;
      g.alpha = cut->alpha;
    }
    rt->Add(k, g, d.vs_float + rt_world_, d.vs_float + (by_world ? rt_world_ : vs.position_matrix),
            by_world ? 1 : rs_[22], rs_[23],  // CULLMODE (NONE: not known), ZFUNC
            cut && cut->on ? &cut->cut : nullptr, view, cut && cut->albedo_on ? &cut->albedo : nullptr);
  }

  // A TLAS of the presented frame so far (RtScene::Snapshot), for effects
  // traced before it ends; built in the frame's work recorded next. Null
  // if there is none. The encoders end; a whole-target clear still pending
  // stays so, the load action of the pass that opens next (no pass of its
  // own), unless it clears depth or stencil and the traced work reads the
  // pass's depth buffer (`reads`).
  // The effects traced from the same depth (`w`: its texture, contents and
  // cameras) in one command buffer share one (RtSnapshotReuse): the shadow
  // mask's and the reflections' TLAS is built once; `fresh`: a new one all
  // the same (the shadows' diagnostics summarise the one they trace).
  const RtFrame* RtSnapshot(__unsafe_unretained id<MTLTexture> reads, const RtShadowWant& w, bool fresh) {
    if (!rt_) return nullptr;
    end_why_ = PassWhy::Compute;
    if (RtShadowSettlesClear(clear_flags_, reads && reads == pass_.depth)) Settle();
    EndEncoders();
    EnsureFrame();
    if (fresh) snap_memo_ = {};
    if (RtSnapshotReuse(&snap_memo_, w, serial_)) {
      if (perf_) pc_.snapshots_shared++;
      return snap_frame_;
    }
    RtScene* rt = rt_.get();
    const uint64_t snap_t0 = perf_ || frame_log_ ? NowNs() : 0;  // (for the perf line and the frame log's)
    const RtFrame* f = rt->Snapshot();
    if (snap_t0) {
      const uint64_t ns = NowNs() - snap_t0;
      if (perf_) pc_.snapshot_ns += ns;
      if (frame_log_) frame_log_counts_.snapshot_ns += ns;
    }
    if (f) Rec([this, rt, f] { rt->EncodeFrame(cmd_, f); });
    if (perf_) pc_.snapshots++;
    snap_frame_ = f;
    snap_target_ = (__bridge const void*)pass_.color[0];
    return f;
  }

  // Whether a mask traced from `depth` takes the frame's jitter (the
  // temporal upscaler's: its points where the jittered scene drew the
  // depth), and if not why, for the perf counts. The first time the depth
  // is not the jittered scene's (while there is one), both said once.
  enum class RtJitter : uint8_t { Jittered, NotTemporal, NoCamera, OtherDepth, Count };
  static_assert(size_t(RtJitter::Count) == 4, "PerfCounters' shadow_jitter and light_jitter");
  RtJitter RtJitterFor(__unsafe_unretained id<MTLTexture> depth, bool light) {
    if (!Temporal()) return RtJitter::NotTemporal;
    if (!HaveCamera()) return RtJitter::NoCamera;
    if (depth == scene_depth_) return RtJitter::Jittered;
    bool& said = light ? jitter_other_said_light_ : jitter_other_said_;
    if (!said && depth) {
      said = true;
      __unsafe_unretained id<MTLTexture> sd = scene_depth_;
      WarnF("gfx: ray-traced %s shadows: traced without the jitter from a depth texture that is not the jittered "
            "scene's: %lux%lu format %lu against %lux%lu format %lu%s\n",
            light ? "local lights'" : "sun", (unsigned long)depth.width, (unsigned long)depth.height,
            (unsigned long)depth.pixelFormat, sd ? (unsigned long)sd.width : 0ul, sd ? (unsigned long)sd.height : 0ul,
            sd ? (unsigned long)sd.pixelFormat : 0ul, sd ? "" : " (none this frame)");
    }
    return RtJitter::OtherDepth;
  }

  // The G-buffer normals a shadow-mask draw names (RayShadow::
  // normal_sampler): a 2D colour texture there, else nil (normals from the
  // depth).
  id<MTLTexture> RtShadowNormals(const ShaderDraw& d, const RayShadow& s) {
    const Tex* t = s.normal_sampler < 16 ? FindTex(d.textures[s.normal_sampler]) : nullptr;
    return t && !t->cube && !t->depth && t->tex.textureType == MTLTextureType2D ? t->tex : nil;
  }

  // The shadow mask a draw of pixel shader `ps` (SetRayShadow) binds, as
  // RtShadowNext chooses: the one chosen last this presented frame for the
  // same depth texture, contents, cameras and light; one traced now from a
  // snapshot of the scene (into the frame slot's mask) where the depth is
  // the scene camera's main view (the jittered scene's: traced sparsely and
  // accumulated over frames, rt_accumulate.h); else all lit (counted).
  id<MTLTexture> RtShadowFor(const ShaderDraw& d, const Shader& ps) {
    const RayShadow& s = ps.shadow;
    __unsafe_unretained id<MTLTexture> white = shadow_masker_->White();
    const Tex* t = FindTex(d.textures[s.depth_sampler]);
    __unsafe_unretained id<MTLTexture> depth = t && !t->cube && t->tex.textureType == MTLTextureType2D ? t->tex : nil;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    if (depth) {
      // A depth texture's writes are its depth buffer's.
      const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
      w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
      w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
      w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);  // (the game's size: scaled targets keep it)
    }
    w.scene_camera = rt_->CameraId();
    const float sign = s.toward_light ? 1.0f : -1.0f;
    if (d.ps_float)
      for (int k = 0; k < 3; k++) w.light[k] = sign * d.ps_float[s.light_register][k];
    bool dim = false;  // (RayShadow::dim_light)
    // The light's colour times its intensity (RayShadow::colour_register):
    // black, nothing the mask scales reaches the image.
    if (d.ps_float && s.colour_register < 224) {
      const float* c = d.ps_float[s.colour_register];
      const float most = std::max({c[0], c[1], c[2]}) * c[3];
      w.dark = !(most > 0.0f);
      dim = most < s.dim_light;
      if (perf_) {
        std::memcpy(pc_.shadow_colour, c, sizeof pc_.shadow_colour);
        pc_.shadow_colour_seen = true;
      }
    }
    switch (RtShadowNext(&shadow_memo_, w)) {
      case RtShadowStep::Reuse: return shadow_memo_.white ? white : shadow_memo_mask_;
      case RtShadowStep::White:
        if (perf_) pc_.shadow_white[size_t(RtShadowWhyWhite(w))]++;
        return white;
      case RtShadowStep::Trace: break;
    }
    const bool depth_tex = t->depth != 0;
    // Diagnostics (RayTracing::shadow_log): the first mask of
    // every shadow_log_every_-th presented frame (120), its instances
    // summarised.
    // Not for perf stats: its full-size trace on a fresh snapshot is a
    // spike every 120 frames, which every measured average then carries.
    const bool diagnose = shadow_log_ && shadow_frames_ % shadow_log_every_ == 0 &&
                          shadow_diag_frame_ != shadow_frames_ && ShadowDiagBuffer(slot_);
    if (diagnose) {
      shadow_diag_frame_ = shadow_frames_;
      rt_->DescribeNextTlas(&shadow_diag_[slot_].summary);
    }
    const RtFrame* f = RtSnapshot(depth, w, diagnose);
    if (diagnose) rt_->DescribeNextTlas(nullptr);
    // The jittered scene's depth (a temporal upscaler's): its pixels moved
    // by this frame's jitter.
    const RtJitter jit = RtJitterFor(depth, false);
    const bool jittered = jit == RtJitter::Jittered;
    RtShadowParams p;
    __strong id<MTLTexture>& mask = shadow_mask_[slot_];
    if (f && f->tlas && f->camera_id == w.scene_camera &&
        (!mask || mask.width != depth.width || mask.height != depth.height)) {
      Keep(mask);  // (recorded work may still use it)
      mask = shadow_masker_->NewMask(uint32_t(depth.width), uint32_t(depth.height));
    }
    if (!f || !f->tlas || f->camera_id != w.scene_camera || !mask ||
        !RtShadowParamsFor(*f, uint32_t(depth.width), uint32_t(depth.height),
                           depth_tex && ReversedDepth(depth.pixelFormat), depth_tex, jittered ? jitter_ : nullptr,
                           w.light, &p)) {
      shadow_memo_.white = true;
      if (perf_) pc_.shadow_white[size_t(RtShadowWhite::NoTlas)]++;
      return white;
    }
    NoteRtSunToward(p.light);
    // Hard shadows, the denoiser on or not: the forward shaders' own rays
    // (decals, glass, fences over the same ground) trace one ray toward the
    // light, and a cone sampled per pixel per frame (kRtShadowCone) left
    // the denoised mask speckled under them, a second shadow differing from
    // theirs.
    __unsafe_unretained id<MTLTexture> m = mask;
    __unsafe_unretained id<MTLBuffer> counts = shadow_counts_[slot_];
    __unsafe_unretained id<MTLBuffer> diag = nil;
    if (diagnose) {
      ShadowDiag& sd = shadow_diag_[slot_];
      p.diag = kRtShadowDiagCounts;
      diag = sd.buf;
      sd.pending = true;
      sd.size[0] = p.size[0];
      sd.size[1] = p.size[1];
      std::memcpy(sd.eye, p.eye, sizeof sd.eye);
      std::memcpy(sd.light, p.light, sizeof sd.light);
      sd.instances = f->instances;
    }
    const RtShadowMasker* sm = shadow_masker_.get();
    __unsafe_unretained id<MTLTexture> normals = RtShadowNormals(d, s);
    const uint32_t step = dim ? kRtShadowDimStep : 2;
    // The game's cascaded shadow map (RayShadow::cascade_*): rays only where
    // it is not sure the point is lit.
    RtShadowCascade cas;
    __unsafe_unretained id<MTLTexture> cmap = nil;
    if (d.ps_float && RtShadowCascadeFor(s, d.ps_float, &cas))
      if (const Tex* ct = FindTex(d.textures[s.cascade_sampler]); ct && !ct->cube) cmap = ct->tex;
    // (RECOMP_RT_SHADOW_CASCADE=0: every texel traced to the light.)
    static const bool cascade_off = [] {
      const char* e = std::getenv("RECOMP_RT_SHADOW_CASCADE");
      return e && *e == '0';
    }();
    if (!cmap || cascade_off) cas.on = 0;
    if (perf_ && cmap) {
      RtShadowCascade c = cas;
      c.atlas[0] = float(cmap.width);
      c.atlas[1] = float(cmap.height);
      for (int k = 0; k < 4; k++) pc_.cascade_texel[k] = float(RtCascadeTexel(c, k));
      pc_.cascade_range = cas.range;
      pc_.cascade_format = uint32_t(cmap.pixelFormat);
      pc_.cascade_size[0] = uint32_t(cmap.width);
      pc_.cascade_size[1] = uint32_t(cmap.height);
    }
    // Where the depth is the jittered scene's (a camera to reproject by):
    // fewer rays, accumulated (rt_accumulate.h); else as before.
    RtAccumulateFrame af;
    const uint32_t astep = dim ? kRtShadowDimStep : sun_step_;
    // (What the sparse trace needs made first: Begin commits the history
    // to this frame. A second trace in a presented frame: as before.)
    bool accumulate = false;
    if (accumulator_ && jittered && !diagnose && sm->PrepareSparse(p.size[0], p.size[1], astep)) {
      std::vector<id> retired;
      accumulate = accumulator_->Begin(&sun_history_, *f, p, RtAccumulateTerm{}, astep, RtLightStride(astep),
                                       uint32_t(shadow_frames_), w.light, false, &af, &retired);
      for (id o : retired) Keep(o);  // (recorded work may still use them)
    }
    // The sun's mask traced at one pixel of every sun_turn_ x sun_turn_,
    // turning each frame, upsampled by depth (the temporal anti-aliasing
    // averages the turns), where the depth is the jittered scene's.
    const bool turned = !accumulate && sun_turn_ >= 2 && jittered && !diagnose && !dim && sm->SparseReady();
    if (turned) {
      const uint32_t ts = sun_turn_, tstride = RtLightStride(sun_turn_), tframe = uint32_t(shadow_frames_);
      Rec([this, sm, f, depth, m, p, counts, normals, ts, tstride, tframe, cas, cmap] {
        sm->EncodeSparse(cmd_, f, depth, p, counts, 0, normals, ts, tstride, tframe, &cas, cmap, m);
      });
    } else if (accumulate) {
      const RtAccumulator* acc = accumulator_.get();
      Rec([this, sm, acc, f, depth, m, p, counts, normals, astep, af, cas, cmap] {
        if (id<MTLTexture> low =
                sm->EncodeSparse(cmd_, f, depth, p, counts, 0, normals, astep, af.p.stride, af.p.frame, &cas, cmap))
          acc->Encode(cmd_, af, low, sm->SparsePoints(), depth, normals, m);
      });
    } else {
      Rec([this, sm, f, depth, m, p, counts, diag, normals, step, cas, cmap] {
        sm->Encode(cmd_, f, depth, m, p, counts, 0, diag, nullptr, normals, step, &cas, cmap);
      });
    }
    // The volumetric fog (RtFog: RECOMP_FOG), from the same scene and
    // camera, lit by this light: its colour times its intensity over 16 (a
    // day's sun about 1).
    if (fog_ && !diagnose) {
      struct Fog {
        float sun[3];
        double prev[16];
        bool have_prev;
      } fg{};
      if (d.ps_float && s.colour_register < 224) {
        const float* c = d.ps_float[s.colour_register];
        for (int k = 0; k < 3; k++) fg.sun[k] = c[k] * c[3] / 16.0f;
      }
      std::memcpy(fg.prev, vp_prev_, sizeof fg.prev);
      fg.have_prev = have_prev_;
      const RtFog* fog = fog_.get();
      const RtFogLook look = fog_look_;
      // (The sky's colour for the fog's sky light: none kept here, a zero
      // with w 0, which the fog reads as no sky seen.)
      if (!sky_buf_) {
        sky_buf_ = [dev_ newBufferWithLength:16 options:MTLResourceStorageModeShared];
        std::memset(sky_buf_.contents, 0, 16);
      }
      __unsafe_unretained id<MTLBuffer> sky = sky_buf_;
      const uint32_t frame = uint32_t(shadow_frames_);
      Rec([this, fog, f, p, fg, look, sky, frame] {
        fog->Encode(cmd_, f, p, fg.sun, sky, fg.have_prev ? fg.prev : nullptr, frame, look);
      });
      fog_traced_ = true;
      fog_frame_ = true;
    }
    if (perf_) {
      pc_.shadow_masks++;
      pc_.shadow_dim += dim ? 1 : 0;
      pc_.shadow_pixels += uint64_t(p.size[0]) * p.size[1];
      pc_.shadow_jitter[size_t(jit)]++;
    }
    shadow_memo_mask_ = m;
    return m;
  }

  // The shadow mask a draw of pixel shader `ps` naming a local light
  // (RayShadow::local) binds: the one traced for that light (RtLocalLight:
  // its registers' values) this presented frame, while the depth texture,
  // its contents and the cameras are the same (RtLightMaskCache: an entry,
  // its memo RtShadowNext's); else one traced now into an entry's mask over
  // the light's pixels on screen (RtLocalLightBounds), from the snapshot the
  // frame's other effects of that depth took (RtSnapshot, else a new one).
  // The trace is compute work: the render pass open ends (a store and a
  // load), so at most RayTracing::local_light_masks lights that stay where
  // they are a presented frame are traced, in turn (RtLightMaskChoose: those
  // reserved for the frame first), and those that moved every frame (up to
  // RtLightMaskCache::kMovingBudget more); a still light past its budget
  // draws with its own mask of an earlier frame. All lit (counted by reason) where nothing can be traced, past the
  // budget without a mask of its own, or off screen; without ending the pass
  // where the snapshot already there says so.
  id<MTLTexture> RtLightShadowFor(const ShaderDraw& d, const Shader& ps) {
    const RayShadow& s = ps.shadow;
    __unsafe_unretained id<MTLTexture> white = shadow_masker_->White();
    if (perf_) pc_.light_draws++;
    const Tex* t = FindTex(d.textures[s.depth_sampler]);
    __unsafe_unretained id<MTLTexture> depth = t && !t->cube && t->tex.textureType == MTLTextureType2D ? t->tex : nil;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    if (depth) {
      const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
      w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
      w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
      w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);  // (the game's size: scaled targets keep it)
    }
    w.scene_camera = rt_->CameraId();
    const RtLocalLight light = RtLocalLightFor(s, d.ps_float);
    // (The light is the entry's; RtShadowNext's light only says there is
    // one: none without a radius.)
    if (RtLocalLightRadius(light) > 0.0) w.light[2] = 1.0f;
    // Its entry: kept from an earlier draw, else one only once it is to be
    // traced (none taken over for a light all lit).
    int e = RtLightMaskFind(&light_cache_, light);
    if (e >= 0) {
      RtShadowMemo& memo = light_cache_.entry[e].memo;
      switch (RtShadowNext(&memo, w)) {
        case RtShadowStep::Reuse:
          if (perf_) pc_.light_reused++;
          return memo.white ? white : light_mask_[e];
        case RtShadowStep::White:
          if (perf_) pc_.light_white[size_t(RtShadowWhyWhite(w))]++;
          return white;
        case RtShadowStep::Trace: break;
      }
    } else if (const RtShadowWhite why = RtShadowWhyWhite(w); why != RtShadowWhite::None) {
      if (perf_) pc_.light_white[size_t(why)]++;
      return white;
    }
    auto refuse = [&](RtShadowWhite why) {
      if (e >= 0) light_cache_.entry[e].memo.white = true;
      if (perf_) {
        if (why == RtShadowWhite::None)
          pc_.light_offscreen++;
        else
          pc_.light_white[size_t(why)]++;
      }
      return white;
    };
    switch (RtLightMaskChoose(&light_cache_, e, light_budget_)) {
      case RtLightStep::Trace: break;
      case RtLightStep::Older:
        // (Of the same size: not across a resolution change.)
        if (light_mask_[e] && light_mask_[e].width == depth.width && light_mask_[e].height == depth.height) {
          light_cache_.entry[e].memo.white = false;
          if (perf_) {
            pc_.light_older++;
            pc_.light_older_age = std::max(pc_.light_older_age, light_cache_.frame - light_cache_.entry[e].traced);
          }
          return light_mask_[e];
        }
        [[fallthrough]];
      case RtLightStep::Budget:
        // (An entry all the same: first in line for the next frame's traces.)
        if (e < 0) {
          e = RtLightMaskTake(&light_cache_, light);
          RtShadowNext(&light_cache_.entry[e].memo, w);
          RtLightMaskChoose(&light_cache_, e, light_budget_);
        }
        return refuse(RtShadowWhite::Budget);
    }
    const bool depth_tex = t->depth != 0;
    const RtJitter jit = RtJitterFor(depth, true);
    const float* jitter = jit == RtJitter::Jittered ? jitter_ : nullptr;
    const uint32_t dw = uint32_t(depth.width), dh = uint32_t(depth.height);
    // The snapshot this depth has already: one that cannot be used, or that
    // puts the light off screen, leaves the render pass open.
    RtSnapshotMemo peek = snap_memo_;
    if (const RtFrame* known = RtSnapshotReuse(&peek, w, serial_) ? snap_frame_ : nullptr) {
      if (!known->tlas || known->camera_id != w.scene_camera) return refuse(RtShadowWhite::NoTlas);
      if (RtLocalLightBounds(*known, dw, dh, jitter, light).Empty()) return refuse(RtShadowWhite::None);
    }
    if (e < 0) {
      e = RtLightMaskTake(&light_cache_, light);
      RtShadowNext(&light_cache_.entry[e].memo, w);
      light_cache_.entry[e].wanted = light_cache_.frame;
    }
    // (The trace is compute work: the render pass open ends.)
    if (enc_open_ && perf_) pc_.light_splits++;
    light_traced_++;
    const RtFrame* f = RtSnapshot(depth, w, false);
    if (!f || !f->tlas || f->camera_id != w.scene_camera) return refuse(RtShadowWhite::NoTlas);
    const RtRect r = RtLocalLightBounds(*f, dw, dh, jitter, light);
    if (r.Empty()) return refuse(RtShadowWhite::None);
    // (One set of masks for all frames in flight: Metal orders a frame's
    // writes after the last one's reads.)
    __strong id<MTLTexture>& mask = light_mask_[e];
    if (!mask || mask.width != depth.width || mask.height != depth.height) {
      Keep(mask);  // (recorded work may still use it)
      mask = shadow_masker_->NewMask(dw, dh);
      light_cache_.entry[e].mask = false;  // (until traced into)
    }
    RtShadowParams p;
    RtLightParams lp;
    if (!mask || !RtLightParamsFor(*f, dw, dh, depth_tex && ReversedDepth(depth.pixelFormat), depth_tex, jitter,
                                   light, r, &p, &lp))
      return refuse(RtShadowWhite::NoMask);
    __unsafe_unretained id<MTLTexture> m = mask;
    __unsafe_unretained id<MTLBuffer> counts = shadow_counts_[slot_];
    const RtShadowMasker* sm = shadow_masker_.get();
    __unsafe_unretained id<MTLTexture> normals = RtShadowNormals(d, s);
    Rec([this, sm, f, depth, m, p, lp, counts, normals] {
      sm->Encode(cmd_, f, depth, m, p, counts, kLightCounts, nil, &lp, normals);
    });
    if (perf_ && light_cache_.entry[e].moved == light_cache_.frame) pc_.light_moving++;
    RtLightMaskTraced(&light_cache_, e, r.Area());
    if (perf_) {
      pc_.light_masks++;
      pc_.light_pixels += r.Area();
      pc_.light_jitter[size_t(jit)]++;
    }
    return m;
  }

  // The reflections a draw of pixel shader `ps` (SetRayReflection) binds,
  // as RtShadowFor chooses a mask (RtShadowNext over the same depth, and
  // the draw's constants: RtReflectionInputs): the ones traced last this
  // presented frame for the same; traced now (into the frame slot's
  // textures) from the snapshot the shadow mask of that depth has, or a new
  // one; else nothing hit (counted).
  id<MTLTexture> RtReflectionFor(const ShaderDraw& d, const Shader& ps) {
    const RayReflection& r = ps.reflection;
    __unsafe_unretained id<MTLTexture> miss = reflector_->Miss();
    auto flat = [&](uint32_t sampler) -> const Tex* {
      const Tex* t = sampler < 16 ? FindTex(d.textures[sampler]) : nullptr;
      return t && !t->cube && t->tex.textureType == MTLTextureType2D ? t : nullptr;
    };
    const Tex* t = flat(r.depth_sampler);
    // (A colour texture holding depth: a float one, D3D's z / w.)
    __unsafe_unretained id<MTLTexture> depth = t && (t->depth || IsFloat(t->fmt)) ? t->tex : nil;
    // (The G-buffer's colour textures: not a depth texture, which reads its
    // depth.)
    auto colour = [&](uint32_t sampler) -> const Tex* {
      const Tex* c = flat(sampler);
      return c && !c->depth ? c : nullptr;
    };
    const Tex* nt = colour(r.normal_sampler);
    const Tex* st = colour(r.surface_sampler);
    const Tex* wt = colour(r.stencil_sampler);
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    if (depth) {
      const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
      w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
      w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
      w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);  // (the game's size: scaled targets keep it)
    }
    w.scene_camera = rt_->CameraId();
    const RtReflectionInputs in = RtReflectionInputsFor(r, d.ps_float);
    // (No light: traced all the same, the hits lit by the ambient term.)
    w.light[0] = in.light[0], w.light[1] = in.light[1], w.light[2] = in.light[2];
    if (w.light[0] == 0.0f && w.light[1] == 0.0f && w.light[2] == 0.0f) w.light[2] = 1.0f;
    RtReflectionKey key;
    key.config = r;
    key.inputs = in;
    key.textures[0] = nt ? (__bridge const void*)nt->tex : nullptr;
    key.textures[1] = st ? (__bridge const void*)st->tex : nullptr;
    key.textures[2] = wt ? (__bridge const void*)wt->tex : nullptr;
    const int e = RtReflectionEntry(&refl_cache_, key);
    RtShadowMemo& memo = refl_cache_.entry[e].memo;
    if (!nt || !st) w.depth = nullptr;  // (no G-buffer: nothing traced, as without depth)
    switch (RtShadowNext(&memo, w)) {
      case RtShadowStep::Reuse: return memo.white ? miss : refl_out_[e];
      case RtShadowStep::White:
        if (perf_) pc_.refl_none++;
        return miss;
      case RtShadowStep::Trace: break;
    }
    const RtFrame* f = RtSnapshot(depth, w, false);
    const bool depth_tex = t->depth != 0;
    const bool jittered = Temporal() && HaveCamera() && depth == scene_depth_;
    // (One set per entry for every frame in flight: Metal orders this
    // frame's writes after the last one's reads.)
    __strong id<MTLTexture>& out = refl_out_[e];
    __strong id<MTLTexture>& dist = refl_dist_[e];
    const bool usable = f && f->tlas && f->camera_id == w.scene_camera;
    if (usable && (!out || out.width != depth.width || out.height != depth.height)) {
      Keep(out);  // (recorded work may still use them)
      Keep(dist);
      out = reflector_->NewOutput(uint32_t(depth.width), uint32_t(depth.height));
      dist = reflector_->NewDistance(uint32_t(depth.width), uint32_t(depth.height));
    }
    RtReflectionParams p;
    if (!usable || !out || !dist ||
        !RtReflectionParamsFor(*f, uint32_t(depth.width), uint32_t(depth.height),
                               depth_tex && ReversedDepth(depth.pixelFormat), depth_tex, jittered ? jitter_ : nullptr,
                               r, in, wt != nullptr, &p)) {
      memo.white = true;
      if (perf_) pc_.refl_none++;
      return miss;
    }
    // The checkerboard where a temporal pass follows: half the pixels
    // traced, the other half filled from them, the halves trading places
    // with the jitter's phase (the history gathers both).
    // (RECOMP_RT_REFL_CHECKER=ab: on and off by turns of 240 passes, for
    // a comparison within one run; the perf line says the share on.)
    const bool ab_off = reflector_->CheckerAb() && (refl_ab_++ / 240) % 2 == 1;
    // Where the far light reached almost nothing a frame ago (under ground,
    // kBounceMinLit), the hits are lit by the ambient alone: no shadow ray
    // from each hit toward it (those cross the city above: 8 ms a frame
    // of reflections in a subway station).
    if (sun_lit_share_ < kBounceMinLit) p.colour[0] = p.colour[1] = p.colour[2] = 0.0f;
    p.checker = jittered && reflector_->Checker() && !ab_off ? 1 : 0;
    // Where it reaches nothing, every reflecting pixel one of 2 x 2 a frame
    // (RtReflectionParams::sparse_all; RECOMP_RT_REFLECT_UNLIT_SPARSE=0 the
    // checkerboard as by day).
    static const bool unlit_sparse = [] {
      const char* e = std::getenv("RECOMP_RT_REFLECT_UNLIT_SPARSE");
      return !e || *e != '0';
    }();
    p.sparse_all = unlit_sparse && sun_lit_share_ < kBounceMinLit ? 1 : 0;
    p.parity = uint32_t(jitter_phase_) & 1;
    p.frame = uint32_t(refl_frame_);
    if (perf_ && p.checker) pc_.refl_checker++;
    if (reflector_->Diag()) {
      const uint64_t turn = (refl_diag_++ / 240) % 3;
      p.diag = turn == 1 ? 1u : turn == 2 ? 2u : 0u;
      if (perf_) pc_.refl_diag[turn]++;
    }
    // Screen space first (RtSsrParams) where the scene's last frame kept
    // its lit scene: the main view's, at its size.
    RtSsrParams sp;
    const bool ssr_ab_off = reflector_->SsrAb() && (refl_ssr_ab_++ / 240) % 2 == 1;
    const bool screen = jittered && !ssr_ab_off && refl_kept_ && refl_kept_frame_ + 1 == refl_frame_ &&
                        refl_kept_.width == depth.width && refl_kept_.height == depth.height &&
                        RtSsrParamsFor(p, refl_kept_vp_, refl_kept_jitter_, uint32_t(refl_kept_.width),
                                       uint32_t(refl_kept_.height), r.output_scale, &sp);
    if (perf_ && screen) pc_.refl_screen_passes++;
    if (jittered) {
      // (This frame's lit scene: the lighting's target, kept for the next.)
      const Tex* lit = pass_.targets > 0 ? FindTex(pass_.ids[0]) : nullptr;
      refl_lit_ = lit && !lit->cube && !lit->depth && lit->levels == 1 && lit->tex.width == depth.width &&
                          lit->tex.height == depth.height
                      ? lit->tex
                      : nil;
      refl_lit_copy_ = 0;
    }
    __unsafe_unretained id<MTLTexture> o = out, di = dist, nm = nt->tex, sf = st->tex;
    __unsafe_unretained id<MTLTexture> sn = wt ? wt->tex : nil;
    __unsafe_unretained id<MTLTexture> prev = screen ? refl_kept_ : nil;
    __unsafe_unretained id<MTLBuffer> counts = refl_counts_[slot_];
    const RtReflector* rf = reflector_.get();
    Rec([this, rf, f, depth, nm, sf, sn, o, di, p, counts, screen, sp, prev] {
      rf->Encode(cmd_, f, depth, nm, sf, sn, o, di, p, counts, 0, screen ? &sp : nullptr, prev);
    });
    if (perf_) {
      pc_.refl_passes++;
      pc_.refl_pixels += uint64_t(p.geo.size[0]) * p.geo.size[1];
    }
    refl_last_ = e;
    refl_last_scale_ = r.output_scale;
    if (jittered) dn_hit_ = di;  // (the scene's: the denoiser's hit distance)
    return o;
  }

  // A post-process draw (SetRayDenoise's post_input) binding this frame's
  // lit scene (refl_lit_, or a whole copy of it) at its input, the first of
  // the frame: the lit scene kept (RtReflector::EncodeKeep) with the
  // camera that drew it, for the next frame's screen-space reflections;
  // only while the reflections cover enough pixels (kReflKeepPixels, the
  // last frame read back) for the march to pay for the copy.
  void ReflectionKeepAt(const ShaderDraw& d, const Shader& ps) {
    if (!refl_lit_ || refl_keep_done_ || !reflector_->Ssr() || ps.denoise.input_sampler >= 16) return;
    const TextureId tid = d.textures[ps.denoise.input_sampler];
    const Tex* t = FindTex(tid);
    if (!t || !(t->tex == refl_lit_ || (refl_lit_copy_ && tid == refl_lit_copy_))) return;
    refl_keep_done_ = true;
    if (refl_last_listed_ < kReflKeepPixels || !HaveCamera() || !(t->tex.usage & MTLTextureUsageShaderRead))
      return;
    const uint32_t w = uint32_t(t->tex.width), h = uint32_t(t->tex.height);
    if (!refl_kept_ || refl_kept_.width != w || refl_kept_.height != h) {
      Keep(refl_kept_);  // (recorded work may still read it)
      refl_kept_ = reflector_->NewKept(w, h);
      if (!refl_kept_) return;
    }
    end_why_ = PassWhy::Compute;
    if (RtShadowSettlesClear(clear_flags_, scene_depth_ == pass_.depth)) Settle();
    EndEncoders();
    EnsureFrame();
    __unsafe_unretained id<MTLTexture> lit = t->tex, kept = refl_kept_;
    const RtReflector* rf = reflector_.get();
    Rec([this, rf, lit, kept] { rf->EncodeKeep(cmd_, lit, kept); });
    std::memcpy(refl_kept_vp_, vp_cur_, sizeof vp_cur_);
    refl_kept_jitter_[0] = jitter_[0];
    refl_kept_jitter_[1] = jitter_[1];
    refl_kept_frame_ = refl_frame_;
    if (perf_) pc_.refl_kept++;
  }

  // What a draw of pixel shader `ps` tracing its reflections itself traces
  // with (rt_fwd_frame_, rt_fwd_params_): the snapshot the frame's traced
  // effects took last in this frame (the backend's: its command buffer; a
  // TLAS built before the pass), where the scene's camera took it and the draw is in its view
  // (RtForwardWhyOff: its own matrices' camera where vertex shader `vs`
  // reads both, else the snapshot's target or the depth's camera); else
  // nothing (counted by reason; with perf stats, the first four refused
  // draws of every 120th presented frame said on stderr).
  void RtForwardFor(const ShaderDraw& d, const Shader& vs, const Shader& ps) {
    const bool local = ps.ray_forward_shadow && ps.forward_shadow.point && ps.forward_shadow.local;
    // A local light's volume: the lights traced at once if not yet (the
    // render pass open ends), before the snapshot is looked at.
    if (local && lights_plan_n_ && LightSetOn() && ps.forward_shadow.depth_sampler < 16) {
      const bool split = enc_open_;
      if (LightSetAt(FindTex(d.textures[ps.forward_shadow.depth_sampler])) && split && perf_) pc_.lset_splits++;
    }
    rt_fwd_frame_ = nullptr;
    rt_fwd_params_ = RtForwardParams{};
    RtForwardWant w;
    w.config = ps.ray_forward || ps.ray_forward_shadow;
    w.snapshot = snap_memo_.valid && snap_memo_.serial == serial_ && snap_frame_ && snap_frame_->tlas;
    w.snapshot_camera = w.snapshot ? snap_frame_->camera_id : 0;
    w.scene_camera = rt_->CameraId();
    const int pm = vs.position_matrix, wr = rt_world_;
    if (d.vs_float && wr >= 0 && pm >= 0 && pm <= 252 && vs.float_reads[size_t(wr)] &&
        vs.float_reads[size_t(wr) + 1] && vs.float_reads[size_t(wr) + 2] && vs.float_reads[size_t(wr) + 3]) {
      w.draw_known = true;
      w.draw_camera = rt_->CameraOf(d.vs_float + wr, d.vs_float + pm);
    }
    w.snapshot_target = snap_target_ && snap_target_ == (__bridge const void*)pass_.color[0];
    if (pass_.depth_id)
      if (auto it = depths_.find(pass_.depth_id); it != depths_.end()) w.depth_camera = it->second.rt_camera;
    RtForwardOff why = RtForwardWhyOff(w);
    if (why == RtForwardOff::None &&
        !RtForwardParamsFor(*snap_frame_, uint32_t(std::max(pass_.w, 1)), uint32_t(std::max(pass_.h, 1)),
                            ps.ray_forward ? &ps.forward : nullptr,
                            ps.ray_forward_shadow ? &ps.forward_shadow : nullptr, &rt_fwd_params_))
      why = RtForwardOff::NoCamera;
    if (why == RtForwardOff::None) frame_log_counts_.fwd_draws++;  // (the frame log's "r")
    if (perf_) {
      pc_.fwd[size_t(why)]++;
      if (why != RtForwardOff::None && why != RtForwardOff::NoConfig && shadow_frames_ % 120 == 0) {
        if (fwd_log_frame_ != shadow_frames_) {
          fwd_log_frame_ = shadow_frames_;
          fwd_logged_ = 0;
        }
        if (fwd_logged_++ < 4)
          std::fprintf(stderr,
                       "gfx: forward ray queries not traced (%s): pixel shader %016llx; snapshot %s, its camera %llu, "
                       "the scene's %llu; the draw's %llu%s; target %dx%d%s; depth %u written last by camera "
                       "%llu; viewport %ux%u at %u,%u, depth %.2f-%.2f\n",
                       RtForwardOffName(why), (unsigned long long)ps.hash, w.snapshot ? "taken" : "none",
                       (unsigned long long)w.snapshot_camera, (unsigned long long)w.scene_camera,
                       (unsigned long long)w.draw_camera, w.draw_known ? "" : " (not known)", pass_.w, pass_.h,
                       w.snapshot_target ? " (the snapshot's)" : " (not the snapshot's)", unsigned(pass_.depth_id),
                       (unsigned long long)w.depth_camera, viewport_.width, viewport_.height, viewport_.x,
                       viewport_.y, double(viewport_.min_z), double(viewport_.max_z));
      }
    }
    if (why != RtForwardOff::None) {
      rt_fwd_params_ = RtForwardParams{};
      return;
    }
    rt_fwd_frame_ = snap_frame_;
    // A draw in one of the plain places (SetRtPlainPlaces: a train's cars):
    // no lamp ray; its reflection and its sun's shadow still. Off unless
    // RECOMP_RT_PLAIN_PLACES=1: on a platform with a train in view the lit
    // pass took as long with these rays as without them (2 to 4 ms more
    // than with no train, either way: its reflection rays, measured over
    // four play sessions of 2026-11-14), so leaving them out only took the
    // lamps' shadows off the train.
    static const bool plain_off = [] {
      const char* e = std::getenv("RECOMP_RT_PLAIN_PLACES");
      return !(e && *e == '1');
    }();
    bool plain = false;
    if (!plain_off && !plain_places_.empty() && d.vs_float && rt_world_ >= 0 &&
        vs.float_reads[size_t(rt_world_) + 3]) {
      const float* t = d.vs_float[rt_world_ + 3];  // (the world matrix's origin: its row 3)
      const float* q = plain_places_.data();
      for (size_t i = 0; i + 5 <= plain_places_.size() && !plain; i += 5) {
        const float dx = t[0] - q[i], dy = t[1] - q[i + 1], dz = t[2] - q[i + 2];
        plain = dx * dx + dy * dy < q[i + 3] * q[i + 3] && std::fabs(dz) < q[i + 4];
      }
    }
    // (RECOMP_RT_PLAIN_REFLECT=0: no reflection ray there either.)
    static const bool plain_no_reflect = [] {
      const char* e = std::getenv("RECOMP_RT_PLAIN_REFLECT");
      return e && *e == '0';
    }();
    if (plain) {
      if (plain_no_reflect) rt_fwd_params_.on = 0;
      rt_fwd_params_.shadow |= kRtForwardNoLamps;
    }
    // Where the far light reached almost nothing a frame ago (kBounceMinLit:
    // under ground), its rays are not traced: in shadow (kRtForwardFarDark).
    // A subway station's walls are drawn by these shaders, a ray to the sun
    // a pixel and one from each reflection's hit, each crossing the city
    // above: 7.5 ms a frame there.
    if (sun_lit_share_ < kBounceMinLit) rt_fwd_params_.shadow |= kRtForwardFarDark;
    // For measuring what each of a forward shader's rays costs (a shop's
    // glass front filling the view: 50 ms a frame): RECOMP_RT_FORWARD_REFLECT=0
    // no reflection ray, RECOMP_RT_FORWARD_FAR=0 the far light's rays
    // untraced (as under ground). Not for play: both change the picture.
    static const int fwd_measure = [] {
      const char* r = std::getenv("RECOMP_RT_FORWARD_REFLECT");
      const char* f = std::getenv("RECOMP_RT_FORWARD_FAR");
      return (r && *r == '0' ? 1 : 0) | (f && *f == '0' ? 2 : 0);
    }();
    if (fwd_measure & 1) rt_fwd_params_.on = 0;
    if (fwd_measure & 2) rt_fwd_params_.shadow |= kRtForwardFarDark;
    // Glass: no lamp rays (kRtForwardNoLamps). RECOMP_RT_FORWARD_GLASS_LAMPS=0
    // rays as before; =2 for every blended draw writing no depth, with a
    // reflection or not (to measure).
    // Blended draws take no lamp rays (kRtForwardNoLamps): the game draws
    // a shop's interior behind its glass front in the blended pass, some
    // 900 draws in layers back to front, each layer's every pixel tracing
    // toward its lamps: 48.7 ms of forward pass at a car showroom by night
    // (13 fps), 24.9 without those rays, and the picture the same to the
    // eye there and in a subway station (whose lamps' shadows are the
    // deferred pass's). RECOMP_RT_FORWARD_NO_LAMPS=<mask> picks the classes
    // of draws left without (default 3): 1 blended without depth writes, 2
    // blended with them, 4 alpha-tested, 8 plain opaque; 16 only those whose
    // shader also traces a reflection; 0 rays for all, as before.
    // RECOMP_RT_FORWARD_DUMP=1 lists every 600th frame's draws by pixel
    // shader and class with their triangles (for measuring).
    static const int no_lamps = [] {
      const char* e = std::getenv("RECOMP_RT_FORWARD_NO_LAMPS");
      return e && *e ? std::atoi(e) : 3;
    }();
    static const bool dump = [] {
      const char* e = std::getenv("RECOMP_RT_FORWARD_DUMP");
      return e && *e == '1';
    }();
    const bool blend = rs_[27] != 0, zwrite = rs_[7] && rs_[14], tested = rs_[15] && rs_[25] != 8;
    const int cls = blend ? (zwrite ? 2 : 1) : tested ? 4 : 8;
    if ((no_lamps & cls) && (!(no_lamps & 16) || ps.ray_forward)) rt_fwd_params_.shadow |= kRtForwardNoLamps;
    if (dump && shadow_frames_ % 600 == 300) {
      if (fwd_dump_frame_ != shadow_frames_) {
        if (!fwd_dump_.empty()) {
          std::vector<std::pair<uint64_t, FwdDump>> v(fwd_dump_.begin(), fwd_dump_.end());
          std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second.triangles > y.second.triangles; });
          std::fprintf(stderr, "gfx: forward ray draws of a frame, by pixel shader and class (1 blended no depth write, 2 "
                               "blended, 4 alpha-tested, 8 opaque):\n");
          for (size_t k = 0; k < v.size() && k < 40; k++)
            std::fprintf(stderr, "  ps %016llx class %d: %u draws, %llu triangles%s%s, target %dx%d\n",
                         (unsigned long long)(v[k].first >> 4), int(v[k].first & 15), v[k].second.draws,
                         (unsigned long long)v[k].second.triangles, v[k].second.reflect ? ", reflection" : "",
                         v[k].second.lamps ? ", lamp rays" : "", v[k].second.w, v[k].second.h);
        }
        fwd_dump_.clear();
        fwd_dump_frame_ = shadow_frames_;
      }
      FwdDump& fd = fwd_dump_[(ps.hash << 4) | uint64_t(cls)];
      fd.draws++;
      fd.triangles += d.primitives;
      fd.reflect = ps.ray_forward;
      fd.lamps = ps.ray_forward_shadow;
      fd.w = pass_.w;
      fd.h = pass_.h;
    }
    // By day, the lamps' rays only where the sun mask holds the pixel in
    // shadow (kRtForwardSunGate; RECOMP_RT_FORWARD_SUN_GATE=0 everywhere):
    // the frame's mask, of this draw's target's size.
    static const bool sun_gate = [] {
      const char* e = std::getenv("RECOMP_RT_FORWARD_SUN_GATE");
      return !e || *e != '0';
    }();
    rt_fwd_sun_mask_ = nil;
    if (sun_gate && sun_lit_share_ >= kBounceMinLit && shadow_memo_mask_ && !shadow_memo_.white &&
        int(shadow_memo_mask_.width) == pass_.w && int(shadow_memo_mask_.height) == pass_.h) {
      rt_fwd_params_.shadow |= kRtForwardSunGate;
      rt_fwd_sun_mask_ = shadow_memo_mask_;
    }
    if (local) LightSetDraw(d, ps.forward_shadow);
  }

  // ---- the far light bounced once, and the sky light's occlusion
  // (RtReflector::EncodeBounce), for the shaders reading the occlusion
  // (RayOcclusion) while reflections are traced

  // What a draw of such a shader binds: the bounce of the depth it binds,
  // traced once a presented frame from the main view's depth, with the
  // reflections' parameters of its own constants (the shader reads both);
  // else none and the sky open (BounceNone: never Miss()'s zero alpha,
  // which would black the ambient out). Where the far light reached almost
  // nothing a frame ago (kBounceMinLit: under ground, at night) its light
  // is none: the rays still give the occlusion, without a shadow ray from
  // each hit (which cross the city above), and end at the occlusion's
  // radius.
  id<MTLTexture> BounceFor(const ShaderDraw& d, const Shader& ps) {
    __unsafe_unretained id<MTLTexture> none = reflector_->BounceNone();
    if (!ps.ray_reflection) return none;
    const RayReflection& r = ps.reflection;
    const Tex* t = r.depth_sampler < 16 ? FindTex(d.textures[r.depth_sampler]) : nullptr;
    if (!t || t->cube || t->tex.textureType != MTLTextureType2D || !(t->depth || IsFloat(t->fmt))) return none;
    __unsafe_unretained id<MTLTexture> depth = t->tex;
    // (The main view's depth only: a water reflection's or a mirror's view
    // takes none and leaves the frame's trace to the main view.)
    if (depth != scene_depth_) return none;
    if (bounce_live_) return bounce_traced_ ? bounce_out_ : none;
    bounce_live_ = true;
    bounce_traced_ = false;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
    w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
    w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
    w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);
    w.scene_camera = rt_->CameraId();
    const RtReflectionInputs in = RtReflectionInputsFor(r, d.ps_float);
    w.light[0] = in.light[0], w.light[1] = in.light[1], w.light[2] = in.light[2];
    if (w.light[0] == 0.0f && w.light[1] == 0.0f && w.light[2] == 0.0f) w.light[2] = 1.0f;
    if (RtShadowWhyWhite(w) != RtShadowWhite::None) return none;
    const RtFrame* f = RtSnapshot(depth, w, false);
    if (!f || !f->tlas || f->camera_id != w.scene_camera) return none;
    const bool depth_tex = t->depth != 0;
    const bool jittered = Temporal() && HaveCamera() && depth == scene_depth_;
    RtReflectionParams p;
    if (!RtReflectionParamsFor(*f, uint32_t(depth.width), uint32_t(depth.height),
                               depth_tex && ReversedDepth(depth.pixelFormat), depth_tex, jittered ? jitter_ : nullptr, r,
                               in, false, &p))
      return none;
    const bool unlit = sun_lit_share_ < kBounceMinLit;
    if (unlit) p.colour[0] = p.colour[1] = p.colour[2] = 0.0f;
    if (unlit && perf_) pc_.bounce_skipped++;
    const uint32_t dw = uint32_t(depth.width), dh = uint32_t(depth.height);
    if (!bounce_out_ || bounce_out_.width != (dw + 1) / 2 || bounce_out_.height != (dh + 1) / 2) {
      Keep(bounce_out_);  // (recorded work may still use it)
      bounce_out_ = reflector_->NewBounceOut(dw, dh);
      if (!bounce_out_) return none;
    }
    RtBounceParams bp;
    bp.frame = uint32_t(shadow_frames_);
    bp.stride = jittered ? RtLightStride(kRtBounceBlock) : 0;
    bp.strength = bounce_strength_;
    bp.tolerance = 6.0f * p.geo.pixel;
    bp.ao_strength = bounce_ao_strength_;
    bp.distance = unlit ? bp.ao_radius : bounce_distance_;
    // The history where each block's point was a frame ago.
    RtAccumulateCamera cam;
    if (RtAccumulateCameraOf(*f, &cam)) {
      bp.history = bounce_cam_.valid && bounce_cam_frame_ + 1 == shadow_frames_ && !RtAccumulateCut(bounce_cam_, cam) &&
                           RtAccumulateReproject(bounce_cam_, cam, bp.prev, bp.eye_delta)
                       ? 1u
                       : 0u;
      bounce_cam_ = cam;
      bounce_cam_frame_ = shadow_frames_;
    }
    bp.size[0] = dw;
    bp.size[1] = dh;
    __unsafe_unretained id<MTLTexture> bo = bounce_out_;
    const RtReflector* rf = reflector_.get();
    Rec([this, rf, f, depth, p, bp, bo] { rf->EncodeBounce(cmd_, f, depth, p, bp, bo); });
    bounce_traced_ = true;
    return bo;
  }

  // ---- ray-traced ambient occlusion (RayOcclusion)

  // What a draw of a shader reading the occlusion binds: the frame's
  // occlusion of the depth it binds, traced now if not yet for that depth
  // and contents (from the main view's depth only, as the shadow masks;
  // ending the render pass open), else open everywhere.
  id<MTLTexture> RtOcclusionFor(const ShaderDraw& d, const Shader& ps) {
    const Tex* t = ps.occlusion.depth_sampler < 16 ? FindTex(d.textures[ps.occlusion.depth_sampler]) : nullptr;
    if (!t || t->cube || t->tex.textureType != MTLTextureType2D) return occluder_->Open();
    __unsafe_unretained id<MTLTexture> depth = t->tex;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
    w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
    if (ao_memo_.traced && ao_memo_.depth == w.depth && ao_memo_.epoch == w.epoch)
      return ao_memo_.open ? occluder_->Open() : ao_out_;
    ao_memo_.traced = true;
    ao_memo_.depth = w.depth;
    ao_memo_.epoch = w.epoch;
    ao_memo_.open = true;
    w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
    w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);
    w.scene_camera = rt_->CameraId();
    w.light[2] = 1.0f;
    if (RtShadowWhyWhite(w) != RtShadowWhite::None) return occluder_->Open();
    const RtFrame* f = RtSnapshot(depth, w, false);
    if (!f || !f->tlas || f->camera_id != w.scene_camera) return occluder_->Open();
    const uint32_t dw = uint32_t(depth.width), dh = uint32_t(depth.height);
    const bool depth_tex = t->depth != 0;
    const bool jittered = RtJitterFor(depth, false) == RtJitter::Jittered;
    const float up[3] = {0.0f, 0.0f, 1.0f};
    RtShadowParams p;
    if (!RtShadowParamsFor(*f, dw, dh, depth_tex && ReversedDepth(depth.pixelFormat), depth_tex,
                           jittered ? jitter_ : nullptr, up, &p))
      return occluder_->Open();
    if (!ao_out_ || ao_out_.width != (dw + 1) / 2 || ao_out_.height != (dh + 1) / 2) {
      Keep(ao_out_);  // (recorded work may still use it)
      ao_out_ = occluder_->NewOut(dw, dh);
      if (!ao_out_) return occluder_->Open();
    }
    const Tex* nt = ps.occlusion.normal_sampler < 16 ? FindTex(d.textures[ps.occlusion.normal_sampler]) : nullptr;
    __unsafe_unretained id<MTLTexture> normals = nt && !nt->cube ? nt->tex : nil;
    RtOcclusionParams op{};
    op.frame = uint32_t(shadow_frames_);
    op.stride = jittered ? RtLightStride(kRtOcclusionBlock) : 0;
    op.radius = ao_radius_;
    __unsafe_unretained id<MTLTexture> out = ao_out_;
    __unsafe_unretained id<MTLBuffer> counts = shadow_counts_[slot_];
    const RtOccluder* oc = occluder_.get();
    Rec([this, oc, f, depth, normals, p, op, out, counts] {
      oc->Encode(cmd_, f, depth, normals, p, op, out, counts, kOcclusionCounts);
    });
    ao_memo_.open = false;
    if (perf_) pc_.ao_passes++;
    return ao_out_;
  }

  // ---- the local lights lit by the device (RayManyLights)

  // A draw of a shader naming the G-buffer: what it binds there, noted.
  void MlNote(const ShaderDraw& d, const RayLightVolume& v) {
    const uint32_t s[5] = {v.depth_sampler, v.albedo_sampler, v.normal_sampler, v.surface_sampler, v.class_sampler};
    for (int k = 0; k < 5; k++)
      if (s[k] < 16 && d.textures[s[k]]) ml_ids_[k] = d.textures[s[k]];
  }

  // Before a presented frame's first light volume: the frame's lights
  // (rt_lights_.Current() where the work runs) binned and lit from the
  // G-buffer noted (its depth the main view's, written by the scene's
  // camera: as the local lights' masks are traced) into the target the
  // volume draws to, against a snapshot of the scene (ending the render
  // pass open). Once a presented frame.
  void MlAt() {
    if (!ml_ || ml_tried_ || !pass_.color[0]) return;
    ml_tried_ = true;
    const Tex* t = FindTex(ml_ids_[0]);
    const Tex* albedo = FindTex(ml_ids_[1]);
    const Tex* normal = FindTex(ml_ids_[2]);
    if (!t || !albedo || !normal || t->cube || t->tex.textureType != MTLTextureType2D) return;
    const Tex* surface = FindTex(ml_ids_[3]);
    const Tex* cls = FindTex(ml_ids_[4]);
    __unsafe_unretained id<MTLTexture> depth = t->tex;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
    w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
    w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
    w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);
    w.scene_camera = rt_->CameraId();
    w.light[2] = 1.0f;
    if (RtShadowWhyWhite(w) != RtShadowWhite::None) return;
    __unsafe_unretained id<MTLTexture> target = pass_.color[0];
    const uint32_t level = uint32_t(pass_.level[0]);
    // (RECOMP_RT_LAMP_PROBE masks bit 0x40000, to measure: every 120 frames
    // the instances whose bounds hold the camera, by kind, the largest.)
    static const char* probe_file = std::getenv("RECOMP_RT_LAMP_PROBE");
    std::vector<RtInstanceSummary> boxes;
    const bool count_boxes = probe_file && *probe_file && (g_lamp_probe_masks & 0x40000u) && shadow_frames_ % 120 == 0;
    if (count_boxes) rt_->DescribeNextTlas(&boxes);
    const RtFrame* f = RtSnapshot(depth, w, count_boxes);
    if (count_boxes) rt_->DescribeNextTlas(nullptr);
    if (count_boxes && f) {
      double e[3] = {};
      RtLightsEye(*f, e);
      uint32_t holding[3] = {}, retained = 0, total = 0;
      std::vector<std::pair<double, const RtInstanceSummary*>> big;
      for (const RtInstanceSummary& b : boxes) {
        if (!b.box) continue;
        total++;
        bool in = true;
        for (int k = 0; k < 3; k++) in = in && e[k] >= b.lo[k] - 1.0 && e[k] <= b.hi[k] + 1.0;
        if (!in) continue;
        holding[b.view % 3]++;
        retained += b.retained ? 1 : 0;
        big.emplace_back(double(b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]), &b);
      }
      std::sort(big.begin(), big.end(), [](auto& a, auto& b) { return a.first > b.first; });
      std::fprintf(stderr,
                   "gfx: lamp probe: camera (%.1f %.1f %.1f): %zu instances (%u with bounds), %u hold it within 1 m "
                   "(%u main view, %u another view, %u captured; %u retained)\n",
                   e[0], e[1], e[2], boxes.size(), total, holding[0] + holding[1] + holding[2], holding[0],
                   holding[1], holding[2], retained);
      for (size_t i = 0; i < big.size() && i < 12; i++) {
        const RtInstanceSummary& b = *big[i].second;
        std::fprintf(stderr, "gfx: lamp probe:   mesh %u (%u triangles) %s%s box (%.0f %.0f %.0f)-(%.0f %.0f %.0f)\n",
                     b.mesh, b.triangles, b.view == 0 ? "main" : b.view == 1 ? "any camera" : "captured",
                     b.retained ? " retained" : "", b.lo[0], b.lo[1], b.lo[2], b.hi[0], b.hi[1], b.hi[2]);
      }
    }
    if (!f || !f->tlas || f->camera_id != w.scene_camera) return;
    // Every kRetainContactEvery-th frame the trace also finds the retained
    // models lying on what the camera sees (RtManyLights::Encode's
    // retain_hits), let go when the slot comes round (MlRetainContact).
    // RECOMP_RT_RETAIN_CONTACT=0: not.
    static const bool contact = [] {
      const char* e = std::getenv("RECOMP_RT_RETAIN_CONTACT");
      return !(e && *e == '0');
    }();
    __unsafe_unretained id<MTLBuffer> retain_hits = nil;
    ml_retained_slots_[slot_].clear();
    if (contact && shadow_frames_ % kRetainContactEvery == 0) {
      if (!ml_retain_hits_[slot_]) {
        ml_retain_hits_[slot_] = [dev_ newBufferWithLength:kRtManyRetainSlots * 4 options:MTLResourceStorageModeShared];
        if (ml_retain_hits_[slot_]) std::memset(ml_retain_hits_[slot_].contents, 0, kRtManyRetainSlots * 4);
      }
      rt_->RetainedSlots(f, &ml_retained_slots_[slot_]);
      if (!ml_retained_slots_[slot_].empty()) retain_hits = ml_retain_hits_[slot_];
    }
    const uint32_t dw = uint32_t(depth.width), dh = uint32_t(depth.height);
    const bool depth_tex = t->depth != 0;
    const bool jittered = RtJitterFor(depth, true) == RtJitter::Jittered;
    const float jitter[2] = {jittered ? jitter_[0] : 0.0f, jittered ? jitter_[1] : 0.0f};
    const float up[3] = {0.0f, 0.0f, 1.0f};
    RtShadowParams p;
    struct Eye {
      double v[3];
    } eye;
    if (!RtShadowParamsFor(*f, dw, dh, depth_tex && ReversedDepth(depth.pixelFormat), depth_tex,
                           jittered ? jitter : nullptr, up, &p) ||
        !RtLightsEye(*f, eye.v))
      return;
    RtManyParams mp{};
    mp.frame = uint32_t(shadow_frames_);
    mp.stride = jittered ? RtLightStride(kRtManyBlock) : 0;
    // (RECOMP_RT_MANY_REST: another share, to compare; above 1 the
    // strongest light alone gets a ray.)
    static const float rest_share = [] {
      const char* e = std::getenv("RECOMP_RT_MANY_REST");
      return e && *e ? float(std::atof(e)) : kRtManyRestShare;
    }();
    mp.rest_share = rest_share;
    for (int k = 0; k < 4; k++) mp.models[k] = RtManyModelFor(ml_cfg_.models[k]);
    std::memcpy(mp.classes, ml_cfg_.classes, sizeof mp.classes);
    RtManyGBuffer g;
    g.depth = depth;
    g.albedo = albedo->tex;
    g.normal = normal->tex;
    g.surface = surface ? surface->tex : nil;
    g.cls = cls ? cls->tex : nil;
    // (RECOMP_RT_LAMP_PROBE: the pass's counters and the blockers by mesh
    // in a buffer of their own, MlProbeLog.)
    static const bool probe = [] {
      const char* e = std::getenv("RECOMP_RT_LAMP_PROBE");
      return e && *e;
    }();
    if (probe && !ml_probe_[slot_]) {
      ml_probe_[slot_] = [dev_ newBufferWithLength:kMlProbeBytes options:MTLResourceStorageModeShared];
      if (ml_probe_[slot_]) std::memset(ml_probe_[slot_].contents, 0, kMlProbeBytes);
    }
    const bool probing = probe && ml_probe_[slot_];
    __unsafe_unretained id<MTLBuffer> counts = probing ? ml_probe_[slot_] : shadow_counts_[slot_];
    const size_t counts_at = probing ? 0 : kManyLightCounts;
    const int slot = slot_;
    const float gap = ml_cfg_.lamp_gap;
    const RtManyLights* ml = ml_.get();
    // (The light added by the next render pass on the target, as its first
    // draw: MlAddPending; RECOMP_RT_MANY_ADD_PASS=1 in a pass of its own.)
    static const bool own_pass = [] {
      const char* e = std::getenv("RECOMP_RT_MANY_ADD_PASS");
      return e && *e == '1';
    }();
    const uint32_t parts = own_pass ? 3u : 3u | RtManyLights::kNoAdd;
    Rec([this, ml, slot, f, g, p, mp, target, level, counts, counts_at, retain_hits, eye, jitter, jittered, dw, dh, gap,
         parts] {
      RtManyLightsBin(*f, dw, dh, jittered ? jitter : nullptr, rt_lights_.Current(), eye.v, gap, &ml_bins_);
      frame_log_lights_ = uint32_t(ml_bins_.lights.size());  // (the frame log's "r", EncodeEndFrame)
      for (int i = 0; i < 3; i++) ml_probe_eye_[i] = eye.v[i];
      ml_enc_lights_ += ml_bins_.lights.size();
      ml_enc_dropped_ += ml_bins_.dropped;
      ml_enc_entries_ += ml_bins_.index.size();
      ml->Encode(cmd_, slot, f, g, p, mp, ml_bins_, target, level, counts, counts_at, parts, retain_hits);
    });
    if (!own_pass) ml_add_ = {true, target, int(level)};
    if (perf_) pc_.ml_passes++;
  }

  // The many lights' light not added yet (MlAt), added now in a render
  // pass of its own (none may be open): before a pass on another target,
  // or one that clears it (then not at all: the clear would take it), at
  // the frame's end.
  void MlAddFlush(bool cleared = false) {
    if (!ml_add_.on) return;
    ml_add_.on = false;
    if (cleared || !ml_) return;
    const RtManyLights* ml = ml_.get();
    __unsafe_unretained id<MTLTexture> target = ml_add_.target;
    const uint32_t level = uint32_t(ml_add_.level);
    Rec([this, ml, target, level] { ml->EncodeAddPass(cmd_, target, level); });
    if (perf_) pc_.ml_add_passes++;
  }

  // ---- the local lights traced at once (RayForwardShadow::local)

  // RECOMP_RT_LIGHT_ALT's phase: 0 on, 1 off, 2 no local light traced, 3
  // on at a step one larger.
  uint32_t LightAlt() const { return lights_alt_ ? uint32_t(shadow_frames_ / 240 % 4) : 0; }
  bool LightSetOn() const {
    return lights_ready_ && lights_step_ > 0 && rt_forward_ && !ml_only_ && (LightAlt() == 0 || LightAlt() == 3);
  }
  uint32_t LightStep() const { return std::max(lights_step_, 1u) + (LightAlt() == 3 ? 1 : 0); }

  // The light a draw of a local light's volume names (as RtLocalLightFor
  // reads a RayShadow's).
  static RtLocalLight LightOfDraw(const RayForwardShadow& fs, const float (*c)[4]) {
    RayShadow s;
    s.local = true;
    s.position_register = fs.light_register;
    s.radius_register = fs.radius_register;
    s.radius_component = fs.radius_component;
    s.cone = fs.cone;
    s.direction_register = fs.direction_register;
    s.cone_offset_register = fs.cone_offset_register;
    s.cone_offset_component = fs.cone_offset_component;
    s.cone_scale_register = fs.cone_scale_register;
    s.cone_scale_component = fs.cone_scale_component;
    s.lamp_gap = fs.lamp_gap;
    return RtLocalLightFor(s, c);
  }

  // The lights of the presented frame before (lights_plan_) traced from
  // depth texture `t` (the main view's, as a far light's mask is traced
  // from: RtShadowWant) into light_masks_ and light_dist_, against the
  // snapshot of that depth (RtSnapshot: one already taken, else a new one,
  // ending the render pass open); once a presented frame for one depth
  // texture and its contents (tried again for another: the lights' draws
  // read the one they read). False where nothing was traced now.
  bool LightSetAt(const Tex* t) {
    if (!lights_plan_n_ || !LightSetOn() || !t || t->cube || t->tex.textureType != MTLTextureType2D) return false;
    __unsafe_unretained id<MTLTexture> depth = t->tex;
    RtShadowWant w;
    w.depth = (__bridge const void*)depth;
    const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
    w.epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
    if (lights_tried_ && lights_tried_depth_ == w.depth && lights_tried_epoch_ == w.epoch) return false;
    w.depth_camera = dit != depths_.end() ? dit->second.rt_camera : t->rt_camera;
    w.main_size = RtMainSize(t->w, t->h, bb_w_, bb_h_);
    w.scene_camera = rt_->CameraId();
    w.light[2] = 1.0f;
    if (RtShadowWhyWhite(w) != RtShadowWhite::None) return false;
    lights_tried_ = true;
    lights_tried_depth_ = w.depth;
    lights_tried_epoch_ = w.epoch;
    const RtFrame* f = RtSnapshot(depth, w, false);
    if (!f || !f->tlas || f->camera_id != w.scene_camera) return false;
    const uint32_t dw = uint32_t(depth.width), dh = uint32_t(depth.height);
    const bool depth_tex = t->depth != 0;
    const float* jitter = RtJitterFor(depth, true) == RtJitter::Jittered ? jitter_ : nullptr;
    const float up[3] = {0.0f, 0.0f, 1.0f};
    RtShadowParams p;
    double eye[3];
    if (!RtShadowParamsFor(*f, dw, dh, depth_tex && ReversedDepth(depth.pixelFormat), depth_tex, jitter, up, &p) ||
        !RtLightsEye(*f, eye))
      return false;
    struct Lights {
      RtLightGpu l[kRtLightSlots];
    } gpu{};
    uint32_t n = 0;
    for (uint32_t k = 0; k < lights_plan_n_; k++)
      if (RtLightGpuFor(lights_plan_[k], eye, &gpu.l[n])) {
        lights_traced_[n] = lights_plan_[k];
        lights_traced_moved_[n] = lights_plan_moved_[k];
        n++;
      }
    if (!n) return false;
    RtLightsParams lp{};
    lp.count = n;
    lp.step = LightStep();
    // The sample turning within its block each frame where the temporal
    // upscaler averages them (the jittered scene's depth), else fixed.
    lp.stride = jitter ? RtLightStride(lp.step) : 0;
    lp.frame = uint32_t(shadow_frames_ % (lp.step * lp.step));
    lp.cap = lights_cap_;
    lp.low[0] = (dw + lp.step - 1) / lp.step;
    lp.low[1] = (dh + lp.step - 1) / lp.step;
    if (!light_masks_ || light_masks_.width != lp.low[0] || light_masks_.height != lp.low[1]) {
      Keep(light_masks_);  // (recorded work may still use them)
      Keep(light_dist_);
      light_masks_ = shadow_masker_->NewLightMasks(lp.low[0], lp.low[1]);
      light_dist_ = shadow_masker_->NewLightDist(lp.low[0], lp.low[1]);
      if (!light_masks_ || !light_dist_) return false;
    }
    __unsafe_unretained id<MTLTexture> masks = light_masks_, dist = light_dist_;
    __unsafe_unretained id<MTLBuffer> counts = shadow_counts_[slot_];
    const RtShadowMasker* sm = shadow_masker_.get();
    Rec([this, sm, f, depth, masks, dist, p, lp, gpu, counts] {
      sm->EncodeLights(cmd_, f, depth, masks, dist, p, lp, gpu.l, counts, kLightSetCounts);
    });
    lights_frame_ = f;
    lights_depth_ = (__bridge const void*)depth;
    lights_epoch_ = w.epoch;
    lights_traced_n_ = n;
    lights_traced_step_ = lp.step;
    lights_traced_frame_ = lp.frame;
    lights_traced_stride_ = lp.stride;
    lights_size_[0] = dw;
    lights_size_[1] = dh;
    if (perf_) {
      pc_.lset_passes++;
      pc_.lset_lights += n;
    }
    return true;
  }

  // A draw of a local light's volume that traces (RtForwardFor's params
  // made): its light noted for the next frame's pass, its reach in the
  // params, and the layer of this frame's pass holding it (RtLightMatch),
  // where that pass traced the depth the draw reads, unchanged, against the
  // snapshot it traces with.
  void LightSetDraw(const ShaderDraw& d, const RayForwardShadow& fs) {
    if (LightAlt() == 2) rt_fwd_params_.shadow = 0;
    const RtLocalLight l = LightOfDraw(fs, d.ps_float);
    if (RtLocalLightRadius(l) > 0.0) {
      if (LightAlt() != 1) rt_fwd_params_.reach = l.inv_sq_radius;
      bool seen = false;
      for (uint32_t k = 0; k < lights_seen_n_ && !seen; k++) seen = RtSameLocalLight(lights_seen_[k], l);
      if (!seen && lights_seen_n_ < kRtLightSlots) lights_seen_[lights_seen_n_++] = l;
    }
    int slot = -1;
    const Tex* t = fs.depth_sampler < 16 ? FindTex(d.textures[fs.depth_sampler]) : nullptr;
    if (lights_traced_n_ && lights_frame_ == rt_fwd_frame_ && t && (__bridge const void*)t->tex == lights_depth_ &&
        LightSetOn()) {
      const auto dit = t->depth ? depths_.find(t->depth) : depths_.end();
      const uint64_t epoch = dit != depths_.end() ? dit->second.rt_epoch : t->rt_epoch;
      for (uint32_t k = 0; k < lights_traced_n_ && slot < 0 && epoch == lights_epoch_; k++)
        if (RtLightMatch(lights_traced_[k], lights_traced_moved_[k], l)) slot = int(k);
    }
    if (slot < 0) {
      if (perf_) pc_.lset_own++;
      return;
    }
    rt_fwd_params_.light_slot = uint32_t(slot);
    rt_fwd_params_.light_step = lights_traced_step_;
    rt_fwd_params_.light_frame = lights_traced_frame_;
    rt_fwd_params_.light_stride = lights_traced_stride_;
    rt_fwd_params_.light_scale[0] = float(lights_size_[0]) / float(std::max(pass_.w, 1));
    rt_fwd_params_.light_scale[1] = float(lights_size_[1]) / float(std::max(pass_.h, 1));
    if (perf_) pc_.lset_draws++;
  }

  // A presented frame ends: its lights are the next pass's, a moved one
  // (RtNearLocalLight to one of the frame before, not the same) where it
  // will be.
  void LightSetEndFrame() {
    if (lights_alt_ && lights_alt_said_ != LightAlt()) {
      static const char* const kPhase[] = {"on", "off (each volume's pixels tracing their own)", "no local light traced",
                                           "on at a step one larger"};
      lights_alt_said_ = LightAlt();
      std::fprintf(stderr, "gfx: local lights traced at once (RECOMP_RT_LIGHT_ALT): %s\n", kPhase[LightAlt()]);
    }
    lights_plan_n_ = 0;
    for (uint32_t k = 0; k < lights_seen_n_; k++) {
      const RtLocalLight& l = lights_seen_[k];
      int same = -1, near = -1;
      for (uint32_t j = 0; j < lights_prev_n_ && same < 0; j++) {
        if (RtSameLocalLight(lights_prev_[j], l))
          same = int(j);
        else if (near < 0 && RtNearLocalLight(lights_prev_[j], l))
          near = int(j);
      }
      const bool moved = same < 0 && near >= 0;
      lights_plan_[lights_plan_n_] = moved ? RtLightPredict(l, lights_prev_[near]) : l;
      lights_plan_moved_[lights_plan_n_] = moved;
      lights_plan_n_++;
    }
    std::copy(lights_seen_, lights_seen_ + lights_seen_n_, lights_prev_);
    lights_prev_n_ = lights_seen_n_;
    lights_seen_n_ = 0;
    lights_tried_ = false;
    lights_tried_depth_ = nullptr;
    lights_frame_ = nullptr;
    lights_depth_ = nullptr;
    lights_traced_n_ = 0;
  }

  // Frame slot k's diagnostics buffer, made (zeroed) when first wanted.
  bool ShadowDiagBuffer(int k) {
    ShadowDiag& d = shadow_diag_[k];
    if (!d.buf) {
      d.buf = [dev_ newBufferWithLength:sizeof(RtShadowDiag) + kRtShadowDiagCounts * 4
                                options:MTLResourceStorageModeShared];
      if (d.buf) std::memset(d.buf.contents, 0, d.buf.length);
    }
    return d.buf != nil;
  }

  // Frame slot k's contact counts (its frame done): the retained models
  // that many of its blocks found lying on what the camera saw let go
  // (RtScene::DropRetained), then zeroed.
  static constexpr uint64_t kRetainContactEvery = 4;
  static constexpr uint32_t kRetainContactMin = 32;  // blocks (2 x 2 pixels) a model must cover
  id<MTLBuffer> ml_retain_hits_[4] = {};  // (kSlots, declared further on)
  std::vector<std::pair<uint32_t, uint64_t>> ml_retained_slots_[4];
  std::vector<uint64_t> ml_retain_drop_;
  uint64_t ml_retain_dropped_ = 0;
  void MlRetainContact(int k) {
    std::vector<std::pair<uint32_t, uint64_t>>& slots = ml_retained_slots_[k];
    if (slots.empty() || !ml_retain_hits_[k]) return;
    auto* n = static_cast<uint32_t*>(ml_retain_hits_[k].contents);
    ml_retain_drop_.clear();
    // (A model in pieces has a slot a piece: its blocks summed over them.
    // The slots of one model come one after another.)
    uint64_t at_id = 0;
    uint32_t sum = 0, all = 0, beyond = 0;
    for (const auto& [at, id] : slots) {
      if (id != at_id) {
        if (at_id && sum >= kRetainContactMin) ml_retain_drop_.push_back(at_id);
        at_id = id, sum = 0;
      }
      if (at < kRtManyRetainSlots - 1) {
        sum += n[at];
        all += n[at];
      } else {
        beyond++;
      }
    }
    if (at_id && sum >= kRetainContactMin) ml_retain_drop_.push_back(at_id);
    // (The frame log: the blocks found lying under retained models, the
    // models let go, and retained slots past the counters, if any.)
    if (FrameLogOn())
      FrameLogLine("e %.4f retained contact: %u blocks, %zu let go, %zu retained slots, %u past the counters",
                   FrameLogSeconds(), all, ml_retain_drop_.size(), slots.size(), beyond);
    if (!ml_retain_drop_.empty()) {
      rt_->DropRetained(ml_retain_drop_.data(), ml_retain_drop_.size());
      ml_retain_dropped_ += ml_retain_drop_.size();
      static const bool say = [] {
        const char* e = std::getenv("RECOMP_RT_LAMP_PROBE");
        return e && *e;
      }();
      if (say)
        std::fprintf(stderr, "gfx: retained models lying on what is seen, let go: %zu (%llu in all)\n",
                     ml_retain_drop_.size(), (unsigned long long)ml_retain_dropped_);
    }
    std::memset(n, 0, kRtManyRetainSlots * 4);
    slots.clear();
  }

  // RECOMP_RT_LAMP_PROBE: frame slot k's lamp-ray blockers (its frame done)
  // added up; every 60 frames on stderr: the centre block's blocker and the
  // meshes that blocked most, then zeroed.
  static constexpr size_t kMlProbeBytes = (64 + 8192 + 64 + 8192) * 4;
  id<MTLBuffer> ml_probe_[4] = {};  // (kSlots, declared further on)
  std::vector<uint64_t> ml_probe_mesh_ = std::vector<uint64_t>(4096), ml_probe_dist_ = std::vector<uint64_t>(64);
  std::vector<uint32_t> ml_probe_tris_ = std::vector<uint32_t>(4096);
  std::vector<uint64_t> ml_probe_contact_ = std::vector<uint64_t>(4096);
  std::vector<uint32_t> ml_probe_kind_ = std::vector<uint32_t>(4096);
  uint64_t ml_probe_rays_ = 0, ml_probe_blocked_ = 0, ml_probe_frames_ = 0;
  uint64_t ml_probe_contact_rays_ = 0, ml_probe_contact_drawn_ = 0, ml_probe_contact_kept_ = 0,
           ml_probe_contact_other_ = 0;
  double ml_probe_eye_[3] = {};
  uint64_t ml_probe_band_[4] = {}, ml_probe_band_hit_[4] = {};
  void MlProbeLog(int k) {
    if (!ml_probe_[k]) return;
    auto* n = static_cast<uint32_t*>(ml_probe_[k].contents);
    ml_probe_rays_ += n[0];
    ml_probe_blocked_ += n[1];
    for (int i = 0; i < 4096; i++) {
      ml_probe_mesh_[i] += n[64 + i];
      if (n[64 + 4096 + i]) ml_probe_tris_[i] = n[64 + 4096 + i];
    }
    for (int i = 0; i < 64; i++) ml_probe_dist_[i] += n[64 + 8192 + i];
    ml_probe_contact_rays_ += n[22];
    ml_probe_contact_drawn_ += n[23];
    ml_probe_contact_kept_ += n[24];
    ml_probe_contact_other_ += n[25];
    for (int i = 0; i < 4; i++) ml_probe_band_[i] += n[26 + i], ml_probe_band_hit_[i] += n[30 + i];
    for (int i = 0; i < 4096; i++) {
      ml_probe_contact_[i] += n[64 + 8192 + 64 + i];
      ml_probe_kind_[i] |= n[64 + 8192 + 64 + 4096 + i];
    }
    if (n[21]) {
      float v[17];
      std::memcpy(v, n + 4, sizeof v);
      std::fprintf(stderr,
                   "gfx: lamp probe: centre blocked at %.4f m by draw %u mesh %u (%u triangles) %s face, info flags "
                   "%#x, triangle %u; ray from (%.3f %.3f %.3f) toward (%.3f %.3f %.3f), normal (%.3f %.3f %.3f), "
                   "offset %.4f\n",
                   v[0], n[5], n[6], n[7], n[8] ? "front" : "back", n[9], n[10], v[7], v[8], v[9], v[10], v[11], v[12],
                   v[13], v[14], v[15], v[16]);
    }
    std::memset(n, 0, kMlProbeBytes);
    if (++ml_probe_frames_ % 60) return;
    std::string line;
    char b[128];
    for (int t = 0; t < 6; t++) {
      int best = -1;
      for (int i = 0; i < 4096; i++)
        if (ml_probe_mesh_[i] && (best < 0 || ml_probe_mesh_[i] > ml_probe_mesh_[best])) best = i;
      if (best < 0) break;
      std::snprintf(b, sizeof b, " mesh %d (%u triangles) %llu;", best, ml_probe_tris_[best],
                    (unsigned long long)ml_probe_mesh_[best]);
      line += b;
      ml_probe_mesh_[best] = 0;
    }
    std::string dist;
    for (int i = 0; i < 64; i++)
      if (ml_probe_dist_[i]) {
        std::snprintf(b, sizeof b, " %dmm:%llu", i, (unsigned long long)ml_probe_dist_[i]);
        dist += b;
      }
    std::fprintf(stderr, "gfx: lamp probe: 60 frames, %llu rays, %llu blocked; most by:%s\ngfx: lamp probe: blocker distance (mm, 63 = beyond):%s\n",
                 (unsigned long long)ml_probe_rays_, (unsigned long long)ml_probe_blocked_, line.c_str(), dist.c_str());
    if (ml_probe_contact_rays_) {
      std::string top;
      for (int t = 0; t < 8; t++) {
        int best = -1;
        for (int i = 0; i < 4096; i++)
          if (ml_probe_contact_[i] && (best < 0 || ml_probe_contact_[i] > ml_probe_contact_[best])) best = i;
        if (best < 0) break;
        const uint32_t k = ml_probe_kind_[best];
        std::snprintf(b, sizeof b, " mesh %d (%u triangles, %s%s%s) %.2f%%;", best, ml_probe_tris_[best],
                      k & 1 ? "retained " : "", k & 2 ? "main-view " : "", k & 4 ? "other-view" : "",
                      100.0 * double(ml_probe_contact_[best]) / double(ml_probe_contact_rays_));
        top += b;
        ml_probe_contact_[best] = 0;
      }
      std::fprintf(stderr,
                   "gfx: contact probe: camera (%.1f %.1f %.1f): of %llu points, %.2f%% lie under a surface within 5 cm "
                   "(%.2f%% retained, %.2f%% drawn, of them %.2f%% another view's); by distance <20 m %.2f%%, 20-60 %.2f%%, 60-150 %.2f%%, >150 "
                   "%.2f%%; by:%s\n",
                   ml_probe_eye_[0], ml_probe_eye_[1], ml_probe_eye_[2], (unsigned long long)ml_probe_contact_rays_,
                   100.0 * double(ml_probe_contact_drawn_ + ml_probe_contact_kept_) / double(ml_probe_contact_rays_),
                   100.0 * double(ml_probe_contact_kept_) / double(ml_probe_contact_rays_),
                   100.0 * double(ml_probe_contact_drawn_) / double(ml_probe_contact_rays_),
                   100.0 * double(ml_probe_contact_other_) / double(ml_probe_contact_rays_),
                   ml_probe_band_[0] ? 100.0 * double(ml_probe_band_hit_[0]) / double(ml_probe_band_[0]) : 0.0,
                   ml_probe_band_[1] ? 100.0 * double(ml_probe_band_hit_[1]) / double(ml_probe_band_[1]) : 0.0,
                   ml_probe_band_[2] ? 100.0 * double(ml_probe_band_hit_[2]) / double(ml_probe_band_[2]) : 0.0,
                   ml_probe_band_[3] ? 100.0 * double(ml_probe_band_hit_[3]) / double(ml_probe_band_[3]) : 0.0,
                   top.c_str());
      for (int i = 0; i < 4; i++) ml_probe_band_[i] = ml_probe_band_hit_[i] = 0;
    }
    std::fill(ml_probe_contact_.begin(), ml_probe_contact_.end(), 0);
    std::fill(ml_probe_kind_.begin(), ml_probe_kind_.end(), 0);
    ml_probe_contact_rays_ = ml_probe_contact_drawn_ = ml_probe_contact_kept_ = ml_probe_contact_other_ = 0;
    std::fill(ml_probe_mesh_.begin(), ml_probe_mesh_.end(), 0);
    std::fill(ml_probe_dist_.begin(), ml_probe_dist_.end(), 0);
    ml_probe_rays_ = ml_probe_blocked_ = 0;
  }

  // Frame slot k's diagnostics, its frame done: on stderr (two lines), then
  // zeroed.
  void ShadowDiagLog(int k) {
    ShadowDiag& d = shadow_diag_[k];
    if (!d.pending) return;
    d.pending = false;
    auto* r = static_cast<RtShadowDiag*>(d.buf.contents);
    auto* counts = reinterpret_cast<uint32_t*>(r + 1);
    auto describe = [&](uint32_t id) {
      char b[256];
      if (id >= d.summary.size()) {
        std::snprintf(b, sizeof b, "#%u (not summarised)", id);
        return std::string(b);
      }
      const RtInstanceSummary& s = d.summary[id];
      static const char* const kView[] = {"main", "any camera", "captured"};
      int n = std::snprintf(b, sizeof b, "#%u %s%s%s draw %u mesh %u %u triangles", id, kView[s.view & 3],
                            s.carried ? " carried" : "", s.retained ? " retained" : "", s.draw, s.mesh, s.triangles);
      if (s.box && n > 0 && size_t(n) < sizeof b)
        std::snprintf(b + n, sizeof b - size_t(n), " box (%.1f %.1f %.1f)-(%.1f %.1f %.1f)", s.lo[0], s.lo[1], s.lo[2],
                      s.hi[0], s.hi[1], s.hi[2]);
      return std::string(b);
    };
    uint32_t views[3] = {}, carried = 0, retained = 0;
    for (const RtInstanceSummary& s : d.summary) {
      views[s.view % 3]++;
      carried += s.carried ? 1 : 0;
      retained += s.retained ? 1 : 0;
    }
    std::string centre;
    char b[512];
    switch (r->state) {
      case 1: std::snprintf(b, sizeof b, "sky (stored %.7g)", r->point[3]); break;
      case 2:
      case 3:
        std::snprintf(b, sizeof b,
                      "stored %.7g point (%.2f %.2f %.2f) normal (%.3f %.3f %.3f) ray from (%.2f %.2f %.2f) offset %.4f: ",
                      r->point[3], r->point[0], r->point[1], r->point[2], r->normal[0], r->normal[1], r->normal[2],
                      r->origin[0], r->origin[1], r->origin[2], r->origin[3]);
        centre = b;
        if (r->state == 2) {
          centre += "lit";
        } else {
          std::snprintf(b, sizeof b, "blocked at %.3f by ", r->distance);
          centre += b + describe(r->instance) + " triangle " + std::to_string(r->primitive);
        }
        b[0] = 0;
        break;
      default: std::snprintf(b, sizeof b, "not written"); break;
    }
    if (b[0]) centre = b;
    std::fprintf(stderr,
                 "gfx: shadow diag: toward the light (%.3f %.3f %.3f), camera (%.2f %.2f %.2f), mask %ux%u, %u "
                 "instances (%u main, %u any camera, %u captured; %u carried, %u retained); centre pixel: %s\n",
                 d.light[0], d.light[1], d.light[2], d.eye[0], d.eye[1], d.eye[2], d.size[0], d.size[1], d.instances,
                 views[0], views[1], views[2], carried, retained, centre.c_str());
    uint32_t top[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint64_t blocked = 0;
    for (uint32_t i = 0; i < kRtShadowDiagCounts; i++) {
      blocked += counts[i];
      for (int t = 0; t < 3; t++)
        if (counts[i] && (top[t] == UINT32_MAX || counts[i] > counts[top[t]])) {
          for (int u = 2; u > t; u--) top[u] = top[u - 1];
          top[t] = i;
          break;
        }
    }
    std::string line = "gfx: shadow diag: " + std::to_string(blocked) + " rays blocked; most by:";
    for (uint32_t t : top)
      if (t != UINT32_MAX) line += " " + std::to_string(counts[t]) + " by " + describe(t) + ";";
    std::fprintf(stderr, "%s\n", line.c_str());
    std::memset(d.buf.contents, 0, d.buf.length);
  }

  // A draw (or a clear) writing the pass's depth and colour targets: their
  // contents' numbers move on (RtShadowWant::epoch); a main-view draw that
  // voted for a camera (RtScene::TakeVote) names it as theirs where it
  // drew over the whole viewport with depth range 0-1, and forgets it
  // otherwise; a clear forgets it.
  void RtNoteWrites(bool depth, bool color, bool clear) {
    const uint64_t vote = clear ? 0 : rt_->TakeVote();
    const bool full = viewport_.x == 0 && viewport_.y == 0 && int(viewport_.width) == pass_.w &&
                      int(viewport_.height) == pass_.h && viewport_.min_z == 0.0f && viewport_.max_z == 1.0f;
    if (!depth && !color) return;
    const uint64_t epoch = ++rt_epoch_seq_;
    auto note = [&](uint64_t& e, uint64_t& c) {
      e = epoch;
      c = RtShadowWriter(c, clear, vote, full);
    };
    if (depth && pass_.depth_id)
      if (auto it = depths_.find(pass_.depth_id); it != depths_.end()) note(it->second.rt_epoch, it->second.rt_camera);
    if (color)
      for (int i = 0; i < pass_.targets; i++)
        if (Tex* t = FindTex(pass_.ids[i])) note(t->rt_epoch, t->rt_camera);
  }

  // The end of a frame: its builds not encoded yet; and if presented, its
  // TLAS and the debug view (a frame a readback ends goes on into the
  // next). Once a frame, with no encoder open.
  void RtEndFrame(bool presented) {
    if (!rt_ || !frame_open_ || rt_ended_ == serial_) return;
    // (For the perf line and the frame log's: up to the scene's EndFrame.)
    const uint64_t end_t0 = perf_ || frame_log_ ? NowNs() : 0;
    MlAddFlush();
    rt_ended_ = serial_;
    // The mask over the frame (RayTracing::Debug::ShadowMask): its last one.
    __unsafe_unretained id<MTLTexture> shown =
        presented && shadow_view_ && shadow_memo_.valid && !shadow_memo_.white ? shadow_memo_mask_ : nil;
    // The reflections over it (reflection_view): their last.
    const bool refl_shown = presented && refl_view_ && refl_last_ >= 0 &&
                            refl_cache_.entry[refl_last_].memo.valid && !refl_cache_.entry[refl_last_].memo.white;
    __unsafe_unretained id<MTLTexture> refl = refl_shown ? refl_out_[refl_last_] : nil;
    __unsafe_unretained id<MTLTexture> refl_dist = refl_shown ? refl_dist_[refl_last_] : nil;
    const float refl_scale = refl_last_scale_;
    if (presented) {
      shadow_memo_ = {};  // a mask is traced again in the next presented frame
      RtLightMaskEndFrame(&light_cache_, light_budget_);  // and the local lights' (their masks kept)
      light_traced_ = 0;
      LightSetEndFrame();
      ml_tried_ = false;
      ao_memo_ = {};
      bounce_live_ = false;
      refl_cache_ = {};   // and reflections
      refl_last_ = -1;
      snap_memo_ = {};
      snap_frame_ = nullptr;
      shadow_frames_++;
    }
    if (shown) {
      const RtShadowMasker* sm = shadow_masker_.get();
      __unsafe_unretained id<MTLTexture> back = backbuffer_;
      Rec([this, sm, shown, back] { sm->EncodeView(cmd_, shown, back); });
    }
    if (refl) {
      const RtReflector* rf = reflector_.get();
      __unsafe_unretained id<MTLTexture> back = backbuffer_;
      Rec([this, rf, refl, refl_dist, refl_scale, back] { rf->EncodeView(cmd_, refl, refl_dist, refl_scale, back); });
    }
    RtScene* rt = rt_.get();
    const RtFrame* f = rt->EndFrame(presented);
    if (end_t0) {
      const uint64_t ns = NowNs() - end_t0;
      if (perf_) pc_.rt_end_ns += ns;
      if (frame_log_) frame_log_counts_.rt_end_ns += ns;
    }
    if (!f) return;
    __unsafe_unretained id<MTLTexture> target = presented ? backbuffer_ : nil;
    Rec([this, rt, f, target] {
      rt->EncodeFrame(cmd_, f);
      if (target) rt->EncodeDebug(cmd_, f, target);
    });
  }

  // A pipeline copying a depth buffer into one of `pf` (DepthProxy, the
  // upscaler's depth).
  id<MTLRenderPipelineState> DepthCopyPipe(MTLPixelFormat pf) {
    auto it = depth_copy_pipes_.find(pf);
    if (it != depth_copy_pipes_.end()) return it->second;
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [internal_lib_ newFunctionWithName:@"tri_vs"];
    pd.fragmentFunction = [internal_lib_ newFunctionWithName:@"depth_copy_fs"];
    pd.depthAttachmentPixelFormat = pf;
    if (pf == MTLPixelFormatDepth32Float_Stencil8) pd.stencilAttachmentPixelFormat = pf;
    NSError* err = nil;
    id<MTLRenderPipelineState> ps = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) FatalF("gfx: depth copy pipeline: %s", err.localizedDescription.UTF8String);
    depth_copy_pipes_[pf] = ps;
    return ps;
  }

  // Encodes a copy of depth buffer `src` into `dst` (any size; stencil
  // cleared).
  void CopyDepth(__unsafe_unretained id<MTLTexture> src, __unsafe_unretained id<MTLTexture> dst) {
    __unsafe_unretained id<MTLRenderPipelineState> pipe = DepthCopyPipe(dst.pixelFormat);
    __unsafe_unretained id<MTLDepthStencilState> ds = depth_copy_ds_;
    __unsafe_unretained id<MTLSamplerState> smp = quad_sampler_[0];
    const bool stencil = dst.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    Rec([=, this] {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = dst;
      rp.depthAttachment.loadAction = MTLLoadActionDontCare;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      if (stencil) {
        rp.stencilAttachment.texture = dst;
        rp.stencilAttachment.loadAction = MTLLoadActionClear;
        rp.stencilAttachment.clearStencil = 0;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
      }
      id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "depth copy");
      [e setRenderPipelineState:pipe];
      [e setDepthStencilState:ds];
      [e setFragmentTexture:src atIndex:0];
      [e setFragmentSamplerState:smp atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
    });
  }

  // A full-size copy (w x h) of depth buffer `src` at the internal
  // resolution, kept per size, copied once a frame (when first wanted:
  // after the scene). From BeginPassEncoder: no encoder is open.
  id<MTLTexture> DepthProxy(id<MTLTexture> src, NSUInteger w, NSUInteger h) {
    std::vector<DepthCopy>& sizes = depth_proxies_[(__bridge void*)src];
    auto px = std::find_if(sizes.begin(), sizes.end(),
                           [&](const DepthCopy& c) { return c.tex.width == w && c.tex.height == h; });
    if (px == sizes.end()) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageRenderTarget;
      px = sizes.insert(sizes.end(), DepthCopy{[dev_ newTextureWithDescriptor:td], 0});
    }
    if (px->frame != serial_) {
      px->frame = serial_;
      CopyDepth(src, px->tex);
    }
    return px->tex;
  }

  // mf: this frame's clip space to the last frame's, from the camera alone
  // (identity without a last frame). True if the history starts over: no
  // last frame, or a cut (cutscenes, respawns, camera switches): a point
  // 20 m into the scene (its depth value from the camera's planes) moved by
  // more than a third of the screen.
  // With `reversed`, mf takes the depth as stored (1 - d, reversed_z.h):
  // R * m, R taking (x, y, 1 - d, 1) to (x, y, d, 1).
  bool CameraMotion(float mf[16], bool reversed) {
    double inv[16], m[16];
    const bool moved = have_prev_ && Invert4(vp_cur_, inv);
    if (moved) Mul4(inv, vp_prev_, m);
    for (int i = 0; i < 16; i++) mf[i] = moved ? float(m[i]) : (i % 5 == 0 ? 1.0f : 0.0f);
    if (reversed)
      for (int k = 0; k < 4; k++) {
        mf[12 + k] += mf[8 + k];
        mf[8 + k] = -mf[8 + k];
      }
    double cut_z = 0.99, n, f;
    if (NearFar(vp_cur_, &n, &f)) {
      const double d = std::clamp(20.0, n * 2.0, f * 0.5);
      cut_z = f / (f - n) * (1.0 - n / d);
    }
    const bool reset = fx_reset_ || !have_prev_ || (moved && CameraCut(m, cut_z));
    fx_reset_ = false;
    return reset;
  }

  // TAA's pipeline, the composite's copy it reads and its history (two),
  // at the back buffer's size. False if the back buffer cannot take its
  // output.
  bool EnsureTaa() {
    if (!(backbuffer_.usage & MTLTextureUsageShaderWrite)) return false;
    if (!EnsureSharpen()) return false;
    const NSUInteger w = NSUInteger(bb_w_), h = NSUInteger(bb_h_);
    if (taa_pipe_ && taa_src_.width == w && taa_src_.height == h) return true;
    if (!taa_pipe_) {
      NSError* err = nil;
      taa_pipe_ = [dev_ newComputePipelineStateWithFunction:[internal_lib_ newFunctionWithName:@"taa_cs"] error:&err];
      if (!taa_pipe_) FatalF("gfx: TAA pipeline: %s", err.localizedDescription.UTF8String);
    }
    for (int i = 0; i < 2; i++) {
      if (taa_history_[i]) Keep(taa_history_[i]);
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      taa_history_[i] = [dev_ newTextureWithDescriptor:td];
    }
    if (taa_src_) Keep(taa_src_);
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:backbuffer_.pixelFormat
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead;
    taa_src_ = [dev_ newTextureWithDescriptor:td];
    fx_reset_ = true;
    std::fprintf(stderr, "gfx: TAA %lux%lu, sharpening %.2f, texture LOD bias %.1f\n", (unsigned long)w,
                 (unsigned long)h, sharpness_, kTaaLodBias);
    return true;
  }

  // RCAS's pipeline (rcas.h), made once; true without it when sharpening
  // is off. Sharpening writes the back buffer from a compute pass: false
  // if the back buffer cannot take that.
  bool EnsureSharpen() {
    if (!RcasActive(sharpness_) || rcas_pipe_) return true;
    if (!(backbuffer_.usage & MTLTextureUsageShaderWrite)) return false;
    NSError* err = nil;
    rcas_pipe_ = [dev_ newComputePipelineStateWithFunction:[internal_lib_ newFunctionWithName:@"rcas_cs"] error:&err];
    if (!rcas_pipe_) FatalF("gfx: sharpening pipeline: %s", err.localizedDescription.UTF8String);
    return true;
  }

  // RCAS from src into the back buffer (the same size), in the frame's
  // command buffer, before the HUD.
  void EncodeSharpen(id<MTLTexture> src, id<MTLTexture> back, float con) {
    id<MTLComputeCommandEncoder> e = [cmd_ computeCommandEncoder];
    [e setComputePipelineState:rcas_pipe_];
    [e setTexture:src atIndex:0];
    [e setTexture:back atIndex:1];
    [e setBytes:&con length:sizeof con atIndex:0];
    [e dispatchThreads:MTLSizeMake(back.width, back.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [e endEncoding];
  }

  // The upscaler for low_color_'s size to the back buffer's, and its
  // inputs' textures. False if MetalFX cannot (bilinear then). It writes
  // into the back buffer itself where that allows (made so for MetalFX),
  // else into fx_out_, copied there; with sharpening, always into fx_out_,
  // which RCAS reads into the back buffer.
  bool EnsureScaler() {
    if (scaler_failed_ || !EnsureSharpen()) return false;
    const NSUInteger iw = low_color_.width, ih = low_color_.height;
    if (scaler_ && scaler_.inputWidth == iw && scaler_.inputHeight == ih &&
        scaler_.outputWidth == NSUInteger(bb_w_) && scaler_.outputHeight == NSUInteger(bb_h_))
      return true;
    if (@available(macOS 13.0, iOS 16.0, *)) {
      MTLFXTemporalScalerDescriptor* sd = [MTLFXTemporalScalerDescriptor new];
      sd.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
      sd.depthTextureFormat = MTLPixelFormatDepth32Float;
      sd.motionTextureFormat = MTLPixelFormatRG16Float;
      sd.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
      sd.inputWidth = iw;
      sd.inputHeight = ih;
      sd.outputWidth = NSUInteger(bb_w_);
      sd.outputHeight = NSUInteger(bb_h_);
      id<MTLFXTemporalScaler> sc = [sd newTemporalScalerWithDevice:dev_];
      if (!sc) {
        WarnF("gfx: MetalFX temporal scaler %lux%lu -> %dx%d unavailable: bilinear\n", (unsigned long)iw,
              (unsigned long)ih, bb_w_, bb_h_);
        scaler_failed_ = true;
        return false;
      }
      sc.motionVectorScaleX = float(iw);
      sc.motionVectorScaleY = float(ih);
      sc.depthReversed = NO;
      for (id o : {(id)scaler_, (id)motion_tex_, (id)fx_depth_, (id)fx_out_})
        if (o) Keep(o);
      scaler_ = sc;
      fx_out_ = nil;
      auto tex = [&](MTLPixelFormat pf, NSUInteger w, NSUInteger h, MTLTextureUsage usage) {
        MTLTextureDescriptor* td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = usage;
        return [dev_ newTextureWithDescriptor:td];
      };
      motion_tex_ = tex(MTLPixelFormatRG16Float, iw, ih,
                        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | sc.motionTextureUsage);
      fx_depth_ = tex(MTLPixelFormatDepth32Float, iw, ih,
                      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | sc.depthTextureUsage);
      // Sharpened, MetalFX writes fx_out_ and RCAS reads it into the back
      // buffer.
      if (RcasActive(sharpness_) || (backbuffer_.usage & sc.outputTextureUsage) != sc.outputTextureUsage)
        fx_out_ = tex(MTLPixelFormatBGRA8Unorm, NSUInteger(bb_w_), NSUInteger(bb_h_),
                      MTLTextureUsageShaderRead | sc.outputTextureUsage);
      EnsureMotionPipe();
      fx_reset_ = true;
      std::fprintf(stderr, "gfx: MetalFX temporal scaler %lux%lu -> %dx%d%s\n", (unsigned long)iw,
                   (unsigned long)ih, bb_w_, bb_h_, fx_out_ && !RcasActive(sharpness_) ? " (then copied)" : "");
      if (RcasActive(sharpness_)) std::fprintf(stderr, "gfx: sharpening %.2f after MetalFX (temporal)\n", sharpness_);
      return true;
    }
    scaler_failed_ = true;
    return false;
  }

  // The pass making the camera's motion and a Depth32Float copy of the
  // scene's depth (motion_fs), for MetalFX's scalers.
  void EnsureMotionPipe() {
    if (motion_pipe_) return;
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [internal_lib_ newFunctionWithName:@"tri_vs"];
    pd.fragmentFunction = [internal_lib_ newFunctionWithName:@"motion_fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRG16Float;
    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    NSError* err = nil;
    motion_pipe_ = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!motion_pipe_) FatalF("gfx: motion pipeline: %s", err.localizedDescription.UTF8String);
  }

  // MetalFX's spatial scaler for low_color_'s size to the back buffer's,
  // and the SMAA'd composite it reads (spatial_in_), its output into the
  // back buffer (with sharpening, into spatial_out_, which RCAS reads
  // into the back buffer). False if MetalFX cannot (bilinear then).
  bool EnsureSpatial() {
    if (spatial_failed_ || !EnsureSharpen()) return false;
    const NSUInteger iw = low_color_.width, ih = low_color_.height;
    if (spatial_ && spatial_.inputWidth == iw && spatial_.inputHeight == ih &&
        spatial_.outputWidth == NSUInteger(bb_w_) && spatial_.outputHeight == NSUInteger(bb_h_))
      return true;
    if (@available(macOS 13.0, iOS 16.0, *)) {
      MTLFXSpatialScalerDescriptor* sd = [MTLFXSpatialScalerDescriptor new];
      sd.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
      sd.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
      sd.inputWidth = iw;
      sd.inputHeight = ih;
      sd.outputWidth = NSUInteger(bb_w_);
      sd.outputHeight = NSUInteger(bb_h_);
      sd.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;  // the game's 8-bit sRGB-ish colours
      id<MTLFXSpatialScaler> sc = [sd newSpatialScalerWithDevice:dev_];
      if (!sc) {
        WarnF("gfx: MetalFX spatial scaler %lux%lu -> %dx%d unavailable: bilinear\n", (unsigned long)iw,
              (unsigned long)ih, bb_w_, bb_h_);
        spatial_failed_ = true;
        return false;
      }
      for (id o : {(id)spatial_, (id)spatial_in_, (id)spatial_out_})
        if (o) Keep(o);
      spatial_ = sc;
      MTLTextureDescriptor* td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:iw height:ih mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | sc.colorTextureUsage;
      spatial_in_ = [dev_ newTextureWithDescriptor:td];
      spatial_out_ = nil;
      if (RcasActive(sharpness_) || (backbuffer_.usage & sc.outputTextureUsage) != sc.outputTextureUsage) {
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                width:NSUInteger(bb_w_)
                                                               height:NSUInteger(bb_h_)
                                                            mipmapped:NO];
        td.storageMode = MTLStorageModePrivate;
        td.usage = MTLTextureUsageShaderRead | sc.outputTextureUsage;
        spatial_out_ = [dev_ newTextureWithDescriptor:td];
      }
      std::fprintf(stderr, "gfx: MetalFX spatial scaler %lux%lu -> %dx%d%s\n", (unsigned long)iw,
                   (unsigned long)ih, bb_w_, bb_h_, spatial_out_ && !RcasActive(sharpness_) ? " (then copied)" : "");
      if (RcasActive(sharpness_)) std::fprintf(stderr, "gfx: sharpening %.2f after MetalFX (spatial)\n", sharpness_);
      return true;
    }
    spatial_failed_ = true;
    return false;
  }

  // ---- SMAA 1x (Jimenez et al., SMAA.hlsl), the scene's antialiasing
  // when RECOMP_UPSCALER=smaa: from the composite to the back buffer
  // through luma edges, blending weights (from the area and search lookup
  // textures), then neighbourhood blending; the HUD draws on after.

  id<MTLRenderPipelineState> SmaaPipeline(int pass, MTLPixelFormat target) {
    static const int kSamplers[3] = {1, 3, 2};  // per pass
    id<MTLFunction> fn[2];
    for (int stage = 0; stage < 2; stage++) {
      const bool vertex = stage == 0;
      std::string glsl = "#version 450\n#define SMAA_GLSL_4 1\n#define SMAA_PRESET_HIGH 1\n"
                         "#define SMAA_RT_METRICS smaa_metrics\n#define PASS " +
                         std::to_string(pass) + "\n";
      glsl += vertex ? "#define SMAA_INCLUDE_PS 0\n" : "#define SMAA_INCLUDE_VS 0\n";
      glsl += "layout(std140, set = 0, binding = 0) uniform SmaaMetrics { vec4 smaa_metrics; };\n";
      for (int i = 0; !vertex && i < kSamplers[pass]; i++)
        glsl += "layout(set = 0, binding = " + std::to_string(i) + ") uniform sampler2D t" +
                std::to_string(i) + ";\n";
      glsl += kSmaaHlsl;
      glsl += vertex ? R"(
layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec2 v_pix;
layout(location = 2) out vec4 v_off[3];
void main() {
  // One triangle over the target; texture y runs down.
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  v_uv = vec2(p.x, 1.0 - p.y);
  v_pix = vec2(0.0);
  v_off[0] = v_off[1] = v_off[2] = vec4(0.0);
#if PASS == 0
  SMAAEdgeDetectionVS(v_uv, v_off);
#elif PASS == 1
  SMAABlendingWeightCalculationVS(v_uv, v_pix, v_off);
#else
  SMAANeighborhoodBlendingVS(v_uv, v_off[0]);
#endif
}
)" : R"(
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pix;
layout(location = 2) in vec4 v_off[3];
layout(location = 0) out vec4 o_color;
void main() {
#if PASS == 0
  o_color = vec4(SMAALumaEdgeDetectionPS(v_uv, v_off, t0), 0.0, 0.0);
#elif PASS == 1
  o_color = SMAABlendingWeightCalculationPS(v_uv, v_pix, v_off, t0, t1, t2, vec4(0.0));
#else
  o_color = SMAANeighborhoodBlendingPS(v_uv, v_off[0], t0, t1);
#endif
}
)";
      std::vector<uint32_t> spirv;
      std::string log, msl;
      if (!GlslToSpirv(glsl, vertex ? ShaderStage::Vertex : ShaderStage::Fragment, &spirv, &log))
        FatalF("gfx: compiling SMAA pass %d: %s", pass, log.c_str());
      if (!SpirvToMsl(spirv, &msl, &log)) FatalF("gfx: SMAA pass %d to MSL: %s", pass, log.c_str());
      NSError* err = nil;
      id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(msl.c_str()) options:nil error:&err];
      if (!lib) FatalF("gfx: SMAA pass %d MSL: %s", pass, err.localizedDescription.UTF8String);
      fn[stage] = [lib newFunctionWithName:@"main0"];
    }
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = fn[0];
    pd.fragmentFunction = fn[1];
    pd.colorAttachments[0].pixelFormat = target;
    if (pass < 2) pd.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;  // smaa_stencil_
    NSError* err = nil;
    id<MTLRenderPipelineState> ps = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) FatalF("gfx: SMAA pass %d pipeline: %s", pass, err.localizedDescription.UTF8String);
    return ps;
  }

  // Pipelines and lookup textures once; the edge and weight targets at
  // the back buffer's size.
  void SmaaPrepare(NSUInteger w, NSUInteger h) {
    auto tex = [&](MTLPixelFormat pf, NSUInteger w, NSUInteger h, bool target) {
      MTLTextureDescriptor* td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
      td.storageMode = target ? MTLStorageModePrivate : MTLStorageModeShared;
      td.usage = MTLTextureUsageShaderRead | (target ? MTLTextureUsageRenderTarget : 0);
      return [dev_ newTextureWithDescriptor:td];
    };
    if (!smaa_pipe_[0]) {
      smaa_pipe_[0] = SmaaPipeline(0, MTLPixelFormatRG8Unorm);
      smaa_pipe_[1] = SmaaPipeline(1, MTLPixelFormatRGBA8Unorm);
      smaa_pipe_[2] = SmaaPipeline(2, MTLPixelFormatBGRA8Unorm);
      // The edge pass sets 1 where it found edges (it discards elsewhere);
      // the weight pass runs only there.
      for (int i = 0; i < 2; i++) {
        MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
        sd.stencilCompareFunction = i ? MTLCompareFunctionEqual : MTLCompareFunctionAlways;
        sd.depthStencilPassOperation = i ? MTLStencilOperationKeep : MTLStencilOperationReplace;
        MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
        dd.frontFaceStencil = dd.backFaceStencil = sd;
        smaa_ds_[i] = [dev_ newDepthStencilStateWithDescriptor:dd];
      }
      smaa_area_ = tex(MTLPixelFormatRG8Unorm, AREATEX_WIDTH, AREATEX_HEIGHT, false);
      [smaa_area_ replaceRegion:MTLRegionMake2D(0, 0, AREATEX_WIDTH, AREATEX_HEIGHT)
                    mipmapLevel:0
                      withBytes:areaTexBytes
                    bytesPerRow:AREATEX_PITCH];
      smaa_search_ = tex(MTLPixelFormatR8Unorm, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, false);
      [smaa_search_ replaceRegion:MTLRegionMake2D(0, 0, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT)
                      mipmapLevel:0
                        withBytes:searchTexBytes
                      bytesPerRow:SEARCHTEX_PITCH];
    }
    if (!smaa_edges_ || smaa_edges_.width != w || smaa_edges_.height != h) {
      for (id t : {(id)smaa_edges_, (id)smaa_blend_, (id)smaa_stencil_})
        if (t) Keep(t);
      smaa_edges_ = tex(MTLPixelFormatRG8Unorm, w, h, true);
      smaa_blend_ = tex(MTLPixelFormatRGBA8Unorm, w, h, true);
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatStencil8
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageRenderTarget;
      smaa_stencil_ = [dev_ newTextureWithDescriptor:td];
    }
  }

  // Encodes SMAA from `src` to `dst` (the frame: back to front buffer; a
  // composite: low_color_ to the back buffer).
  void SmaaEncode(__unsafe_unretained id<MTLTexture> src, __unsafe_unretained id<MTLTexture> dst) {
    const NSUInteger sw = src.width, sh = src.height;
    SmaaPrepare(sw, sh);
    struct Metrics {
      float v[4];
    } m{{1.0f / float(sw), 1.0f / float(sh), float(sw), float(sh)}};
    __unsafe_unretained id<MTLTexture> edges = smaa_edges_, blend = smaa_blend_, area = smaa_area_,
                                       search = smaa_search_;
    __unsafe_unretained id<MTLRenderPipelineState> p0 = smaa_pipe_[0], p1 = smaa_pipe_[1], p2 = smaa_pipe_[2];
    __unsafe_unretained id<MTLTexture> stencil = smaa_stencil_;
    __unsafe_unretained id<MTLDepthStencilState> ds0 = smaa_ds_[0], ds1 = smaa_ds_[1];
    __unsafe_unretained id<MTLSamplerState> lin = quad_sampler_[1], point = quad_sampler_[0];
    Rec([=, this] {
      struct Pass {
        __unsafe_unretained id<MTLTexture> target;
        __unsafe_unretained id<MTLRenderPipelineState> pipe;
        __unsafe_unretained id<MTLTexture> in[3];
        __unsafe_unretained id<MTLSamplerState> smp[3];
        int inputs;
      };
      const Pass passes[3] = {
          {edges, p0, {src, nil, nil}, {lin, nil, nil}, 1},
          {blend, p1, {edges, area, search}, {lin, lin, point}, 3},
          {dst, p2, {src, blend, nil}, {lin, lin, nil}, 2},
      };
      for (int i = 0; i < 3; i++) {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = passes[i].target;
        // Edges and weights start at 0 (the edge pass discards where there
        // is none); the last pass writes every pixel.
        rp.colorAttachments[0].loadAction = i < 2 ? MTLLoadActionClear : MTLLoadActionDontCare;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        if (i < 2) {
          rp.stencilAttachment.texture = stencil;
          rp.stencilAttachment.loadAction = i ? MTLLoadActionLoad : MTLLoadActionClear;
          rp.stencilAttachment.clearStencil = 0;
          rp.stencilAttachment.storeAction = i ? MTLStoreActionDontCare : MTLStoreActionStore;
        }
        id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "smaa");
        [e setRenderPipelineState:passes[i].pipe];
        if (i < 2) {
          [e setDepthStencilState:i ? ds1 : ds0];
          [e setStencilReferenceValue:1];
        }
        [e setVertexBytes:m.v length:sizeof m.v atIndex:0];
        [e setFragmentBytes:m.v length:sizeof m.v atIndex:0];
        for (int t = 0; t < passes[i].inputs; t++) {
          [e setFragmentTexture:passes[i].in[t] atIndex:NSUInteger(t)];
          [e setFragmentSamplerState:passes[i].smp[t] atIndex:NSUInteger(t)];
        }
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
      }
    });
  }

  // SMAA over the finished frame, from the back buffer straight into the
  // front one (what Present shows; the back buffer's contents after
  // Present are undefined in D3D9). On the frame as a whole (its HUD too):
  // drawing the scene into another target for the filter (the upscalers'
  // composite redirect, every full-size target marked) cost GTA IV as much
  // as 20 ms a frame at 3440x1440.
  void SmaaFrame() { SmaaEncode(backbuffer_, front_); }

  // The composite is done (at the internal resolution): upscaled into the
  // back buffer, which the HUD then draws on at full size.
  void Composite() {
    composited_ = true;
    Settle();
    EnsureFrame();
    MlAddFlush();
    // TAA's composite is the back buffer itself.
    __unsafe_unretained id<MTLTexture> low = FullSizeScene() ? backbuffer_ : low_color_,
                                       back = backbuffer_;
    const int bw = bb_w_, bh = bb_h_;
    const int lw = int(low.width), lh = int(low.height);
    if (upscaler_ == Upscaler::Smaa) {
      SmaaEncode(low, back);
    } else if (upscaler_ == Upscaler::Taa && scene_depth_ && HaveCamera() && EnsureTaa()) {
      struct {  // taa_cs's TaaParams
        float m[16];
        float inv_size[2];
        float jitter[2];
        float w_min, w_max, w_per_px, gamma;
        uint32_t reset, direct, reversed, pad;
      } tp = {};
      static_assert(sizeof tp == 112);
      const bool reversed = ReversedDepth(scene_depth_.pixelFormat);
      tp.reversed = reversed ? 1 : 0;
      tp.reset = CameraMotion(tp.m, reversed) ? 1 : 0;
      tp.inv_size[0] = 1.0f / float(lw);
      tp.inv_size[1] = 1.0f / float(lh);
      tp.jitter[0] = jitter_[0];  // r_position: +x right, +y down, in pixels
      tp.jitter[1] = jitter_[1];
      // 5% of this frame when still, up to 20% at 10 pixels of motion (as
      // FusionFix's TAA for this game); the clip box one deviation (Salvi's
      // variance clipping: tighter keeps less blurred history).
      tp.w_min = 0.05f;
      tp.w_max = 0.2f;
      tp.w_per_px = 0.1f;
      tp.gamma = 1.0f;
      const float sharp = sharpness_, con = RcasConstant(sharp);
      tp.direct = RcasActive(sharp) ? 0 : 1;
      __unsafe_unretained id<MTLTexture> depth = scene_depth_, h_in = taa_history_[taa_cur_],
                                         h_out = taa_history_[taa_cur_ ^ 1], src = taa_src_;
      __unsafe_unretained id<MTLComputePipelineState> pipe = taa_pipe_, sharpen = rcas_pipe_;
      taa_cur_ ^= 1;
      Rec([=, this] {
        // Sharpened (the default), the resolve writes only the history and
        // the sharpening the back buffer, after it (the encoder's dispatches
        // run in order): it reads the composite where it is. Else the
        // composite is copied first and resolved from the copy into the
        // back buffer (0.5 ms a frame at 3440 x 1440).
        __unsafe_unretained id<MTLTexture> color = back;
        if (sharp <= 0.0f) {
          id<MTLBlitCommandEncoder> bl = TimedBlit(cmd_, "taa copy");
          [bl copyFromTexture:back toTexture:src];
          [bl endEncoding];
          color = src;
        }
        id<MTLComputeCommandEncoder> e = TimedCompute(cmd_, "taa");
        [e setComputePipelineState:pipe];
        [e setTexture:color atIndex:0];
        [e setTexture:depth atIndex:1];
        [e setTexture:h_in atIndex:2];
        [e setTexture:h_out atIndex:3];
        // (Not written but where direct: the back buffer, read at 0 when
        // sharpened, not bound for writing too.)
        [e setTexture:sharp > 0.0f ? h_out : back atIndex:4];
        [e setBytes:&tp length:sizeof tp atIndex:0];
        [e dispatchThreads:MTLSizeMake(NSUInteger(lw), NSUInteger(lh), 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        if (RcasActive(sharp)) {  // the same encoder: Metal orders the two dispatches' texture use
          [e setComputePipelineState:sharpen];
          [e setTexture:h_out atIndex:0];
          [e setTexture:back atIndex:1];
          [e setBytes:&con length:sizeof con atIndex:0];
          [e dispatchThreads:MTLSizeMake(NSUInteger(lw), NSUInteger(lh), 1)
              threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        }
        [e endEncoding];
      });
    } else if (upscaler_ == Upscaler::MetalFX && scene_depth_ && HaveCamera() && EnsureScaler()) {
      struct M16 {
        float v[16];
      } mm;
      // The depth MetalFX takes is the scene's as stored.
      const bool reversed = ReversedDepth(scene_depth_.pixelFormat);
      const bool reset = CameraMotion(mm.v, reversed);
      __unsafe_unretained id<MTLTexture> depth = scene_depth_, motion = motion_tex_, fxd = fx_depth_,
                                         out = fx_out_ ? fx_out_ : back;
      __unsafe_unretained id<MTLRenderPipelineState> mp = motion_pipe_;
      __unsafe_unretained id<MTLDepthStencilState> ds = depth_copy_ds_;
      __unsafe_unretained id<MTLFXTemporalScaler> sc = scaler_;
      const float jx = jitter_[0] * jitter_sign_[0], jy = jitter_[1] * jitter_sign_[1];
      const float mvs = mv_sign_;
      const float con = RcasActive(sharpness_) ? RcasConstant(sharpness_) : 0.0f;
      Rec([=, this] {
        // Motion, and the depth MetalFX takes, in one pass.
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = motion;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.depthAttachment.texture = fxd;
        rp.depthAttachment.loadAction = MTLLoadActionDontCare;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "metalfx motion");
        [e setRenderPipelineState:mp];
        [e setDepthStencilState:ds];
        [e setFragmentTexture:depth atIndex:0];
        [e setFragmentBytes:mm.v length:sizeof mm.v atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [e endEncoding];
        sc.colorTexture = low;
        sc.depthTexture = fxd;
        sc.motionTexture = motion;
        sc.outputTexture = out;
        sc.jitterOffsetX = jx;
        sc.jitterOffsetY = jy;
        sc.motionVectorScaleX = mvs * float(lw);
        sc.motionVectorScaleY = mvs * float(lh);
        sc.depthReversed = reversed;
        sc.reset = reset;
        [sc encodeToCommandBuffer:cmd_];
        TimedMarkerAfter(cmd_, "metalfx upscale", out, gp_scratch_);
        if (out == back) return;
        if (con > 0.0f) return EncodeSharpen(out, back, con);
        id<MTLBlitCommandEncoder> bl = [cmd_ blitCommandEncoder];
        [bl copyFromTexture:out toTexture:back];
        [bl endEncoding];
      });
    } else if (upscaler_ == Upscaler::MetalFXSpatial && EnsureSpatial()) {
      // SMAA at the internal resolution, then the spatial upscale.
      SmaaEncode(low, spatial_in_);
      __unsafe_unretained id<MTLTexture> in = spatial_in_, out = spatial_out_ ? spatial_out_ : back;
      __unsafe_unretained id<MTLFXSpatialScaler> sc = spatial_;
      const float con = RcasActive(sharpness_) ? RcasConstant(sharpness_) : 0.0f;
      Rec([=, this] {
        sc.colorTexture = in;
        sc.outputTexture = out;
        [sc encodeToCommandBuffer:cmd_];
        TimedMarkerAfter(cmd_, "metalfx spatial", out, gp_scratch_);
        if (out == back) return;
        if (con > 0.0f) return EncodeSharpen(out, back, con);
        id<MTLBlitCommandEncoder> bl = [cmd_ blitCommandEncoder];
        [bl copyFromTexture:out toTexture:back];
        [bl endEncoding];
      });
    } else if (low != back) {
      Rec([=, this] {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = back;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "composite quad");
        DrawQuad(e, MTLPixelFormatBGRA8Unorm, low, 0, float(lw), float(lh), {0, 0, lw, lh}, float(bw),
                 float(bh), {0, 0, bw, bh}, true);
        [e endEncoding];
      });
    }
    FogOver();
    BindTarget();  // the back buffer itself now
  }

  // The volumetric fog (RtFog) over the finished scene in the back buffer,
  // in the frames that traced it: after the anti-aliasing or upscaling (the
  // history stays without it), before the HUD. A draw blended into the back
  // buffer: its colour times the transmittance, plus the scattered light
  // (fog_over_fs).
  void FogOver() {
    const bool traced = fog_frame_;
    fog_frame_ = false;
    double hn = 0, hf = 0;
    if (!traced || !fog_ || fog_over_failed_ || !scene_depth_ || !HaveCamera() || !NearFar(vp_cur_, &hn, &hf)) return;
    __unsafe_unretained id<MTLTexture> back = backbuffer_;
    if (!fog_over_pipe_) {
      MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
      pd.vertexFunction = [internal_lib_ newFunctionWithName:@"tri_vs"];
      pd.fragmentFunction = [internal_lib_ newFunctionWithName:@"fog_over_fs"];
      pd.colorAttachments[0].pixelFormat = back.pixelFormat;
      pd.colorAttachments[0].blendingEnabled = YES;
      pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
      pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorSourceAlpha;
      pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
      pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
      NSError* err = nil;
      fog_over_pipe_ = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
      if (!fog_over_pipe_) {
        fog_over_failed_ = true;
        WarnF("gfx: ray-traced volumetric fog: its pass over the scene: %s: not shown\n",
              err ? err.localizedDescription.UTF8String : "no pipeline");
        return;
      }
    }
    struct {  // (MSL: FogOver)
      float near_plane, far_plane;
      uint32_t reversed, pad;
      float near_z, far_z, slices, pad1;
    } fp = {float(hn), float(hf), ReversedDepth(scene_depth_.pixelFormat) ? 1u : 0u, 0u,
            fog_look_.near_z, fog_look_.far_z, float(kRtFogD), 0.0f};
    static_assert(sizeof fp == 32);
    __unsafe_unretained id<MTLTexture> depth = scene_depth_, vol = fog_->Integrated();
    __unsafe_unretained id<MTLRenderPipelineState> pipe = fog_over_pipe_;
    if (!vol) return;
    Rec([=, this] {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = back;
      rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "rt fog over");
      [e setRenderPipelineState:pipe];
      [e setFragmentTexture:depth atIndex:0];
      [e setFragmentTexture:vol atIndex:1];
      [e setFragmentBytes:&fp length:sizeof fp atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
    });
  }

  // At the end of a frame: the next one's jitter and camera history.
  void EndUpscalerFrame() {
    if (denoiser_) DenoiseEndFrame();
    refl_frame_++;
    refl_lit_ = nil;
    refl_lit_copy_ = 0;
    refl_keep_done_ = false;
    if (upscaler_ == Upscaler::Off) return;
    if (frame_scene_ && !composited_ && pass_.low) Composite();  // never covered: as it is
    if (have_vp_) std::memcpy(vp_prev_, vp_cur_, sizeof vp_cur_);
    have_prev_ = have_vp_ && composited_;
    have_vp_ = false;
    // Where the frame's camera came from (the perf counts), and the first
    // time draws before it gave another, said once.
    if (perf_) {
      const CameraVote::State st = cam_vote_.state();
      (st == CameraVote::State::Confirmed ? pc_.camera_confirmed
       : st == CameraVote::State::Unconfirmed ? pc_.camera_unconfirmed
                                              : pc_.camera_none)++;
      pc_.camera_rejected += cam_vote_.rejected();
    }
    if (cam_vote_.state() != CameraVote::State::None && cam_vote_.rejected() && !camera_rejected_said_) {
      camera_rejected_said_ = true;
      WarnF("gfx: the frame's camera: %llu draws before it gave another world^-1 * world-view-projection (up to "
            "%.3g of its largest entry off: their world register holds something else); left out\n",
            (unsigned long long)cam_vote_.rejected(), cam_vote_.worst());
    }
    cam_vote_.Reset();
    frame_scene_ = composited_ = false;
    scene_depth_ = nil;
    // 8 phases for TAA (a short cycle shimmers less when the history is
    // clipped) and the denoiser (Apple's 8 x scale^2 at 1:1), 16 for
    // MetalFX.
    jitter_phase_ = (jitter_phase_ + 1) % (FullSizeScene() ? 8 : 16);
    jitter_[0] = Halton(jitter_phase_ + 1, 2) - 0.5f;
    jitter_[1] = Halton(jitter_phase_ + 1, 3) - 0.5f;
  }

  // ---- MetalFX's denoiser of the lit scene (rt_denoise.h)

  // Lit frames in a row whose lit scene no post-process draw read before
  // the denoiser gives up (DenoiseOff): the game's hook names the wrong
  // shaders, or reads a copy made otherwise than by StretchRect.
  static constexpr int kDenoiseMissLimit = 120;

  // Where shadows are traced and the game names its shaders: the denoiser,
  // in place of the post-process anti-aliasing or upscaler asked for (the
  // scene jittered at the back buffer's size; that one, or TAA in place of
  // an upscaler, is what DenoiseOff falls back to).
  // The lit texture of a game's own shadow map (RayShadow::own_map_lit,
  // RayTracing::own_maps_lit): 1x1 R32Float (read through the mask's
  // nearest sampler), one per value; false if it cannot be made.
  bool OwnMapLit(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if (own_map_tex_.count(bits)) return true;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                  width:1
                                                                                 height:1
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [dev_ newTextureWithDescriptor:td];
    if (!t) return false;
    [t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&value bytesPerRow:4];
    own_map_tex_[bits] = t;
    return true;
  }

  // A draw of pixel shader `ps` still reading the game's own shadow map
  // (RayShadow::own_map): counted (RtOwnMapDraw: the first kRtOwnMapsSaid
  // shaders said on stderr, once each); what is bound at its sampler: the
  // lit texture with own_maps_lit_, else the draw's own (nil).
  id<MTLTexture> OwnMapDraw(const Shader& ps) {
    if (perf_) pc_.own_map_draws++;
    if (RtOwnMapDraw(&own_maps_, ps.hash))
      WarnF("gfx: ray-traced shadows: pixel shader %016llx still reads the game's own shadow map%s (%zu of the first "
            "%zu such shaders)\n",
            (unsigned long long)ps.hash, ps.shadow_claimed ? ", bound lit" : "", own_maps_.count, kRtOwnMapsSaid);
    if (!ps.shadow_claimed) return nil;
    uint32_t bits;
    std::memcpy(&bits, &ps.shadow.own_map_lit, 4);
    auto it = own_map_tex_.find(bits);
    return it == own_map_tex_.end() ? nil : it->second;
  }

  void StartDenoiser() {
    float lo = 0, hi = 0;
    std::string why;
    if (!RtDenoiseSupported(dev_, &why, &lo, &hi) || !(denoiser_ = RtDenoiser::Make(dev_, &why))) {
      dn_why_ = why;
      std::fprintf(stderr, "gfx: MetalFX's denoiser is off: %s\n", why.c_str());
      return;
    }
    for (int k = 0; k < kSlots; k++)
      if ((dn_counts_[k] = [dev_ newBufferWithLength:16 options:MTLResourceStorageModeShared]))
        std::memset(dn_counts_[k].contents, 0, 16);
      else {
        denoiser_.reset();
        dn_why_ = "no memory for its counters";
        return;
      }
    dn_fallback_ = upscaler_ == Upscaler::Off || upscaler_ == Upscaler::Smaa || upscaler_ == Upscaler::Taa
                       ? upscaler_
                       : Upscaler::Taa;
    upscaler_ = Upscaler::Denoise;
    render_scale_ = 1.0f;
    dn_on_ = true;
    std::fprintf(stderr,
                 "gfx: MetalFX's denoiser on the lit scene before post-processing (input scales %.3g to %.3g; at 1), "
                 "soft sun shadows; it replaces the anti-aliasing asked for\n",
                 lo, hi);
  }

  // The denoiser stops for good (why on stderr and in the perf stats): the
  // anti-aliasing it replaced instead (TAA in place of an upscaler), shadows
  // hard again.
  void DenoiseOff(const char* why) {
    if (!dn_on_) return;
    dn_on_ = false;
    dn_why_ = why;
    upscaler_ = dn_fallback_;
    std::fprintf(stderr, "gfx: MetalFX's denoiser is off: %s; %s instead\n", why,
                 upscaler_ == Upscaler::Taa ? "TAA" : upscaler_ == Upscaler::Smaa ? "SMAA" : "no anti-aliasing");
  }

  // A draw of a shader SetRayDenoise took: the lighting pass's first of the
  // presented frame takes the guides (DenoiseGuides); a post-process one
  // binding the lit scene (or a whole copy of it) at its input denoises it
  // there, once a frame.
  void DenoiseAt(const ShaderDraw& d, const Shader& ps) {
    if (ps.denoise_guides) {
      if (!dn_lit_ && !dn_done_) DenoiseGuides(d, ps.denoise);
      return;
    }
    if (!dn_lit_ || dn_done_ || ps.denoise.input_sampler >= 16) return;
    const TextureId tid = d.textures[ps.denoise.input_sampler];
    Tex* t = FindTex(tid);
    if (t && (t->tex == dn_lit_ || (dn_copy_id_ && tid == dn_copy_id_))) Denoise(t);
  }

  // The lit scene (the colour target the lighting draws into: one level, the
  // scene depth's size) and its guides from the G-buffer the draw binds,
  // made now (the G-buffer may be drawn over before the denoise), with the
  // scene's depth as it is (the reactive mask's reference). Nothing where
  // any of it is not there (the frame then not denoised).
  void DenoiseGuides(const ShaderDraw& d, const RayDenoise& g) {
    Tex* lt = pass_.targets > 0 ? FindTex(pass_.ids[0]) : nullptr;
    __unsafe_unretained id<MTLTexture> depth = scene_depth_;
    if (!lt || !lt->rt || lt->cube || lt->levels != 1 || lt->opaque || lt->depth || pass_.level[0] != 0 || !depth ||
        !HaveCamera() || lt->tex.width != depth.width || lt->tex.height != depth.height ||
        !(depth.usage & MTLTextureUsageShaderRead))
      return;
    auto flat = [&](uint32_t sampler) -> const Tex* {
      const Tex* t = sampler < 16 ? FindTex(d.textures[sampler]) : nullptr;
      return t && !t->cube && !t->depth && t->tex.textureType == MTLTextureType2D ? t : nullptr;
    };
    const Tex *at = flat(g.albedo_sampler), *nt = flat(g.normal_sampler), *st = flat(g.surface_sampler);
    if (!at || !nt || !st) return;
    std::vector<id> retired;
    const bool made = denoiser_->Ensure(uint32_t(lt->tex.width), uint32_t(lt->tex.height), lt->tex.pixelFormat,
                                        reflector_ != nullptr, &retired);
    for (id o : retired) Keep(o);
    if (!made) return DenoiseOff("MetalFX could not make it for the lit scene's size and format");
    if ((lt->tex.usage & denoiser_->ColorUsage()) != denoiser_->ColorUsage())
      return DenoiseOff("the lit scene's texture lacks the usage MetalFX reads it with");
    double inv[16];
    if (!Invert4(vp_cur_, inv)) return;
    RtDenoiseGuideParams p{};
    for (int i = 0; i < 16; i++) p.inv_view_proj[i] = float(inv[i]);
    p.jitter[0] = jitter_[0];  // (the scene's draws are jittered: Temporal)
    p.jitter[1] = jitter_[1];
    p.exponent_scale = g.exponent_scale;
    p.spec_scale = g.spec_scale;
    p.fresnel_min = g.fresnel_min;
    p.fresnel_power = g.fresnel_power;
    p.spec_channel = g.spec_channel;
    p.gloss_channel = g.gloss_channel;
    // (Compute work: the render pass open ends, as a trace's; a pending
    // clear of the depth runs first.)
    end_why_ = PassWhy::Compute;
    if (RtShadowSettlesClear(clear_flags_, depth == pass_.depth)) Settle();
    if (enc_open_ && perf_) pc_.dn_splits++;
    EndEncoders();
    EnsureFrame();
    const RtDenoiser* dn = denoiser_.get();
    const RtDenoiser::Set set = dn->Current();
    __unsafe_unretained id<MTLTexture> a = at->tex, n = nt->tex, su = st->tex;
    Rec([this, dn, set, a, n, su, depth, p] { dn->EncodeGuides(cmd_, set, a, n, su, depth, p); });
    dn_lit_ = lt->tex;
  }

  // The denoise of texture `t` (the lit scene, or a whole copy of it):
  // MetalFX's inputs (the camera's motion and a copy of the depth, the
  // reactive mask and hit distance), the denoise into dn_out_, and then
  // `t`'s texture and dn_out_ trade places, so the post-processing reads the
  // output and the next frame draws into the old input.
  void Denoise(Tex* t) {
    dn_done_ = true;
    const bool fresh = !dn_last_;  // (the last frame was not denoised)
    dn_last_ = false;
    const RtDenoiser* dn = denoiser_.get();
    const RtDenoiser::Set set = dn->Current();
    __unsafe_unretained id<MTLTexture> depth = scene_depth_;
    if (!set.scaler || !depth || !HaveCamera() || !t->rt || t->levels != 1 || t->cube || t->opaque ||
        t->tex.width != set.w || t->tex.height != set.h || t->tex.pixelFormat != dn->ColorFormat() ||
        (t->tex.usage & dn->ColorUsage()) != dn->ColorUsage())
      return;
    RtDenoiseFrame f;
    struct M16 {
      float v[16];
    } mm;
    const bool reversed = ReversedDepth(depth.pixelFormat);
    f.reset = CameraMotion(mm.v, reversed) || fresh;
    if (!RtSplitCamera(vp_cur_, f.view, f.proj)) return;
    // The projection's depth as the depth is stored: 1 - d (z' = w - z).
    if (reversed)
      for (int r = 0; r < 4; r++) f.proj[r * 4 + 2] = f.proj[r * 4 + 3] - f.proj[r * 4 + 2];
    f.depth_reversed = reversed;
    f.jitter[0] = jitter_[0] * jitter_sign_[0];
    f.jitter[1] = jitter_[1] * jitter_sign_[1];
    f.motion_scale[0] = mv_sign_ * float(set.w);
    f.motion_scale[1] = mv_sign_ * float(set.h);
    // (The texture traded for: one of the game's the first time, which
    // MetalFX may not write; then the outputs alternate.)
    const MTLTextureUsage need = t->tex.usage | dn->OutputUsage();
    if (!dn_out_ || dn_out_.width != set.w || dn_out_.height != set.h ||
        dn_out_.pixelFormat != t->tex.pixelFormat || (dn_out_.usage & need) != need) {
      Keep(dn_out_);
      dn_out_ = dn->NewOutput(t->tex.usage);
      if (!dn_out_) return;
    }
    EnsureMotionPipe();
    end_why_ = PassWhy::Compute;
    if (RtShadowSettlesClear(clear_flags_, depth == pass_.depth)) Settle();
    EndEncoders();
    EnsureFrame();
    __unsafe_unretained id<MTLTexture> color = t->tex, out = dn_out_, hit = dn_hit_;
    __unsafe_unretained id<MTLRenderPipelineState> mp = motion_pipe_;
    __unsafe_unretained id<MTLDepthStencilState> ds = depth_copy_ds_;
    __unsafe_unretained id<MTLBuffer> counts = dn_counts_[slot_];
    Rec([=, this] {
      // Motion, and the depth MetalFX takes, in one pass (as Composite's).
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = set.motion;
      rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.depthAttachment.texture = set.depth;
      rp.depthAttachment.loadAction = MTLLoadActionDontCare;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "denoise motion");
      [e setRenderPipelineState:mp];
      [e setDepthStencilState:ds];
      [e setFragmentTexture:depth atIndex:0];
      [e setFragmentBytes:mm.v length:sizeof mm.v atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
      dn->EncodeMasks(cmd_, set, depth, hit, counts, 0);
      RtDenoiser::EncodeDenoise(cmd_, set, color, out, f);
    });
    // (Both stay held: one by the texture, the other here.)
    id<MTLTexture> old = t->tex;
    t->tex = dn_out_;
    dn_out_ = old;
    for (int i = 0; i < pass_.targets; i++)
      if (pass_.color[i] == old) {
        BindTarget();  // (its pass drew into the old texture)
        break;
      }
    dn_last_ = true;
    if (perf_) {
      pc_.dn_frames++;
      if (f.reset) pc_.dn_resets++;
    }
  }

  // At the end of a presented frame: whether it was denoised (a lit frame
  // that was not starts the next one's history over, and too many in a
  // row turn the denoiser off), the next frame's number.
  void DenoiseEndFrame() {
    if (dn_on_ && dn_lit_ && !dn_done_) {
      if (perf_) pc_.dn_missed++;
      if (++dn_missed_ >= kDenoiseMissLimit)
        DenoiseOff("no post-process draw read the lit scene (the game's hook names its shaders)");
    } else if (dn_done_) {
      dn_missed_ = 0;
    }
    if (!dn_done_) dn_last_ = false;
    dn_lit_ = dn_hit_ = nil;
    dn_copy_id_ = 0;
    dn_done_ = false;
    dn_frame_++;
  }
  struct Tex {
    id<MTLTexture> tex = nil;
    int w = 0, h = 0, levels = 1;
    Format fmt = Format::Unknown;
    bool rt = false;
    bool cube = false;
    DepthId depth = 0;  // CreateDepthTexture: the depth buffer sharing `tex`
    uint64_t used = 0;  // the last frame (serial_) that drew with or into it
    bool opaque = false;  // alpha read as 1: X8R8G8B8 data uploaded as is
    // The least alpha its uploaded texels read as (MinAlpha), and whether
    // its whole top level has been uploaded once (before that: unknown).
    uint8_t alpha_lo = 255;
    bool alpha_whole = false;
    bool scaled = false;  // made at another size than w x h (ScaledSize)
    bool scene = false;   // one of the scene's buffers (the back buffer's size or a fraction)
    // Ray-traced shadows (RtNoteWrites): its contents' number, the camera
    // of the last main-view draw over its whole viewport into it.
    uint64_t rt_epoch = 0, rt_camera = 0;
  };
  struct Depth {
    id<MTLTexture> tex = nil;
    int w = 0, h = 0;
    bool stencil = false;
    float bias_scale = 8388608.0f;  // float depth: 2^23 per unit of depth range
    bool scaled = false;  // at the internal resolution (ScaledSize)
    WDepth wd;            // the projection draws from w use (reversed_z.h)
    uint64_t rt_epoch = 0, rt_camera = 0;  // as Tex's (RtNoteWrites)
  };
  // A texture by id (null if none): through tex_at_, not the map.
  Tex* FindTex(TextureId tid) const { return tid < tex_at_.size() ? tex_at_[tid] : nullptr; }
  // The id of a new texture: a destroyed one's, whose row in tex_at_ is there,
  // else the next (a stream of textures would otherwise grow tex_at_ for
  // ever). Textures, depth buffers and shaders share next_id_'s numbering, so
  // only a texture's id comes back.
  TextureId NewTexId() {
    if (free_tex_ids_.empty()) return next_id_++;
    const TextureId tid = free_tex_ids_.back();
    free_tex_ids_.pop_back();
    return tid;
  }
  void AddTex(TextureId tid, const Tex& t) {
    Tex& e = textures_[tid] = t;
    if (tex_at_.size() <= tid) tex_at_.resize(size_t(tid) + 1);
    tex_at_[tid] = &e;
  }
  struct Shader {
    id<MTLFunction> fn = nil;
    hlsl::Stage stage = hlsl::Stage::Vertex;
    size_t inputs = 0;
    std::array<hlsl::ParamType, 16> sampler_type{};  // per register; Void if unused
    uint32_t samplers = 0;  // highest register used + 1
    ShaderId without_depth = 0;  // (PixelShaderWithoutDepth)
    uint32_t float_regs = 0;
    int position_matrix = -1;  // hlsl::Program's, vertex shaders
    std::string position_note;
    int pos_slot = -1;   // vertex shaders: the input slot of POSITION0 (-1: none)
    bool blend = false;  // vertex shaders: blend weights or indices among the inputs
    // Vertex shaders: the input slots of TEXCOORD0-7 and COLOR0 (-1: none);
    // while ray tracing is on, the varying lanes moved from inputs
    // (hlsl::Program::moved_lanes).
    std::array<int8_t, 8> tex_slot{-1, -1, -1, -1, -1, -1, -1, -1};
    int color0_slot = -1;
    std::vector<hlsl::MovedLane> moved;
    bool discards = false;  // pixel shaders: texkill, clip or discard
    // Made while ray tracing is on, with what the scene reads of shaders
    // (hlsl::TranslateOptions::coverage, or HLSL source, which has none).
    bool analyzed = false;
    std::bitset<256> float_reads;
    int rel_const_base = -1;  // hlsl::Program's
    uint64_t hash = 0;  // of its translation: the same from run to run (0: the device's own)
    uint32_t varyings = 0;  // locations written (vertex) or read (pixel): hlsl::VaryingMask
    // Pixel shaders: the translation, and its variants for vertex shaders
    // that leave some of its inputs unwritten, by the inputs they do write
    // (PixelShaderFor).
    std::shared_ptr<const hlsl::Program> program;
    std::unordered_map<uint32_t, ShaderId> variants;
    bool has_lean = false;  // also made lean (MslOptions(false)): kLeanPs
    // Vertex shaders: the capture variant (CaptureShaderFor) and its
    // pipeline, or that it has none.
    ShaderId capture = 0;
    id<MTLRenderPipelineState> capture_pipe = nil;
    bool capture_none = false;
    // Pixel shaders: reads the traced shadow mask (SetRayShadow): a 2D
    // sampler claimed for it (bound: the mask or all lit), traced (the
    // other slots usable), where.
    bool shadow_claimed = false, ray_shadow = false;
    RayShadow shadow;
    // Pixel shaders: reads the traced reflections (SetRayReflection), as
    // the mask above.
    bool reflection_claimed = false, ray_reflection = false;
    RayReflection reflection;
    // Pixel shaders: its MSL has the ray queries of its r_rt_reflect calls
    // (rt_forward_reflection.h: its draws bind what they trace with); and
    // taken by SetRayForwardReflection (traced), with what.
    bool rt_forward = false, ray_forward = false;
    RayForwardReflection forward;
    // With rt_forward: its plain MSL (Give; a shader of its translation
    // made later shares it, as its function).
    std::shared_ptr<const std::string> fwd_plain;
    id<MTLFunction> fwd_plain_fn = nil;  // its function, if it could be made (StandIn)
    // Pixel shaders: reads of ray_shadow_sampler became shadow calls (hlsl::
    // RayShadowCall; -1: none); taken by SetRayForwardShadow (traced, where
    // rt_forward), with what.
    int ray_shadow_sampler = -1;
    bool ray_forward_shadow = false;
    RayForwardShadow forward_shadow;
    // Pixel shaders: still read a shadow map of the game's own (RayShadow::
    // own_map): counted, and with own_maps_lit_ (shadow_claimed) bound a lit
    // one at shadow.mask_sampler.
    bool own_map = false;
    // Pixel shaders: the lighting pass whose target is the lit scene, or a
    // post-process pass reading it (SetRayDenoise).
    bool denoise_guides = false, denoise_input = false;
    RayDenoise denoise;
    // Pixel shaders: a deferred lighting's G-buffer or local lights'
    // volumes (SetRayLightVolume).
    bool light_volume_on = false;
    RayLightVolume light_volume;
    // Pixel shaders: reads the ray-traced ambient occlusion (SetRayOcclusion).
    bool ray_occlusion = false;
    RayOcclusion occlusion;
  };

  // The pixel shader `sid` (ps) as drawn after a vertex shader writing
  // the varying locations `written`: itself if that covers its inputs,
  // else a variant made once with the others as constants (Metal rejects
  // a pipeline whose fragment inputs the vertex function lacks; D3D9 does
  // not).
  ShaderId PixelShaderFor(ShaderId sid, Shader& ps, uint32_t written) {
    written &= ps.varyings;
    if (written == ps.varyings || !ps.program) return sid;
    auto it = ps.variants.find(written);
    if (it != ps.variants.end()) return it->second;
    hlsl::Program v = *ps.program;
    v.glsl = hlsl::GlslWithDefaultInputs(v, written);
    // (Without the ray's start, no ray queries: the stub's "nothing hit".)
    if (v.ray_reflection_location >= 0 && !(written & (1u << v.ray_reflection_location)))
      v.ray_reflection_sampler = -1;
    std::erase_if(v.varyings, [&](hlsl::Semantic s) {
      int l = hlsl::VaryingLocation(s);
      return l >= 0 && !(written & (1u << l));
    });
    ShaderId vid = MakeShader(v, false, false);  // drawn at once; ps stays valid: shaders_ is node-based
    ps.variants[written] = vid;
    return vid;
  }

  // A pixel shader writing depth for a pass without a depth buffer: D3D9
  // ignores what it writes, a Metal pipeline cannot be made with it.
  ShaderId PixelShaderWithoutDepth(ShaderId sid) {
    Shader& ps = shaders_.find(sid)->second;
    if (!ps.program || !ps.program->writes_depth) return sid;
    if (ps.without_depth) return ps.without_depth;
    hlsl::Program v = *ps.program;
    v.writes_depth = false;
    std::string glsl;
    size_t at = 0;
    while (at < v.glsl.size()) {
      size_t end = v.glsl.find('\n', at);
      end = end == std::string::npos ? v.glsl.size() : end + 1;
      if (v.glsl.find("gl_FragDepth", at) >= end) glsl.append(v.glsl, at, end - at);
      at = end;
    }
    v.glsl = std::move(glsl);
    const ShaderId vid = MakeShader(v, false, false);  // drawn at once; ps stays valid: shaders_ is node-based
    ps.without_depth = vid;
    return vid;
  }

  // The capture variant of vertex shader `vs` (hlsl/capture_position.h),
  // made once; 0 if it has none (its GLSL not in the shape, or the shader
  // made before ray tracing was on).
  // Its pipeline (rasterization off, no fragment function, no attachment
  // formats: Metal takes it in a pass with any) is made here too, so a
  // capture that cannot be made is known before its draw is.
  ShaderId CaptureShaderFor(Shader& vs) {
    if (vs.capture || vs.capture_none) return vs.capture;
    vs.capture_none = true;
    if (!vs.program) return 0;
    hlsl::Program v = *vs.program;
    vs.program.reset();  // (only needed for this)
    v.glsl = hlsl::VertexWithCapturedPosition(v);
    if (v.glsl.empty()) return 0;
    const ShaderId cap = MakeShader(v, false, true);  // vs stays valid: shaders_ is node-based
    if (!cap) return 0;
    @autoreleasepool {
      const Shader& c = shaders_.at(cap);
      MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
      pd.vertexFunction = c.fn;
      pd.rasterizationEnabled = NO;
      NSError* err = nil;
      id<MTLRenderPipelineState> pipe = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
      if (!pipe) {
        CaptureFailed(c.hash, "pipeline", err.localizedDescription.UTF8String);
        DestroyShader(cap);
        return 0;
      }
      vs.capture_pipe = pipe;
    }
    vs.capture = cap;
    vs.capture_none = false;
    return cap;
  }

  // A capture variant (hash) that could not be made at `stage`: noted
  // once, its draws then left out (RtSkip::NoCapture). Returns 0.
  ShaderId CaptureFailed(uint64_t hash, const char* stage, const char* log) {
    if (!capture_failed_) {
      capture_failed_ = true;
      WarnF("gfx: ray tracing: a vertex shader's capture variant (%016llx) failed at %s, its draws are left out "
            "(further ones not noted): %s\n",
            (unsigned long long)hash, stage, log);
    }
    return 0;
  }

  // The render targets drawn to.
  struct Pass {
    id<MTLTexture> color[4] = {};
    int level[4] = {};
    MTLPixelFormat fmt[4] = {};
    TextureId ids[4] = {};
    int targets = 0;
    id<MTLTexture> depth = nil;
    MTLPixelFormat depth_fmt = MTLPixelFormatInvalid;
    bool stencil = false, alpha = false;
    float bias_scale = 1.0f;
    int w = 0, h = 0;  // the game's size of the target
    // A target at the internal resolution: its pixels per game pixel.
    float sx = 1.0f, sy = 1.0f;
    bool scaled = false;
    bool low = false;    // the back buffer, drawn to low_color_ for the upscaler
    bool scene = false;  // a scene buffer (not a shadow map or reflection)
    bool proxy = false;  // depth drawn as its full-size copy (DepthProxy)
    bool reversed = false;  // depth stored as 1 - d (ReversedDepth)
    DepthId depth_id = 0;
    bool Same(const Pass& o) const {
      for (int i = 0; i < 4; i++)
        if (color[i] != o.color[i] || level[i] != o.level[i]) return false;
      return depth == o.depth && proxy == o.proxy;
    }
  };
  struct Slice {
    __unsafe_unretained id<MTLBuffer> buf;  // owned by the arena
    size_t off;
    uint8_t* cpu;
  };
  // A command buffer's per-draw memory: chunks used in turn.
  struct Arena {
    std::vector<id<MTLBuffer>> chunks;
    size_t chunk = 0, used = 0;
  };
  struct Read {
    Rect r{};
    std::vector<uint32_t> px;  // A8R8G8B8
  };
  // What the render encoder has bound, so unchanged state is not sent again.
  struct Bound {
    PipeKey pipe{};
    bool pipe_set = false;
    void* ds = nullptr;
    int cull = -1, fill = -1;
    float bias = NAN, slope = NAN;
    MTLViewport vp{};
    bool has_vp = false;
    MTLScissorRect sc{};
    bool has_sc = false;
    int stencil_ref = -1;
    float blend[4] = {NAN, NAN, NAN, NAN};
    void* vbuf[4] = {};
    size_t voff[4] = {};
    void* fbuf[3] = {};
    size_t foff[3] = {};
    void* vtex[4] = {};
    void* vsmp[4] = {};
    void* ftex[16] = {};
    void* fsmp[16] = {};
    const RtFrame* rt_frame = nullptr;  // resident for fragment ray queries (BindForward)
    bool rt_null = false;               // the null scene's BLAS resident (BindForward)
    // What BindForward bound last at the fragment buffers 3-6.
    bool fwd_set = false;
    const RtFrame* fwd_frame = nullptr;
    RtForwardParams fwd_params{};
    bool fwd_lights = false;  // the local lights' textures bound (BindForward: which ones)
    bool fwd_sun = false;     // the sun mask's (BindForward: fwd_smask_)
  };

  // DrawShader's second half when its vertices are in shared memory: byte
  // offsets from the shared buffer's start, base vertex and indices given
  // to Metal (indices in shared memory read in place too).
  void DrawInPlace(const ShaderDraw& d, const Shader& vs, MTLPrimitiveType type, uint32_t n,
                   uint32_t count) {
    const Shader& ps = shaders_.find(d.ps)->second;
    VsState st{};
    for (size_t i = 0; i < vs.inputs; i++) {
      VertexInput in = i < d.input_count ? d.inputs[i] : VertexInput{};
      if (in.type >= 17 || in.stream >= 16 || !d.streams[in.stream].data) {
        st.in[i][2] = -1;
        continue;
      }
      const VertexStream& vsm = d.streams[in.stream];
      st.in[i][0] = int32_t(vsm.data - shared_lo_) + int32_t(in.offset);
      st.in[i][1] = int32_t(vsm.stride);
      st.in[i][2] = int32_t(in.type);
    }
    PixelState(st.pixel);
    ClipState(st, d, vs);
    VertexBuffer(3, shared_buf_, 0);
    ShaderResources(d, vs, ps, st);

    const auto* idx = static_cast<const uint8_t*>(d.indices);
    if (d.prim == 6) {
      Slice ib = FanIndices(d, count);
      const int32_t base = d.base_vertex;
      Rec([this, type, count, ib, base] {
        GpDraw(count * 3);
        [enc_ drawIndexedPrimitives:type
                         indexCount:count * 3
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:ib.buf
                  indexBufferOffset:ib.off
                      instanceCount:1
                         baseVertex:base
                       baseInstance:0];
      });
    } else if (idx) {
      const MTLIndexType it = d.index_size == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
      const size_t bytes = size_t(n) * d.index_size;
      __unsafe_unretained id<MTLBuffer> buf = shared_buf_;
      size_t off = size_t(idx - shared_lo_);
      // In place when there and 4-byte aligned (Metal's offset rule).
      if (idx < shared_lo_ || idx + bytes > shared_hi_ || (off & 3)) {
        Slice s = Alloc(bytes, 4);
        std::memcpy(s.cpu, idx, bytes);
        buf = s.buf;
        off = s.off;
      }
      const int32_t base = d.base_vertex;
      Rec([this, type, n, it, buf, off, base] {
        GpDraw(n);
        [enc_ drawIndexedPrimitives:type
                         indexCount:n
                          indexType:it
                        indexBuffer:buf
                  indexBufferOffset:off
                      instanceCount:1
                         baseVertex:base
                       baseInstance:0];
      });
    } else {
      const NSUInteger start = NSUInteger(std::max(d.base_vertex, 0));
      Rec([this, type, n, start] {
        GpDraw(n);
        [enc_ drawPrimitives:type vertexStart:start vertexCount:n];
      });
    }
  }

  // A fan's triangles as a 32-bit index list (vertex numbers before the
  // base vertex).
  Slice FanIndices(const ShaderDraw& d, uint32_t count) {
    Slice ib = Alloc(size_t(count) * 12, 4);
    auto* out = reinterpret_cast<uint32_t*>(ib.cpu);
    auto at = [&](uint32_t i) -> uint32_t {
      return !d.indices ? i
             : d.index_size == 4 ? static_cast<const uint32_t*>(d.indices)[i]
                                 : static_cast<const uint16_t*>(d.indices)[i];
    };
    for (uint32_t t = 0; t < count; t++) {
      out[t * 3] = at(0);
      out[t * 3 + 1] = at(t + 1);
      out[t * 3 + 2] = at(t + 2);
    }
    return ib;
  }

  // A shader draw's uniform blocks, textures and samplers.
  // `tex` is one of the pass's targets or its depth buffer.
  bool Attached(__unsafe_unretained id<MTLTexture> tex) const {
    for (int i = 0; i < pass_.targets; i++)
      if (tex == pass_.color[i]) return true;
    return tex == pass_.depth;
  }

  void ShaderResources(const ShaderDraw& d, const Shader& vs, const Shader& ps, const VsState& st) {
    auto ints = [](const int32_t (*i)[4], const uint32_t* b) {
      std::array<int32_t, 80> u{};
      if (i) std::memcpy(u.data(), i, 16 * 16);
      for (int k = 0; b && k < 16; k++) u[64 + k] = b[k] ? 1 : 0;
      return u;
    };
    Uniform(0, d.vs_float, vs.float_regs * 16, d.vs_float_version);
    // Int and bool registers: rebuilt only when they changed.
    if (!d.vs_ib_version || d.vs_ib_version != ib_version_[0] || d.vs_int != ib_src_[0]) {
      ib_[0] = ints(d.vs_int, d.vs_bool);
      ib_version_[0] = d.vs_ib_version;
      ib_src_[0] = d.vs_int;
      ib_serial_[0]++;
    }
    Uniform(1, ib_[0].data(), sizeof ib_[0], ib_serial_[0]);
    Uniform(2, &st, sizeof st);
    Uniform(3, d.ps_float, ps.float_regs * 16, d.ps_float_version);
    if (!d.ps_ib_version || d.ps_ib_version != ib_version_[1] || d.ps_int != ib_src_[1]) {
      ib_[1] = ints(d.ps_int, d.ps_bool);
      ib_version_[1] = d.ps_ib_version;
      ib_src_[1] = d.ps_int;
      ib_serial_[1]++;
    }
    Uniform(4, ib_[1].data(), sizeof ib_[1], ib_serial_[1]);
    PsState pst{};
    pst.alpha[0] = rs_[15] ? int32_t(rs_[25]) : 8;
    pst.alpha[1] = int32_t(rs_[24] & 0xFF);
    pst.alpha[3] = pass_.reversed ? 1 : 0;
    // Depth textures sampled: read back as D3D9 stores depth (R_DEPTH_TEX).
    // (Not the traced shadow mask, bound in place of what the draw has at
    // its sampler.)
    const int mask_reg = RtShadowSampler(ps.shadow_claimed, ps.shadow.mask_sampler, rt_mask_ != nil);
    const int refl_reg = RtShadowSampler(ps.reflection_claimed, ps.reflection.output_sampler, rt_refl_ != nil);
    const int ao_reg = ps.ray_occlusion && rt_ao_ ? int(ps.occlusion.occlusion_sampler) : -1;
    for (uint32_t r = 0; r < ps.samplers; r++)
      if (const Tex* t = FindTex(d.textures[r]); int(r) != mask_reg && int(r) != refl_reg && int(r) != ao_reg && t &&
                                                 t->depth && ReversedDepth(t->tex.pixelFormat))
        pst.alpha[2] |= int32_t(1u << r);
    const float lod_bias = jitter_draw_ && upscaler_ == Upscaler::Taa ? kTaaLodBias : 0.0f;
    for (uint32_t r = 0; r < 16 && d.sampler_states; r++) pst.bias[r] = AsFloat(d.sampler_states[r][8]) + lod_bias;
    pst.vscale[0] = 1.0f / pass_.sx;
    pst.vscale[1] = 1.0f / pass_.sy;
    Uniform(5, &pst, sizeof pst);

    // Samplers: pixel 0-15, then vertex 0-3.
    for (uint32_t r = 0; r < 20; r++) {
      const bool vertex = r >= 16;
      const Shader& sh = vertex ? vs : ps;
      const uint32_t reg = vertex ? r - 16 : r;
      if (reg >= sh.samplers) continue;
      if (!vertex && (int(reg) == mask_reg || int(reg) == refl_reg || int(reg) == ao_reg)) {
        // The mask, the reflections: at the pixel drawn (their size the
        // depth's, whose texels the draw's coordinates fall on), nearest;
        // the occlusion (half the size) linear.
        SamplerKey k{};
        k.mip = 0xFF;
        k.u = k.v = k.w = uint8_t(MTLSamplerAddressModeClampToEdge);
        k.aniso = 1;
        if (int(reg) == ao_reg) k.min = k.mag = 1;
        TextureAndSampler(false, reg,
                          int(reg) == mask_reg ? rt_mask_
                          : int(reg) == ao_reg ? rt_ao_
                                               : rt_refl_,
                          Sampler(k));
        continue;
      }
      const hlsl::ParamType want = sh.sampler_type[reg];
      // Unretained: textures and samplers outlive the draw (their tables
      // hold them; command buffers retain what they use).
      __unsafe_unretained id<MTLTexture> tex = want == hlsl::ParamType::SamplerCube ? dummy_cube_
                           : want == hlsl::ParamType::Sampler3D ? dummy_3d_
                                                                : dummy_2d_;
      Tex* t = FindTex(d.textures[r]);
      bool reversed = false;  // a reversed depth texture: its border colour mirrored
      if (t && t->cube == (want == hlsl::ParamType::SamplerCube) &&
          want != hlsl::ParamType::Sampler3D) {
        tex = t->tex;
        t->used = serial_;
        reversed = t->depth && ReversedDepth(tex.pixelFormat);
      }
      // The sampler made from the row last time, while the row's change
      // counter has not moved.
      const uint32_t* row = d.sampler_states ? d.sampler_states[r] : nullptr;
      const uint64_t version = d.sampler_versions ? d.sampler_versions[r] : 0;
      SlotSampler& cached = slot_sampler_[r];
      if (!version || cached.version != version || cached.row != row)
        cached = {row, version, Sampler(D3D9Sampler(row))};
      const bool nearest = pass_.scaled && Attached(tex);
      if (nearest || reversed) {
        SamplerKey k = D3D9Sampler(row);
        // A target the pass draws to, read at the pixel being drawn (the
        // game's lighting reads its accumulation and depth so): at the
        // internal resolution the game's coordinates fall off the texel
        // centres, and linear filtering then reads the neighbours, which
        // across a GPU tile's edge another tile may be writing (NaN rows
        // and columns on a 32-pixel grid). Nearest, as at full size.
        if (nearest) k.min = k.mag = uint8_t(MTLSamplerMinMagFilterNearest);
        // Read as 1 - rgb (R_DEPTH_TEX): the border's black and white
        // swapped, so it reads as D3D9's.
        if (reversed) k.border = ReversedDepthBorder(k.border);
        TextureAndSampler(vertex, reg, tex, Sampler(k));
        continue;
      }
      TextureAndSampler(vertex, reg, tex, cached.smp);
    }
    if (rt_fwd_bind_) BindForward();
  }

  // What a draw of a shader with ray queries (Shader::rt_forward) binds:
  // the scene (its BLASes and textures resident in the render encoder,
  // once), its materials and instances, the parameters; where it traces
  // nothing (on 0), the null scene (RtForwardNullScene).
  void BindForward() {
    const RtFrame* f = rt_fwd_frame_;
    const RtForwardParams p = rt_fwd_params_;
    // The local lights' masks for a draw reading them, else the stand-ins
    // (a patched shader calling r_rt_shadow declares them).
    if (shadow_masker_) {
      const bool lit = p.light_slot != kRtForwardNoLightSlot && light_masks_ && light_dist_;
      __unsafe_unretained id<MTLTexture> lm = lit ? light_masks_ : shadow_masker_->NoLightMasks();
      __unsafe_unretained id<MTLTexture> ld = lit ? light_dist_ : shadow_masker_->NoLightDist();
      if (!bound_.fwd_lights || fwd_lmask_ != lm || fwd_ldist_ != ld) {
        Rec([this, lm, ld] {
          [enc_ setFragmentTexture:lm atIndex:kRtForwardLightMasksTexture];
          [enc_ setFragmentTexture:ld atIndex:kRtForwardLightDistTexture];
        });
        bound_.fwd_lights = true;
        fwd_lmask_ = lm;
        fwd_ldist_ = ld;
      }
    }
    // The sun mask for the lamps' rays (kRtForwardSunGate), else a stand-in
    // (a patched shader calling r_rt_shadow_to declares it; unread then).
    {
      __unsafe_unretained id<MTLTexture> sm = rt_fwd_sun_mask_ ? rt_fwd_sun_mask_ : dummy_2d_;
      if (!bound_.fwd_sun || fwd_smask_ != sm) {
        Rec([this, sm] { [enc_ setFragmentTexture:sm atIndex:kRtForwardSunMaskTexture]; });
        bound_.fwd_sun = true;
        fwd_smask_ = sm;
      }
    }
    // (Unchanged since the encoder's last draw of such a shader: bound.)
    if (bound_.fwd_set && bound_.fwd_frame == f && std::memcmp(&bound_.fwd_params, &p, sizeof p) == 0) return;
    bound_.fwd_set = true;
    bound_.fwd_frame = f;
    bound_.fwd_params = p;
    if (f && bound_.rt_frame != f) {
      Rec([this, f] { RtUseResources(enc_, f, MTLRenderStageFragment); });
      bound_.rt_frame = f;
    }
    if (f) {
      Rec([this, f, p] {
        [enc_ setFragmentAccelerationStructure:f->tlas atBufferIndex:kRtForwardSceneBuffer];
        [enc_ setFragmentBuffer:f->materials offset:f->materials_offset atIndex:kRtForwardMaterialBuffer];
        [enc_ setFragmentBuffer:f->info offset:f->info_offset atIndex:kRtForwardInfoBuffer];
        [enc_ setFragmentBytes:&p length:sizeof p atIndex:kRtForwardParamsBuffer];
      });
      return;
    }
    __unsafe_unretained id<MTLAccelerationStructure> tlas = rt_fwd_null_.tlas, blas = rt_fwd_null_.blas;
    __unsafe_unretained id<MTLBuffer> mat = rt_fwd_null_.materials, info = rt_fwd_null_.info;
    const bool resident = bound_.rt_null;
    bound_.rt_null = true;
    Rec([this, tlas, blas, mat, info, p, resident] {
      if (!resident) [enc_ useResource:blas usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
      [enc_ setFragmentAccelerationStructure:tlas atBufferIndex:kRtForwardSceneBuffer];
      [enc_ setFragmentBuffer:mat offset:0 atIndex:kRtForwardMaterialBuffer];
      [enc_ setFragmentBuffer:info offset:0 atIndex:kRtForwardInfoBuffer];
      [enc_ setFragmentBytes:&p length:sizeof p atIndex:kRtForwardParamsBuffer];
    });
  }

  // ---- the encoding thread
  //
  // Recorded work: closures (trivially copyable, so the bytes can move)
  // in a batch, run in order on the encoding thread. They capture what
  // they need by value, Metal objects unretained: their owners keep them,
  // and an owner letting one go while recorded work may still use it hands
  // it to Keep, which holds it until that batch has run.
  struct Op {
    void (*run)(void*);
    uint32_t at;
  };
  struct Batch {
    std::vector<uint8_t> bytes;
    std::vector<Op> ops;
    std::vector<id> keep;
  };

  template <typename F>
  void Rec(F f) {
    static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
                  "recorded closures capture plain values and unretained objects");
    if (sync_) {  // RECOMP_METAL_SYNC: encoded on the spot, no encoding thread
      @autoreleasepool {
        f();
      }
      return;
    }
    static_assert(alignof(F) <= 16);
    const size_t at = (rec_.bytes.size() + alignof(F) - 1) & ~(alignof(F) - 1);
    rec_.bytes.resize(at + sizeof(F));
    new (rec_.bytes.data() + at) F(f);
    rec_.ops.push_back({[](void* p) { (*static_cast<F*>(p))(); }, uint32_t(at)});
    // A long frame is handed over in parts, so encoding overlaps it.
    if (rec_.ops.size() >= 2048) Flush();
  }

  void Keep(id o) {
    if (o && !sync_) rec_.keep.push_back(o);
  }

  // Hands the recorded batch to the encoding thread.
  void Flush() {
    if (rec_.ops.empty() && rec_.keep.empty()) return;
    if (perf_) pc_.ops += rec_.ops.size();
    frame_log_ops_ += rec_.ops.size();  // (never zeroed: FrameLogPresent takes its difference)
    // The next batch: one the encoding thread is done with, its memory
    // kept (DXVK's pool of CS chunks), else a new one.
    Batch next;
    bool reused = false;
    {
      std::lock_guard<std::mutex> l(q_mu_);
      batches_.push_back(std::move(rec_));
      submitted_++;
      if (!free_batches_.empty()) {
        next = std::move(free_batches_.back());
        free_batches_.pop_back();
        reused = true;
      }
    }
    if (!reused) next.bytes.reserve(1 << 20);
    rec_ = std::move(next);
    q_cv_.notify_one();
  }

  // Everything recorded so far encoded.
  void WaitIdle() {
    Flush();
    std::unique_lock<std::mutex> l(q_mu_);
    done_cv_.wait(l, [&] { return done_ == submitted_; });
  }

  void EncoderMain() {
    // Its frames are the game's: scheduled like a time-critical thread.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    // The helpers' compute and acceleration-structure passes timed with the
    // render passes (RECOMP_GPU_PASSES; gpu_timer.h).
    if (gp_on_) CurrentGpuTimer() = GpuTimer{this, &TimedPassNext};
    for (;;) {
      Batch b;
      {
        std::unique_lock<std::mutex> l(q_mu_);
        q_cv_.wait(l, [&] { return !batches_.empty() || stop_; });
        if (batches_.empty()) return;
        b = std::move(batches_.front());
        batches_.pop_front();
      }
      const uint64_t t0 = perf_.load(std::memory_order_relaxed) ? NowNs() : 0;
      @autoreleasepool {
        for (const Op& o : b.ops) o.run(b.bytes.data() + o.at);
        b.keep.clear();
      }
      if (t0) enc_busy_ns_.fetch_add(NowNs() - t0, std::memory_order_relaxed);
      b.bytes.clear();
      b.ops.clear();
      {
        std::lock_guard<std::mutex> l(q_mu_);
        done_++;
        if (free_batches_.size() < 4) free_batches_.push_back(std::move(b));
      }
      done_cv_.notify_all();
    }
  }

  // Frames recorded but not finished on the GPU: recording, encoding and
  // the GPU each hold one, and one more absorbs their jitter.
  static constexpr int kSlots = 4;
  static_assert(RtScene::kSlots >= kSlots, "the ray-traced scene keeps a frame's work as long");
  static constexpr size_t kChunk = 32u << 20;
  static constexpr size_t kKeptChunks = 2;  // per slot, between frames

  void WarnOnce(uint32_t bit, const char* what) {
    if (warned_ & bit) return;
    warned_ |= bit;
    WarnF("gfx: %s is not in the Metal backend\n", what);
  }

  void WarnUnsupportedShaderState() {
    if (rs_[194] && !(warned_ & 0x200)) {
      warned_ |= 0x200;
      WarnF("gfx: SRGBWRITEENABLE is not supported yet\n");
    }
  }

  // `opaque`: alpha read as 1 (X8 formats uploaded as they are).
  int64_t GpuAllocatedBytes() const override { return int64_t(dev_.currentAllocatedSize); }

  void SetBackBufferAlpha(bool on) override { back_buffer_alpha_ = on; }

  std::string GpuMemoryText() const override {
    std::lock_guard<std::recursive_mutex> l(mu_);
    uint64_t tex = 0, rt = 0, depth = 0, draw = 0;
    for (const auto& [id, t] : textures_)
      if (t.tex) (t.rt ? rt : tex) += t.tex.allocatedSize;
    for (const auto& [id, d] : depths_)
      if (d.tex) depth += d.tex.allocatedSize;
    for (const Arena& a : arenas_)
      for (id<MTLBuffer> b : a.chunks) draw += b.length;
    char buf[224];
    std::snprintf(buf, sizeof buf, "textures %llu MB (%zu), targets %llu MB, depth %llu MB, draw data %llu MB",
                  (unsigned long long)(tex >> 20), textures_.size(), (unsigned long long)(rt >> 20),
                  (unsigned long long)(depth >> 20), (unsigned long long)(draw >> 20));
    std::string out = buf;
    if (rt_) {
      const RtStats st = rt_->Stats();
      std::snprintf(buf, sizeof buf, ", ray tracing %llu MB (%llu BLAS; captured draws %llu MB)",
                    (unsigned long long)(rt_->MemoryBytes() >> 20), (unsigned long long)st.cached,
                    (unsigned long long)((st.capture_blas_bytes + st.capture_buffer_bytes) >> 20));
      out += buf;
    }
    return out;
  }

  // DXT is stored as B8G8R8A8 (decoded on upload) where BC cannot be sampled.
  MTLPixelFormat TexFormat(Format f) const {
    return IsCompressed(f) && !bc_ ? MTLPixelFormatBGRA8Unorm : PixelFormat(f);
  }

  void DecodeDxtOnUpload() override {
    if (!textures_.empty()) return;  // those already made keep their format
    bc_ = false;
    WarnF("gfx: DXT decoded on upload\n");
  }

  id<MTLTexture> NewTexture(MTLTextureType type, MTLPixelFormat pf, int w, int h, int levels,
                            bool rt, bool opaque = false, MTLTextureUsage extra_usage = 0) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = type;
    td.pixelFormat = pf;
    td.width = NSUInteger(w);
    td.height = NSUInteger(h);
    td.mipmapLevelCount = NSUInteger(levels);
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | (rt ? MTLTextureUsageRenderTarget : 0) | extra_usage;
    if (opaque)
      td.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                                 MTLTextureSwizzleBlue, MTLTextureSwizzleOne);
    return [dev_ newTextureWithDescriptor:td];
  }

  Depth NewDepth(int w, int h, Format fmt, bool sampled) {
    Depth d;
    d.w = w;
    d.h = h;
    MTLPixelFormat pf;
    switch (fmt) {
      case Format::D16: pf = MTLPixelFormatDepth16Unorm; d.bias_scale = 65535.0f; break;
      case Format::D24X8: case Format::D32: pf = MTLPixelFormatDepth32Float; break;
      default: pf = MTLPixelFormatDepth32Float_Stencil8; d.stencil = true; break;  // D24S8, D15S1
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = pf;
    td.width = NSUInteger(w);
    td.height = NSUInteger(h);
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | (sampled ? MTLTextureUsageShaderRead : 0);
    d.tex = [dev_ newTextureWithDescriptor:td];
    if (!d.tex) WarnF("gfx: depth buffer %dx%d fmt %u failed\n", w, h, unsigned(fmt));
    return d;
  }

  // ---- frames: command buffers and their arenas

  // A frame being recorded, its command buffers made on the encoding
  // thread.
  void EnsureFrame() {
    if (!frame_open_) BeginFrame();
  }

  void BeginFrame() {
    // (How long the render thread waits here for one of the frames in
    // flight to finish: for the perf line and for the frame log, whose
    // frame's recording starts when the first of its waits is over: a frame
    // a readback cut comes here again.)
    const uint64_t wait_t0 = perf_ || frame_log_ ? NowNs() : 0;
    dispatch_semaphore_wait(frames_free_, DISPATCH_TIME_FOREVER);
    if (wait_t0) {
      const uint64_t got = NowNs();
      const uint64_t waited = got - wait_t0;
      if (perf_) pc_.slot_wait_ns += waited;
      if (frame_log_) {
        FrameLogCounts& c = frame_log_counts_;
        c.slot_wait_ns += waited;
        if (c.begun_ns)
          c.later_wait_ns += waited;
        else
          c.begun_ns = got;
      }
    }
    frame_first_pass_ = true;
    slot_ = (slot_ + 1) % kSlots;
    // The slot's last frame is done: a frame that took more (a level's
    // texture uploads: hundreds of MB, several times that with DXT decoded)
    // does not keep it. Two standard chunks stay; bigger ones go.
    Arena& arena = arenas_[slot_];
    std::erase_if(arena.chunks, [](id<MTLBuffer> b) { return b.length > kChunk; });
    if (arena.chunks.size() > kKeptChunks) arena.chunks.resize(kKeptChunks);
    arena.chunk = arena.used = 0;
    serial_++;
    frame_open_ = true;
    draw_serial_.store(serial_, std::memory_order_release);
    if (rt_) rt_->BeginFrame(serial_, CompletedSerial());
    if (shadow_masker_) {
      ShadowDiagLog(slot_);  // (before the scene reuses the slot)
      MlProbeLog(slot_);
      MlRetainContact(slot_);
      // The rays this slot's last frame traced (done now).
      auto* n = static_cast<uint32_t*>(shadow_counts_[slot_].contents);
      const uint32_t* ln = n + kLightCounts / 4;
      // (The share of the far light's rays that reached it: the bounce's
      // gate. No rays: nothing known lit.)
      sun_lit_share_ = n[0] ? float(n[0] - std::min(n[1], n[0])) / float(n[0]) : 0.0f;
      if (perf_) {
        pc_.shadow_rays += n[0];
        pc_.shadow_hits += n[1];
        pc_.shadow_retraced += n[2];
        pc_.shadow_cascade_lit += n[3];
        pc_.light_rays += ln[0];
        pc_.light_hits += ln[1];
        pc_.light_outside += ln[2];
        pc_.light_in_gap += ln[3];
        const uint32_t* sn = n + kLightSetCounts / 4;
        pc_.lset_rays += sn[0];
        pc_.lset_hits += sn[1];
        pc_.lset_capped += sn[2];
        pc_.lset_again += sn[3];
        const uint32_t* mn = n + kManyLightCounts / 4;
        pc_.ml_rays += mn[0];
        pc_.ml_blocked += mn[1];
        const uint32_t* on = n + kOcclusionCounts / 4;
        pc_.ao_rays += on[0];
        pc_.ao_blocked += on[1];
      }
      std::memset(n, 0, kShadowCountsSize);
    }
    if (reflector_) {
      // The reflections this slot's last frame traced (done now).
      auto* n = static_cast<RtReflectionCounts*>(refl_counts_[slot_].contents);
      // (Rays and fills and the march's: the pixels reflected, a frame.)
      refl_last_listed_ = uint64_t(n->traced) + n->filled + n->screen;
      if (perf_) {
        pc_.refl_rays += n->traced;
        pc_.refl_hits += n->hits;
        pc_.refl_misses += n->misses;
        pc_.refl_bare += n->bare;
        pc_.refl_distance += n->distance;
        pc_.refl_back += n->back;
        pc_.refl_back_captured += n->back_captured;
        pc_.refl_other_back += n->other_back;
        pc_.refl_filled += n->filled;
        pc_.refl_dense += n->dense;
        pc_.refl_screen += n->screen;
        pc_.refl_screen_mixed += n->screen_mixed;
      }
      *n = RtReflectionCounts{};
    }
    if (denoiser_) {
      // The reactive pixels this slot's last frame counted (done now).
      auto* n = static_cast<uint32_t*>(dn_counts_[slot_].contents);
      if (perf_) pc_.dn_reactive += n[0];
      n[0] = 0;
    }
    for (auto& u : last_uniform_) u = {};
    Rec([this] {
      // Uploads run before the frame's draws: their command buffer is
      // queued first.
      upload_cb_ = [queue_ commandBuffer];
      [upload_cb_ enqueue];
      cmd_ = [queue_ commandBuffer];
      [cmd_ enqueue];
    });
  }

  void EndFrame() {
    if (!frame_open_) return;
    // A frame slot filled past 3/4: one more chunk made ahead, for the
    // frame that outgrows it.
    const Arena& a = arenas_[slot_];
    size_t used = a.used, capacity = 0;
    for (size_t i = 0; i < a.chunks.size(); i++) {
      capacity += a.chunks[i].length;
      if (i < a.chunk) used += a.chunks[i].length;
    }
    if (used * 4 > capacity * 3) {
      bool none;
      {
        std::lock_guard<std::mutex> l(spares_->mu);
        none = spares_->ready.empty() && !spares_->pending;
      }
      if (none) MakeSpareChunks(1);
    }
    Settle();
    RtEndFrame(false);
    const uint64_t serial = serial_;
    const bool presented = presented_;
    presented_ = false;
    frame_open_ = false;
    draw_serial_.store(serial_ + 1, std::memory_order_release);
    const FrameLogRay ray = frame_log_ray_;
    Rec([this, serial, presented, ray] { EncodeEndFrame(serial, presented, ray); });
    Flush();
  }

  // The frame log's "r" line: the ray scene's counts of a presented frame,
  // taken on the recording thread at Present (FrameLogPresent) and handed
  // with the frame's end to the encoding thread, which adds the lights the
  // many lights' pass binned on screen (known only there) and writes it.
  struct FrameLogRay {
    uint32_t instances = 0;  // in the TLAS the frame's effects traced
    uint32_t builds = 0;     // BLASes built anew (no refits)
    uint32_t captured = 0;   // captured draws' instances in that TLAS
    uint32_t fwd_draws = 0;  // draws that traced their own rays (RtForwardFor)
  };
  FrameLogRay frame_log_ray_;       // recording thread
  uint32_t frame_log_lights_ = 0;   // encoding thread: the frame's MlAt pass's lights on screen

  // On the encoding thread: commits the frame's command buffers.
  void EncodeEndFrame(uint64_t serial, bool presented, FrameLogRay ray) {
    if (frame_log_ && presented) {
      FrameLogLine("r %llu %u %u %u %u %u", (unsigned long long)serial, ray.instances, ray.builds, ray.captured,
                   frame_log_lights_, ray.fwd_draws);
      frame_log_lights_ = 0;  // (a frame without the pass has none)
    }
    if (upload_blit_) [upload_blit_ endEncoding];
    upload_blit_ = nil;
    [upload_cb_ commit];
    dispatch_semaphore_t sem = frames_free_;
    // RECOMP_PACE_PRESENT: a frame that showed a drawable frees its slot
    // when the display shows it (Show), not when the GPU is done, so the
    // CPU never runs more frames ahead of the screen than are in flight
    // (DXVK's frame latency limit): no growing queue to the display.
    const bool freed_by_display = pace_present_ && frame_drawable_;
    frame_drawable_ = false;
    std::atomic<uint64_t>* completed = &completed_;
    std::atomic<uint64_t>* gpu_ns = &gpu_ns_;
    std::atomic<uint64_t>* gpu_frames = &gpu_frames_;
    id<MTLCommandBuffer> uploads = upload_cb_;
    PassTimes* pt = gp_on_ ? &gp_[gp_slot_] : nullptr;
    gp_slot_ = (gp_slot_ + 1) % gp_.size();
    [cmd_ addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      // The GPU's busy time for the frame's work (its uploads included):
      // the part of each command buffer's span not already counted, so
      // frames whose spans overlap (one queued behind another, a tiled
      // GPU running two at once) are not counted twice.
      const double t = BusyTime(uploads.GPUStartTime, uploads.GPUEndTime) + BusyTime(cb.GPUStartTime, cb.GPUEndTime);
      // The frame log's "g": a presented frame's, with the busy time of the
      // parts a readback cut off before it, as its "q" line has their
      // passes (ReportPassTimes); the start and end are this command
      // buffer's, the frame's last.
      if (FrameLogOn()) {
        const double all = FrameLogBusy(t, presented);
        if (presented)
          FrameLogLine("g %llu %.3f %.4f %.4f", (unsigned long long)serial, all * 1e3, cb.GPUStartTime, cb.GPUEndTime);
      }
      if (pt) ReportPassTimes(pt, t, serial, presented);
      if (t > 0) gpu_ns->fetch_add(uint64_t(t * 1e9));
      if (presented) gpu_frames->fetch_add(1);
      if (cb.status == MTLCommandBufferStatusError)
        std::fprintf(stderr, "gfx: command buffer failed: %s\n",
                     cb.error.localizedDescription.UTF8String);
      uint64_t c = completed->load();
      while (c < serial && !completed->compare_exchange_weak(c, serial)) {
      }
      if (!freed_by_display) dispatch_semaphore_signal(sem);
    }];
    [cmd_ commit];
    pass_no_ = 0;
    last_cb_ = cmd_;
    cmd_ = nil;
    upload_cb_ = nil;
  }

  // Completion handlers: the GPU time in [start, end] not yet counted.
  double BusyTime(double start, double end) {
    if (!(end > start)) return 0;
    std::lock_guard<std::mutex> l(busy_mu_);
    const double from = std::max(start, busy_until_);
    busy_until_ = std::max(busy_until_, end);
    return end > from ? end - from : 0;
  }
  std::mutex busy_mu_;
  double busy_until_ = 0;

  // Completion handlers, for the frame log's "g" line: `busy_s` of a part
  // a readback cut off (not `presented`) is kept for the frame it belongs
  // to, the next presented one, which returns its own with them. So a
  // frame with a readback in it (a screenshot, 2 or 3 parts) has one busy
  // time for all of its GPU work, the same work its "q" line lists.
  double FrameLogBusy(double busy_s, bool presented) {
    std::lock_guard<std::mutex> l(busy_mu_);
    frame_log_cut_busy_s_ += busy_s;
    if (!presented) return 0;
    const double all = frame_log_cut_busy_s_;
    frame_log_cut_busy_s_ = 0;
    return all;
  }
  // (ReportPassTimes, when it lets go of the passes kept of parts that no
  // Present followed: their busy time goes with them.)
  void FrameLogBusyForget() {
    std::lock_guard<std::mutex> l(busy_mu_);
    frame_log_cut_busy_s_ = 0;
  }
  double frame_log_cut_busy_s_ = 0;  // under busy_mu_

  // `n` arena chunks made on a background queue, their pages touched, so
  // that neither the allocation nor first-use page faults fall on a frame.
  void MakeSpareChunks(int n) {
    std::shared_ptr<SparePool> pool = spares_;
    id<MTLDevice> dev = dev_;
    {
      std::lock_guard<std::mutex> l(pool->mu);
      pool->pending += n;
    }
    for (int i = 0; i < n; i++)
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        id<MTLBuffer> b = [dev newBufferWithLength:kChunk
                                           options:MTLResourceStorageModeShared |
                                                   MTLResourceCPUCacheModeWriteCombined];
        if (b) {
          volatile uint8_t* p = static_cast<uint8_t*>(b.contents);
          for (size_t o = 0; o < kChunk; o += 4096) p[o] = 0;
        }
        std::lock_guard<std::mutex> l(pool->mu);
        pool->pending--;
        if (b) pool->ready.push_back(b);
      });
  }

  id<MTLBuffer> TakeSpareChunk() {
    std::lock_guard<std::mutex> l(spares_->mu);
    if (spares_->ready.empty()) return nil;
    id<MTLBuffer> b = spares_->ready.back();
    spares_->ready.pop_back();
    return b;
  }

  // `bytes` of this frame's arena, aligned to `align` (a power of two).
  // Without `fatal`, a failed allocation returns a null slice.
  Slice Alloc(size_t bytes, size_t align, bool fatal = true) {
    EnsureFrame();
    Arena& a = arenas_[slot_];
    for (;;) {
      if (a.chunk < a.chunks.size()) {
        id<MTLBuffer> b = a.chunks[a.chunk];
        size_t off = (a.used + align - 1) & ~(align - 1);
        if (off + bytes <= b.length) {
          a.used = off + bytes;
          return {b, off, static_cast<uint8_t*>(b.contents) + off};
        }
        if (a.used == 0 && bytes > b.length) {  // too small for this one: replace it
          if (perf_) {
            pc_.arena_buffers++;
            pc_.arena_bytes += std::max(bytes, kChunk);
          }
          id<MTLBuffer> big = [dev_ newBufferWithLength:std::max(bytes, kChunk)
                                                options:MTLResourceStorageModeShared |
                                                        MTLResourceCPUCacheModeWriteCombined];
          if (!big) {
            if (fatal) FatalF("gfx: cannot allocate %zu bytes for draw data", bytes);
            return {nil, 0, nullptr};
          }
          Keep(a.chunks[a.chunk]);
          a.chunks[a.chunk] = big;
          continue;
        }
        a.chunk++;
        a.used = 0;
        continue;
      }
      if (id<MTLBuffer> ready = bytes <= kChunk ? TakeSpareChunk() : nil) {
        if (perf_) pc_.arena_ahead++;
        a.chunks.push_back(ready);
        continue;
      }
      if (perf_) {
        pc_.arena_buffers++;
        pc_.arena_bytes += std::max(bytes, kChunk);
      }
      id<MTLBuffer> fresh = [dev_ newBufferWithLength:std::max(bytes, kChunk)
                                              options:MTLResourceStorageModeShared |
                                                      MTLResourceCPUCacheModeWriteCombined];
      if (!fresh) {
        if (fatal) FatalF("gfx: cannot allocate %zu bytes for draw data", bytes);
        return {nil, 0, nullptr};
      }
      a.chunks.push_back(fresh);
    }
  }

  // ---- render targets and encoders

  id<MTLTexture> Target(ColorTarget t, int* w, int* h, Tex** tex) {
    *tex = nullptr;
    if (!t.tex) {
      *w = bb_w_;
      *h = bb_h_;
      return backbuffer_;
    }
    Tex* e = FindTex(t.tex);
    if (!e || !e->rt || t.level < 0 || t.level >= e->levels) return nil;
    *w = std::max(1, e->w >> t.level);
    *h = std::max(1, e->h >> t.level);
    *tex = e;
    return e->tex;
  }

  // The pass for the D3D render targets.
  void BindTarget() {
    Pass p{};
    Tex* t0 = nullptr;
    int w = 0, h = 0;
    id<MTLTexture> c0 = Target({current_color_, current_level_}, &w, &h, &t0);
    if (!c0) c0 = backbuffer_, w = bb_w_, h = bb_h_, t0 = nullptr;
    if (!t0 && c0 == backbuffer_ && upscaler_ != Upscaler::Off && upscaler_ != Upscaler::Smaa &&
        frame_scene_ && !composited_) {
      // The frame's first drawing to the back buffer after its scene: the
      // composite, at the internal resolution, upscaled when done. TAA's
      // is the back buffer itself, resolved in place when done.
      p.low = p.scene = true;
      if (!FullSizeScene()) {
        EnsureLowColor();
        c0 = low_color_;
        p.scaled = true;
      }
    } else if (t0 && (t0->scaled || t0->scene)) {
      p.scaled = t0->scaled;
      p.scene = t0->scene;
    }
    if (p.scaled) {
      p.sx = float(std::max<NSUInteger>(1, c0.width >> (t0 ? current_level_ : 0))) / float(w);
      p.sy = float(std::max<NSUInteger>(1, c0.height >> (t0 ? current_level_ : 0))) / float(h);
    }
    p.color[0] = c0;
    p.level[0] = t0 ? current_level_ : 0;
    p.fmt[0] = c0 ? c0.pixelFormat : MTLPixelFormatInvalid;
    p.ids[0] = t0 ? current_color_ : 0;
    p.alpha = t0 ? HasAlpha(t0->fmt) : back_buffer_alpha_;
    p.w = w;
    p.h = h;
    p.targets = c0 ? 1 : 0;
    for (int i = 0; i < 3 && p.color[i]; i++) {
      int ew, eh;
      Tex* te = nullptr;
      const ColorTarget& e = current_extra_[i];
      if (!e.tex) break;
      id<MTLTexture> c = Target(e, &ew, &eh, &te);
      if (!c) break;
      p.color[i + 1] = c;
      p.level[i + 1] = e.level;
      p.fmt[i + 1] = c.pixelFormat;
      p.ids[i + 1] = e.tex;
      p.targets = i + 2;
    }
    auto dit = depths_.find(current_depth_);
    if (dit != depths_.end()) {
      p.depth = dit->second.tex;
      p.depth_id = current_depth_;
      p.depth_fmt = p.depth.pixelFormat;
      p.reversed = ReversedDepth(p.depth_fmt);
      p.stencil = dit->second.stencil;
      p.bias_scale = dit->second.bias_scale;
      // A depth buffer at the internal resolution with a full-size target
      // (the HUD's markers on the back buffer): a full-size copy of it, made
      // when the pass opens (the game also binds its targets one at a
      // time, such pairs then never drawn to).
      const NSUInteger cw = std::max<NSUInteger>(1, c0.width >> p.level[0]);
      const NSUInteger ch = std::max<NSUInteger>(1, c0.height >> p.level[0]);
      p.proxy = dit->second.scaled && p.depth.width < cw && p.depth.height < ch;
    }
    SetPass(p);
    bound_serial_ = frame_open_ ? serial_ : 0;
    bound_scene_ = frame_scene_;
    bound_composited_ = composited_;
  }

  // Before a draw or a clear: BindTarget again if what it chose by,
  // besides the bound targets, has moved since (front ends bind only when
  // their bindings change): the frame (DepthProxy copies once a frame;
  // none open yet means the draw opens the next one) and the upscaler's
  // stage (the back buffer stands for low_color_ until the composite).
  void EnsureTarget() {
    if (!frame_open_ || bound_serial_ != serial_ || bound_scene_ != frame_scene_ ||
        bound_composited_ != composited_)
      BindTarget();
  }

  // Why a render pass had to be opened, for the pass report (RECOMP_GPU_
  // PASSES): the frame's first; other colour targets than the render pass
  // before it; the same colour targets with another depth buffer; or, on
  // the same targets, what ended that pass: other targets bound and these
  // again with nothing drawn there (Rebind), compute work (the traces),
  // a copy or an upload, a draw needing the depth buffer the pass was
  // opened without (Split: BareDraw), anything else.
  enum class PassWhy : uint8_t { None, Frame, Target, Depth, Rebind, Compute, Blit, Split, Other, Count };
  static const char* PassWhyName(PassWhy w) {
    static constexpr const char* kNames[size_t(PassWhy::Count)] = {"",        "frame", "target", "depth", "rebind",
                                                                   "compute", "blit",  "split",  "other"};
    return kNames[size_t(w) < size_t(PassWhy::Count) ? size_t(w) : 0];
  }
  // A texture's number in the pass report: 20 bits of its address, enough
  // to tell a frame's targets apart.
  static uint32_t TexTag(const void* p) { return uint32_t((reinterpret_cast<uintptr_t>(p) >> 4) & 0xFFFFFu); }

  // RECOMP_PASS_LOADS (on unless 0). A render pass loads and stores what it
  // attaches for every tile its draws touch: under one full-target quad that
  // tests neither, 24.8 MB of depth and stencil each way at 3440x1440. An
  // attachment not stored is left undefined (MTLStoreActionDontCare), so
  // leaving the buffer's contents alone means leaving it out of the pass,
  // and a pass's attachments are fixed when it opens, before its draws are
  // known. So passes are remembered: one opened by a draw (PassSig: its
  // targets and that draw's pixel shader) that ended kBareAfter times with
  // nothing in it having tested or written depth or stencil, and never
  // otherwise, is opened without the depth buffer from then on ("bare":
  // pipelines of no depth format, pixel shaders without their depth
  // output, which D3D9 ignores for such draws). The first draw or clear
  // that needs the buffer ends a bare pass (BareSplit), the next one opens
  // with it, and that signature keeps it for good.
  //
  // The same draws write the same pixels, and the split shows in one place
  // only: a draw after it that samples the pass's own colour target reads
  // what the first part drew, where in one pass it read what was there
  // before the pass (the game's lighting reads its accumulation so). Such a
  // draw cannot be told ahead, so only passes whose colour target no draw
  // can sample are opened bare: the back buffer and its stand-in at the
  // internal resolution, which are no texture of the game's (BareTargets).
  // RECOMP_PASS_LOADS=2, for measuring: the game's textures too, the frame
  // of a split then not exact for such a draw.
  //
  // Not bare either: a pass whose depth is a full-size copy (DepthProxy:
  // made when the pass opens) or smaller than its colour target, one that
  // opens with a clear of depth or stencil, one the many lights' light is
  // drawn into first (its pipeline is the pass's), and draws of pixel
  // shaders that may have ray queries (their pipelines take seconds to
  // compile, the draws meanwhile tracing nothing: StandIn).
  //
  // pass_idle_'s values: how often the signature's pass has ended without
  // having needed its depth buffer, up to kBareAfter (bare from there), or
  // kPassNeedsDepth once one has. 30 endings and not one: a pass that
  // needed nothing only because a trace cut it short in some frames is not
  // taken for one that never does, to be split in the next frame.
  static constexpr uint8_t kBareAfter = 30, kPassNeedsDepth = 0xFF;
  static constexpr size_t kPassSigs = 4096;  // pass_idle_ starts over at this many

  // The signature of the pass a draw of pixel shader `ps` (the game's)
  // would open on the targets bound; never 0.
  uint64_t PassSig(ShaderId ps) const {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    auto mix = [&h](uint64_t v) {
      h = (h ^ v) * 0x100000001B3ull;
      h ^= h >> 29;
    };
    for (int i = 0; i < 4; i++) {
      mix(uint64_t(reinterpret_cast<uintptr_t>((__bridge const void*)pass_.color[i])));
      mix(uint64_t(uint32_t(pass_.level[i])));
    }
    mix(uint64_t(reinterpret_cast<uintptr_t>((__bridge const void*)pass_.depth)));
    mix(uint64_t(ps));
    return h | 1;
  }

  // Whether pixel shader `sid` (as drawn into a pass with a depth buffer)
  // may be drawn into a bare pass: a translated one whose translation has
  // no ray queries to be given.
  bool BareShader(ShaderId sid) const {
    const Shader& sh = shaders_.at(sid);
    return sh.program && sh.program->ray_reflection_sampler < 0 && sh.program->ray_shadow_sampler < 0;
  }

  // Whether the many lights' light (MlAt) is drawn first in the pass about
  // to open: it is on that pass's target, which the pass keeps.
  bool MlInto() const {
    return ml_add_.on && pass_.color[0] == ml_add_.target && pass_.level[0] == ml_add_.level && !(clear_flags_ & 1);
  }

  // Whether the targets bound allow a bare pass at all: a depth buffer
  // that is not drawn as its full-size copy (DepthProxy) and not smaller
  // than the colour target (a pass draws no further than its smallest
  // attachment: without the buffer it would draw further than with it);
  // and, unless RECOMP_PASS_LOADS=2, no colour target that is a texture of
  // the game's (pass_.ids), which a draw after a split could sample.
  bool BareTargets() const {
    if (!pass_.color[0] || !pass_.depth || pass_.proxy) return false;
    if (pass_loads_ < 2)
      for (int i = 0; i < pass_.targets; i++)
        if (pass_.ids[i]) return false;
    __unsafe_unretained id<MTLTexture> c0 = pass_.color[0], dz = pass_.depth;
    return dz.width >= std::max<NSUInteger>(1, c0.width >> pass_.level[0]) &&
           dz.height >= std::max<NSUInteger>(1, c0.height >> pass_.level[0]);
  }

  // Whether a pass opened now by a draw of pixel shader `ps` (the game's)
  // that needs no depth buffer is opened bare: nothing at its opening
  // needs the buffer (as BeginPassEncoder asks again), and its signature
  // has ended kBareAfter times without, never needing it.
  bool BareOpens(ShaderId ps) const {
    if (!BareTargets() || (clear_flags_ & 6) || MlInto()) return false;
    const auto it = pass_idle_.find(PassSig(ps));
    return it != pass_idle_.end() && it->second == kBareAfter;
  }

  // Whether the draw being recorded (pixel shader `ps`, the game's;
  // `keeps_depth`: it needs the depth buffer attached) goes into a bare
  // pass: the one open if it is bare, else the one it would open.
  bool BareDraw(ShaderId ps, bool keeps_depth) const {
    if (!pass_loads_ || !pass_.depth || keeps_depth) return false;
    return enc_open_ ? pass_bare_ : BareOpens(ps);
  }

  // Ends the bare pass open, for a draw or clear that needs the depth
  // buffer: its signature keeps the buffer from now on (PassEnded).
  void BareSplit() {
    pass_depth_used_ = true;
    if (perf_) pc_.bare_splits++;
    end_why_ = PassWhy::Split;
    EndEncoders();
  }

  // The open render pass ends: why (for the next one's PassWhy), and what
  // its signature has shown.
  void PassEnded() {
    ended_why_ = end_why_;
    if (pass_sig_) {
      if (pass_idle_.size() >= kPassSigs) pass_idle_.clear();
      uint8_t& seen = pass_idle_[pass_sig_];  // (0 when new)
      if (pass_depth_used_) seen = kPassNeedsDepth;
      else if (seen < kBareAfter) seen++;  // (kPassNeedsDepth stays)
    }
    if (perf_ && pass_has_depth_ && !pass_bare_ && !pass_depth_used_) pc_.idle_passes++;
    pass_bare_ = false;
    pass_sig_ = 0;
  }

  // Whether the draw being recorded could make its pass's colour load
  // needless as the pass's first: one or two filled triangles, unblended,
  // with no alpha test or discard, writing every channel of every target,
  // with no scissor or clip plane, their viewport the whole target, and no
  // depth or stencil test to fail (`depth_draw`). That they cover the
  // viewport is not known here (their positions are the vertex shader's),
  // so such passes are only counted (PerfCounts::color_candidates) and
  // load their colour as the others.
  bool ColorLoadCandidate(const Shader& ps, bool tri, uint32_t count, bool depth_draw) const {
    if (!tri || count > 2 || depth_draw || rs_[27] || (rs_[15] && rs_[25] != 8) || ps.discards) return false;
    if (rs_[174] || (rs_[152] & 0x3F) || rs_[8] != 3) return false;
    if ((rs_[168] & 0xF) != 0xF) return false;
    for (int i = 1; i < pass_.targets; i++)
      if ((rs_[189 + i] & 0xF) != 0xF) return false;
    return viewport_.x == 0 && viewport_.y == 0 && int(viewport_.width) >= pass_.w &&
           int(viewport_.height) >= pass_.h;
  }

  void SetPass(const Pass& p) {
    if (p.Same(pass_) && p.targets == pass_.targets) return;
    end_why_ = PassWhy::Target;
    Settle();
    // BLAS builds the draws so far asked for, between two render passes
    // (no encoder open): the frame's TLAS, built last, needs them done.
    if (rt_ && frame_open_ && rt_->PendingBuilds()) {
      RtScene* rt = rt_.get();
      const RtBuildBatch* b = rt->TakeBuilds();
      Rec([this, rt, b] { rt->EncodeBuilds(cmd_, b); });
    }
    pass_ = p;
  }

  // Ends the encoders, first running a pending whole-target clear.
  void Settle() {
    if (!enc_open_ && clear_flags_ && pass_.color[0]) BeginPassEncoder();
    EndEncoders();
  }

  // Ends the encoders (a pending clear left pending).
  void EndEncoders() {
    if (enc_open_) PassEnded();
    if (enc_open_) Rec([this] {
      [enc_ endEncoding];
      enc_ = nil;
    });
    enc_open_ = false;
    if (blit_open_) Rec([this] {
      [blit_ endEncoding];
      blit_ = nil;
    });
    blit_open_ = false;
    end_why_ = PassWhy::Other;  // (a caller's reason is for this call)
  }

  // A render pass's targets and load actions, as recorded for the
  // encoding thread.
  struct PassDesc {
    __unsafe_unretained id<MTLTexture> color[4];
    int level[4];
    int targets;
    __unsafe_unretained id<MTLTexture> depth;  // nil too for a bare pass (BareDraw)
    bool stencil;
    uint32_t clear;
    MTLClearColor clear_color;
    double clear_depth;
    uint8_t clear_stencil;
    bool ml_into;  // the many lights' light drawn first (MlAddFlush)
    const RtManyLights* ml;
    // For the pass report: why it was opened, and the numbers (TexTag) of
    // its first colour target and of the depth buffer bound (a bare pass's
    // too).
    PassWhy why;
    uint32_t tag[2];
  };

  void BeginPassEncoder() {
    if (perf_) pc_.passes++;
    frame_log_counts_.passes++;
    if (blit_open_) Rec([this] {
      [blit_ endEncoding];
      blit_ = nil;
    });
    blit_open_ = false;
    EnsureFrame();
    // The many lights' light (MlAt): drawn first in this pass where it is
    // on their target and keeps it; else added on its own before it.
    const bool ml_into = MlInto();
    if (ml_add_.on && !ml_into) MlAddFlush(pass_.color[0] == ml_add_.target && (clear_flags_ & 1));
    ml_add_.on = false;
    // Opened without the depth buffer where the draw opening it says so
    // (bare_next_: BareDraw) and nothing here needs it (as BareOpens asked);
    // what the pass shows of its signature starts over (PassEnded), a pass
    // on targets never opened bare having none to remember.
    const bool bare_targets = BareTargets();
    const bool bare = bare_next_ && bare_targets && !(clear_flags_ & 6) && !ml_into;
    pass_bare_ = bare;
    pass_has_depth_ = pass_.depth != nil;
    pass_sig_ = open_ps_ && bare_targets ? PassSig(open_ps_) : 0;
    pass_depth_used_ = pass_.proxy || (clear_flags_ & 6) || ml_into;
    if (perf_ && pass_.depth) {
      pc_.depth_passes++;
      if (bare) pc_.bare_passes++;
    }
    // Why it had to be opened (PassWhy), by the render pass before it.
    const void* now_color[4];
    bool same_color = true;
    for (int i = 0; i < 4; i++) {
      now_color[i] = (__bridge const void*)pass_.color[i];
      same_color = same_color && now_color[i] == prev_color_[i] && pass_.level[i] == prev_level_[i];
    }
    const void* now_depth = (__bridge const void*)pass_.depth;
    const PassWhy why = frame_first_pass_          ? PassWhy::Frame
                        : !same_color              ? PassWhy::Target
                        : now_depth != prev_depth_ ? PassWhy::Depth
                        : ended_why_ == PassWhy::Target ? PassWhy::Rebind
                                                        : ended_why_;
    frame_first_pass_ = false;
    for (int i = 0; i < 4; i++) {
      prev_color_[i] = now_color[i];
      prev_level_[i] = pass_.level[i];
    }
    prev_depth_ = now_depth;
    PassDesc pd{};
    for (int i = 0; i < pass_.targets; i++) {
      pd.color[i] = pass_.color[i];
      pd.level[i] = pass_.level[i];
      if (pass_.ids[i]) {
        if (Tex* t = FindTex(pass_.ids[i])) t->used = serial_;
      }
    }
    pd.targets = pass_.targets;
    pd.depth = pass_.depth;
    if (pass_.proxy) {
      id<MTLTexture> c0 = pass_.color[0];
      pd.depth = DepthProxy(pass_.depth, std::max<NSUInteger>(1, c0.width >> pass_.level[0]),
                            std::max<NSUInteger>(1, c0.height >> pass_.level[0]));
    }
    if (bare) pd.depth = nil;
    pd.why = why;
    pd.tag[0] = TexTag(now_color[0]);
    pd.tag[1] = TexTag(now_depth);
    pd.stencil = pass_.stencil;
    pd.clear = clear_flags_;
    pd.clear_color = clear_color_;
    pd.clear_depth = clear_depth_;
    pd.clear_stencil = clear_stencil_;
    pd.ml_into = ml_into;
    pd.ml = ml_.get();
    clear_flags_ = 0;
    Rec([this, pd] {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      for (int i = 0; i < pd.targets; i++) {
        MTLRenderPassColorAttachmentDescriptor* a = rp.colorAttachments[NSUInteger(i)];
        a.texture = pd.color[i];
        a.level = NSUInteger(pd.level[i]);
        // A clear clears every target, as D3D9's does.
        a.loadAction = (pd.clear & 1) ? MTLLoadActionClear : MTLLoadActionLoad;
        a.clearColor = pd.clear_color;
        a.storeAction = MTLStoreActionStore;
      }
      if (pd.depth) {
        rp.depthAttachment.texture = pd.depth;
        rp.depthAttachment.loadAction = (pd.clear & 2) ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.depthAttachment.clearDepth = pd.clear_depth;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        if (pd.stencil) {
          rp.stencilAttachment.texture = pd.depth;
          rp.stencilAttachment.loadAction = (pd.clear & 4) ? MTLLoadActionClear : MTLLoadActionLoad;
          rp.stencilAttachment.clearStencil = pd.clear_stencil;
          rp.stencilAttachment.storeAction = MTLStoreActionStore;
        }
      }
      if (gp_on_) TimePass(rp, pd);
      enc_ = [cmd_ renderCommandEncoderWithDescriptor:rp];
      [enc_ setFrontFacingWinding:MTLWindingClockwise];
      if (pd.ml_into) {
        MTLPixelFormat colors[4] = {MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid,
                                    MTLPixelFormatInvalid};
        for (int i = 0; i < pd.targets && i < 4; i++) colors[i] = pd.color[i].pixelFormat;
        __unsafe_unretained id<MTLTexture> dt = pd.depth;
        const MTLPixelFormat df = dt ? dt.pixelFormat : MTLPixelFormatInvalid;
        const bool has_stencil = dt && pd.stencil;
        if (!pd.ml->EncodeAddInto(enc_, colors, pd.targets, df, has_stencil ? df : MTLPixelFormatInvalid,
                                  pd.color[0].sampleCount) &&
            !ml_add_failed_) {
          ml_add_failed_ = true;
          WarnF("gfx: many lights: their light could not be drawn into the pass that follows them\n");
        }
      }
      if (labels_) {
        // For GPU traces: the pass's place in the frame, its size and what
        // it loads (L), clears (C) or neither (-): colour, depth.
        id<MTLTexture> t = pd.color[0];
        enc_.label = [NSString
            stringWithFormat:@"pass %d %lux%lu x%d fmt %lu %c%c", pass_no_++,
                             (unsigned long)(t.width >> pd.level[0]),
                             (unsigned long)(t.height >> pd.level[0]), pd.targets,
                             (unsigned long)t.pixelFormat, (pd.clear & 1) ? 'C' : 'L',
                             pd.depth ? ((pd.clear & 2) ? 'C' : 'L') : '-'];
      }
    });
    enc_open_ = true;
    bound_ = {};
  }

  // An encoder for the D3D targets; false if there is no colour target.
  bool BeginDraw() {
    if (!pass_.color[0]) return false;
    if (!enc_open_) BeginPassEncoder();
    return true;
  }

  PipeKey PassKey() const {
    PipeKey k;
    std::memset(&k, 0, sizeof k);
    for (int i = 0; i < 4; i++) k.color[i] = i < pass_.targets ? uint16_t(pass_.fmt[i]) : 0;
    // (A bare pass, BareDraw, has no depth buffer attached: pass_bare_ is
    // only set while its encoder is open, as it is for every caller.)
    k.depth = pass_bare_ ? uint16_t(0) : uint16_t(pass_.depth_fmt);
    k.stencil = pass_.stencil && !pass_bare_ ? uint16_t(pass_.depth_fmt) : 0;
    k.src = k.src_a = uint8_t(MTLBlendFactorOne);
    k.dst = k.dst_a = uint8_t(MTLBlendFactorZero);
    k.op = k.op_a = uint8_t(MTLBlendOperationAdd);
    return k;
  }

  id<MTLRenderPipelineState> Pipeline(const PipeKey& k) {
    auto it = pipelines_.find(k);
    if (it != pipelines_.end()) return it->second;
    {
      // Built ahead from an earlier run's recipe (PrebuildFor)?
      std::lock_guard<std::mutex> l(prebuilt_mu_);
      auto pb = prebuilt_.find(k);
      if (pb != prebuilt_.end()) {
        id<MTLRenderPipelineState> p = pb->second;
        prebuilt_.erase(pb);
        pipelines_[k] = p;
        prebuilt_used_++;
        if (perf_.load(std::memory_order_relaxed)) pipe_prebuilt_.fetch_add(1, std::memory_order_relaxed);
        // (One made in the background for a draw that could not wait:
        // StandIn. Kept as a recipe like one compiled here.)
        if (standin_started_.erase(k)) {
          standin_.erase(k);
          RecordRecipe(k);
        }
        return p;
      }
    }
    id<MTLFunction> ps = enc_shaders_.at(k.ps);
    if (k.lean & kLeanPs) {
      id<MTLFunction> lean = FoundLean(enc_shader_hash_.at(k.ps));
      if (!lean) {
        // Not made yet: the full shader (its alpha test then ALWAYS
        // passes) meanwhile, not kept under this key.
        PipeKey full = k;
        full.lean &= uint8_t(~kLeanPs);
        return Pipeline(full);
      }
      ps = lean;
    }
    // A pixel shader with ray queries takes seconds to compile: made in the
    // background, the draws meanwhile with its plain MSL (nothing traced).
    if (id<MTLRenderPipelineState> s = StandIn(k, ps)) return s;
    @autoreleasepool {
      MTLRenderPipelineDescriptor* pd = Describe(k, enc_shaders_.at(k.vs), ps);
      NSError* err = nil;
      const auto t0 = std::chrono::steady_clock::now();
      bool archived = false;
      id<MTLRenderPipelineState> p = archive_ ? archive_->Build(pd, ArchiveKey(k), &err, &archived)
                                              : [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
      if (!p) p = PlainPipeline(k, err);
      if (!p) FatalF("gfx: render pipeline: %s", err.localizedDescription.UTF8String);
      // Compiling while drawing stalls the frame: the slow ones are named
      // (one found in the pipeline archive was not compiled: its own count).
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (archived) {
        pipelines_[k] = p;
        RecordRecipe(k);
        return p;
      }
      if (ms > 2)
        std::fprintf(stderr, "gfx: pipeline %zu compiled in %.1f ms (%zu built ahead so far)\n",
                     pipelines_.size() + 1, ms, prebuilt_used_);
      if (perf_.load(std::memory_order_relaxed)) {
        const uint64_t ns = uint64_t(ms * 1e6);
        pipe_compiles_.fetch_add(1, std::memory_order_relaxed);
        pipe_ns_.fetch_add(ns, std::memory_order_relaxed);
        uint64_t m = pipe_max_ns_.load(std::memory_order_relaxed);
        while (ns > m && !pipe_max_ns_.compare_exchange_weak(m, ns, std::memory_order_relaxed)) {
        }
      }
      pipelines_[k] = p;
      RecordRecipe(k);
      return p;
    }
  }

  // Pipelines a draw needs now and cannot wait for: one at a time, above
  // the work done ahead (AheadQueue), below the frame.
  static dispatch_queue_t NowQueue() {
    static dispatch_queue_t q = dispatch_queue_create(
        "recomp.gfx.now", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
    return q;
  }

  // For key `k` whose pixel shader has ray queries (its plain MSL kept:
  // fwd_src_), not compiled yet: the pipeline itself (function `ps`) is
  // compiled on NowQueue (through the pipeline archive, which keeps it),
  // where Pipeline finds it as one built ahead; what is returned is the
  // same pipeline with the shader's plain MSL (the stubs: lit, nothing
  // reflected), quick to make, drawn with until then and not kept under
  // the key. Nil (the caller builds it as before) for another shader, one
  // the pipeline archive has, with RECOMP_RT_FORWARD_STANDIN=0, or if the
  // plain one cannot be made. A pipeline whose background compile fails
  // stays with its plain one. Encoding side.
  id<MTLRenderPipelineState> StandIn(const PipeKey& k, id<MTLFunction> ps) {
    static const bool on = [] {
      // (RECOMP_RT_FORWARD_STANDIN=0: none, the pipeline compiled by the
      // draw that first needs it.)
      const char* e = std::getenv("RECOMP_RT_FORWARD_STANDIN");
      return !e || std::atoi(e) != 0;
    }();
    if (!on) return nil;
    auto have = standin_.find(k);
    if (have != standin_.end()) return have->second;
    // One the pipeline archive has is found there at once (Pipeline).
    if (archive_ && archive_->Listed(ArchiveKey(k))) return nil;
    const auto t0 = std::chrono::steady_clock::now();
    id<MTLFunction> plain = nil;
    auto pf = standin_fn_.find(k.ps);
    if (pf != standin_fn_.end()) {
      plain = pf->second;
    } else {
      std::string src;
      bool lean = false;
      {
        std::lock_guard<std::mutex> l(fwd_src_mu_);
        auto it = fwd_src_.find(k.ps);
        if (it == fwd_src_.end()) return nil;
        src = it->second.first;
        lean = it->second.second;
      }
      // (The alpha test's function constant made a macro, as every MSL
      // compiled here, if the kept text still has it: a function with the
      // constant left cannot make a pipeline.)
      if (MetalLeanMacros(&src, false)) lean = true;
      NSError* err = nil;
      id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(src.c_str()) options:lean ? MslOptions(true) : nil error:&err];
      plain = [lib newFunctionWithName:@"main0"];
      standin_fn_[k.ps] = plain;  // (nil kept: not tried again)
    }
    if (!plain) return nil;
    NSError* err = nil;
    id<MTLRenderPipelineState> s = [dev_ newRenderPipelineStateWithDescriptor:Describe(k, enc_shaders_.at(k.vs), plain)
                                                                        error:&err];
    if (!s) return nil;
    standin_[k] = s;
    // (Made while drawing: the slow ones are named, as Pipeline's.)
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 2)
      std::fprintf(stderr, "gfx: stand-in pipeline %zu compiled in %.1f ms (%llu of them replaced so far)\n",
                   standin_.size(), ms, (unsigned long long)standin_made_.load(std::memory_order_relaxed));
    if (standin_started_.insert(k).second) {
      MTLRenderPipelineDescriptor* pd = Describe(k, enc_shaders_.at(k.vs), ps);
      id<MTLDevice> dev = dev_;
      std::mutex* mu = &prebuilt_mu_;
      auto* done = &prebuilt_;
      std::atomic<bool>* closing = &closing_;
      std::atomic<uint64_t>* made = &standin_made_;
      // (Through the pipeline archive, as Pipeline's: kept for later runs.)
      std::shared_ptr<PipelineArchive> archive = archive_;
      const uint64_t archive_key = ArchiveKey(k);
      dispatch_async(NowQueue(), ^{
        if (closing->load()) return;
        NSError* e = nil;
        id<MTLRenderPipelineState> p = archive ? archive->Build(pd, archive_key, &e)
                                               : [dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!p) return;
        made->fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> l(*mu);
        (*done)[k] = p;
      });
    }
    return s;
  }

  // A pipeline the pixel shader with ray queries (Shader::rt_forward) of
  // `k` could not make (`why`): made with its plain MSL (the stub's
  // "nothing hit"), which it is drawn with from then on (its full and lean
  // functions replaced); nil for another shader or if that fails too.
  // Encoding side.
  id<MTLRenderPipelineState> PlainPipeline(const PipeKey& k, NSError* why) {
    std::string src;
    bool lean = false;
    {
      std::lock_guard<std::mutex> l(fwd_src_mu_);
      auto it = fwd_src_.find(k.ps);
      if (it == fwd_src_.end()) return nil;
      src = std::move(it->second.first);
      lean = it->second.second;
      fwd_src_.erase(it);
    }
    WarnF("gfx: ray-traced forward reflections: a pipeline of pixel shader %016llx: %s; made without them\n",
          (unsigned long long)enc_shader_hash_.at(k.ps), why.localizedDescription.UTF8String);
    fwd_pipe_plain_.fetch_add(1, std::memory_order_relaxed);
    NSError* err = nil;
    NSString* text = @(src.c_str());
    id<MTLLibrary> full = [dev_ newLibraryWithSource:text options:lean ? MslOptions(true) : nil error:&err];
    id<MTLFunction> fn = [full newFunctionWithName:@"main0"];
    if (!fn) return nil;
    enc_shaders_[k.ps] = fn;
    id<MTLFunction> ps = fn;
    if (lean) {
      id<MTLLibrary> ll = [dev_ newLibraryWithSource:text options:MslOptions(false) error:&err];
      id<MTLFunction> lf = [ll newFunctionWithName:@"main0"];
      if (lf) {
        std::lock_guard<std::mutex> l(lean_mu_);
        lean_[enc_shader_hash_.at(k.ps)] = lf;
      }
      if (k.lean & kLeanPs) ps = lf;
    }
    if (!ps) return nil;
    return [dev_ newRenderPipelineStateWithDescriptor:Describe(k, enc_shaders_.at(k.vs), ps) error:&err];
  }

  // A pipeline's descriptor from its key and functions (metal_cache.h).
  static MTLRenderPipelineDescriptor* Describe(const PipeKey& k, id<MTLFunction> vs, id<MTLFunction> ps) {
    return DescribePipeline(k, vs, ps);
  }

  // ---- pipelines built ahead: each pipeline compiled in a run is kept as
  // a recipe (its shaders' hashes and the rest of its key) in the user's
  // caches; in later runs, once both shaders of a recipe exist, it is
  // compiled in the background and drawing finds it ready.

  using Recipe = PipelineRecipe;  // metal_cache.h

  // The recipe of a pipeline of key `k` whose shaders' translations are
  // `vs` and `ps` (the key without the run's shader ids).
  static Recipe RecipeOf(uint64_t vs, uint64_t ps, const PipeKey& k) {
    Recipe r{vs, ps, k};
    r.key.vs = r.key.ps = 0;
    return r;
  }

  static std::string RecipePath() { return RecipeFile(MetalCacheDir()); }

  // ---- shaders' lean variants (MetalLeanMacros): pixel shaders without
  // the alpha test, from the same MSL with the macro R_ALPHA_TEST false (a
  // function constant left in it would need every function specialised).

  static MTLCompileOptions* MslOptions(bool full) { return LeanMslOptions(full); }

  // RtForwardCompile's compiler: `ctx` the device (id<MTLDevice>).
  static id<MTLLibrary> CompileMsl(void* ctx, const std::string& msl, bool lean, NSError** error) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx;
    return [dev newLibraryWithSource:@(msl.c_str()) options:lean ? MslOptions(true) : nil error:error];
  }

  // The lean variant of the shader of `hash` (MSL `src`), made once; any
  // thread.
  id<MTLFunction> Lean(uint64_t hash, NSString* src) {
    if (id<MTLFunction> fn = FoundLean(hash)) return fn;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:src options:MslOptions(false) error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
    if (!fn) return nil;
    std::lock_guard<std::mutex> l(lean_mu_);
    return lean_.emplace(hash, fn).first->second;
  }

  id<MTLFunction> FoundLean(uint64_t hash) {
    std::lock_guard<std::mutex> l(lean_mu_);
    auto it = lean_.find(hash);
    return it == lean_.end() ? nil : it->second;
  }

  // ---- translated shaders' MSL from earlier runs, by Shader::hash (one
  // file of records: hash, length, text; metal_cache.h).

  void LoadMslCache() {
    msl_path_ = MslCacheFile(MetalCacheDir());
    if (FILE* f = std::fopen(msl_path_.c_str(), "rb")) {
      uint64_t h;
      uint32_t n;
      while (std::fread(&h, 8, 1, f) == 1 && std::fread(&n, 4, 1, f) == 1 && n < (16u << 20)) {
        std::string text(n, '\0');
        if (std::fread(text.data(), 1, n, f) != n) break;
        msl_cache_[h] = std::make_shared<const std::string>(std::move(text));
      }
      std::fclose(f);
    }
    if (!msl_cache_.empty()) std::fprintf(stderr, "gfx: %zu translated shaders from earlier runs\n", msl_cache_.size());
    msl_file_ = std::fopen(msl_path_.c_str(), "ab");
    // Used: a run that only reads it must keep it from being taken for
    // another generation's, unused (RemoveOldGenerations goes by time).
    utimes(msl_path_.c_str(), nullptr);
  }

  // Null if none (shared: a shader's job keeps it until it has run).
  std::shared_ptr<const std::string> CachedMsl(uint64_t hash) {
    auto it = msl_cache_.find(hash);
    return it == msl_cache_.end() ? nullptr : it->second;
  }

  void CacheMsl(uint64_t hash, const std::string& msl) {
    msl_cache_[hash] = std::make_shared<const std::string>(msl);
    if (!msl_file_) return;
    const uint32_t n = uint32_t(msl.size());
    std::fwrite(&hash, 8, 1, msl_file_);
    std::fwrite(&n, 4, 1, msl_file_);
    std::fwrite(msl.data(), 1, n, msl_file_);
    std::fflush(msl_file_);
  }

  void LoadRecipes() {
    // This translation generation's files only (metal_cache.h).
    RemoveOldGenerations(MetalCacheDir());
    recipe_path_ = RecipePath();
    if (FILE* f = std::fopen(recipe_path_.c_str(), "rb")) {
      Recipe r;
      while (std::fread(&r, sizeof r, 1, f) == 1) {
        if (!recipe_known_.insert(Fnv64(&r, sizeof r)).second) continue;
        recipes_.push_back(r);
        recipe_by_shader_.emplace(r.vs, recipes_.size() - 1);
        if (r.ps != r.vs) recipe_by_shader_.emplace(r.ps, recipes_.size() - 1);
      }
      std::fclose(f);
    }
    if (!recipes_.empty())
      std::fprintf(stderr, "gfx: %zu pipeline recipes from earlier runs\n", recipes_.size());
    recipe_file_ = std::fopen(recipe_path_.c_str(), "ab");
    utimes(recipe_path_.c_str(), nullptr);  // used (as the MSL cache)
  }

  // On the encoding thread: a pipeline just compiled, kept for later runs.
  void RecordRecipe(const PipeKey& k) {
    auto vh = enc_shader_hash_.find(k.vs), ph = enc_shader_hash_.find(k.ps);
    if (vh == enc_shader_hash_.end() || ph == enc_shader_hash_.end() || !vh->second || !ph->second) return;
    Recipe r = RecipeOf(vh->second, ph->second, k);
    std::lock_guard<std::mutex> l(recipe_mu_);
    if (!recipe_file_ || !recipe_known_.insert(Fnv64(&r, sizeof r)).second) return;
    std::fwrite(&r, sizeof r, 1, recipe_file_);
    std::fflush(recipe_file_);
  }

  // ---- the pipeline archive (metal_archive.h): pipelines kept compiled
  // between runs, in metal-archives-v1/ next to the recipes. Pipeline and
  // PrebuildFor look there first and add what they had to compile.
  // RECOMP_PIPELINE_ARCHIVE=0: none (pipelines compiled as before). On
  // iOS and tvOS its work runs on AheadQueue, one thing at a time with the
  // compiles ahead (as the shader lanes there); on macOS on a queue of its
  // own at utility priority.
  void SetUpArchive() {
    const char* e = std::getenv("RECOMP_PIPELINE_ARCHIVE");
    if (e && *e == '0') {
      std::fprintf(stderr, "gfx: no pipeline archive (RECOMP_PIPELINE_ARCHIVE=0)\n");
      return;
    }
    PipelineArchive::Options o;
    o.dir = ArchiveFolder(MetalCacheDir());
    o.scope = ArchiveScope(dev_);  // with the translation's version (metal_cache.h)
#if TARGET_OS_IOS || TARGET_OS_TV
    o.target = AheadQueue();
    // The platform budget is shared with prewarm in Options' defaults.
    o.save_idle_s = 15.0;
#endif
    archive_ = PipelineArchive::Open(dev_, std::move(o));
  }

  // On the encoding thread: the archive's name for the pipeline of `k`,
  // its recipe's (the device's own shaders, which have no translation, by
  // their fixed ids).
  uint64_t ArchiveKey(const PipeKey& k) const {
    auto vh = enc_shader_hash_.find(k.vs), ph = enc_shader_hash_.find(k.ps);
    const Recipe r = RecipeOf(vh != enc_shader_hash_.end() && vh->second ? vh->second : k.vs,
                              ph != enc_shader_hash_.end() && ph->second ? ph->second : k.ps, k);
    return gfx::ArchiveKey(r);  // (metal_cache.h: with the ray queries' patch version)
  }

  // Work done ahead (pipelines from earlier runs' recipes, lean shader
  // variants): on serial lanes at background priority (Ahead), never on
  // the global queues: there GCD ran one per core, and on an iPad the first
  // run with a Mac's 1,700 recipes compiled on every core at once (and
  // Metal's compiler service at their priority) and the game crawled for
  // minutes. iOS and tvOS: one lane, AheadQueue (the shader lanes and the
  // pipeline archive run on it too): one Metal compile at a time. macOS: two
  // lanes, AheadQueue the first (RECOMP_AHEAD_LANES=1..4).
  //
  // Until the first scene (FirstScene), or for kBoostSeconds at most, the
  // work is queued at user-initiated priority (RECOMP_AHEAD_BOOST=0: not):
  // there is nothing to play yet, and what is built ahead by then is not
  // compiled while drawing. Work still waiting when the boost ends is
  // queued again at the lanes' own background priority (a block keeps the
  // priority it was queued with).
  static dispatch_queue_t AheadQueue() {
    static dispatch_queue_t q = dispatch_queue_create(
        "recomp.gfx.ahead",
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_BACKGROUND, 0));
    return q;
  }

  // The first scene: the first frame presented with at least kSceneDraws
  // depth-writing draws to targets of the back buffer's size (a 3D scene;
  // loading screens and menus draw few or none), told once, in seconds
  // since the process started.
  static constexpr uint32_t kSceneDraws = 100;

  void FirstScene() {
    const uint32_t n = frame_depth_draws_;
    frame_depth_draws_ = 0;
    if (first_scene_ || n < kSceneDraws) return;
    first_scene_ = true;
    ahead_boost_ = false;  // the work ahead back to background priority
    std::fprintf(stderr, "gfx: first scene at t=%.2f s (since the process started; the device made at t=%.2f s)\n",
                 ProcessSeconds(), device_made_s_);
  }

  // Seconds since this process started (its start time from the kernel),
  // or since the first call if that is not to be had.
  static double ProcessSeconds() {
    static const double start = [] {
      struct kinfo_proc kp {};
      size_t len = sizeof kp;
      int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};
      timeval now{};
      gettimeofday(&now, nullptr);
      if (sysctl(mib, 4, &kp, &len, nullptr, 0) == 0 && len == sizeof kp && kp.kp_proc.p_starttime.tv_sec) {
        const timeval st = kp.kp_proc.p_starttime;
        // As a point on the steady clock: now less the time since the start.
        const double ago = double(now.tv_sec - st.tv_sec) + double(now.tv_usec - st.tv_usec) / 1e6;
        return double(NowNs()) / 1e9 - ago;
      }
      return double(NowNs()) / 1e9;
    }();
    return double(NowNs()) / 1e9 - start;
  }

  void SetUpAheadLanes() {
    ahead_lanes_.push_back(AheadQueue());
#if TARGET_OS_OSX
    int n = 2;
    if (const char* e = std::getenv("RECOMP_AHEAD_LANES")) n = std::clamp(std::atoi(e), 1, 4);
    for (int i = 1; i < n; i++)
      ahead_lanes_.push_back(dispatch_queue_create(
          "recomp.gfx.ahead",
          dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_BACKGROUND, 0)));
#endif
    const char* b = std::getenv("RECOMP_AHEAD_BOOST");
    if (b && *b == '0') ahead_boost_ = false;
  }

  // `work` on a lane, counted (ahead_queued_, ahead_done_): the lane of
  // `shader` (a pixel shader's Shader::hash), so that the work for one
  // shader (its lean variant, made once, and the pipelines that use it)
  // is done in turn, not twice at once.
  void Ahead(uint64_t shader, dispatch_block_t work) {
    ahead_queued_.fetch_add(1, std::memory_order_relaxed);
    EndBoostLate();
    dispatch_queue_t q = ahead_lanes_.empty() ? AheadQueue() : ahead_lanes_[shader % ahead_lanes_.size()];
    QueueAhead(q, work, ahead_boost_.load(), &ahead_boost_, &closing_, &ahead_done_);
  }

  // The boost ended after kBoostSeconds without a first scene (a game
  // that draws its scene to targets of another size, or a session in its
  // menus). Not after a number of presents: loading screens present
  // thousands a second.
  static constexpr double kBoostSeconds = 60.0;

  void EndBoostLate() {
    if (!ahead_boost_.load(std::memory_order_relaxed) || ProcessSeconds() - device_made_s_ < kBoostSeconds) return;
    ahead_boost_ = false;
    std::fprintf(stderr, "gfx: no first scene %.0f s after the device was made: work ahead at background priority\n",
                 kBoostSeconds);
  }

  // `work` on lane `q`, `boosted`: at user-initiated priority, and queued
  // again without it if it is still waiting once `boost` is off (but not
  // when closing: the device's drain of the lanes is to find it done).
  static void QueueAhead(dispatch_queue_t q, dispatch_block_t work, bool boosted, std::atomic<bool>* boost,
                         std::atomic<bool>* closing, std::atomic<uint64_t>* done) {
    dispatch_block_t run = ^{
      if (boosted && !boost->load() && !closing->load()) {
        QueueAhead(q, work, false, boost, closing, done);
        return;
      }
      work();
      done->fetch_add(1, std::memory_order_relaxed);
    };
    if (boosted) run = dispatch_block_create_with_qos_class(DISPATCH_BLOCK_ENFORCE_QOS_CLASS, QOS_CLASS_USER_INITIATED, 0, run);
    dispatch_async(q, run);
  }

  void PrebuildFor(uint64_t hash) {
    auto [lo, hi] = recipe_by_shader_.equal_range(hash);
    for (auto it = lo; it != hi; ++it) {
      const Recipe& r = recipes_[it->second];
      auto vs = sid_by_hash_.find(r.vs), ps = sid_by_hash_.find(r.ps);
      if (vs == sid_by_hash_.end() || ps == sid_by_hash_.end()) continue;
      PipeKey k = r.key;
      k.vs = vs->second;
      k.ps = ps->second;
      // (Shaders a recipe names are compiled on the lanes as soon as they
      // are made; this runs again when each is done, Finish. First: their
      // compile settles has_lean and rt_forward.)
      if (!shaders_.at(k.vs).fn || !shaders_.at(k.ps).fn) continue;
      // (Recorded with the ray queries or without: drawn now the same way,
      // else no draw would find it.)
      if (bool(k.lean & kRtForwardPs) != shaders_.at(k.ps).rt_forward) continue;
      NSString* src = nil;  // the pixel shader's MSL, for its lean variant
      if (k.lean & kLeanPs) {
        auto m = msl_cache_.find(r.ps);
        if (m == msl_cache_.end() || !shaders_.at(k.ps).has_lean) continue;
        // (The cache's MSL is plain: given the ray queries as the shader
        // was.)
        std::string text = *m->second;
        if (shaders_.at(k.ps).rt_forward && !RtForwardReflectionPatch(&text)) continue;
        MetalLeanMacros(&text, false);
        src = @(text.c_str());
      }
      if (!prebuild_started_.insert(k).second) continue;
      MTLRenderPipelineDescriptor* pd = Describe(k, shaders_.at(k.vs).fn, shaders_.at(k.ps).fn);
      id<MTLDevice> dev = dev_;
      std::mutex* mu = &prebuilt_mu_;
      auto* done = &prebuilt_;
      const uint64_t ps_hash = r.ps;
      std::atomic<bool>* closing = &closing_;
      std::shared_ptr<PipelineArchive> archive = archive_;
      const uint64_t archive_key = gfx::ArchiveKey(r);
      Ahead(ps_hash, ^{
        if (closing->load()) return;
        if (src) {
          id<MTLFunction> ps = Lean(ps_hash, src);
          if (!ps || closing->load()) return;
          pd.fragmentFunction = ps;
        }
        NSError* err = nil;
        id<MTLRenderPipelineState> p = archive ? archive->Build(pd, archive_key, &err)
                                               : [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!p) return;
        std::lock_guard<std::mutex> l(*mu);
        (*done)[k] = p;
      });
    }
  }

  void BindPipeline(const PipeKey& k) {
    if (bound_.pipe_set && k == bound_.pipe) return;
    bound_.pipe = k;
    bound_.pipe_set = true;
    // Found (or compiled) on the encoding thread; consecutive draws mostly
    // share theirs, unhashed.
    Rec([this, k] {
      if (!last_pipe_ || !(k == last_pipe_key_)) {
        last_pipe_ = Pipeline(k);
        last_pipe_key_ = k;
      }
      [enc_ setRenderPipelineState:last_pipe_];
    });
  }

  void BindDepth(const DepthKey& k) {
    if (last_depth_ && k == last_depth_key_) {
      if ((__bridge void*)last_depth_ != bound_.ds) {
        __unsafe_unretained id<MTLDepthStencilState> ds = last_depth_;
        Rec([this, ds] { [enc_ setDepthStencilState:ds]; });
        bound_.ds = (__bridge void*)last_depth_;
      }
      return;
    }
    auto it = depth_states_.find(k);
    if (it == depth_states_.end()) {
      MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
      dd.depthCompareFunction = k.test ? MTLCompareFunction(k.func) : MTLCompareFunctionAlways;
      dd.depthWriteEnabled = k.write != 0;
      if (k.stencil) {
        MTLStencilDescriptor* f = [MTLStencilDescriptor new];
        f.stencilFailureOperation = MTLStencilOperation(k.fail);
        f.depthFailureOperation = MTLStencilOperation(k.zfail);
        f.depthStencilPassOperation = MTLStencilOperation(k.pass);
        f.stencilCompareFunction = MTLCompareFunction(k.sfunc);
        f.readMask = k.read;
        f.writeMask = k.wmask;
        MTLStencilDescriptor* b = [MTLStencilDescriptor new];
        b.stencilFailureOperation = MTLStencilOperation(k.bfail);
        b.depthFailureOperation = MTLStencilOperation(k.bzfail);
        b.depthStencilPassOperation = MTLStencilOperation(k.bpass);
        b.stencilCompareFunction = MTLCompareFunction(k.bfunc);
        b.readMask = k.read;
        b.writeMask = k.wmask;
        dd.frontFaceStencil = f;
        dd.backFaceStencil = b;
      }
      it = depth_states_.emplace(k, [dev_ newDepthStencilStateWithDescriptor:dd]).first;
    }
    last_depth_ = it->second;
    last_depth_key_ = k;
    if ((__bridge void*)it->second == bound_.ds) return;
    __unsafe_unretained id<MTLDepthStencilState> ds = it->second;
    Rec([this, ds] { [enc_ setDepthStencilState:ds]; });
    bound_.ds = (__bridge void*)it->second;
  }

  void Raster(MTLCullMode cull, MTLTriangleFillMode fill, float bias, float slope) {
    if (int(cull) != bound_.cull) {
      Rec([this, cull] { [enc_ setCullMode:cull]; });
      bound_.cull = int(cull);
    }
    if (int(fill) != bound_.fill) {
      Rec([this, fill] { [enc_ setTriangleFillMode:fill]; });
      bound_.fill = int(fill);
    }
    if (bias != bound_.bias || slope != bound_.slope) {
      Rec([this, bias, slope] { [enc_ setDepthBias:bias slopeScale:slope clamp:0]; });
      bound_.bias = bias;
      bound_.slope = slope;
    }
  }

  // Game coordinates; scaled to the target (pass_.sx, sy).
  void ViewportOf(MTLViewport v) {
    v.originX *= pass_.sx;
    v.width *= pass_.sx;
    v.originY *= pass_.sy;
    v.height *= pass_.sy;
    if (bound_.has_vp && std::memcmp(&v, &bound_.vp, sizeof v) == 0) return;
    Rec([this, v] { [enc_ setViewport:v]; });
    bound_.vp = v;
    bound_.has_vp = true;
  }

  // Game coordinates; scaled to the target, outwards.
  void ScissorOf(MTLScissorRect s) {
    if (pass_.sx != 1.0f || pass_.sy != 1.0f) {
      const NSUInteger w = pass_.color[0] ? std::max<NSUInteger>(1, pass_.color[0].width >> pass_.level[0]) : 1;
      const NSUInteger h = pass_.color[0] ? std::max<NSUInteger>(1, pass_.color[0].height >> pass_.level[0]) : 1;
      const NSUInteger x0 = std::min(w, NSUInteger(std::floor(s.x * pass_.sx)));
      const NSUInteger y0 = std::min(h, NSUInteger(std::floor(s.y * pass_.sy)));
      const NSUInteger x1 = std::min(w, NSUInteger(std::ceil((s.x + s.width) * pass_.sx)));
      const NSUInteger y1 = std::min(h, NSUInteger(std::ceil((s.y + s.height) * pass_.sy)));
      s = {x0, y0, std::max(x1, x0) - x0, std::max(y1, y0) - y0};
    }
    if (bound_.has_sc && std::memcmp(&s, &bound_.sc, sizeof s) == 0) return;
    Rec([this, s] { [enc_ setScissorRect:s]; });
    bound_.sc = s;
    bound_.has_sc = true;
  }

  void StencilRef(uint8_t r) {
    if (int(r) == bound_.stencil_ref) return;
    Rec([this, r] { [enc_ setStencilReferenceValue:r]; });
    bound_.stencil_ref = r;
  }

  void BlendColor(const float c[4]) {
    if (std::memcmp(c, bound_.blend, sizeof bound_.blend) == 0) return;
    const float r = c[0], g = c[1], b = c[2], a = c[3];
    Rec([this, r, g, b, a] { [enc_ setBlendColorRed:r green:g blue:b alpha:a]; });
    std::memcpy(bound_.blend, c, sizeof bound_.blend);
  }

  void VertexBuffer(int index, __unsafe_unretained id<MTLBuffer> b, size_t off) {
    void* p = (__bridge void*)b;
    const NSUInteger i = NSUInteger(index);
    if (bound_.vbuf[index] == p) {
      if (bound_.voff[index] != off) Rec([this, off, i] { [enc_ setVertexBufferOffset:off atIndex:i]; });
    } else {
      Rec([this, b, off, i] { [enc_ setVertexBuffer:b offset:off atIndex:i]; });
    }
    bound_.vbuf[index] = p;
    bound_.voff[index] = off;
  }

  void FragmentBuffer(int index, __unsafe_unretained id<MTLBuffer> b, size_t off) {
    void* p = (__bridge void*)b;
    const NSUInteger i = NSUInteger(index);
    if (bound_.fbuf[index] == p) {
      if (bound_.foff[index] != off) Rec([this, off, i] { [enc_ setFragmentBufferOffset:off atIndex:i]; });
    } else {
      Rec([this, b, off, i] { [enc_ setFragmentBuffer:b offset:off atIndex:i]; });
    }
    bound_.fbuf[index] = p;
    bound_.foff[index] = off;
  }

  // Uniform block `slot` of a shader draw (0-2 vertex, 3-5 pixel): the
  // previous draw's copy when equal (by `version`, the source's change
  // counter, when it has one; else compared with a copy in ordinary
  // memory, as the arena's write-combined memory is slow to read). At least
  // 16 bytes. The shadow copy is only of the buffer this cache holds, so an
  // upload that does not write it invalidates it.
  void Uniform(int slot, const void* data, uint32_t len, uint64_t version = 0) {
    static const uint8_t kZero[16] = {};
    if (!data || !len) {
      data = kZero;
      len = sizeof kZero;
      version = 0;
    }
    LastUniform& last = last_uniform_[slot];
    const bool shadowed = !version && len <= sizeof last.shadow;
    const bool same = last.len == len &&
                      (version    ? last.version == version && last.src == data
                       : shadowed ? last.shadow_valid && std::memcmp(last.shadow, data, len) == 0
                                  : std::memcmp(last.cpu, data, len) == 0);
    if (!same) {
      Slice s = Alloc(len, uniform_align_);
      std::memcpy(s.cpu, data, len);
      last.buf = s.buf;
      last.off = s.off;
      last.cpu = s.cpu;
      last.len = len;
      last.src = data;
      last.version = version;
      // The shadow copies the buffer, so only an upload that writes it keeps
      // it good: a versioned or larger one leaves a shadow of another buffer,
      // which an unversioned upload must not compare against.
      last.shadow_valid = shadowed;
      if (shadowed) std::memcpy(last.shadow, data, len);
    }
    if (slot < 3)
      VertexBuffer(slot, last.buf, last.off);
    else
      FragmentBuffer(slot - 3, last.buf, last.off);
  }

  void TextureAndSampler(bool vertex, uint32_t reg, __unsafe_unretained id<MTLTexture> t,
                         __unsafe_unretained id<MTLSamplerState> s) {
    void* tp = (__bridge void*)t;
    void* sp = (__bridge void*)s;
    void*& bt = vertex ? bound_.vtex[reg] : bound_.ftex[reg];
    void*& bs = vertex ? bound_.vsmp[reg] : bound_.fsmp[reg];
    if (bt != tp) {
      Rec([this, vertex, t, reg] {
        if (vertex) [enc_ setVertexTexture:t atIndex:reg];
        else [enc_ setFragmentTexture:t atIndex:reg];
      });
      bt = tp;
    }
    if (bs != sp) {
      Rec([this, vertex, s, reg] {
        if (vertex) [enc_ setVertexSamplerState:s atIndex:reg];
        else [enc_ setFragmentSamplerState:s atIndex:reg];
      });
      bs = sp;
    }
  }

  // D3DSAMPLERSTATETYPE values (index 1-13; null: D3D9's defaults).
  static SamplerKey D3D9Sampler(const uint32_t* ss) {
    static const uint32_t kDefaults[14] = {0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0};
    if (!ss) ss = kDefaults;
    SamplerKey k{};
    // D3DTEXF: NONE 0, POINT 1, LINEAR 2, ANISOTROPIC 3, ...
    k.mag = ss[5] <= 1 ? 0 : 1;
    k.min = ss[6] <= 1 ? 0 : 1;
    k.mip = ss[7] == 1 ? 0 : ss[7] >= 2 ? 1 : 0xFF;
    k.u = uint8_t(AddressMode(ss[1]));
    k.v = uint8_t(AddressMode(ss[2]));
    k.w = uint8_t(AddressMode(ss[3]));
    // BORDERCOLOR: Metal's nearest fixed one.
    const uint32_t b = ss[4];
    k.border = (b >> 24) < 0x80 ? 0 : ((b & 0xFFFFFF) == 0 || ((b >> 8) & 0xFF) < 0x80) ? 1 : 2;
    bool aniso = ss[5] == 3 || ss[6] == 3;
    k.aniso = uint8_t(aniso ? std::clamp<uint32_t>(ss[10], 1, 16) : 1);
    k.min_lod = float(ss[9]);  // MAXMIPLEVEL: the most detailed level used
    return k;
  }

  id<MTLSamplerState> Sampler(const SamplerKey& k) {
    auto it = samplers_.find(k);
    if (it != samplers_.end()) return it->second;
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = k.min ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.magFilter = k.mag ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    // MIPFILTER NONE samples only the top level (or MAXMIPLEVEL).
    sd.mipFilter = k.mip == 1 ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    sd.lodMinClamp = k.min_lod;
    sd.lodMaxClamp = k.mip == 0xFF ? k.min_lod : FLT_MAX;
    sd.sAddressMode = MTLSamplerAddressMode(k.u);
    sd.tAddressMode = MTLSamplerAddressMode(k.v);
    sd.rAddressMode = MTLSamplerAddressMode(k.w);
    sd.borderColor = k.border == 0   ? MTLSamplerBorderColorTransparentBlack
                     : k.border == 1 ? MTLSamplerBorderColorOpaqueBlack
                                     : MTLSamplerBorderColorOpaqueWhite;
    int aniso = k.aniso;
    if (min_aniso_ > aniso && k.min && k.mip != 0xFF) aniso = min_aniso_;
    sd.maxAnisotropy = NSUInteger(std::max(aniso, 1));
    id<MTLSamplerState> s = [dev_ newSamplerStateWithDescriptor:sd];
    samplers_[k] = s;
    return s;
  }

  // ---- uploads

  void Update(TextureId tid, int slice, int level, const Rect& r, const void* src, uint32_t pitch) {
    auto it = textures_.find(tid);
    if (it == textures_.end() || level < 0 || level >= it->second.levels) return;
    Tex& t = it->second;
    Rect c = Clip(r, std::max(1, t.w >> level), std::max(1, t.h >> level));
    if (Empty(c)) return;
    Upload(t, slice, level, c, src, pitch, t.fmt);
  }

  // `c` (clipped) of `level` / `slice` from linear memory in `fmt`.
  void Upload(Tex& t, int slice, int level, const Rect& c, const void* src, uint32_t pitch,
              Format fmt) {
    @autoreleasepool {
      const int w = c.right - c.left, h = c.bottom - c.top;
      const auto* s = static_cast<const uint8_t*>(src);
      // The ray-traced scene's alpha (RtShadersOf): a texture whose every
      // texel passes a draw's alpha test is traced as opaque.
      if (!t.rt) {
        t.alpha_lo = std::min(t.alpha_lo, MinAlpha(s, pitch, fmt, w, h));
        if (level == 0 && slice == 0 && c.left == 0 && c.top == 0 && w == std::max(1, t.w) && h == std::max(1, t.h))
          t.alpha_whole = true;
      }
      const bool in_order = t.used == serial_;
      if (perf_) (in_order ? pc_.uploads_in_order : pc_.uploads_ahead)++;
      // Data in shared memory, in the texture's own layout: copied by the
      // GPU from there (the front end keeps it until this frame is done).
      const bool decode = IsCompressed(fmt) && !bc_;
      const bool raw = (IsCompressed(fmt) && !decode) || IsFloat(fmt) || fmt == Format::A8R8G8B8 ||
                       (fmt == Format::X8R8G8B8 && t.opaque);
      const uint32_t unit = IsCompressed(fmt) ? RowPitch(fmt, 4) : BitsPerPixel(fmt) / 8;
      if (raw && shared_buf_ && s >= shared_lo_ && unit && size_t(s - shared_lo_) % unit == 0 &&
          pitch % 4 == 0) {
        const uint32_t n = IsCompressed(fmt) ? RowCount(fmt, uint32_t(h)) : uint32_t(h);
        if (s + size_t(pitch) * n <= shared_hi_) {
          frame_upload_bytes_ += size_t(pitch) * n;
          if (perf_ && in_order) pc_.in_order_bytes += size_t(pitch) * n;
          if (in_order) {
            if (enc_open_) end_why_ = PassWhy::Blit;
            if (enc_open_ || clear_flags_) Settle();
            if (!blit_open_) Rec([this] { blit_ = TimedBlit(cmd_, "upload blit"); });
            blit_open_ = true;
          }
          EnsureFrame();
          __unsafe_unretained id<MTLTexture> tex = t.tex;
          __unsafe_unretained id<MTLBuffer> buf = shared_buf_;
          const size_t off = size_t(s - shared_lo_);
          const MTLOrigin origin = MTLOriginMake(NSUInteger(c.left), NSUInteger(c.top), 0);
          Rec([=, this] {
            id<MTLBlitCommandEncoder> bl = blit_;
            if (!in_order) {
              if (!upload_blit_) upload_blit_ = [upload_cb_ blitCommandEncoder];
              bl = upload_blit_;
            }
            [bl copyFromBuffer:buf
                       sourceOffset:off
                  sourceBytesPerRow:pitch
                sourceBytesPerImage:size_t(pitch) * n
                         sourceSize:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1)
                          toTexture:tex
                   destinationSlice:NSUInteger(slice)
                   destinationLevel:NSUInteger(level)
                  destinationOrigin:origin];
          });
          return;
        }
      }
      uint32_t row, rows;
      if ((IsCompressed(fmt) && !decode) || IsFloat(fmt)) {  // as the GPU format stores it
        row = RowPitch(fmt, uint32_t(w));
        rows = RowCount(fmt, uint32_t(h));
      } else {
        row = uint32_t(w) * 4;
        rows = uint32_t(h);
      }
      Slice dst = Alloc(size_t(row) * rows, 16);
      frame_upload_bytes_ += size_t(row) * rows;
      if (perf_ && in_order) pc_.in_order_bytes += size_t(row) * rows;
      if (decode) {
        DecodeDxt(s, pitch, fmt, w, h, reinterpret_cast<uint32_t*>(dst.cpu));
      } else if (IsCompressed(fmt) || IsFloat(fmt)) {
        for (uint32_t y = 0; y < rows; y++)
          std::memcpy(dst.cpu + size_t(y) * row, s + size_t(y) * pitch, row);
      } else {
        ConvertToArgb(s, pitch, fmt, w, h, reinterpret_cast<uint32_t*>(dst.cpu));
      }
      // Draws already recorded with or into it must see the old contents:
      // then in order. Otherwise ahead of the frame.
      if (in_order) {
        if (enc_open_) end_why_ = PassWhy::Blit;
        if (enc_open_ || clear_flags_) Settle();
        if (!blit_open_) Rec([this] { blit_ = TimedBlit(cmd_, "upload blit"); });
        blit_open_ = true;
      }
      __unsafe_unretained id<MTLTexture> tex = t.tex;
      const MTLOrigin origin = MTLOriginMake(NSUInteger(c.left), NSUInteger(c.top), 0);
      Rec([=, this] {
        id<MTLBlitCommandEncoder> bl = blit_;
        if (!in_order) {
          if (!upload_blit_) upload_blit_ = [upload_cb_ blitCommandEncoder];
          bl = upload_blit_;
        }
        [bl copyFromBuffer:dst.buf
                   sourceOffset:dst.off
              sourceBytesPerRow:row
            sourceBytesPerImage:size_t(row) * rows
                     sourceSize:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1)
                      toTexture:tex
               destinationSlice:NSUInteger(slice)
               destinationLevel:NSUInteger(level)
              destinationOrigin:origin];
      });
    }
  }

  // ---- quads: StretchRect and the window

  // On the encoding thread: draws `sr` of level `level` of `src` (sw x
  // sh) over `dr` of a dw x dh target of format `pf`, through `curve` if
  // it is one.
  void DrawQuad(id<MTLRenderCommandEncoder> e, MTLPixelFormat pf, id<MTLTexture> src, int level,
                float sw, float sh, const Rect& sr, float dw, float dh, const Rect& dr, bool linear,
                const OutputCurve& curve = {}) {
    PipeKey k;
    std::memset(&k, 0, sizeof k);
    k.vs = kQuadVs;
    k.ps = curve.Active() ? kQuadCurvePs : kQuadPs;
    k.color[0] = uint16_t(pf);
    k.src = k.src_a = uint8_t(MTLBlendFactorOne);
    k.dst = k.dst_a = uint8_t(MTLBlendFactorZero);
    k.mask[0] = MTLColorWriteMaskAll;
    [e setRenderPipelineState:Pipeline(k)];
    const float r[12] = {float(dr.left) / dw * 2 - 1,  1 - float(dr.top) / dh * 2,
                         float(dr.right) / dw * 2 - 1, 1 - float(dr.bottom) / dh * 2,
                         float(sr.left) / sw,          float(sr.top) / sh,
                         float(sr.right) / sw,         float(sr.bottom) / sh,
                         float(level),                 curve.ShaderMode(),
                         curve.exponent,               0};
    [e setVertexBytes:r length:sizeof r atIndex:0];
    [e setFragmentBytes:r length:sizeof r atIndex:0];
    [e setFragmentTexture:src atIndex:0];
    [e setFragmentSamplerState:quad_sampler_[linear ? 1 : 0] atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
  }

  // Premultiplied RGBA on top of the frame (the achievement toast).
  void DrawAlphaQuad(id<MTLRenderCommandEncoder> e, id<MTLTexture> src, float dw, float dh, const Rect& dr) {
    PipeKey k;
    std::memset(&k, 0, sizeof k);
    k.vs = kQuadVs;
    k.ps = kQuadPs;
    k.color[0] = uint16_t(MTLPixelFormatBGRA8Unorm);
    k.blend = 1;
    k.src = k.src_a = uint8_t(MTLBlendFactorOne);
    k.dst = k.dst_a = uint8_t(MTLBlendFactorOneMinusSourceAlpha);
    k.mask[0] = MTLColorWriteMaskAll;
    [e setRenderPipelineState:Pipeline(k)];
    const float r[12] = {float(dr.left) / dw * 2 - 1, 1 - float(dr.top) / dh * 2,
                         float(dr.right) / dw * 2 - 1, 1 - float(dr.bottom) / dh * 2,
                         0, 0, 1, 1, 0, 0, 0, 0};
    [e setVertexBytes:r length:sizeof r atIndex:0];
    [e setFragmentBytes:r length:sizeof r atIndex:0];
    [e setFragmentTexture:src atIndex:0];
    [e setFragmentSamplerState:quad_sampler_[1] atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
  }

  // A bitmap the port supplied (SetPresentOverlay), on top of this present.
  // `scene` was captured when the frame was presented, after FirstScene().
  // With no overlay installed this returns without drawing.
  void DrawAchievementOverlay(id<MTLRenderCommandEncoder> e, int pw, int ph, float scale, bool scene) {
    const PresentOverlayFn fn = ::recomp::PresentOverlay();
    if (!fn) return;
    static bool logged = false;
    if (!logged) {
      logged = true;
      // Same channel as the renderer's other lines: stderr.
      std::fprintf(stderr, "achievements: overlay active\n");
      std::fflush(stderr);
    }
    PresentOverlayBitmap bm{};
    if (!fn(pw, ph, scale, scene, &bm) || !bm.rgba || bm.width <= 0 || bm.height <= 0) return;
    if (bm.y >= ph || bm.x >= pw || bm.y + bm.height <= 0 || bm.x + bm.width <= 0) return;
    if (!toast_tex_ || toast_w_ != bm.width || toast_h_ != bm.height) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:NSUInteger(bm.width)
                                                                                    height:NSUInteger(bm.height)
                                                                                 mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead;
      td.storageMode = MTLStorageModeShared;
      toast_tex_ = [dev_ newTextureWithDescriptor:td];
      toast_w_ = bm.width;
      toast_h_ = bm.height;
      toast_serial_ = 0;
    }
    if (!toast_tex_) return;
    if (toast_serial_ != bm.serial) {
      [toast_tex_ replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(bm.width), NSUInteger(bm.height))
                    mipmapLevel:0
                      withBytes:bm.rgba
                    bytesPerRow:NSUInteger(bm.width * 4)];
      toast_serial_ = bm.serial;
    }
    DrawAlphaQuad(e, toast_tex_, float(pw), float(ph), Rect{bm.x, bm.y, bm.x + bm.width, bm.y + bm.height});
  }

  // On the encoding thread: the front buffer (bw x bh) to the window,
  // scaled to fit with its aspect kept, black around it (FitRect). Not
  // while the window cannot be seen.
  // t_present: when the game called Present (CACurrentMediaTime), for the
  // latency from that call to the display (0: not measured).
  void Show(__unsafe_unretained id<MTLTexture> front, int bw, int bh, double t_present = 0,
            bool scene = false) {
    // (The frame log has an "s" line for every image handed here, shown or
    // not: a frame that got no drawable is written as dropped, 0, so that a
    // reader counting them stays in step with the "p" lines when the window
    // is hidden or covered for a while.)
    if (SDL_GetWindowFlags(window_) &
        (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED | SDL_WINDOW_OCCLUDED)) {
      FrameLogLine("s 0");
      return;
    }
    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(window_, &pw, &ph);
    if (pw <= 0 || ph <= 0) {
      FrameLogLine("s 0");
      return;
    }
    if (layer_.drawableSize.width != pw || layer_.drawableSize.height != ph)
      layer_.drawableSize = CGSizeMake(pw, ph);
    id<CAMetalDrawable> drawable = nil;
    // (For the perf line: how long the encoding thread waits here.)
    const uint64_t wait_t0 = perf_.load(std::memory_order_relaxed) ? NowNs() : 0;
    if (link_on_) {
      // The display's drawable for its next refresh: wait at most ~3
      // refreshes (none comes while the window cannot be seen).
      std::unique_lock<std::mutex> l(link_slot_.mu);
      link_slot_.cv.wait_for(l, std::chrono::milliseconds(50), [&] { return link_slot_.drawable != nil; });
      drawable = link_slot_.drawable;
      link_slot_.drawable = nil;
    } else {
      drawable = [layer_ nextDrawable];
    }
    if (wait_t0) show_wait_ns_.fetch_add(NowNs() - wait_t0, std::memory_order_relaxed);
    if (!drawable) {
      FrameLogLine("s 0");
      return;
    }
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = TimedRender(cmd_, rp, "show");
    const Rect r = FitRect(bw, bh, pw, ph);
    const bool same = r.right - r.left == bw && r.bottom - r.top == bh;
    DrawQuad(e, MTLPixelFormatBGRA8Unorm, front, 0, float(bw), float(bh), {0, 0, bw, bh},
             float(pw), float(ph), r, !same, output_curve_);
    int lw = 0, lh = 0;
    SDL_GetWindowSize(window_, &lw, &lh);
    const float scale = lw > 0 ? std::max(1.f, float(pw) / float(lw)) : 1.f;
    DrawAchievementOverlay(e, pw, ph, scale, scene);
    [e endEncoding];
    // When the display really showed it (what Metal's HUD graphs).
    std::mutex* mu = &shown_mu_;
    std::vector<double>* shown = &shown_;
    std::vector<double>* latency = &latency_;
    dispatch_semaphore_t slot = pace_present_ ? frames_free_ : nullptr;
    frame_drawable_ = true;
    [drawable addPresentedHandler:^(id<MTLDrawable> shown_drawable) {
      if (slot) dispatch_semaphore_signal(slot);  // RECOMP_PACE_PRESENT
      const double t = shown_drawable.presentedTime;
      FrameLogLine("s %.4f", t);  // (the frame log's; 0: dropped)
      if (t <= 0) return;  // dropped
      std::lock_guard<std::mutex> l(*mu);
      shown->push_back(t);
      if (t_present > 0 && t > t_present) latency->push_back((t - t_present) * 1e3);
    }];
    // With vsync, shown no sooner than one refresh of the fastest rate
    // after the last: a variable-refresh display (ProMotion, Adaptive-Sync
    // set to Variable) then shows each frame when it is ready instead of
    // on a fixed 120 Hz grid (an 85 fps game shown 8.3 or 16.7 ms apart).
    if (vsync_ && !link_on_)
      [cmd_ presentDrawable:drawable afterMinimumDuration:MinFrameDuration()];
    else
      [cmd_ presentDrawable:drawable];
    NotePresentation({bw, bh, pw, ph, r});
  }

  // One refresh at the display's highest rate, in seconds.
  double MinFrameDuration() {
    NSInteger hz = 0;
#if TARGET_OS_OSX
    for (NSScreen* scr in NSScreen.screens)
      if (@available(macOS 12.0, *)) hz = std::max(hz, scr.maximumFramesPerSecond);
#else
    hz = UIScreen.mainScreen.maximumFramesPerSecond;
#endif
    return 1.0 / double(hz > 0 ? hz : 60);
  }

  // ---- readback

  // `r` of level `level` of `tex` (tw x th), as everything recorded so far
  // leaves it, into dst in `fmt`. Waits for the GPU.
  void ReadTexture(__unsafe_unretained id<MTLTexture> tex, int tw, int th, int level, const Rect& r, void* dst,
                   uint32_t pitch, Format fmt) {
    @autoreleasepool {
      if (!tex) return;
      if (ml_add_.on && tex == ml_add_.target) {  // (the many lights' light first)
        Settle();
        MlAddFlush();
      }
      Rect c = Clip(r, tw, th);
      const int w = c.right - c.left, h = c.bottom - c.top;
      if (w <= 0 || h <= 0) return;
      const size_t bytes = size_t(w) * h * 4;
      if (!readback_ || readback_.length < bytes) {
        Keep(readback_);
        readback_ = [dev_ newBufferWithLength:std::max<size_t>(bytes, 1u << 20)
                                      options:MTLResourceStorageModeShared];
      }
      Settle();
      EnsureFrame();
      __unsafe_unretained id<MTLTexture> src = tex;
      __unsafe_unretained id<MTLBuffer> out = readback_;
      Rec([=, this] {
        id<MTLBlitCommandEncoder> bl = TimedBlit(cmd_, "readback");
        [bl copyFromTexture:src
                         sourceSlice:0
                         sourceLevel:NSUInteger(level)
                        sourceOrigin:MTLOriginMake(NSUInteger(c.left), NSUInteger(c.top), 0)
                          sourceSize:MTLSizeMake(NSUInteger(w), NSUInteger(h), 1)
                            toBuffer:out
                   destinationOffset:0
              destinationBytesPerRow:NSUInteger(w) * 4
            destinationBytesPerImage:bytes];
        [bl endEncoding];
      });
      EndFrame();
      WaitIdle();  // the encoding thread has committed it: last_cb_ is it
      [last_cb_ waitUntilCompleted];
      const auto* px = static_cast<const uint32_t*>(readback_.contents);
      const uint32_t bpp = BitsPerPixel(fmt) / 8;
      // Rows of the requested rectangle outside the target stay as they were.
      const int ox = c.left - r.left, oy = c.top - r.top;
      for (int y = 0; y < h; y++) {
        const uint32_t* s = px + size_t(y) * uint32_t(w);
        uint8_t* out = static_cast<uint8_t*>(dst) + size_t(y + oy) * pitch + size_t(ox) * bpp;
        if (fmt == Format::X8R8G8B8 || fmt == Format::A8R8G8B8) {
          auto* o = reinterpret_cast<uint32_t*>(out);
          for (int x = 0; x < w; x++) o[x] = s[x] | 0xFF000000u;
        } else {
          for (int x = 0; x < w; x++) EncodePixel(out + size_t(x) * bpp, fmt, s[x] | 0xFF000000u);
        }
      }
    }
  }

  // ---- rectangles

  Rect ViewportRect() const {
    Rect v{int32_t(viewport_.x), int32_t(viewport_.y), int32_t(viewport_.x + viewport_.width),
           int32_t(viewport_.y + viewport_.height)};
    return Clip(v, pass_.w, pass_.h);
  }

  // Scissor of a draw: the target and the scissor rect if enabled. False if
  // nothing is left.
  bool DrawScissor(MTLScissorRect* s) const {
    Rect r{0, 0, pass_.w, pass_.h};
    if (rs_[174]) r = Intersect(r, scissor_);
    if (Empty(r)) return false;
    *s = {NSUInteger(r.left), NSUInteger(r.top), NSUInteger(r.right - r.left),
          NSUInteger(r.bottom - r.top)};
    return true;
  }

  // ---- members

  mutable std::recursive_mutex mu_;
  SDL_Window* window_;
  SDL_MetalView view_ = nullptr;
  id<MTLTexture> toast_tex_ = nil;
  int toast_w_ = 0, toast_h_ = 0;
  uint64_t toast_serial_ = 0;
  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  CAMetalLayer* layer_ = nil;
  size_t uniform_align_ = 256;
  bool back_buffer_alpha_ = false;  // SetBackBufferAlpha
  bool bc_ = true;  // the GPU samples BC1-3 (else DXT is decoded on upload)

  // The encoding thread, its queue of recorded batches and the batch being
  // recorded.
  std::thread encoder_;
  id<NSObject> activity_ = nil;  // beginActivityWithOptions:, held while alive
  std::mutex q_mu_;
  std::condition_variable q_cv_, done_cv_;
  std::deque<Batch> batches_;
  std::vector<Batch> free_batches_;  // run, their memory kept for reuse
  uint64_t submitted_ = 0, done_ = 0;
  bool stop_ = false;
  bool sync_ = false;  // RECOMP_METAL_SYNC: closures run as recorded
  bool reversed_z_ = true;  // RECOMP_REVERSED_Z: float depth stored as 1 - d
  bool depth_from_w_ = true;  // RECOMP_DEPTH_FROM_W
  int wvp_reg_ = -1;  // SetDepthFromW's; -1 also with depth from w off
  DepthPlanes depth_planes_ = nullptr;
  int w_logged_ = 0;
  std::unordered_set<uint64_t> w_noted_;  // NoteNotFromW's shaders
  // The draw being recorded: a reversed buffer's depth bias for
  // r_position (DepthBiasOf's clip), and whether it tests or writes depth.
  float clip_bias_ = 0;
  bool depth_used_ = false;
  // RECOMP_GPU_PASSES: a timestamp buffer per frame in flight (one more
  // than the slots), the passes it timed, and the totals by pass number.
  static constexpr size_t kNoPass = SIZE_MAX;
  struct PassTimes {
    id<MTLCounterSampleBuffer> buf = nil;
    std::vector<std::string> names;  // by pass number in the frame
    std::vector<uint32_t> draws, verts;  // of each: its draws and their vertices
    // Of the render passes among them (TimePass; up to the last one, 0 for
    // the others): why each was opened (PassWhy) and its targets' numbers
    // (PassDesc::tag).
    std::vector<uint8_t> why;
    std::vector<std::array<uint32_t, 2>> tags;
    size_t current = kNoPass;            // the pass being encoded, if timed
  };
  static constexpr size_t kTimedPasses = 256;
  bool gp_on_ = false;
  bool gp_report_ = false;  // RECOMP_GPU_PASSES itself: the report (gp_on_ without it: the frame log's times)
  std::vector<PassTimes> gp_;
  size_t gp_slot_ = 0;  // encoding thread
  std::mutex gp_mu_;
  // The frame log's (ReportPassTimes, under gp_mu_): the pass names' ids
  // ("n" lines; some tens of names a run), and the "q" line being put
  // together (kept: its memory is the next frame's).
  std::unordered_map<std::string, uint32_t> gp_ids_;
  std::string gp_line_;
  // Totals by name and the occurrence within its frame ("rt TLAS build"
  // twice a frame: #1 and #2), not by position: a frame's passes vary
  // (BLAS batches, light masks, diagnostics), so a position is no pass.
  struct PassSum {
    double ms = 0;
    double order = 0;  // position in its frames, summed (display order)
    int frames = 0;    // frames it ran in
    double draws = 0, verts = 0;
    // A render pass's: how often it was opened for each reason (PassWhy),
    // and its targets' numbers the last time (PassDesc::tag).
    int why[size_t(PassWhy::Count)] = {};
    uint32_t tag[2] = {};
  };
  std::map<std::string, PassSum> gp_sum_;
  double gp_busy_ms_ = 0, gp_timed_ms_ = 0;
  int gp_frames_ = 0;
  double gp_tick_ns_ = 1;  // GPU timestamp tick in ns
  id<MTLBuffer> gp_scratch_ = nil;  // TimedMarkerAfter's texel

  void SetUpPassTimes() {
    if (![dev_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
      WarnF("gfx: RECOMP_GPU_PASSES: no stage-boundary counters on this GPU\n");
      return;
    }
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> c in dev_.counterSets)
      if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
    if (!ts) return;
    gp_.resize(kSlots + 1);
    for (PassTimes& p : gp_) {
      MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
      d.counterSet = ts;
      d.storageMode = MTLStorageModeShared;
      d.sampleCount = kTimedPasses * 2;
      NSError* err = nil;
      p.buf = [dev_ newCounterSampleBufferWithDescriptor:d error:&err];
      if (!p.buf) {
        WarnF("gfx: RECOMP_GPU_PASSES: %s\n", err.localizedDescription.UTF8String);
        gp_.clear();
        return;
      }
    }
    // GPU ticks to ns: two CPU/GPU timestamp pairs 50 ms apart.
    MTLTimestamp c0, g0, c1, g1;
    [dev_ sampleTimestamps:&c0 gpuTimestamp:&g0];
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    [dev_ sampleTimestamps:&c1 gpuTimestamp:&g1];
    // The CPU side is in ns here (measured: with mach_timebase applied, the
    // passes of a 29 ms frame added up to 41.7 x that).
    if (g1 > g0) gp_tick_ns_ = double(c1 - c0) / double(g1 - g0);
    gp_scratch_ = [dev_ newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    gp_on_ = true;
    WarnF("gfx: GPU pass times on (1 GPU tick = %.3f ns)\n", gp_tick_ns_);
  }

  // On the encoding thread: the pass about to be made samples the time its
  // vertex work starts and its fragment work ends.
  void TimePass(MTLRenderPassDescriptor* rp, const PassDesc& pd) {
    PassTimes& p = gp_[gp_slot_];
    const size_t n = p.names.size();
    p.current = kNoPass;  // (past kTimedPasses its draws are counted nowhere)
    if (n >= kTimedPasses) return;
    p.current = n;
    id<MTLTexture> t = pd.color[0];
    char name[96];
    std::snprintf(name, sizeof name, "%lux%lu x%d fmt %lu %c%c",
                  (unsigned long)(t.width >> pd.level[0]), (unsigned long)(t.height >> pd.level[0]),
                  pd.targets, (unsigned long)t.pixelFormat, (pd.clear & 1) ? 'C' : 'L',
                  pd.depth ? ((pd.clear & 2) ? 'C' : 'L') : '-');
    p.names.push_back(name);
    p.draws.resize(p.names.size());
    p.verts.resize(p.names.size());
    p.why.resize(p.names.size());
    p.why.back() = uint8_t(pd.why);
    p.tags.resize(p.names.size());
    p.tags.back() = {pd.tag[0], pd.tag[1]};
    MTLRenderPassSampleBufferAttachmentDescriptor* a = rp.sampleBufferAttachments[0];
    a.sampleBuffer = p.buf;
    a.startOfVertexSampleIndex = n * 2;
    a.endOfVertexSampleIndex = MTLCounterDontSample;
    a.startOfFragmentSampleIndex = MTLCounterDontSample;
    a.endOfFragmentSampleIndex = n * 2 + 1;
  }

  // On the encoding thread: a draw of `n` vertices in the pass being timed.
  void GpDraw(NSUInteger n) {
    if (!gp_on_) return;
    PassTimes& p = gp_[gp_slot_];
    if (p.current >= p.draws.size()) return;
    p.draws[p.current]++;
    p.verts[p.current] += uint32_t(n);
  }

  // GpuTimer::next: the encoding thread's next pair of samples, named.
  static bool TimedPassNext(void* ctx, const char* name, __autoreleasing id<MTLCounterSampleBuffer>* buf,
                            NSUInteger* start, NSUInteger* end) {
    auto* self = static_cast<MetalDevice*>(ctx);
    if (!self->gp_on_ || self->gp_.empty()) return false;
    PassTimes& p = self->gp_[self->gp_slot_];
    const size_t n = p.names.size();
    p.current = kNoPass;  // (a pass of its own: no draws)
    if (n >= kTimedPasses) return false;
    p.names.push_back(name);
    p.draws.resize(p.names.size());
    p.verts.resize(p.names.size());
    *buf = p.buf;
    *start = n * 2;
    *end = n * 2 + 1;
    return true;
  }

  // When the frame's GPU work is done (`serial`'s; `presented`: it ended at
  // Present, not at a readback). Its passes' times go to the frame log
  // (RECOMP_FRAME_LOG): a "q" line a presented frame, every pass in the
  // order it ran with its ms, the passes of the parts a readback cut off
  // before it first (they have no line of their own: a reader needs no
  // serial without a "p"); a pass's name as a number, given once in an "n"
  // line before its first use (a name is up to 95 characters, a frame some
  // 100 passes). And with RECOMP_GPU_PASSES they are added up; every 120
  // frames the passes by GPU time, with the GPU's busy time a frame
  // (`busy_s`, overlapping frames counted once).
  void ReportPassTimes(PassTimes* p, double busy_s, uint64_t serial, bool presented) {
    const size_t n = p->names.size();
    std::vector<double> ms(n, 0);
    if (n) {
      NSData* d = [p->buf resolveCounterRange:NSMakeRange(0, n * 2)];
      const auto* r = static_cast<const MTLCounterResultTimestamp*>(d.bytes);
      // A pass's cost: from the end of the one before (the first: its own
      // vertex start) to its fragment end. Starts alone mislead on a tiled
      // GPU, where the next pass's vertex work runs under this one's
      // fragments; ends are in order, so the costs add up to the frame.
      uint64_t prev = d && r[0].timestamp != MTLCounterErrorValue ? r[0].timestamp : 0;
      for (size_t i = 0; d && i < n; i++) {
        const uint64_t end = r[2 * i + 1].timestamp;
        if (end == MTLCounterErrorValue || !end || !prev) continue;
        if (end > prev) ms[i] = double(end - prev) * gp_tick_ns_ / 1e6;
        prev = std::max(prev, end);
      }
    }
    std::lock_guard<std::mutex> l(gp_mu_);
    if (frame_log_) {
      // (Completion handlers run in the frames' order; were two ever to
      // swap, a cut-off part's passes would go to the frame before its own.)
      char item[32];
      for (size_t i = 0; i < n; i++) {
        const auto [it, fresh] = gp_ids_.try_emplace(p->names[i], uint32_t(gp_ids_.size()));
        if (fresh) FrameLogLine("n %u %s", it->second, it->first.c_str());
        const int len = std::snprintf(item, sizeof item, " %u:%.3f", it->second, ms[i]);
        if (len > 0 && size_t(len) < sizeof item) gp_line_.append(item, size_t(len));
      }
      if (presented) {
        FrameLogLine("q %llu%s", (unsigned long long)serial, gp_line_.c_str());
        gp_line_.clear();
      } else if (gp_line_.size() > (64u << 10)) {
        // Readbacks with no Present after them (a loading loop): not kept
        // for ever, and their busy time dropped with them, so the next "g"
        // and "q" are about the same parts.
        gp_line_.clear();
        FrameLogBusyForget();
      }
    }
    std::map<std::string, int> seen;
    for (size_t i = 0; gp_report_ && i < n; i++) {
      const int k = ++seen[p->names[i]];
      std::string key = p->names[i];
      if (k > 1) key += " #" + std::to_string(k);
      PassSum& s = gp_sum_[key];
      s.ms += ms[i];
      s.order += double(i);
      s.frames++;
      if (i < p->draws.size()) {
        s.draws += p->draws[i];
        s.verts += p->verts[i];
      }
      if (i < p->why.size() && i < p->tags.size() && p->why[i] && p->why[i] < uint8_t(PassWhy::Count)) {
        s.why[p->why[i]]++;
        s.tag[0] = p->tags[i][0];
        s.tag[1] = p->tags[i][1];
      }
      gp_timed_ms_ += ms[i];
    }
    p->names.clear();
    p->draws.clear();
    p->verts.clear();
    p->why.clear();
    p->tags.clear();
    p->current = kNoPass;
    if (!gp_report_) return;  // (timed for the frame log alone)
    gp_busy_ms_ += busy_s * 1e3;
    if (++gp_frames_ < 120) return;
    std::vector<std::pair<std::string, PassSum>> all(gp_sum_.begin(), gp_sum_.end());
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    std::string out;
    char line[320];
    std::snprintf(line, sizeof line,
                  "gfx: GPU busy %.2f ms a frame, passes %.2f (each from the previous pass's end to its own; "
                  "ms a frame over all %d frames):\n",
                  gp_busy_ms_ / gp_frames_, gp_timed_ms_ / gp_frames_, gp_frames_);
    out += line;
    for (size_t k = 0; k < all.size() && k < 32; k++) {
      const PassSum& s = all[k].second;
      char in[32] = "";
      if (s.frames < gp_frames_) std::snprintf(in, sizeof in, "  (in %d frames)", s.frames);
      // A render pass's line ends with why it was opened (PassWhy: the
      // commonest reason, with its share where there were others) and the
      // numbers of its first colour target and depth buffer (TexTag): after
      // the columns the line had, so the pass names and their #n stay.
      char after[72] = "";
      int opened = 0, top = 0;
      for (int w = 1; w < int(PassWhy::Count); w++) {
        opened += s.why[w];
        if (s.why[w] > s.why[top]) top = w;
      }
      if (opened) {
        char share[16] = "";
        if (s.why[top] < opened) std::snprintf(share, sizeof share, " %d%%", int(100.0 * s.why[top] / opened + 0.5));
        std::snprintf(after, sizeof after, "  after %s%s, c%05x d%05x", PassWhyName(PassWhy(top)), share, s.tag[0],
                      s.tag[1]);
      }
      std::snprintf(line, sizeof line, "  %6.2f ms  @%5.1f  %s%s  %.0f draws %.0fk vertices%s\n", s.ms / gp_frames_,
                    s.order / s.frames, all[k].first.c_str(), in, s.draws / s.frames, s.verts / s.frames / 1e3, after);
      out += line;
    }
    std::fputs(out.c_str(), stderr);
    gp_sum_.clear();
    gp_busy_ms_ = 0;
    gp_timed_ms_ = 0;
    gp_frames_ = 0;
  }

  bool labels_ = false;  // RECOMP_METAL_LABELS
  float clip_planes_[6][4] = {};  // SetClipPlane
  // The upscaler (RECOMP_UPSCALER): recording side, except as noted.
  Upscaler upscaler_ = Upscaler::Off;
  std::vector<RtBoost> rt_boost_;  // RECOMP_RT_BOOST (target_size.h)
  // Ray tracing (SetRayTracing; null: off): the scene, the world matrix's
  // register, the frame whose scene has ended (RtEndFrame), the position
  // types logged.
  std::unique_ptr<RtScene> rt_;
  // The frame's local lights (SetRtLights; gfx/rt_light_list.h): given on
  // the recording side, Current() on the encoding side for the frame it
  // encodes.
  RtLightFrames rt_lights_;
  int rt_world_ = -1;
  std::vector<float> plain_places_;  // SetRtPlainPlaces's, five floats a place
  bool (*rt_filter_)(const float (*)[4], uint32_t, bool, bool) = nullptr;  // RayTracing::filter
  bool (*rt_alpha_source_)(uint64_t, uint32_t*, uint32_t*) = nullptr;  // RayTracing::alpha_source
  bool (*rt_albedo_source_)(uint64_t, uint32_t*, uint32_t*) = nullptr;  // RayTracing::albedo_source
  bool rt_hit_data_ = false;  // RtSceneOptions::hit_data: draws' albedos are resolved
  uint64_t rt_ended_ = 0;
  // RtActorOf's actor now: its registers' hash, its first bone's place,
  // its number (0: none), actors numbered.
  uint64_t actor_state_ = 0, actor_id_ = 0, actor_seq_ = 0;
  float actor_anchor_[3] = {};
  bool rt_other_passes_ = true, rt_other_color_ = false;  // RayTracing::other_passes, other_color_passes
  // Ray-traced shadows (RtShadowFor; null: not traced): the kernel, a mask
  // and the ray counters per frame slot (the mask the size of the depth
  // last traced in it), the one the draw being recorded binds (nil: none),
  // the presented frame's last choice (RtShadowNext) and its mask, and the
  // numbers of targets' contents (Tex/Depth::rt_epoch).
  std::unique_ptr<RtShadowMasker> shadow_masker_;
  id<MTLTexture> shadow_mask_[kSlots] = {};
  // (The far light's counters at 0, the local lights' at kLightCounts:
  // rays, rays blocked, pixels outside the lights' reach, pixels within
  // their lamp gaps.)
  // (And the local lights' pass's at kLightSetCounts: rays, rays blocked,
  // texels reached by more lights than the cap, rays traced again; the many
  // lights' at kManyLightCounts: rays, rays blocked.)
  static constexpr size_t kLightCounts = 16, kLightSetCounts = 32, kManyLightCounts = 48, kOcclusionCounts = 64,
                          kShadowCountsSize = 80;
  id<MTLBuffer> shadow_counts_[kSlots] = {};
  __unsafe_unretained id<MTLTexture> rt_mask_ = nil;
  RtShadowMemo shadow_memo_;
  bool shadow_log_ = false, shadow_view_ = false;  // RayTracing::shadow_log, Debug::ShadowMask
  uint32_t shadow_log_every_ = 120;                // RECOMP_RT_SHADOW_LOG_EVERY
  // RayTracing::own_maps_lit; the lit textures (OwnMapLit, by the value's
  // bits); the draws and shaders said (OwnMapDraw).
  bool own_maps_lit_ = false;
  std::unordered_map<uint32_t, id<MTLTexture>> own_map_tex_;
  RtOwnMaps own_maps_;
  uint64_t shadow_frames_ = 0, shadow_diag_frame_ = UINT64_MAX;  // presented frames; the last diagnosed
  // A frame slot's diagnostics (RtShadowDiag and the blocker counts), what
  // the CPU knew of that trace, and the TLAS's instances.
  struct ShadowDiag {
    id<MTLBuffer> buf = nil;
    bool pending = false;
    uint32_t size[2] = {};
    float eye[4] = {}, light[4] = {};
    uint32_t instances = 0;
    std::vector<RtInstanceSummary> summary;
  };
  ShadowDiag shadow_diag_[kSlots];
  __unsafe_unretained id<MTLTexture> shadow_memo_mask_ = nil;
  // The far light's mask accumulated over frames (gfx/rt_accumulate.h)
  // where the depth is the jittered scene's: traced at one pixel of every
  // sun_step_ x sun_step_ (a dim light's every kRtShadowDimStep), the pixel
  // turning; null: off (the default; RECOMP_RT_ACCUMULATE=1 turns it on),
  // each frame's mask traced and upsampled on its own; and its history.
  std::unique_ptr<RtAccumulator> accumulator_;
  RtAccumulateHistory sun_history_;
  uint32_t sun_step_ = kRtAccumulateStep;
  // The sun's mask where the depth is jittered: traced at one pixel of
  // every sun_turn_ x sun_turn_, turning each frame (the temporal
  // anti-aliasing averages the turns: a caster thinner than the step,
  // which a fixed pixel catches in some frames and not others as the view
  // moves, shadows in proportion instead of flickering). 2 by default (the
  // rays of the fixed 2 x 2 pass before); RECOMP_RT_SUN_TURN=3 or 4 fewer
  // (1.5 ms a frame at 3440 x 1440 in the city at 4 x 4 against 2.7, seen
  // as noise in play), 0 every 2 x 2's top-left pixel, as before.
  uint32_t sun_turn_ = 2;
  // Local lights' masks (RtLightShadowFor): the choices by light (kept
  // across presented frames), a mask per entry (one set for all frames in
  // flight, each the size of the depth it was last traced from).
  RtLightMaskCache light_cache_;
  id<MTLTexture> light_mask_[RtLightMaskCache::kEntries] = {};
  // Lights traced (render passes ended for them) this presented frame and
  // the most allowed (RayTracing::local_light_masks); pixel shaders flagged
  // as reading a local light's mask, and those flagged that cannot be.
  uint32_t light_traced_ = 0, light_budget_ = 8;
  uint64_t light_shaders_ = 0, light_shaders_refused_ = 0;
  // The local lights traced at once (LightSetAt; RayForwardShadow::local):
  // the pass's settings (RayTracing::local_light_step, local_light_cap;
  // lights_ready_: the kernel made), the lights drawn this presented frame
  // and the one before (in draw order, kRtLightSlots at most), and those
  // the next pass traces (the last frame's, a moved one where it will be:
  // RtLightPredict); the pass of this presented frame (tried once), the
  // snapshot, depth texture and its contents it traced, its lights and
  // their masks (one set for all frames in flight: Metal orders a frame's
  // writes after the last one's reads).
  uint32_t lights_step_ = 3, lights_cap_ = 4;
  // The local lights lit by the device (RayManyLights, MlAt): its passes,
  // the configuration, the G-buffer's textures as last named (depth,
  // albedo, normal, surface, class), whether a presented frame has tried;
  // the bins (encoding side) and what they held (the report's).
  std::unique_ptr<RtManyLights> ml_;
  RayManyLights ml_cfg_;
  bool ml_only_ = false, ml_tried_ = false;
  TextureId ml_ids_[5] = {};
  RtManyBins ml_bins_;
  std::atomic<uint64_t> ml_enc_lights_{0}, ml_enc_dropped_{0}, ml_enc_entries_{0};
  // Their light not added yet (MlAt, MlAddFlush): into which target.
  struct MlAdd {
    bool on = false;
    __unsafe_unretained id<MTLTexture> target = nil;
    int level = 0;
  } ml_add_;
  bool ml_add_failed_ = false;  // (encoding side: warned once)
  // Ray-traced ambient occlusion (RtOcclusionFor): its passes, the rays'
  // length, the texture traced (kept per size), the one bound for a draw,
  // and the memo of what was traced this presented frame (depth, contents).
  // The far light's bounce (RtReflector::EncodeBounce): its rays' length
  // (0: off), its output, whether traced this presented frame.
  float bounce_distance_ = 0.0f, bounce_strength_ = 1.0f;
  float sun_lit_share_ = 1.0f;  // of the far light's rays, those that reached it (a frame slot's)
  float bounce_ao_strength_ = 1.0f;  // RECOMP_RT_AO_STRENGTH
  static constexpr float kBounceMinLit = 0.03f;
  id<MTLTexture> bounce_out_ = nil;
  bool bounce_live_ = false, bounce_traced_ = false;
  RtAccumulateCamera bounce_cam_;  // the last bounce's camera (its frame: bounce_cam_frame_)
  uint64_t bounce_cam_frame_ = 0;  // BounceFor ran this presented frame; traced
  std::unique_ptr<RtOccluder> occluder_;
  float ao_radius_ = 0.0f;
  id<MTLTexture> ao_out_ = nil;
  __unsafe_unretained id<MTLTexture> rt_ao_ = nil;
  struct AoMemo {
    const void* depth = nullptr;
    uint64_t epoch = 0;
    bool traced = false, open = false;
  } ao_memo_;
  bool lights_ready_ = false, lights_alt_ = false;
  RtLocalLight lights_seen_[kRtLightSlots], lights_prev_[kRtLightSlots], lights_plan_[kRtLightSlots];
  bool lights_plan_moved_[kRtLightSlots] = {};
  uint32_t lights_seen_n_ = 0, lights_prev_n_ = 0, lights_plan_n_ = 0;
  bool lights_tried_ = false;
  const void* lights_tried_depth_ = nullptr;
  uint64_t lights_tried_epoch_ = 0;
  const RtFrame* lights_frame_ = nullptr;
  const void* lights_depth_ = nullptr;
  uint64_t lights_epoch_ = 0;
  RtLocalLight lights_traced_[kRtLightSlots];
  bool lights_traced_moved_[kRtLightSlots] = {};
  uint32_t lights_traced_n_ = 0, lights_traced_step_ = 1, lights_size_[2] = {};
  uint32_t lights_traced_frame_ = 0, lights_traced_stride_ = 0;  // their samples' turn (RtLightSample)
  uint32_t lights_alt_said_ = UINT32_MAX;  // the RECOMP_RT_LIGHT_ALT phase said last
  id<MTLTexture> light_masks_ = nil, light_dist_ = nil;
  // What BindForward bound at kRtForwardLightMasksTexture and
  // kRtForwardLightDistTexture (render-encoder state, as Bound's).
  __unsafe_unretained id<MTLTexture> fwd_lmask_ = nil, fwd_ldist_ = nil;
  __unsafe_unretained id<MTLTexture> fwd_smask_ = nil;
  // RECOMP_RT_FORWARD_DUMP: a frame's forward ray draws by (pixel shader's hash << 4 | class).
  struct FwdDump {
    uint32_t draws = 0;
    uint64_t triangles = 0;
    bool reflect = false, lamps = false;
    int w = 0, h = 0;
  };
  std::unordered_map<uint64_t, FwdDump> fwd_dump_;
  uint64_t fwd_dump_frame_ = 0;
  // The sun mask the draw's lamp rays are gated by (kRtForwardSunGate), or nil.
  __unsafe_unretained id<MTLTexture> rt_fwd_sun_mask_ = nil;
  // Ray-traced reflections (RtReflectionFor; null: not traced): the
  // kernel, the textures and counters per frame slot, the one the draw
  // being recorded binds (nil: none), the presented frame's last choice,
  // what it was traced with, its textures and scale (the debug view's),
  // whether that view is drawn.
  std::unique_ptr<RtReflector> reflector_;
  uint64_t refl_ab_ = 0;  // passes traced (RECOMP_RT_REFL_CHECKER=ab's turns)
  uint64_t refl_ssr_ab_ = 0;  // likewise RECOMP_RT_REFL_SSR=ab's
  uint64_t refl_diag_ = 0;    // and RECOMP_RT_REFL_DIAG's
  // Screen-space reflections (ReflectionKeepAt): this frame's lit scene
  // (the lighting's target where the scene's reflections were traced) or a
  // whole copy of it, whether it was kept; the kept one, the camera and
  // jitter that drew it and its frame (refl_frame_ counts presented
  // frames); the pixels the reflections covered in the last frame read
  // back.
  static constexpr uint64_t kReflKeepPixels = 150000;
  __unsafe_unretained id<MTLTexture> refl_lit_ = nil;
  TextureId refl_lit_copy_ = 0;
  bool refl_keep_done_ = false;
  id<MTLTexture> refl_kept_ = nil;
  double refl_kept_vp_[16] = {};
  float refl_kept_jitter_[2] = {};
  uint64_t refl_kept_frame_ = ~0ull, refl_frame_ = 0;
  uint64_t refl_last_listed_ = 0;
  id<MTLTexture> refl_out_[RtReflectionCache::kEntries] = {}, refl_dist_[RtReflectionCache::kEntries] = {};
  id<MTLBuffer> refl_counts_[kSlots] = {};
  __unsafe_unretained id<MTLTexture> rt_refl_ = nil;
  RtReflectionCache refl_cache_;
  int refl_last_ = -1;  // the entry traced last (the debug view's)
  float refl_last_scale_ = 1.0f;
  bool refl_view_ = false;  // RayTracing::reflection_view
  // MetalFX's denoiser of the lit scene (StartDenoiser; null: off), on
  // (dn_on_) until it falls back to dn_fallback_ (DenoiseOff: why in
  // dn_why_, also why it never started), each frame slot's counter of
  // reactive pixels. This presented frame's lit scene (the colour target
  // the lighting drew into; nil: none yet), the id of a whole copy of it
  // (StretchRect), the reflections' hit distance of the scene's
  // depth (nil: none), whether it was denoised; whether the last presented
  // frame was (else the history starts over), lit frames in a row whose
  // lit scene no post-process draw read, the presented frames' count, the
  // output that trades places with the lit scene's texture.
  std::unique_ptr<RtDenoiser> denoiser_;
  bool dn_on_ = false;
  Upscaler dn_fallback_ = Upscaler::Off;
  std::string dn_why_;
  id<MTLBuffer> dn_counts_[kSlots] = {};
  __unsafe_unretained id<MTLTexture> dn_lit_ = nil, dn_hit_ = nil;
  TextureId dn_copy_id_ = 0;
  bool dn_done_ = false, dn_last_ = false;
  int dn_missed_ = 0;
  uint64_t dn_frame_ = 0;
  id<MTLTexture> dn_out_ = nil;
  // The snapshot the effects of one depth share (RtSnapshot).
  RtSnapshotMemo snap_memo_;
  const RtFrame* snap_frame_ = nullptr;
  // Forward pixel shaders tracing their reflections themselves (rt_
  // forward_reflection.h; false: not on this GPU or mode): what draws
  // tracing nothing bind; the draw being recorded binds what it traces with
  // (BindForward), the snapshot and parameters (RtForwardFor); shaders made
  // without the ray queries.
  bool rt_forward_ = false;
  // The flight recorder (RecorderBegin): made once, its frames counted.
  std::unique_ptr<FlightRecorder> recorder_;
  bool recorder_tried_ = false, rec_pad_down_ = false;
  uint64_t rec_frame_ = 0;
  double rec_last_t_ = 0;
  uint64_t rec_skip_[3] = {};  // RecorderNote's last skip counts
  // The frames' notes (RecorderNote) until RecordFrame takes them.
  static constexpr uint64_t kRecNotes = 16;
  std::mutex rec_note_mu_;
  std::string rec_notes_[kRecNotes];
  RtForwardNullScene rt_fwd_null_;
  bool rt_fwd_bind_ = false;
  const RtFrame* rt_fwd_frame_ = nullptr;
  RtForwardParams rt_fwd_params_{};
  uint64_t fwd_plain_ = 0, fwd_stub_calls_ = 0;
  // The patched shaders' plain MSL and whether it has the lean macro (for
  // PlainPipeline, on the encoding side); pipelines made with it.
  std::mutex fwd_src_mu_;
  std::unordered_map<ShaderId, std::pair<std::string, bool>> fwd_src_;
  // Pipelines drawn with a shader's plain MSL while the one with its ray
  // queries compiles in the background (StandIn): encoding side.
  std::unordered_map<PipeKey, id<MTLRenderPipelineState>, BytesHash<PipeKey>> standin_;
  std::unordered_map<ShaderId, id<MTLFunction>> standin_fn_;
  std::unordered_set<PipeKey, BytesHash<PipeKey>> standin_started_;
  std::atomic<uint64_t> standin_made_{0};
  std::atomic<uint64_t> fwd_pipe_plain_{0};
  // The colour target of the pass the last snapshot was taken in (its
  // traced effects'), for RtForwardFor; refused draws logged (RtForwardFor).
  const void* snap_target_ = nullptr;
  uint64_t fwd_log_frame_ = UINT64_MAX;
  int fwd_logged_ = 0;
  uint64_t rt_epoch_seq_ = 0;
  bool capture_failed_ = false;  // a capture variant failed (CaptureFailed)
  uint32_t rt_types_seen_ = 0;
  float render_scale_ = 1.0f;
  id<MTLLibrary> internal_lib_ = nil;
  id<MTLDepthStencilState> depth_copy_ds_ = nil;
  std::unordered_map<NSUInteger, id<MTLRenderPipelineState>> depth_copy_pipes_;
  struct DepthCopy {
    id<MTLTexture> tex = nil;
    uint64_t frame = 0;  // serial_ of the last copy
  };
  std::unordered_map<void*, std::vector<DepthCopy>> depth_proxies_;  // by the scaled depth buffer
  id<MTLTexture> low_color_ = nil;  // the composite at the internal resolution
  bool frame_scene_ = false;        // a scene buffer drawn to this frame
  bool composited_ = false;         // and upscaled to the back buffer
  // What BindTarget chose by (EnsureTarget): serial_ (0: no frame open),
  // frame_scene_ and composited_ then.
  uint64_t bound_serial_ = 0;
  bool bound_scene_ = false, bound_composited_ = false;
  bool jitter_draw_ = false;        // the draw being recorded is jittered
  int jitter_phase_ = 0;
  float jitter_[2] = {};            // this frame's, in pixels of the internal resolution
  float jitter_sign_[2] = {1, 1}, mv_sign_ = 1;  // conventions (RECOMP_MFX_SIGNS)
  double vp_cur_[16] = {}, vp_prev_[16] = {};  // the camera, world to clip (row vectors)
  CameraVote cam_vote_;                   // this frame's camera, from the scene's draws
  bool camera_rejected_said_ = false;     // (said once: draws whose product was no camera)
  bool have_vp_ = false, have_prev_ = false;
  __unsafe_unretained id<MTLTexture> scene_depth_ = nil;  // the jittered scene's depth buffer
  bool jitter_other_said_ = false, jitter_other_said_light_ = false;  // (RtJitterFor: said once)
  id<MTLFXTemporalScaler> scaler_ = nil;
  id<MTLFXSpatialScaler> spatial_ = nil;                 // EnsureSpatial
  id<MTLTexture> spatial_in_ = nil, spatial_out_ = nil;  // its input (SMAA's output); its output if not the back buffer
  bool spatial_failed_ = false;
  bool scaler_failed_ = false, fx_reset_ = true;
  id<MTLTexture> motion_tex_ = nil, fx_depth_ = nil, fx_out_ = nil;
  id<MTLRenderPipelineState> motion_pipe_ = nil;
  id<MTLComputePipelineState> taa_pipe_ = nil, rcas_pipe_ = nil;  // EnsureTaa; EnsureSharpen
  float sharpness_ = 0.5f;  // RECOMP_SHARPNESS
  OutputCurve output_curve_;  // RECOMP_OUTPUT_CURVE
  // The volumetric fog (RtFog), its look, whether a frame has traced it.
  std::unique_ptr<RtFog> fog_;
  RtFogLook fog_look_;
  bool fog_traced_ = false;
  bool fog_frame_ = false;       // this presented frame traced it (FogOver)
  id<MTLBuffer> sky_buf_ = nil;  // (RtFog::Encode's sky colour: zero)
  id<MTLRenderPipelineState> fog_over_pipe_ = nil;  // FogOver
  bool fog_over_failed_ = false;
  // The scene's textures under TAA: a mip sharper, as FSR 2 and DLSS
  // advise for temporal antialiasing (log2(render / display) - 1; the
  // accumulated jittered samples keep it from aliasing).
  static constexpr float kTaaLodBias = 0.0f;  // -1 sharpened textures but cost ~2 ms a frame (bandwidth) on the M3
  id<MTLTexture> taa_history_[2] = {};  // read one, write the other
  id<MTLTexture> taa_src_ = nil;        // the composite, as TAA reads it
  int taa_cur_ = 0;
  id<MTLRenderPipelineState> smaa_pipe_[3] = {};  // SmaaPrepare
  id<MTLTexture> smaa_area_ = nil, smaa_search_ = nil, smaa_edges_ = nil, smaa_blend_ = nil;
  id<MTLTexture> smaa_stencil_ = nil;            // SmaaPrepare: the edge pass's pixels
  id<MTLDepthStencilState> smaa_ds_[2] = {};     // SmaaPrepare: edge, weight pass
  int pass_no_ = 0;      // labels_: render passes in this frame (encoding thread)
  Batch rec_;
  // Recording side: a frame is open (BeginFrame), a render or blit encoder
  // is (as the encoding thread will have it).
  bool frame_open_ = false, enc_open_ = false, blit_open_ = false;
  // RECOMP_PASS_LOADS (BareDraw), recording side: the switch (0 off, 1 on,
  // 2 on for the game's textures too); the open pass has no depth buffer
  // attached; RecordDraw's word to the BeginPassEncoder it calls (open it
  // bare; the pixel shader of the draw opening it, 0 otherwise); the open
  // pass's signature (0: not opened by a draw, or on targets never opened
  // bare: BareTargets), whether a depth buffer is bound, whether anything
  // in the pass needed it; and what each signature has shown (a count up
  // to kBareAfter, or kPassNeedsDepth).
  int pass_loads_ = 1;
  bool pass_bare_ = false, bare_next_ = false;
  ShaderId open_ps_ = 0;
  uint64_t pass_sig_ = 0;
  bool pass_has_depth_ = false, pass_depth_used_ = false;
  std::unordered_map<uint64_t, uint8_t> pass_idle_;
  // The pass report's PassWhy, recording side: a caller's reason for the
  // EndEncoders it is about to call, why the last render pass ended, that
  // none has opened in this frame yet, and the last one's targets (compared
  // only).
  PassWhy end_why_ = PassWhy::Other, ended_why_ = PassWhy::Other;
  bool frame_first_pass_ = true;
  const void* prev_color_[4] = {};
  int prev_level_[4] = {};
  const void* prev_depth_ = nullptr;
  uint32_t rt_timed_tick_ = 0;  // RecordDraw: every 32nd draw's ray-scene work is timed
  // Encoding thread: the shaders' functions (CreateShader hands them over)
  // and the device's own quad samplers.
  std::unordered_map<ShaderId, id<MTLFunction>> enc_shaders_;
  std::unordered_map<ShaderId, uint64_t> enc_shader_hash_;  // the encoding thread's Shader::hash
  // Pipelines built ahead (Recipe): the recording side's index, the
  // background compiles' results.
  std::unordered_map<uint64_t, ShaderId> sid_by_hash_;
  std::vector<Recipe> recipes_;
  std::unordered_multimap<uint64_t, size_t> recipe_by_shader_;
  std::unordered_set<PipeKey, BytesHash<PipeKey>> prebuild_started_;
  std::mutex prebuilt_mu_;
  std::unordered_map<PipeKey, id<MTLRenderPipelineState>, BytesHash<PipeKey>> prebuilt_;
  size_t prebuilt_used_ = 0;  // encoding thread
  std::mutex recipe_mu_;
  std::unordered_set<uint64_t> recipe_known_;
  std::mutex lean_mu_;
  std::unordered_map<uint64_t, id<MTLFunction>> lean_;  // by Shader::hash (Lean)
  std::string recipe_path_;
  std::shared_ptr<PipelineArchive> archive_;  // set in Init; null: none
  FILE* recipe_file_ = nullptr;
  std::unordered_map<uint64_t, std::shared_ptr<const std::string>> msl_cache_;  // recording side (CreateShader)
  // Shaders compiled when first drawn (MakeShader): the jobs, by Shader::hash
  // (under their own lock; taken after mu_, never before it), the shaders
  // waiting for each; the rest under mu_.
  ShaderJobs shader_jobs_;
  Waiters<uint64_t, ShaderId> waiters_;
  bool lazy_ = true;          // RECOMP_LAZY_SHADERS
  bool prefetching_ = false;  // the lanes have started (AdoptFinished)
  std::vector<dispatch_queue_t> lanes_;  // RECOMP_SHADER_LANES
  size_t next_lane_ = 0;
  std::string msl_path_;
  FILE* msl_file_ = nullptr;
  id<MTLSamplerState> quad_sampler_[2] = {};

  // Encoding thread: the frame's command buffers and encoders.
  id<MTLCommandBuffer> cmd_ = nil;
  id<MTLCommandBuffer> upload_cb_ = nil;  // queued before cmd_
  id<MTLCommandBuffer> last_cb_ = nil;    // the last one committed
  id<MTLRenderCommandEncoder> enc_ = nil;
  id<MTLBlitCommandEncoder> blit_ = nil;         // in cmd_, between render encoders
  id<MTLBlitCommandEncoder> upload_blit_ = nil;  // in upload_cb_
  uint64_t serial_ = 0;
  // FrameSerial: the frame the next draw goes to (serial_ while one is
  // open, else serial_ + 1); written under mu_.
  std::atomic<uint64_t> draw_serial_{1};
  std::atomic<uint64_t> completed_{0};  // the last frame the GPU finished
  // GPU time of finished command buffers and the presented frames among
  // them (TakeGpuFrameMs); whether the one being recorded is presented.
  std::atomic<uint64_t> gpu_ns_{0}, gpu_frames_{0};
  // Counters for tuning (EnablePerfStats): the recording side's under mu_,
  // the encoding thread's atomic.
  struct PerfCounts {
    uint64_t frames = 0, draws = 0, draw_ns = 0, ops = 0, passes = 0;
    uint64_t w_draws = 0;  // with depth from w
    uint64_t w_other = 0;  // that would be, but have another projection than the buffer content's
    uint64_t uploads_ahead = 0, uploads_in_order = 0, upload_bytes = 0, in_order_bytes = 0;
    // Arena buffers: made while drawing, and taken ready (made ahead).
    uint64_t textures = 0, depths = 0, arena_buffers = 0, arena_bytes = 0, arena_ahead = 0;
    uint64_t shaders = 0, shader_ns = 0, msl_compiles = 0;
    // Of shader_ns: GLSL to SPIR-V to MSL, and Metal's compile of the MSL
    // (newLibraryWithSource).
    uint64_t translate_ns = 0, library_ns = 0;
    // Ray-traced shadows (RtShadowFor): masks traced, their pixels (a ray
    // each at most: none for the sky), and draws given the all-lit mask.
    uint64_t shadow_masks = 0, shadow_pixels = 0;
    uint64_t shadow_dim = 0;  // masks of a dim light (RayShadow::dim_light)
    uint64_t shadow_white[size_t(RtShadowWhite::Count)] = {};  // by reason
    uint64_t shadow_jitter[4] = {};  // masks traced by RtJitter: with the jitter, or why not
    // Read from the GPU's counters as frames finish; the far light's rays
    // traced again next to a captured mesh (rt_shadow_blocked) apart.
    uint64_t shadow_rays = 0, shadow_hits = 0, shadow_retraced = 0;
    // Texels the game's cascaded shadow map was sure were lit (a short ray); the
    // map the last mask had: its cascades' texel sizes, range, format, size.
    uint64_t shadow_cascade_lit = 0;
    float cascade_texel[4] = {}, cascade_range = 0;
    uint32_t cascade_format = 0, cascade_size[2] = {};
    // The far light's colour register (rgb, w) the last draw had.
    float shadow_colour[4] = {};
    bool shadow_colour_seen = false;
    // Local lights' masks (RtLightShadowFor): traced, their pixels
    // dispatched, draws reusing one, draws given the all-lit mask (by
    // reason; off screen apart), render passes ended to trace one; the GPU's
    // counters as frames finish (rays, blocked, pixels outside the light).
    uint64_t light_draws = 0;  // draws of shaders naming a local light (each counted once below)
    uint64_t light_masks = 0, light_pixels = 0, light_reused = 0, light_offscreen = 0, light_splits = 0;
    uint64_t light_white[size_t(RtShadowWhite::Count)] = {};
    uint64_t light_jitter[4] = {};  // as shadow_jitter
    // Draws of lights past the budget given their own mask of an earlier
    // presented frame (RtLightStep::Older), and the oldest such mask's age.
    uint64_t light_older = 0, light_older_age = 0;
    uint64_t light_moving = 0;  // masks traced of lights that moved (RtLightMaskCache::kMovingBudget)
    uint64_t light_rays = 0, light_hits = 0, light_outside = 0, light_in_gap = 0;
    uint64_t snapshots = 0;  // TLASes built for traced effects (RtSnapshot)
    uint64_t own_map_draws = 0;  // draws of shaders still reading a game's own shadow map (RayShadow::own_map)
    // Where frames' cameras came from (CameraVote): two draws agreeing, one
    // unconfirmed, none; and the draws whose product was not the camera.
    uint64_t camera_confirmed = 0, camera_unconfirmed = 0, camera_none = 0, camera_rejected = 0;
    // Ray-traced reflections (RtReflectionFor): passes traced, their
    // pixels, draws given nothing hit; the GPU's counters (RtReflection
    // Counts) as frames finish. Snapshots shared by effects of one depth.
    uint64_t refl_passes = 0, refl_pixels = 0, refl_none = 0;
    uint64_t bounce_skipped = 0;  // presented frames whose bounce the gate left out
    uint64_t refl_rays = 0, refl_hits = 0, refl_misses = 0, refl_bare = 0, refl_back = 0;
    uint64_t refl_back_captured = 0, refl_other_back = 0, refl_filled = 0, refl_checker = 0;
    uint64_t refl_screen = 0, refl_screen_mixed = 0, refl_screen_passes = 0, refl_kept = 0, refl_dense = 0;
    uint64_t refl_diag[3] = {};  // passes as traced, without hit shadow rays, all opaque
    double refl_distance = 0;
    uint64_t snapshots_shared = 0;
    // Draws of forward shaders tracing their reflections (RtForwardFor), by
    // whether they traced and why not.
    uint64_t fwd[size_t(RtForwardOff::Count)] = {};
    // The local lights traced at once (LightSetAt): passes, their lights,
    // render passes ended for them; the lights' draws reading them, and
    // those tracing their own (not in the pass: new, moved off their
    // prediction, past kRtLightSlots, or none traced); the GPU's counters.
    uint64_t lset_passes = 0, lset_lights = 0, lset_splits = 0, lset_draws = 0, lset_own = 0;
    uint64_t lset_rays = 0, lset_hits = 0, lset_capped = 0, lset_again = 0;
    // The many lights (MlAt): passes, the volumes' draws left out (Only),
    // the GPU's counters (rays, blocked).
    uint64_t ml_passes = 0, ml_skipped = 0, ml_rays = 0, ml_blocked = 0, ml_add_passes = 0;
    // The occlusion (RtOcclusionFor): passes, the GPU's counters.
    uint64_t ao_passes = 0, ao_rays = 0, ao_blocked = 0;
    // The denoiser (DenoiseAt): frames denoised, history resets, reactive
    // pixels (the GPU's count), lit scenes no post-process draw was found
    // reading, render passes ended for the guides.
    uint64_t dn_frames = 0, dn_resets = 0, dn_reactive = 0, dn_missed = 0, dn_splits = 0;
    // The frame's local lights (SetRtLights): lists handed over, lights in
    // them, those kept (rt_light_list.h), by kind.
    uint64_t rtl_lists = 0, rtl_given = 0, rtl_kept = 0, rtl_point = 0, rtl_spot = 0;
    // Shaders compiled when first drawn: here, and waited for on a lane;
    // compiled on the lanes (and the lanes' time on them).
    uint64_t first_drawn = 0, joins = 0, join_ns = 0, ahead_compiles = 0, ahead_ns = 0;
    // The render thread's time: blocked on a frame slot (BeginFrame), in
    // the ray scene's Snapshot, in RtEndFrame; and in RtNoteDraw and
    // RtCaptureDraw: their calls, how many were timed (1 draw in 32) and
    // those calls' mach ticks.
    uint64_t slot_wait_ns = 0, snapshot_ns = 0, rt_end_ns = 0;
    uint64_t rt_note_calls = 0, rt_note_timed = 0, rt_note_ticks = 0;
    uint64_t rt_capture_calls = 0, rt_capture_timed = 0, rt_capture_ticks = 0;
    // RECOMP_PASS_LOADS (BareDraw): render passes with a depth buffer
    // bound, those opened without it, the ones of those ended because it
    // was needed, those that had it attached and never used it; and the
    // passes whose first draw may cover the target (ColorLoadCandidate).
    uint64_t depth_passes = 0, bare_passes = 0, bare_splits = 0, idle_passes = 0, color_candidates = 0;
  };
  std::atomic<bool> perf_{false};
  std::atomic<bool> closing_{false};  // the device is going: prebuild blocks do nothing
  PerfCounts pc_;
  uint64_t perf_t0_ = 0;
  std::atomic<uint64_t> enc_busy_ns_{0}, pipe_compiles_{0}, pipe_ns_{0}, pipe_max_ns_{0},
      pipe_prebuilt_{0};
  std::atomic<uint64_t> show_wait_ns_{0};  // the encoding thread's wait for a drawable (Show)
  // Work handed to AheadQueue, and how much of it has run (Ahead): the
  // difference is its backlog.
  std::atomic<uint64_t> ahead_queued_{0}, ahead_done_{0};
  std::vector<dispatch_queue_t> ahead_lanes_;  // Ahead's (SetUpAheadLanes)
  std::atomic<bool> ahead_boost_{true};        // until the first scene (EndBoostLate)
  // The first scene (FirstScene): the frame's depth-writing draws to
  // targets of the back buffer's size, and whether it was reported.
  uint32_t frame_depth_draws_ = 0;
  bool first_scene_ = false;
  double device_made_s_ = 0;  // ProcessSeconds() when Init ran
  static uint64_t NowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
  }
  // RECOMP_FRAME_LOG=<file> (frame_log.h): every frame's numbers, a line
  // each, not the report's averages over seconds, which hide a frame time
  // that swings between 17 and 37 ms. This backend's lines: "p" and "c" at
  // Present on the recording thread (FrameLogPresent), "r" at the frame's
  // end on the encoding thread (EncodeEndFrame), "g", "n" and "q" when the
  // GPU finished the frame (its completion handler, ReportPassTimes), "s"
  // when the display showed a drawable (Show).
  const bool frame_log_ = FrameLogOn();
  // A frame's numbers for "p" and "c", from the last Present to this one
  // (so all its parts where a readback cut it), on the recording thread
  // under mu_. Apart from PerfCounts, which holds the same times: that is
  // zeroed every report (2 s), so no frame's share can be taken from it.
  struct FrameLogCounts {
    uint64_t draws = 0, passes = 0;
    uint64_t begun_ns = 0;       // NowNs() when its first BeginFrame had its slot; 0: none yet
    uint64_t slot_wait_ns = 0;   // its BeginFrames' waits for a slot
    uint64_t later_wait_ns = 0;  // of those, the ones after begun_ns
    uint64_t snapshot_ns = 0, rt_end_ns = 0;  // RtSnapshot's scene work, RtEndFrame
    uint64_t note_calls = 0, note_timed = 0, note_ticks = 0;  // RecordDraw's RtNoteDraw, as PerfCounts'
    uint32_t fwd_draws = 0;  // draws that traced their own rays (RtForwardFor)
  };
  FrameLogCounts frame_log_counts_;
  // NowNs() at the last "p"; the ops handed to the encoding thread in all
  // (Flush) and those the last "p" had counted.
  uint64_t frame_log_last_ = 0, frame_log_ops_ = 0, frame_log_ops_seen_ = 0;
  // The ray scene's counters (RtStats) as the last "p" saw them, for the
  // next one's differences: those TakeStats zeroes (TakePerfStats zeroes
  // these with them), and total_builds, which nothing zeroes.
  struct FrameLogRtSeen {
    uint64_t instances = 0, captured = 0, capture_builds = 0, capture_refits = 0;
  };
  FrameLogRtSeen frame_log_rt_seen_;
  uint64_t frame_log_rt_builds_ = 0;
  bool presented_ = false;
  bool vsync_ = false;  // ResetSwapChain's: frames paced by the display
  // RECOMP_DISPLAY_LINK: the link runs on a thread of its own with its own
  // run loop (the game's main thread pumps events only between frames).
  bool link_on_ = false;
  RecompLinkSlot link_slot_;
  std::thread link_thread_;
  CFRunLoopRef link_loop_ = nullptr;
  std::mutex link_mu_;
  std::condition_variable link_cv_;

  void StartDisplayLink() {
    if (@available(macOS 14.0, *)) {
      link_on_ = true;
      link_thread_ = std::thread([this] {
        @autoreleasepool {
          RecompDisplayLinkTarget* target = [RecompDisplayLinkTarget new];
          target.slot = &link_slot_;
          CAMetalDisplayLink* link = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer_];
          link.delegate = target;
          // RECOMP_LINK_LATENCY: frames the display may hold (default 1).
          float latency = 1.0f;
          if (const char* e = std::getenv("RECOMP_LINK_LATENCY")) latency = std::clamp(float(std::atof(e)), 1.0f, 2.0f);
          link.preferredFrameLatency = latency;
          const float hz = float(1.0 / MinFrameDuration());
          link.preferredFrameRateRange = CAFrameRateRangeMake(30.0f, hz, hz);
          [link addToRunLoop:NSRunLoop.currentRunLoop forMode:NSDefaultRunLoopMode];
          {
            std::lock_guard<std::mutex> l(link_mu_);
            link_loop_ = CFRunLoopGetCurrent();
          }
          link_cv_.notify_all();
          CFRunLoopRun();
          [link invalidate];
        }
      });
      std::unique_lock<std::mutex> l(link_mu_);
      link_cv_.wait(l, [&] { return link_loop_ != nullptr; });
      std::fprintf(stderr, "gfx: frames paced by CAMetalDisplayLink (up to %.0f Hz)\n", 1.0 / MinFrameDuration());
    } else {
      std::fprintf(stderr, "gfx: RECOMP_DISPLAY_LINK needs macOS 14\n");
    }
  }

  void StopDisplayLink() {
    if (!link_on_) return;
    {
      std::lock_guard<std::mutex> l(link_mu_);
      if (link_loop_) CFRunLoopStop(link_loop_);
    }
    if (link_thread_.joinable()) link_thread_.join();
    link_on_ = false;
  }
  bool pace_present_ = false;   // RECOMP_PACE_PRESENT=1
  bool frame_drawable_ = false; // encoding thread: this frame showed a drawable
  // Times the display showed frames (presented handlers), for
  // TakeShownIntervals.
  std::mutex shown_mu_;
  std::vector<double> shown_;
  std::vector<double> latency_;  // ms from the game's Present to the display
  size_t frame_upload_bytes_ = 0;  // texture data uploaded since the last Present
  dispatch_semaphore_t frames_free_ = nil;
  // Guest memory read in place (MapSharedMemory).
  id<MTLBuffer> shared_buf_ = nil;
  const uint8_t* shared_lo_ = nullptr;
  const uint8_t* shared_hi_ = nullptr;
  Arena arenas_[kSlots];
  int slot_ = 0;
  // Arena chunks made ahead (MakeSpareChunks), shared with the blocks
  // that make them.
  struct SparePool {
    std::mutex mu;
    std::vector<id<MTLBuffer>> ready;
    int pending = 0;
  };
  std::shared_ptr<SparePool> spares_ = std::make_shared<SparePool>();
  Bound bound_;
  struct LastUniform {
    __unsafe_unretained id<MTLBuffer> buf;  // owned by the arena
    size_t off;
    uint8_t* cpu;
    uint32_t len;
    const void* src;   // where it was copied from, and that source's
    uint64_t version;  // change counter (0: none)
    // Without a counter: the data, if it fits (VsState, PsState), and only
    // while it is `buf`'s (shadow_valid).
    alignas(16) uint8_t shadow[sizeof(VsState)];
    bool shadow_valid;
  };
  LastUniform last_uniform_[6] = {};
  // Per sampler register (pixel 0-15, vertex 0-3): the sampler last made
  // from its D3D9 states, the row and the row's change counter.
  struct SlotSampler {
    const uint32_t* row;
    uint64_t version;
    __unsafe_unretained id<MTLSamplerState> smp;  // samplers_ holds it
  };
  SlotSampler slot_sampler_[20] = {};
  // Int and bool register blocks of the last draws (vertex, pixel), the
  // front end's change counter and source they were built from, and a
  // serial of each rebuild (Uniform's version for them).
  std::array<int32_t, 80> ib_[2] = {};
  uint64_t ib_version_[2] = {}, ib_serial_[2] = {1, 1};
  const void* ib_src_[2] = {};

  // The pass and its pending whole-target clear (flags as Clear's).
  Pass pass_;
  uint32_t clear_flags_ = 0;
  float clear_rgba_[4] = {0, 0, 0, 1};
  MTLClearColor clear_color_ = {0, 0, 0, 1};
  double clear_depth_ = 1;
  uint8_t clear_stencil_ = 0;

  id<MTLTexture> backbuffer_ = nil;  // the game's back buffer
  id<MTLTexture> front_ = nil;       // the last presented frame
  int bb_w_ = 0, bb_h_ = 0;
  id<MTLTexture> dummy_2d_ = nil, dummy_cube_ = nil, dummy_3d_ = nil;
  id<MTLBuffer> readback_ = nil;
  std::unordered_map<uint32_t, Read> reads_;  // StartReadBack tickets
  uint32_t next_read_ = 1;
  int min_aniso_ = 0;
  uint32_t warned_ = 0;

  std::unordered_map<TextureId, Tex> textures_;
  // The same by id, for lookups on every draw: pointers into textures_,
  // whose elements do not move. Sized by the highest live id, and the ids of
  // destroyed textures are handed out again (NewTexId).
  std::vector<Tex*> tex_at_;
  std::vector<TextureId> free_tex_ids_;
  std::unordered_map<DepthId, Depth> depths_;
  std::unordered_map<ShaderId, Shader> shaders_;
  uint32_t next_id_ = 1;
  TextureId current_color_ = 0;
  int current_level_ = 0;
  std::array<ColorTarget, 3> current_extra_{};
  DepthId current_depth_ = 0;
  std::unordered_map<PipeKey, id<MTLRenderPipelineState>, BytesHash<PipeKey>> pipelines_;
  id<MTLRenderPipelineState> last_pipe_ = nil;  // Pipeline(last_pipe_key_)
  PipeKey last_pipe_key_{};
  id<MTLDepthStencilState> last_depth_ = nil;
  DepthKey last_depth_key_{};
  std::unordered_map<DepthKey, id<MTLDepthStencilState>, BytesHash<DepthKey>> depth_states_;
  std::unordered_map<SamplerKey, id<MTLSamplerState>, BytesHash<SamplerKey>> samplers_;

  // D3D state.
  std::array<uint32_t, 256> rs_{};
  Viewport viewport_{0, 0, 1, 1, 0, 1};
  Rect scissor_{};
};

}  // namespace

std::unique_ptr<Device> CreateMetalDevice(SDL_Window* window) {
  auto d = std::make_unique<MetalDevice>(window);
  if (!d->Init()) return nullptr;
  return d;
}

}  // namespace recomp::gfx
