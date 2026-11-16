// MetalFX's denoiser over the lit scene (rt_denoise.h).

#import "runtime/gfx/rt_denoise.h"

#import <MetalFX/MetalFX.h>
#include <simd/simd.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/rt_reflection.h"

namespace recomp::gfx {

namespace {

// The guides' and masks' kernels: a thread a pixel of the lit scene.
const char* const kRtDenoiseMsl = R"(
#include <metal_stdlib>
using namespace metal;
struct RtDenoiseGuideParams {
  float4x4 inv_view_proj;  // as uploaded (rows of the row-vector matrix): m * v is v's product
  uint2 size;
  float2 jitter;
  float exponent_scale, spec_scale, fresnel_min, fresnel_power;
  uint spec_channel, gloss_channel, pad0, pad1;
};
struct RtDenoiseMaskParams {
  uint2 size;
  float sky_distance;
  uint hit;
};
// Texel of t at pixel gid of a size-sized grid (nearest, scaled).
template <typename T>
uint2 rt_dn_at(T t, uint2 gid, uint2 size) {
  const uint2 ts = uint2(t.get_width(), t.get_height());
  return min(uint2((float2(gid) + 0.5) * float2(ts) / float2(size)), ts - 1u);
}
float3 rt_dn_world(constant RtDenoiseGuideParams& p, float x, float y, float z) {
  const float4 h = p.inv_view_proj * float4(x, y, z, 1.0);
  return h.xyz / h.w;
}
kernel void rt_denoise_guides_cs(constant RtDenoiseGuideParams& p [[buffer(0)]],
                                 texture2d<float, access::read> albedo [[texture(0)]],
                                 texture2d<float, access::read> normal [[texture(1)]],
                                 texture2d<float, access::read> surface [[texture(2)]],
                                 depth2d<float, access::read> depth [[texture(3)]],
                                 texture2d<half, access::write> out_albedo [[texture(4)]],
                                 texture2d<half, access::write> out_normal [[texture(5)]],
                                 texture2d<half, access::write> out_roughness [[texture(6)]],
                                 texture2d<half, access::write> out_specular [[texture(7)]],
                                 texture2d<float, access::write> depth_lit [[texture(8)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.size.x || gid.y >= p.size.y) return;
  out_albedo.write(half4(half3(albedo.read(rt_dn_at(albedo, gid, p.size)).rgb), 1.0h), gid);
  float3 n = normal.read(rt_dn_at(normal, gid, p.size)).xyz * 2.0 - 1.0;
  n = length_squared(n) > 0.0 ? normalize(n) : float3(0.0, 0.0, 1.0);
  out_normal.write(half4(half3(n), 0.0h), gid);
  const float4 s = surface.read(rt_dn_at(surface, gid, p.size));
  const float spec = s[p.spec_channel & 3u], gloss = s[p.gloss_channel & 3u];
  const float e = p.exponent_scale * gloss * gloss;
  out_roughness.write(half4(half(sqrt(2.0 / (e + 2.0)))), gid);
  // The view direction at the pixel (D3D9's pixel centre, less the jitter).
  const float x = 2.0 * (float(gid.x) - p.jitter.x) / float(p.size.x) - 1.0;
  const float y = 1.0 - 2.0 * (float(gid.y) - p.jitter.y) / float(p.size.y);
  const float3 i = normalize(rt_dn_world(p, x, y, 1.0) - rt_dn_world(p, x, y, 0.0));
  const float c = min(abs(dot(i, n)), 1.0);
  const float f = p.fresnel_min + (1.0 - p.fresnel_min) * pow(1.0 - c, p.fresnel_power);
  out_specular.write(half4(half3(saturate(p.spec_scale * spec * f)), 1.0h), gid);
  depth_lit.write(float4(depth.read(rt_dn_at(depth, gid, p.size))), gid);
}
kernel void rt_denoise_masks_cs(constant RtDenoiseMaskParams& p [[buffer(0)]],
                                device atomic_uint* counts [[buffer(1)]],
                                depth2d<float, access::read> depth [[texture(0)]],
                                texture2d<float, access::read> depth_lit [[texture(1)]],
                                texture2d<half, access::read> hit [[texture(2)]],
                                texture2d<half, access::write> reactive [[texture(3)]],
                                texture2d<half, access::write> hit_out [[texture(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
  const bool in = gid.x < p.size.x && gid.y < p.size.y;
  bool moved = false;
  if (in) {
    moved = depth.read(rt_dn_at(depth, gid, p.size)) != depth_lit.read(gid).r;
    reactive.write(half4(moved ? 1.0h : 0.0h), gid);
    float d = p.hit != 0u ? float(hit.read(rt_dn_at(hit, gid, p.size)).r) : 0.0;
    hit_out.write(half4(half(d > 0.0 ? d : p.sky_distance)), gid);
  }
  const uint n = simd_sum(moved ? 1u : 0u);
  if (simd_is_first() && n) atomic_fetch_add_explicit(counts, n, memory_order_relaxed);
}
)";

struct RtDenoiseMaskParams {
  uint32_t size[2];
  float sky_distance;
  uint32_t hit;
};
static_assert(sizeof(RtDenoiseMaskParams) == 16, "kRtDenoiseMsl's RtDenoiseMaskParams");

id<MTLTexture> PrivateTexture(id<MTLDevice> dev, MTLPixelFormat pf, uint32_t w, uint32_t h, MTLTextureUsage usage) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = usage;
  return [dev newTextureWithDescriptor:td];
}

// Row-vector matrix m (row-major) as simd's column-vector one: its
// transpose, which is m's memory read column by column.
simd_float4x4 Simd(const double* m) {
  simd_float4x4 r;
  for (int i = 0; i < 4; i++) r.columns[i] = simd_make_float4(float(m[i * 4]), float(m[i * 4 + 1]),
                                                              float(m[i * 4 + 2]), float(m[i * 4 + 3]));
  return r;
}

}  // namespace

bool RtSplitCamera(const double vp[16], double view[16], double proj[16]) {
  auto at = [&](int r, int c) { return vp[r * 4 + c]; };
  auto dot3 = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto len3 = [&](const double* a) { return std::sqrt(dot3(a, a)); };
  // Column 3 (w) is k times the view's forward axis, plus k tz.
  double f[3] = {at(0, 3), at(1, 3), at(2, 3)};
  const double k = len3(f);
  if (!(k > 1e-12) || !std::isfinite(k)) return false;
  for (double& v : f) v /= k;
  const double tz = at(3, 3) / k;
  // Column 0: sx right + cx forward.
  double r[3], c0[3] = {at(0, 0), at(1, 0), at(2, 0)};
  const double cx = dot3(c0, f);
  for (int i = 0; i < 3; i++) r[i] = c0[i] - cx * f[i];
  const double sx = len3(r);
  if (!(sx > 1e-12)) return false;
  for (double& v : r) v /= sx;
  // Column 1: shear right + sy up + cy forward.
  double u[3], c1[3] = {at(0, 1), at(1, 1), at(2, 1)};
  const double s01 = dot3(c1, r), cy = dot3(c1, f);
  for (int i = 0; i < 3; i++) u[i] = c1[i] - s01 * r[i] - cy * f[i];
  const double sy = len3(u);
  if (!(sy > 1e-12)) return false;
  for (double& v : u) v /= sy;
  const double tx = (at(3, 0) - cx * tz) / sx;
  const double ty = (at(3, 1) - s01 * tx - cy * tz) / sy;
  // Column 2: its parts along the axes, and what is left of row 3.
  const double c2[3] = {at(0, 2), at(1, 2), at(2, 2)};
  const double p02 = dot3(c2, r), p12 = dot3(c2, u), p22 = dot3(c2, f);
  const double p32 = at(3, 2) - (p02 * tx + p12 * ty + p22 * tz);
  const double v[16] = {r[0], u[0], f[0], 0, r[1], u[1], f[1], 0, r[2], u[2], f[2], 0, tx, ty, tz, 1};
  const double p[16] = {sx, s01, p02, 0, 0, sy, p12, 0, cx, cy, p22, k, 0, 0, p32, 0};
  for (int i = 0; i < 16; i++) {
    if (!std::isfinite(v[i]) || !std::isfinite(p[i])) return false;
    view[i] = v[i];
    proj[i] = p[i];
  }
  return true;
}

bool RtDenoiseSupported(id<MTLDevice> dev, std::string* why, float* min_scale, float* max_scale) {
  auto no = [&](const char* w) {
    if (why) *why = w;
    return false;
  };
  if (!dev) return no("no GPU");
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
    if (![MTLFXTemporalDenoisedScalerDescriptor supportsDevice:dev]) return no("the GPU does not have it");
    const float lo = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMinScaleForDevice:dev];
    const float hi = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMaxScaleForDevice:dev];
    if (min_scale) *min_scale = lo;
    if (max_scale) *max_scale = hi;
    if (!(lo <= 1.0f && hi >= 1.0f)) {
      if (why) {
        char b[96];
        std::snprintf(b, sizeof b, "it takes input scales %.3g to %.3g, not 1", lo, hi);
        *why = b;
      }
      return false;
    }
    return true;
  }
  return no("it needs macOS 26 or iOS 26");
}

std::unique_ptr<RtDenoiser> RtDenoiser::Make(id<MTLDevice> dev, std::string* why) {
  if (!RtDenoiseSupported(dev, why)) return nullptr;
  auto m = std::make_unique<RtDenoiser>();
  m->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(kRtDenoiseMsl) options:nil error:&err];
  if (lib) {
    m->guide_pipe_ = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rt_denoise_guides_cs"]
                                                        error:&err];
    if (m->guide_pipe_)
      m->mask_pipe_ = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rt_denoise_masks_cs"]
                                                         error:&err];
  }
  if (!m->guide_pipe_ || !m->mask_pipe_) {
    if (why) *why = std::string("its kernels: ") + err.localizedDescription.UTF8String;
    return nullptr;
  }
  return m;
}

bool RtDenoiser::Ensure(uint32_t w, uint32_t h, MTLPixelFormat color, bool hit_distance, std::vector<id>* retired) {
  if (scaler_ && w == set_.w && h == set_.h && color == color_ && hit_distance == set_.hit_on) return true;
  if (retired)
    for (id o : {scaler_, (id)motion_, (id)depth_, (id)albedo_, (id)normal_, (id)roughness_, (id)specular_, (id)hit_,
                 (id)reactive_, (id)depth_lit_})
      if (o) retired->push_back(o);
  scaler_ = nil;
  motion_ = depth_ = albedo_ = normal_ = roughness_ = specular_ = hit_ = reactive_ = depth_lit_ = nil;
  set_ = Set{};
  if (!w || !h) return false;
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
    MTLFXTemporalDenoisedScalerDescriptor* sd = [MTLFXTemporalDenoisedScalerDescriptor new];
    sd.colorTextureFormat = color;
    sd.depthTextureFormat = MTLPixelFormatDepth32Float;
    sd.motionTextureFormat = MTLPixelFormatRG16Float;
    sd.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    sd.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    sd.normalTextureFormat = MTLPixelFormatRGBA16Float;
    sd.roughnessTextureFormat = MTLPixelFormatR16Float;
    sd.specularHitDistanceTextureFormat = MTLPixelFormatR16Float;
    sd.specularHitDistanceTextureEnabled = hit_distance;
    sd.reactiveMaskTextureEnabled = YES;
    sd.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
    sd.denoiseStrengthMaskTextureEnabled = NO;
    sd.transparencyOverlayTextureEnabled = NO;
    sd.autoExposureEnabled = YES;
    sd.outputTextureFormat = color;
    sd.inputWidth = sd.outputWidth = w;
    sd.inputHeight = sd.outputHeight = h;
    id<MTLFXTemporalDenoisedScaler> sc = [sd newTemporalDenoisedScalerWithDevice:dev_];
    if (!sc) {
      std::fprintf(stderr, "gfx: MetalFX's denoiser for %ux%u (pixel format %lu) could not be made\n", w, h,
                   (unsigned long)color);
      return false;
    }
    MTLTextureUsage reactive_usage;
    if (@available(macOS 27.0, iOS 27.0, tvOS 27.0, *))
      reactive_usage = sc.reactiveMaskTextureUsage;
    else
      reactive_usage = sc.reactiveTextureUsage;
    const MTLTextureUsage rw = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    motion_ = PrivateTexture(dev_, MTLPixelFormatRG16Float, w, h,
                             MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | sc.motionTextureUsage);
    depth_ = PrivateTexture(dev_, MTLPixelFormatDepth32Float, w, h,
                            MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | sc.depthTextureUsage);
    albedo_ = PrivateTexture(dev_, MTLPixelFormatRGBA16Float, w, h, rw | sc.diffuseAlbedoTextureUsage);
    normal_ = PrivateTexture(dev_, MTLPixelFormatRGBA16Float, w, h, rw | sc.normalTextureUsage);
    roughness_ = PrivateTexture(dev_, MTLPixelFormatR16Float, w, h, rw | sc.roughnessTextureUsage);
    specular_ = PrivateTexture(dev_, MTLPixelFormatRGBA16Float, w, h, rw | sc.specularAlbedoTextureUsage);
    hit_ = PrivateTexture(dev_, MTLPixelFormatR16Float, w, h, rw | sc.specularHitDistanceTextureUsage);
    reactive_ = PrivateTexture(dev_, MTLPixelFormatR8Unorm, w, h, rw | reactive_usage);
    depth_lit_ = PrivateTexture(dev_, MTLPixelFormatR32Float, w, h, rw);
    if (!motion_ || !depth_ || !albedo_ || !normal_ || !roughness_ || !specular_ || !hit_ || !reactive_ ||
        !depth_lit_)
      return false;
    color_usage_ = sc.colorTextureUsage;
    output_usage_ = sc.outputTextureUsage;
    scaler_ = sc;
    color_ = color;
    set_.scaler = scaler_;
    set_.motion = motion_;
    set_.depth = depth_;
    set_.albedo = albedo_;
    set_.normal = normal_;
    set_.roughness = roughness_;
    set_.specular = specular_;
    set_.hit = hit_;
    set_.reactive = reactive_;
    set_.depth_lit = depth_lit_;
    set_.w = w;
    set_.h = h;
    set_.hit_on = hit_distance;
    std::fprintf(stderr, "gfx: MetalFX's denoiser %ux%u (pixel format %lu)%s\n", w, h, (unsigned long)color,
                 hit_distance ? ", with the reflections' hit distance" : "");
    return true;
  }
  return false;
}

id<MTLTexture> RtDenoiser::NewOutput(MTLTextureUsage extra) const {
  if (!scaler_) return nil;
  return PrivateTexture(dev_, color_, set_.w, set_.h, extra | OutputUsage());
}

void RtDenoiser::EncodeGuides(id<MTLCommandBuffer> cmd, const Set& t, id<MTLTexture> albedo, id<MTLTexture> normal,
                              id<MTLTexture> surface, id<MTLTexture> depth, const RtDenoiseGuideParams& p) const {
  if (!t.scaler || !albedo || !normal || !surface || !depth) return;
  RtDenoiseGuideParams q = p;
  q.size[0] = t.w;
  q.size[1] = t.h;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "denoise guides");
  [e setComputePipelineState:guide_pipe_];
  [e setBytes:&q length:sizeof q atIndex:0];
  [e setTexture:albedo atIndex:0];
  [e setTexture:normal atIndex:1];
  [e setTexture:surface atIndex:2];
  [e setTexture:depth atIndex:3];
  [e setTexture:t.albedo atIndex:4];
  [e setTexture:t.normal atIndex:5];
  [e setTexture:t.roughness atIndex:6];
  [e setTexture:t.specular atIndex:7];
  [e setTexture:t.depth_lit atIndex:8];
  [e dispatchThreads:MTLSizeMake(t.w, t.h, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
  [e endEncoding];
}

void RtDenoiser::EncodeMasks(id<MTLCommandBuffer> cmd, const Set& t, id<MTLTexture> depth, id<MTLTexture> hit,
                             id<MTLBuffer> counts, size_t offset) const {
  if (!t.scaler || !depth || !counts) return;
  RtDenoiseMaskParams q{{t.w, t.h}, kRtReflectionSkyDistance, t.hit_on && hit ? 1u : 0u};
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "denoise mask");
  [e setComputePipelineState:mask_pipe_];
  [e setBytes:&q length:sizeof q atIndex:0];
  [e setBuffer:counts offset:offset atIndex:1];
  [e setTexture:depth atIndex:0];
  [e setTexture:t.depth_lit atIndex:1];
  [e setTexture:q.hit ? hit : t.roughness atIndex:2];  // (unread without one)
  [e setTexture:t.reactive atIndex:3];
  [e setTexture:t.hit atIndex:4];
  [e dispatchThreads:MTLSizeMake(t.w, t.h, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
  [e endEncoding];
}

void RtDenoiser::EncodeDenoise(id<MTLCommandBuffer> cmd, const Set& t, id<MTLTexture> color, id<MTLTexture> out,
                               const RtDenoiseFrame& f) {
  if (!t.scaler || !color || !out) return;
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
    id<MTLFXTemporalDenoisedScaler> sc = t.scaler;
    sc.colorTexture = color;
    sc.depthTexture = t.depth;
    sc.motionTexture = t.motion;
    sc.diffuseAlbedoTexture = t.albedo;
    sc.specularAlbedoTexture = t.specular;
    sc.normalTexture = t.normal;
    sc.roughnessTexture = t.roughness;
    sc.specularHitDistanceTexture = t.hit_on ? t.hit : nil;
    sc.reactiveMaskTexture = t.reactive;
    sc.outputTexture = out;
    sc.preExposure = 1.0f;
    sc.jitterOffsetX = f.jitter[0];
    sc.jitterOffsetY = f.jitter[1];
    sc.motionVectorScaleX = f.motion_scale[0];
    sc.motionVectorScaleY = f.motion_scale[1];
    sc.depthReversed = f.depth_reversed;
    sc.shouldResetHistory = f.reset;
    sc.worldToViewMatrix = Simd(f.view);
    sc.viewToClipMatrix = Simd(f.proj);
    [sc encodeToCommandBuffer:cmd];
  }
}

}  // namespace recomp::gfx
