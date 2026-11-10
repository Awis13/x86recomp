#pragma once
#import <Metal/Metal.h>
namespace recomp::gfx {
inline MTLRenderPipelineDescriptor* MetalDepthCopyDescriptor(
    MTLPixelFormat format, id<MTLFunction> vertex, id<MTLFunction> fragment) {
  auto descriptor = [MTLRenderPipelineDescriptor new];
  descriptor.vertexFunction = vertex;
  descriptor.fragmentFunction = fragment;
  descriptor.depthAttachmentPixelFormat = format;
  if (format == MTLPixelFormatDepth32Float_Stencil8)
    descriptor.stencilAttachmentPixelFormat = format;
  return descriptor;
}

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

// Sharpening after TAA, contrast-adaptive (after AMD's CAS): the cross of
// neighbours pulled away from the centre, less where the local contrast
// is already high (no halos on edges). sharpness 0..1.
kernel void taa_sharpen_cs(texture2d<half, access::read> src [[texture(0)]],
                           texture2d<half, access::write> out [[texture(1)]],
                           constant float& sharpness [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const uint2 size = uint2(src.get_width(), src.get_height());
  if (gid.x >= size.x || gid.y >= size.y) return;
  const int2 last = int2(size) - 1, g = int2(gid);
  const half3 e = src.read(gid).rgb;
  const half3 b = src.read(uint2(clamp(g + int2(0, -1), 0, last))).rgb;
  const half3 d = src.read(uint2(clamp(g + int2(-1, 0), 0, last))).rgb;
  const half3 f = src.read(uint2(clamp(g + int2(1, 0), 0, last))).rgb;
  const half3 h = src.read(uint2(clamp(g + int2(0, 1), 0, last))).rgb;
  const half3 mn = min(e, min(min(b, d), min(f, h))), mx = max(e, max(max(b, d), max(f, h)));
  const half3 amp = sqrt(saturate(min(mn, 2.0h - mx) / max(mx, 1.0e-4h)));
  const half3 w = amp * half(-1.0f / mix(8.0f, 5.0f, sharpness));
  out.write(half4(saturate((e + (b + d + f + h) * w) / (1.0h + 4.0h * w)), 1.0h), gid);
}
)";
}
