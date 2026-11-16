// Ray-traced volumetric fog (rt_fog.h).

#import "runtime/gfx/rt_fog.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "runtime/gfx/gpu_timer.h"

namespace recomp::gfx {

const char* const kRtFogMsl = R"(
struct RtFogParams {
  uint4 dims;      // W, H, D, frame
  float4 range;    // near_z, far_z, density, height
  float4 look;     // anisotropy, sun, sky, jitter
  float4 sun;
  float4x4 prev_vp;  // world to last frame's clip: prev_vp * float4(world, 1)
  float4 prev;     // w: 1 with a last frame
  float4 blend;    // x: this frame's weight
};
// The view depth of slice coordinate s (0..D: the slices' near sides).
float rt_fog_z(constant RtFogParams& fp, float s) {
  return fp.range.x * pow(fp.range.y / fp.range.x, s / float(fp.dims.z));
}
// The slice coordinate of view depth z.
float rt_fog_s(constant RtFogParams& fp, float z) {
  return log(max(z, 1e-3) / fp.range.x) / log(fp.range.y / fp.range.x) * float(fp.dims.z);
}
// A point on the camera's ray through screen position c (pixels, at the
// depth's size), relative to the eye.
float3 rt_fog_ray(constant RtShadowParams& p, float2 c) {
  const float x = 2.0 * (c.x - p.jitter.x) / float(p.size.x) - 1.0;
  const float y = 1.0 - 2.0 * (c.y - p.jitter.y) / float(p.size.y);
  const float4 h = x * p.rx + y * p.ry + 0.5 * p.rv + p.rc;
  return h.xyz / h.w;
}
kernel void rt_fog_trace_cs(instance_acceleration_structure scene [[buffer(0)]], constant RtShadowParams& p [[buffer(1)]],
                            const device RtMaterial* materials [[buffer(2)]], constant RtFogParams& fp [[buffer(3)]],
                            RtShadowTable table [[buffer(4)]], const device RtInfo* info [[buffer(5)]],
                            const device float4* sky [[buffer(6)]], texture3d<half> history [[texture(0)]],
                            texture3d<half, access::write> out [[texture(1)]], uint3 tid [[thread_position_in_grid]]) {
  if (any(tid >= fp.dims.xyz)) return;
  const float2 c = (float2(tid.xy) + 0.5) / float2(fp.dims.xy) * float2(p.size);
  const float3 dir = normalize(rt_fog_ray(p, c));
  const float3 fwd = normalize(rt_fog_ray(p, float2(p.size) * 0.5));
  const float z = rt_fog_z(fp, float(tid.z) + fp.look.w);
  const float3 world = p.eye.xyz + dir * (z / max(dot(dir, fwd), 0.05));
  const float sigma = fp.range.z * exp(-max(world.z, 0.0) / fp.range.w);
  bool again = false;
  const float vis =
      rt_shadow_blocked(scene, table, world, p.light.xyz, p.light.w, 0.0, world, dir, materials, info, again) ? 0.0
                                                                                                             : 1.0;
  const float g = fp.look.x, ct = dot(dir, p.light.xyz);
  const float phase = (1.0 - g * g) / (12.566371 * pow(max(1.0 + g * g - 2.0 * g * ct, 1e-4), 1.5));
  const float3 sky_c = sky[0].w > 0.0 ? sky[0].rgb : float3(0.0);
  const float3 lin = vis * phase * 12.566371 * fp.look.y * fp.sun.rgb + fp.look.z * sky_c;
  float4 cell = float4(sigma * lin, sigma);
  // Last frame's at the same world point, blended (where it was in view).
  if (fp.prev.w > 0.5) {
    const float4 h = fp.prev_vp * float4(world, 1.0);
    if (h.w > 0.0) {
      const float2 uv = float2(h.x / h.w * 0.5 + 0.5, 0.5 - h.y / h.w * 0.5);
      const float w = rt_fog_s(fp, h.w) / float(fp.dims.z);
      if (all(uv > 0.0) && all(uv < 1.0) && w > 0.0 && w < 1.0) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        cell = mix(float4(history.sample(s, float3(uv, w))), cell, fp.blend.x);
      }
    }
  }
  out.write(half4(cell), tid);
}
kernel void rt_fog_integrate_cs(constant RtFogParams& fp [[buffer(3)]], texture3d<half, access::read> cells [[texture(0)]],
                                texture3d<half, access::write> out [[texture(1)]], uint2 tid [[thread_position_in_grid]]) {
  if (any(tid >= fp.dims.xy)) return;
  float3 acc = float3(0.0);
  float tr = 1.0;
  for (uint s = 0u; s < fp.dims.z; s++) {
    const float4 v = float4(cells.read(uint3(tid, s)));
    const float dz = rt_fog_z(fp, float(s) + 1.0) - rt_fog_z(fp, float(s));
    const float t = exp(-v.a * dz);
    // (Hillaire's: the light scattered over the slice, as it fades.)
    acc += tr * (v.a > 1e-7 ? (v.rgb - v.rgb * t) / v.a : v.rgb * dz);
    tr *= t;
    out.write(half4(half3(acc), half(tr)), uint3(tid, s));
  }
}
)";

bool ParseRtFog(const char* text, RtFogLook* out) {
  RtFogLook l;
  std::string s(text);
  size_t at = 0;
  while (at < s.size()) {
    const size_t end = std::min(s.find(';', at), s.size());
    const std::string part = s.substr(at, end - at);
    at = end + 1;
    float v;
    if (std::sscanf(part.c_str(), "density=%f", &v) == 1) l.density = std::clamp(v, 0.0f, 1.0f);
    else if (std::sscanf(part.c_str(), "height=%f", &v) == 1) l.height = std::max(v, 1.0f);
    else if (std::sscanf(part.c_str(), "anisotropy=%f", &v) == 1) l.anisotropy = std::clamp(v, -0.95f, 0.95f);
    else if (std::sscanf(part.c_str(), "sun=%f", &v) == 1) l.sun = std::max(v, 0.0f);
    else if (std::sscanf(part.c_str(), "sky=%f", &v) == 1) l.sky = std::max(v, 0.0f);
    else if (std::sscanf(part.c_str(), "far=%f", &v) == 1) l.far_z = std::max(v, 10.0f);
    else return false;
  }
  *out = l;
  return true;
}

std::unique_ptr<RtFog> RtFog::Make(id<MTLDevice> dev) {
  if (!RtSupported(dev)) return nullptr;
  auto m = std::make_unique<RtFog>();
  m->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(kRtMsl) + kRtShadowMsl + kRtFogMsl).c_str())
                                         options:nil
                                           error:&err];
  if (!lib) {
    std::fprintf(stderr, "gfx: ray-traced fog: %s\n", err.localizedDescription.UTF8String);
    return nullptr;
  }
  id<MTLFunction> alpha = [lib newFunctionWithName:@"rt_shadow_alpha"];
  id<MTLFunction> trace = [lib newFunctionWithName:@"rt_fog_trace_cs"];
  id<MTLFunction> integ = [lib newFunctionWithName:@"rt_fog_integrate_cs"];
  if (!alpha || !trace || !integ) return nullptr;
  MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = trace;
  MTLLinkedFunctions* linked = [MTLLinkedFunctions linkedFunctions];
  linked.functions = @[ alpha ];
  pd.linkedFunctions = linked;
  m->trace_ = [dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&err];
  if (!m->trace_) {
    std::fprintf(stderr, "gfx: ray-traced fog's trace: %s\n", err.localizedDescription.UTF8String);
    return nullptr;
  }
  MTLIntersectionFunctionTableDescriptor* td = [MTLIntersectionFunctionTableDescriptor new];
  td.functionCount = 1;
  m->table_ = [m->trace_ newIntersectionFunctionTableWithDescriptor:td];
  id<MTLFunctionHandle> h = [m->trace_ functionHandleWithFunction:alpha];
  if (!m->table_ || !h) return nullptr;
  [m->table_ setFunction:h atIndex:0];
  m->integrate_ = [dev newComputePipelineStateWithFunction:integ error:&err];
  if (!m->integrate_) return nullptr;
  auto vol = [&]() {
    MTLTextureDescriptor* d = [MTLTextureDescriptor new];
    d.textureType = MTLTextureType3D;
    d.pixelFormat = MTLPixelFormatRGBA16Float;
    d.width = kRtFogW, d.height = kRtFogH, d.depth = kRtFogD;
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [dev newTextureWithDescriptor:d];
  };
  m->cells_[0] = vol(), m->cells_[1] = vol(), m->integrated_ = vol();
  m->no_sky_ = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  if (!m->cells_[0] || !m->cells_[1] || !m->integrated_ || !m->no_sky_) return nullptr;
  std::memset(m->no_sky_.contents, 0, 16);
  return m;
}

void RtFog::Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, const RtShadowParams& p, const float sun[3],
                   id<MTLBuffer> sky, const double* prev_vp, uint32_t frame, const RtFogLook& look) const {
  if (!f || !f->tlas) return;
  RtFogParams fp{};
  fp.dims[0] = kRtFogW, fp.dims[1] = kRtFogH, fp.dims[2] = kRtFogD, fp.dims[3] = frame;
  fp.range[0] = look.near_z, fp.range[1] = look.far_z, fp.range[2] = look.density, fp.range[3] = look.height;
  // (The slice's jitter: a Halton (2) sequence over 8 frames.)
  const uint32_t j = frame % 8u + 1u;
  float hal = 0.0f, fr = 0.5f;
  for (uint32_t i = j; i; i >>= 1, fr *= 0.5f) hal += fr * float(i & 1u);
  fp.look[0] = look.anisotropy, fp.look[1] = look.sun, fp.look[2] = look.sky, fp.look[3] = hal;
  for (int k = 0; k < 3; k++) fp.sun[k] = std::max(sun[k], 0.0f);
  if (prev_vp && have_) {
    for (int k = 0; k < 16; k++) fp.prev_vp[k] = float(prev_vp[k]);
    fp.prev[3] = 1.0f;
  }
  fp.blend[0] = 0.1f;
  RtShadowParams q = p;
  q.diag = 0;
  const int last = cur_, now = cur_ ^ 1;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt fog trace");
  [e setComputePipelineState:trace_];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBytes:&fp length:sizeof fp atIndex:3];
  [e setIntersectionFunctionTable:table_ atBufferIndex:4];
  [e setBuffer:f->info offset:f->info_offset atIndex:5];
  [e setBuffer:sky ? sky : no_sky_ offset:0 atIndex:6];
  RtUseResources(e, f);
  [e setTexture:cells_[last] atIndex:0];
  [e setTexture:cells_[now] atIndex:1];
  [e dispatchThreads:MTLSizeMake(kRtFogW, kRtFogH, kRtFogD) threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
  [e endEncoding];
  e = TimedCompute(cmd, "rt fog integrate");
  [e setComputePipelineState:integrate_];
  [e setBytes:&fp length:sizeof fp atIndex:3];
  [e setTexture:cells_[now] atIndex:0];
  [e setTexture:integrated_ atIndex:1];
  [e dispatchThreads:MTLSizeMake(kRtFogW, kRtFogH, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
  [e endEncoding];
  cur_ = now;
  have_ = true;
}

}  // namespace recomp::gfx
