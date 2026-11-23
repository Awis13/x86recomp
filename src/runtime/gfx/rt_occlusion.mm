// Ray-traced ambient occlusion (rt_occlusion.h).

#import "runtime/gfx/rt_occlusion.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include "runtime/gfx/gpu_timer.h"

namespace recomp::gfx {

const char* const kRtOcclusionMsl = R"(
struct RtOcclusionParams {
  uint2 size, blocks;
  uint frame, stride;
  float radius;
  uint normals;
};
constant uint RT_OCCLUSION_RAYS = 1u;   // (kRtOcclusionRays)
constant uint RT_OCCLUSION_BLOCK = 4u;  // (kRtOcclusionBlock)
uint rt_ao_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Pixel c's geometric normal (as the shadow masks make it), facing the
// camera; the G-buffer's where the depth gives none.
template <typename D>
float3 rt_ao_normal(D t, texture2d<float, access::read> normals, constant RtShadowParams& p,
                    constant RtOcclusionParams& op, int2 c, float v, float3 q) {
  float3 n = -q;
  float3 dx, dy;
  bool geometric = false;
  if (rt_step(t, p, c, int2(1, 0), v, q, dx) && rt_step(t, p, c, int2(0, 1), v, q, dy)) {
    const float3 m = cross(dx, dy);
    if (length_squared(m) > 0.0) n = m, geometric = true;
  }
  if (!geometric && op.normals != 0u) {
    const float3 g = normals.read(uint2(c)).xyz * 2.0 - 1.0;
    if (length_squared(g) > 0.01) n = g;
  }
  n = normalize(n);
  return dot(n, q) > 0.0 ? -n : n;
}
// The nearest blocker along unit l from o within tmax (a static instance
// from o on, a captured one beyond `gap` where the depth's point `surface`,
// seen along unit `view`, is on a person or a vehicle itself:
// rt_on_captured, tested once a ray meets one within the gap, `probed`
// keeping the answer in `on`), as rt_shadow_blocked finds one; tmax where
// none.
float rt_ao_hit(instance_acceleration_structure scene, RtShadowTable table, float3 o, float3 l, float tmax, float gap,
                float3 surface, float3 view, thread bool& probed, thread bool& on,
                const device RtMaterial* materials, const device RtInfo* info) {
  intersector<triangle_data, instancing> it;
  it.assume_geometry_type(geometry_type::triangle);
  it.accept_any_intersection(true);
  // Cut-out geometry (leaves, railings, fences) as solid: no alpha test
  // per hit, which made the rays several times dearer in the game's
  // scenes; within a few metres a canopy or a fence darkens the sky light
  // much as a solid one would.
  it.force_opacity(forced_opacity::opaque);
  RtShadowPayload pl;
  pl.materials = materials;
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  auto h = it.intersect(ray(o, l, 0.0, tmax), scene, fixed | moving, table, pl);
  if (h.type == intersection_type::none) return tmax;
  if ((info[h.user_instance_id + h.geometry_id].flags & RT_SHADOW_INFO_CAPTURED) == 0u || h.distance >= gap)
    return h.distance;
  if (!probed) probed = true, on = rt_on_captured(scene, table, surface, view, gap, pl);
  if (!on) return h.distance;
  float best = tmax;
  h = it.intersect(ray(o, l, 0.0, tmax), scene, fixed, table, pl);
  if (h.type != intersection_type::none) best = h.distance;
  if (gap < best) {
    h = it.intersect(ray(o, l, gap, best), scene, moving, table, pl);
    if (h.type != intersection_type::none) best = h.distance;
  }
  return best;
}
template <typename D>
void rt_ao_trace(D t, texture2d<float, access::read> normals, instance_acceleration_structure scene,
                 RtShadowTable table, constant RtShadowParams& p, constant RtOcclusionParams& op,
                 const device RtMaterial* materials, const device RtInfo* info, texture2d<float, access::write> out,
                 texture2d<half, access::write> out_n, device atomic_uint* counts, uint2 tid) {
  uint traced = 0u, blocked = 0u;
  if (all(tid < op.blocks)) {
    const uint2 gid = min(rt_light_sample(tid, op.frame, op.stride, RT_OCCLUSION_BLOCK), op.size - 1u);
    const float v = rt_depth(t, gid);
    float4 res = float4(-1.0, 0.0, 0.0, 0.0);  // (the sky)
    float3 nrm = float3(0.0);
    if (!rt_sky(p, v)) {
      const int2 c = int2(gid);
      const float3 q = rt_point(p, c, v);
      const float d = length(q);
      const float3 n = rt_ao_normal(t, normals, p, op, c, v, q);
      const float3 world = p.eye.xyz + q;
      const float off = d * (p.offset_pixels * p.pixel + p.offset_view) + length(world) * p.offset_world;
      const float3 o = world + n * max(off, 0.01 * op.radius);
      const float gap = min(d * p.captured_gap, op.radius);
      bool probed = false, on = false;
      // A tangent frame around n.
      const float3 a = abs(n.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
      const float3 tx = normalize(cross(a, n)), ty = cross(n, tx);
      uint h = rt_ao_hash(gid.x * 0x8da6b343u ^ gid.y * 0xd8163841u ^ op.frame * 0x9e3779b9u);
      float open = 0.0;
      for (uint k = 0u; k < RT_OCCLUSION_RAYS; k++) {
        h = rt_ao_hash(h + k * 0x632be5abu);
        const float u1 = float(h & 0xFFFFu) / 65536.0, u2 = float(h >> 16) / 65536.0;
        // Cosine-weighted: a point of the unit disc lifted onto the hemisphere.
        const float r = sqrt(u1), phi = 6.28318530718 * u2;
        const float3 l = normalize(tx * (r * cos(phi)) + ty * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
        const float hit = rt_ao_hit(scene, table, o, l, op.radius, gap, world, q / d, probed, on, materials, info);
        traced++;
        blocked += hit < op.radius ? 1u : 0u;
        open += hit / op.radius;  // (a blocker at t takes 1 - t / radius of the ray's share)
      }
      res = float4(open / float(RT_OCCLUSION_RAYS), q);
      nrm = n;
    }
    out.write(res, tid);
    out_n.write(half4(half3(nrm), 0.0h), tid);
  }
  const uint nt = simd_sum(traced), nb = simd_sum(blocked);
  if (simd_is_first() && nt) {
    atomic_fetch_add_explicit(&counts[0], nt, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], nb, memory_order_relaxed);
  }
}
// The occlusion at texel gid of the half-size output (at full-size pixel
// 2 gid): the samples (each its occlusion, its point and its normal, as
// traced) of the 3 x 3 blocks around its own on its surface, by place and
// by how far the pixel's point lies off each sample's plane for its
// distance from it; open where none is.
template <typename D>
void rt_ao_filter(D t, texture2d<half, access::read> sample_n, texture2d<float, access::read> samples,
                  constant RtShadowParams& p, constant RtOcclusionParams& op, texture2d<half, access::write> out,
                  uint2 gid) {
  if (any(gid * 2u >= op.size + 1u)) return;
  const uint2 c = min(gid * 2u, op.size - 1u);
  const float v = rt_depth(t, c);
  if (rt_sky(p, v)) {
    out.write(half4(1.0h), gid);
    return;
  }
  const float3 q = rt_point(p, int2(c), v);
  const int2 b0 = int2(c / RT_OCCLUSION_BLOCK);
  float sum = 0.0, wsum = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      const int2 b = b0 + int2(dx, dy);
      if (any(b < 0) || any(uint2(b) >= op.blocks)) continue;
      const float4 s = samples.read(uint2(b));
      if (s.x < 0.0) continue;
      const float3 n = float3(sample_n.read(uint2(b)).xyz);
      const float3 dv = s.yzw - q;
      const float len = length(dv);
      const float z = len > 0.0 ? saturate(1.0 - 5.0 * abs(dot(dv, n)) / len) : 1.0;
      const float w = (dx == 0 && dy == 0 ? 1.0 : (dx == 0 || dy == 0) ? 0.6 : 0.36) * z * z;
      sum += w * s.x;
      wsum += w;
    }
  out.write(half4(half(wsum > 0.0 ? sum / wsum : 1.0)), gid);
}
#define RT_AO_KERNELS(KIND, DEPTH)                                                                              \
  kernel void rt_ao_trace_##KIND##_cs(                                                                          \
      instance_acceleration_structure scene [[buffer(0)]], constant RtShadowParams& p [[buffer(1)]],            \
      const device RtMaterial* materials [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],               \
      constant RtOcclusionParams& op [[buffer(4)]], RtShadowTable table [[buffer(5)]],                          \
      const device RtInfo* info [[buffer(6)]], DEPTH depth [[texture(0)]],                                      \
      texture2d<float, access::read> normals [[texture(1)]], texture2d<float, access::write> out [[texture(2)]], \
      texture2d<half, access::write> out_n [[texture(3)]], uint2 tid [[thread_position_in_grid]]) {             \
    rt_ao_trace(depth, normals, scene, table, p, op, materials, info, out, out_n, counts, tid);                 \
  }                                                                                                             \
  kernel void rt_ao_filter_##KIND##_cs(                                                                         \
      constant RtShadowParams& p [[buffer(1)]], constant RtOcclusionParams& op [[buffer(4)]],                   \
      DEPTH depth [[texture(0)]], texture2d<half, access::read> sample_n [[texture(1)]],                        \
      texture2d<float, access::read> samples [[texture(2)]], texture2d<half, access::write> out [[texture(3)]],  \
      uint2 gid [[thread_position_in_grid]]) {                                                                  \
    rt_ao_filter(depth, sample_n, samples, p, op, out, gid);                                                    \
  }
typedef depth2d<float, access::read> RtAoDepthTex;
typedef texture2d<float, access::read> RtAoColorTex;
RT_AO_KERNELS(depth, RtAoDepthTex)
RT_AO_KERNELS(color, RtAoColorTex)
)";

std::unique_ptr<RtOccluder> RtOccluder::Make(id<MTLDevice> dev) {
  if (!RtSupported(dev)) return nullptr;
  auto m = std::make_unique<RtOccluder>();
  m->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(kRtMsl) + kRtShadowMsl + kRtOcclusionMsl).c_str())
                                         options:nil
                                           error:&err];
  if (!lib) {
    std::fprintf(stderr, "gfx: ray-traced occlusion: %s\n", err.localizedDescription.UTF8String);
    return nullptr;
  }
  id<MTLFunction> alpha = [lib newFunctionWithName:@"rt_shadow_alpha"];
  auto tracing = [&](NSString* name, id<MTLIntersectionFunctionTable> __strong* table) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn || !alpha) return nil;
    MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    MTLLinkedFunctions* linked = [MTLLinkedFunctions linkedFunctions];
    linked.functions = @[ alpha ];
    pd.linkedFunctions = linked;
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithDescriptor:pd
                                                                        options:MTLPipelineOptionNone
                                                                     reflection:nil
                                                                          error:&err];
    if (!ps) return nil;
    MTLIntersectionFunctionTableDescriptor* td = [MTLIntersectionFunctionTableDescriptor new];
    td.functionCount = 1;
    *table = [ps newIntersectionFunctionTableWithDescriptor:td];
    id<MTLFunctionHandle> h = [ps functionHandleWithFunction:alpha];
    if (!*table || !h) return nil;
    [*table setFunction:h atIndex:0];
    return ps;
  };
  auto pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    return fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
  };
  m->trace_depth_ = tracing(@"rt_ao_trace_depth_cs", &m->table_depth_);
  if (m->trace_depth_) m->trace_color_ = tracing(@"rt_ao_trace_color_cs", &m->table_color_);
  if (m->trace_color_) m->filter_depth_ = pipe(@"rt_ao_filter_depth_cs");
  if (m->filter_depth_) m->filter_color_ = pipe(@"rt_ao_filter_color_cs");
  if (!m->filter_color_) {
    std::fprintf(stderr, "gfx: ray-traced occlusion's passes: %s\n",
                 err ? err.localizedDescription.UTF8String : "none");
    return nullptr;
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  m->open_ = [dev newTextureWithDescriptor:td];
  MTLTextureDescriptor* wd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  wd.storageMode = MTLStorageModeShared;
  wd.usage = MTLTextureUsageShaderRead;
  m->white_ = [dev newTextureWithDescriptor:wd];
  if (!m->open_ || !m->white_) return nullptr;
  const uint8_t one = 0xFF;
  [m->open_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:1];
  const uint32_t white = 0xFFFFFFFFu;
  [m->white_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&white bytesPerRow:4];
  return m;
}

id<MTLTexture> RtOccluder::NewOut(uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:std::max((w + 1) / 2, 1u)
                                                                               height:std::max((h + 1) / 2, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

void RtOccluder::Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> normals,
                        const RtShadowParams& p, const RtOcclusionParams& op_in, id<MTLTexture> out,
                        id<MTLBuffer> counts, size_t counts_offset) const {
  if (!f || !f->tlas || !depth || !out || !counts) return;
  const uint32_t w = uint32_t(depth.width), h = uint32_t(depth.height);
  if (out.width != (w + 1) / 2 || out.height != (h + 1) / 2) return;
  if (normals && (normals.width != w || normals.height != h)) normals = nil;
  RtOcclusionParams op = op_in;
  op.size[0] = w, op.size[1] = h;
  op.blocks[0] = (w + kRtOcclusionBlock - 1) / kRtOcclusionBlock;
  op.blocks[1] = (h + kRtOcclusionBlock - 1) / kRtOcclusionBlock;
  op.normals = normals ? 1 : 0;
  if (!samples_ || samples_.width != op.blocks[0] || samples_.height != op.blocks[1]) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:op.blocks[0]
                                                                                 height:op.blocks[1]
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    samples_ = [dev_ newTextureWithDescriptor:td];
    td.pixelFormat = MTLPixelFormatRGBA16Float;
    sample_n_ = [dev_ newTextureWithDescriptor:td];
    if (!samples_ || !sample_n_) return;
  }
  RtShadowParams q = p;
  q.diag = 0;
  const bool depth_tex = q.depth_texture != 0;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt occlusion trace");
  [e setComputePipelineState:depth_tex ? trace_depth_ : trace_color_];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setBytes:&op length:sizeof op atIndex:4];
  [e setIntersectionFunctionTable:depth_tex ? table_depth_ : table_color_ atBufferIndex:5];
  [e setBuffer:f->info offset:f->info_offset atIndex:6];
  RtUseResources(e, f);
  [e setTexture:depth atIndex:0];
  [e setTexture:normals ? normals : white_ atIndex:1];
  [e setTexture:samples_ atIndex:2];
  [e setTexture:sample_n_ atIndex:3];
  [e dispatchThreads:MTLSizeMake(op.blocks[0], op.blocks[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  e = TimedCompute(cmd, "rt occlusion filter");
  [e setComputePipelineState:depth_tex ? filter_depth_ : filter_color_];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBytes:&op length:sizeof op atIndex:4];
  [e setTexture:depth atIndex:0];
  [e setTexture:sample_n_ atIndex:1];
  [e setTexture:samples_ atIndex:2];
  [e setTexture:out atIndex:3];
  [e dispatchThreads:MTLSizeMake(out.width, out.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [e endEncoding];
}

}  // namespace recomp::gfx
