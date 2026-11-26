// Ray-traced reflections in forward pixel shaders (rt_forward_reflection.h).

#import "runtime/gfx/rt_forward_reflection.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/rt_reflection.h"
#include "runtime/gfx/rt_shadow.h"

namespace recomp::gfx {

namespace {

// The start of a patched shader's text.
constexpr const char* kMarker = "// gfx: ray-traced forward reflections, version ";

// After kRtMsl and kRtReflectRayMsl: what the calls of r_rt_reflect become.
constexpr const char* kRtForwardReflectionMsl = R"(
struct RtForwardParams {
  float4 eye;
  float4 up;
  uint on, toward_light;
  float scale, wrap;
  float max_distance, pixel, captured_gap;
  uint shadow;     // the shadow calls: bit 0 traced, bit 1 the light's register points toward it, bit 2 a position
  float4 offsets;  // a ray's start off the point: per pixel footprint, of the camera's distance, of |point|; min slope
  float lamp_gap;  // a local light's ray ends this far short of it
  float reach;     // a local light's 1 / radius^2 (0: not known)
  uint light_slot, light_step;  // its layer of the frame's masks (0xFFFFFFFF: none), their step
  float2 light_scale;           // the target's pixels to the masks' depth's
  uint light_frame, light_stride;  // the masks' samples' turn (rt_light_sample)
};
// The reflection from origin.xyz along dir.xyz (hlsl::RayReflectionCall's
// arguments; light, colour and the ambient the shader's registers): rgb
// times the scale, w 1 where a ray hit an instance with an albedo.
float4 r_rt_reflect_rt(instance_acceleration_structure scene, const device RtMaterial* materials,
                       const device RtInfo* info, constant RtForwardParams& p, float4 pixel, float4 origin,
                       float4 dir, float4 light, float4 colour, float4 ambient0, float4 ambient1) {
  if (p.on == 0u || simd_is_helper_thread()) return float4(0.0);
  const float3 o = origin.xyz;
  const float dl = length(dir.xyz);
  if (!(dl > 0.0) || !all(isfinite(o)) || !isfinite(dl)) return float4(0.0);
  const float3 r = dir.xyz / dl;
  const float dist = length(o - p.eye.xyz);
  const float3 l = p.toward_light != 0u ? light.xyz : -light.xyz;
  const float ll = length(l);
  const bool lit = ll > 0.0 && isfinite(ll) && (p.shadow & 8u) == 0u;  // (kRtForwardFarDark: ambient alone)
  RtReflectLight lt;
  lt.light = float4(lit ? l / ll : p.up.xyz, INFINITY);
  lt.colour = float4(lit ? colour.rgb * colour.w : float3(0.0), 0.0);
  lt.ambient0 = ambient0;
  lt.ambient1 = ambient1;
  lt.up = p.up;
  lt.wrap = p.wrap;
  lt.pixel = p.pixel;
  lt.captured_gap = p.captured_gap;
  lt.pad = 0.0;
  lt.offsets = p.offsets;
  const float tmin = dist * (p.offsets.x * p.pixel + p.offsets.y) + length(o) * p.offsets.z;
  const float gap = max(dist * p.captured_gap, tmin);
  float3 rgb;
  float hd;
  const uint s = RtReflectRay(scene, materials, info, o, r, tmin, p.max_distance, gap, dist, lt, rgb, hd);
  if (s != 1u && s != 6u) return float4(0.0);
  return float4(clamp(rgb * p.scale, 0.0, 65504.0), 1.0);
}
// A local light's shadow at the pixel from the frame's masks (p.light_slot's
// layer): the texels of the pixel's block (one pixel of every light_step x
// light_step traced, which one turning with the frame: RtLightSample) and
// of the three beside it toward the pixel, each weighted by a tent of the
// step's width around the pixel it traced and by how near its point's
// distance from the camera is to the point's own, `d` (0: the sky, no
// weight); where they agree, theirs. (The pixel's own block's pixel is
// within the tent: one weight at least is not 0.)
float r_rt_light_mask(texture2d_array<half, access::read> masks, texture2d<float, access::read> dist,
                      constant RtForwardParams& p, float4 pixel, float d) {
  const int2 hs = int2(dist.get_width(), dist.get_height());
  const uint s = max(p.light_step, 1u);
  const float2 c = floor(pixel.xy * p.light_scale);
  const int2 own = clamp(int2(c) / int(s), int2(0), hs - 1);
  const int2 dir = select(int2(-1), int2(1), c >= float2(own * int(s)) + 0.5 * float(s - 1u));
  half m4[4];
  uint2 at[4];
  for (int k = 0; k < 4; k++) {
    at[k] = uint2(clamp(own + int2(k & 1, k >> 1) * dir, int2(0), hs - 1));
    m4[k] = masks.read(at[k], p.light_slot).r;
  }
  if (m4[0] == m4[1] && m4[0] == m4[2] && m4[0] == m4[3]) return float(m4[0]);
  float sum = 0.0, wsum = 0.0;
  for (int k = 0; k < 4; k++) {
    uint2 q = at[k] * s;
    if (p.light_stride != 0u) {
      uint h = at[k].x * 0x8da6b343u ^ at[k].y * 0xd8163841u;
      h ^= h >> 16;
      h *= 0x7feb352du;
      h ^= h >> 15;
      const uint n = s * s, cell = (h % n + (p.light_frame % n) * p.light_stride) % n;
      q += uint2(cell % s, cell / s);
    }
    const float2 t = saturate(1.0 - abs(float2(q) - c) / float(s));
    const float tent = t.x * t.y + 1e-3;
    const float dk = dist.read(at[k]).r;
    float near = 0.0;
    if (dk > 0.0) {
      const float rel = (dk - d) / d;
      near = 1.0 / (1e-4 + rel * rel * 400.0);
    }
    sum += tent * near * float(m4[k]);
    wsum += tent * near;
  }
  return wsum > 0.0 ? sum / wsum : float(m4[0]);
}
// Whether world.xyz (hlsl::RayShadowCall's point) sees the far light along
// light.xyz (the shader's register), or with bit 2 the local light at
// light.xyz (to lamp_gap short of it): 1 lit, 0 in shadow. A local light's
// from the frame's masks where it is in them (p.light_slot), none beyond
// its radius (p.reach).
float4 r_rt_shadow_rt(instance_acceleration_structure scene, const device RtMaterial* materials,
                      const device RtInfo* info, constant RtForwardParams& p,
                      texture2d_array<half, access::read> light_masks, texture2d<float, access::read> light_dist,
                      float4 pixel, float4 world, float4 light) {
  if ((p.shadow & 1u) == 0u || simd_is_helper_thread()) return float4(1.0);
  const float3 o = world.xyz;
  const bool point = (p.shadow & 4u) != 0u;
  if (!point && (p.shadow & 8u) != 0u) return float4(0.0);  // (kRtForwardFarDark)
  const float3 l = point ? light.xyz - o : (p.shadow & 2u) != 0u ? light.xyz : -light.xyz;
  const float ll = length(l);
  if (!(ll > 0.0) || !isfinite(ll) || !all(isfinite(o))) return float4(1.0);
  if (point && !(ll * ll * p.reach < 1.0)) return float4(1.0);
  const float dist = length(o - p.eye.xyz);
  if (point && p.light_slot != 0xFFFFFFFFu && dist > 0.0)
    return float4(r_rt_light_mask(light_masks, light_dist, p, pixel, dist));
  const float3 u = l / ll;
  const float off = dist * (p.offsets.x * p.pixel + p.offsets.y) + length(o) * p.offsets.z;
  const float start = off / max(p.offsets.w, 1e-3);
  const float gap = max(dist * p.captured_gap, start);
  const float tmax = point ? ll - p.lamp_gap - start : INFINITY;
  if (!(tmax > 0.0)) return float4(1.0);
  return float4(rt_shadowed(scene, o + u * start, u, tmax, gap - start, materials, info) ? 0.0 : 1.0);
}
// Whether world.xyz sees the local light at world.xyz + to.xyz (hlsl::
// RayShadowCall's texldd), to lamp_gap short of it: 1 lit, 0 in shadow; lit
// untraced where the light's weight there is not above 0.
// By day (bit 4, kRtForwardSunGate) none where the sun mask holds the pixel
// lit.
float4 r_rt_shadow_to_rt(instance_acceleration_structure scene, const device RtMaterial* materials,
                         const device RtInfo* info, constant RtForwardParams& p,
                         texture2d<half, access::read> sun_mask, float4 pixel, float4 world, float4 to, float4 weight) {
  if ((p.shadow & 1u) == 0u || (p.shadow & 32u) != 0u || simd_is_helper_thread()) return float4(1.0);
  if (!(weight.x > RT_FORWARD_MIN_WEIGHT)) return float4(1.0);
  if ((p.shadow & 16u) != 0u && pixel.x >= 0.0 && pixel.y >= 0.0) {
    const uint2 at = uint2(pixel.xy);
    if (at.x < sun_mask.get_width() && at.y < sun_mask.get_height() && sun_mask.read(at).r > 0.5h)
      return float4(1.0);
  }
  const float3 o = world.xyz;
  const float ll = length(to.xyz);
  if (!(ll > 0.0) || !isfinite(ll) || !all(isfinite(o))) return float4(1.0);
  const float dist = length(o - p.eye.xyz);
  const float3 u = to.xyz / ll;
  const float off = dist * (p.offsets.x * p.pixel + p.offsets.y) + length(o) * p.offsets.z;
  const float start = off / max(p.offsets.w, 1e-3);
  const float gap = max(dist * p.captured_gap, start);
  const float tmax = ll - p.lamp_gap - start;
  if (!(tmax > 0.0)) return float4(1.0);
  return float4(rt_shadowed(scene, o + u * start, u, tmax, gap - start, materials, info) ? 0.0 : 1.0);
}
)";

// kRtMsl's functions the patch puts out of line, as their definitions start.
constexpr const char* kOutOfLine[] = {
    "RtHit RtTraceClosest(instance_acceleration_structure as, ray r, uint mask, const device RtMaterial* table,",
    "bool RtTraceAny(instance_acceleration_structure as, ray r, uint mask, const device RtMaterial* table,",
    "bool rt_shadowed(instance_acceleration_structure scene, float3 o, float3 l, float tmax, float gap,\n"
    "                 const device RtMaterial* materials, const device RtInfo* info) {",
};

// The stubs whose calls the patch sends to the ray queries.
struct Stub {
  const char* name;  // "r_rt_reflect("
  const char* rt;    // what main0's calls become
};
constexpr Stub kStubs[] = {{"r_rt_reflect(", "r_rt_reflect_rt"}, {"r_rt_shadow(", "r_rt_shadow_rt"},
                           {"r_rt_shadow_to(", "r_rt_shadow_to_rt"}};
// What the shadow calls take beyond the four (the local lights' masks).
constexpr const char* kShadowArgs = "r_rt_lmask, r_rt_ldist, ";
// And the local lights' (the sun mask).
constexpr const char* kShadowToArgs = "r_rt_smask, ";

// The position of the parenthesis closing the one at `open`; npos if none.
size_t Closing(const std::string& s, size_t open) {
  int depth = 0;
  for (size_t i = open; i < s.size(); i++) {
    if (s[i] == '(') depth++;
    if (s[i] == ')' && --depth == 0) return i;
  }
  return std::string::npos;
}

}  // namespace

bool RtForwardParamsFor(const RtFrame& f, uint32_t w, uint32_t h, const RayForwardReflection& r,
                        RtForwardParams* out) {
  return RtForwardParamsFor(f, w, h, &r, nullptr, out);
}

bool RtForwardParamsFor(const RtFrame& f, uint32_t w, uint32_t h, const RayForwardReflection* rp,
                        const RayForwardShadow* s, RtForwardParams* out) {
  *out = RtForwardParams{};
  const RayForwardReflection r = rp ? *rp : RayForwardReflection{};
  float up[3] = {r.up[0], r.up[1], r.up[2]};
  const float ul = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
  if (!(ul > 0) || !std::isfinite(ul)) {
    up[0] = up[1] = 0;
    up[2] = 1;
  } else {
    for (float& v : up) v /= ul;
  }
  // The camera's position and the pixels' angle, as the shadow masks have
  // them (the light's direction does not matter here).
  RtShadowParams g;
  if (!RtShadowParamsFor(f, w, h, false, false, nullptr, up, &g)) return false;
  for (int k = 0; k < 3; k++) {
    out->eye[k] = g.eye[k];
    out->up[k] = up[k];
  }
  out->on = rp ? 1 : 0;
  out->toward_light = r.toward_light ? 1 : 0;
  out->shadow = s ? 1u | (s->toward_light ? 2u : 0u) | (s->point ? 4u : 0u) : 0u;
  out->lamp_gap = s ? std::max(s->lamp_gap, 0.0f) : 0.0f;  // (a point light's, and the shadows' toward a point)
  out->reach = 0.0f;
  out->light_slot = kRtForwardNoLightSlot;
  out->light_step = 1;
  out->light_scale[0] = out->light_scale[1] = 1.0f;
  out->scale = r.output_scale;
  out->wrap = std::min(r.diffuse_wrap, 0.999f);
  out->max_distance = std::clamp(r.max_distance, 0.0f, kRtReflectionLongest);
  out->pixel = g.pixel;
  out->captured_gap = kRtShadowCapturedGap;
  out->offsets[0] = kRtShadowOffsetPixels;
  out->offsets[1] = kRtShadowOffsetView;
  out->offsets[2] = kRtShadowOffsetWorld;
  out->offsets[3] = kRtShadowMinSlope;
  return true;
}

bool RtForwardReflectionPatched(const std::string& msl) { return msl.rfind(kMarker, 0) == 0; }

bool RtForwardReflectionPatch(std::string* msl) {
  const std::string& m = *msl;
  if (RtForwardReflectionPatched(m)) return true;
  // The entry point: "fragment <type> main0(" on one line.
  size_t entry = std::string::npos, open = std::string::npos;
  for (size_t at = m.find("fragment "); at != std::string::npos; at = m.find("fragment ", at + 1)) {
    if (at > 0 && m[at - 1] != '\n') continue;
    const size_t eol = m.find('\n', at), name = m.find(" main0(", at);
    if (name != std::string::npos && name < eol) {
      entry = at;
      open = name + 6;
      break;
    }
  }
  if (entry == std::string::npos) return false;
  const size_t close = Closing(m, open);
  if (close == std::string::npos) return false;
  size_t body = close + 1;
  while (body < m.size() && std::isspace(static_cast<unsigned char>(m[body]))) body++;
  if (body >= m.size() || m[body] != '{') return false;
  // A stub defined before it, called in it.
  bool called[std::size(kStubs)] = {}, any = false;
  for (size_t k = 0; k < std::size(kStubs); k++) {
    const size_t def = m.find(kStubs[k].name);
    called[k] = def != std::string::npos && def < entry && m.find(kStubs[k].name, body) != std::string::npos;
    any = any || called[k];
  }
  if (!any) return false;
  for (uint32_t b = kRtForwardSceneBuffer; b <= kRtForwardParamsBuffer; b++)
    if (m.find("[[buffer(" + std::to_string(b) + ")]]") != std::string::npos) return false;
  if (called[1])
    for (uint32_t t : {kRtForwardLightMasksTexture, kRtForwardLightDistTexture})
      if (m.find("[[texture(" + std::to_string(t) + ")]]") != std::string::npos) return false;
  if (called[2] && m.find("[[texture(" + std::to_string(kRtForwardSunMaskTexture) + ")]]") != std::string::npos)
    return false;
  const auto buf = [](uint32_t b) { return " [[buffer(" + std::to_string(b) + ")]]"; };
  bool empty = true;
  for (size_t i = open + 1; i < close && empty; i++) empty = std::isspace(static_cast<unsigned char>(m[i])) != 0;
  const std::string params = std::string(empty ? "" : ", ") + "instance_acceleration_structure r_rt_scene" +
                             buf(kRtForwardSceneBuffer) + ", const device RtMaterial* r_rt_mat" +
                             buf(kRtForwardMaterialBuffer) + ", const device RtInfo* r_rt_info" +
                             buf(kRtForwardInfoBuffer) + ", constant RtForwardParams& r_rt_params" +
                             buf(kRtForwardParamsBuffer) +
                             (called[1] ? ", texture2d_array<half, access::read> r_rt_lmask [[texture(" +
                                              std::to_string(kRtForwardLightMasksTexture) +
                                              ")]], texture2d<float, access::read> r_rt_ldist [[texture(" +
                                              std::to_string(kRtForwardLightDistTexture) + ")]]"
                                        : "") +
                             (called[2] ? ", texture2d<half, access::read> r_rt_smask [[texture(" +
                                              std::to_string(kRtForwardSunMaskTexture) + ")]]"
                                        : "");
  // kRtMsl's rays out of line here (the closest hit, any hit, and a shadow's
  // one traversal with the instances' info): every ray of the shader then
  // goes through those three copies, and the pipeline compiles three
  // traversals, not up to five a reflection and three a light (inlined, a
  // glass shader's pipeline took 2 s to compile, one with eight lights 3 s;
  // a pipeline first met in a frame is compiled in it). The traversal itself
  // stays inline in each, so that each keeps only what it returns.
  // (RECOMP_RT_FORWARD_INLINE=1: all of it inline as before, to measure what
  // the calls cost a frame.)
  static const bool inline_all = [] {
    const char* e = std::getenv("RECOMP_RT_FORWARD_INLINE");
    return e && *e == '1';
  }();
  std::string rt = kRtMsl;
  for (const char* decl : kOutOfLine) {
    const size_t at = rt.find(decl);
    if (at == std::string::npos) return false;
    if (!inline_all) rt.insert(at, "__attribute__((noinline))\n");
  }
  // The least weight of a local light at a pixel that gets a ray (below it
  // the light stays unshadowed there): RECOMP_RT_FORWARD_MIN, to find the
  // value by eye and by the pass's time.
  static const std::string min_weight = [] {
    const char* e = std::getenv("RECOMP_RT_FORWARD_MIN");
    char b[64];
    std::snprintf(b, sizeof b, "#define RT_FORWARD_MIN_WEIGHT %.6f\n", e && *e ? std::atof(e) : double(kRtForwardMinWeight));
    return std::string(b);
  }();
  std::string out = std::string(kMarker) + std::to_string(kRtForwardVersion) + "\n" + min_weight + rt + kRtReflectRayMsl +
                    kRtForwardReflectionMsl;
  out.append(m, 0, entry);
  for (size_t k = 0; k < std::size(kStubs); k++) {
    if (!called[k]) continue;
    const std::string name(kStubs[k].name, std::strlen(kStubs[k].name) - 1);
    out += "#define " + name + "(...) " + kStubs[k].rt + "(r_rt_scene, r_rt_mat, r_rt_info, r_rt_params, " +
           (k == 1 ? kShadowArgs : k == 2 ? kShadowToArgs : "") + "__VA_ARGS__)\n";
  }
  out.append(m, entry, close - entry);
  out += params;
  out.append(m, close, std::string::npos);
  *msl = std::move(out);
  return true;
}

bool RtForwardMakeNullScene(id<MTLDevice> dev, id<MTLCommandQueue> queue, RtForwardNullScene* out) {
  *out = RtForwardNullScene{};
  if (!dev || !queue) return false;
  const float tri[9] = {};
  id<MTLBuffer> vb = [dev newBufferWithBytes:tri length:sizeof tri options:MTLResourceStorageModeShared];
  MTLAccelerationStructureTriangleGeometryDescriptor* g = [MTLAccelerationStructureTriangleGeometryDescriptor
      descriptor];
  g.vertexBuffer = vb;
  g.vertexStride = 12;
  g.triangleCount = 1;
  g.opaque = YES;
  MTLPrimitiveAccelerationStructureDescriptor* bd = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  bd.geometryDescriptors = @[ g ];
  MTLAccelerationStructureInstanceDescriptor inst{};
  for (int k = 0; k < 3; k++) inst.transformationMatrix.columns[k].elements[k] = 1.0f;
  inst.mask = 0xFF;
  inst.accelerationStructureIndex = 0;
  id<MTLBuffer> ib = [dev newBufferWithBytes:&inst length:sizeof inst options:MTLResourceStorageModeShared];
  const MTLAccelerationStructureSizes bs = [dev accelerationStructureSizesWithDescriptor:bd];
  id<MTLAccelerationStructure> blas = [dev newAccelerationStructureWithSize:bs.accelerationStructureSize];
  MTLInstanceAccelerationStructureDescriptor* td = [MTLInstanceAccelerationStructureDescriptor descriptor];
  td.instancedAccelerationStructures = blas ? @[ blas ] : @[];
  td.instanceCount = 1;
  td.instanceDescriptorBuffer = ib;
  const MTLAccelerationStructureSizes ts = [dev accelerationStructureSizesWithDescriptor:td];
  id<MTLAccelerationStructure> tlas = [dev newAccelerationStructureWithSize:ts.accelerationStructureSize];
  id<MTLBuffer> scratch = [dev newBufferWithLength:std::max<NSUInteger>(std::max(bs.buildScratchBufferSize,
                                                                                 ts.buildScratchBufferSize), 16)
                                           options:MTLResourceStorageModePrivate];
  const uint8_t zero[64] = {};
  id<MTLBuffer> materials = [dev newBufferWithBytes:zero
                                              length:sizeof(RtMaterial)
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> info = [dev newBufferWithBytes:zero
                                         length:sizeof(RtInstanceInfo)
                                        options:MTLResourceStorageModeShared];
  if (!vb || !ib || !blas || !tlas || !scratch || !materials || !info) return false;
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
  [e buildAccelerationStructure:blas descriptor:bd scratchBuffer:scratch scratchBufferOffset:0];
  [e endEncoding];
  e = [cb accelerationStructureCommandEncoder];
  [e buildAccelerationStructure:tlas descriptor:td scratchBuffer:scratch scratchBufferOffset:0];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) return false;
  out->tlas = tlas;
  out->blas = blas;
  out->materials = materials;
  out->info = info;
  return true;
}

RtForwardBuild RtForwardCompile(const std::string& plain, bool forward, RtCompileMsl compile, void* ctx) {
  RtForwardBuild b;
  b.msl = plain;
  if (forward) {
    b.patched = RtForwardReflectionPatch(&b.msl);
    b.refused = !b.patched;
  }
  b.lean = MetalLeanMacros(&b.msl, false);
  NSError* err = nil;
  b.lib = compile(ctx, b.msl, b.lean, &err);
  if (!b.lib && b.patched) {
    b.patch_error = err;
    b.msl = plain;
    b.patched = false;
    b.lean = MetalLeanMacros(&b.msl, false);
    err = nil;
    b.lib = compile(ctx, b.msl, b.lean, &err);
  }
  b.error = err;
  return b;
}

RtForwardOff RtForwardWhyOff(const RtForwardWant& w) {
  if (!w.config) return RtForwardOff::NoConfig;
  if (!w.snapshot) return RtForwardOff::NoSnapshot;
  if (!w.scene_camera || w.snapshot_camera != w.scene_camera) return RtForwardOff::NoCamera;
  if (w.draw_known) return w.draw_camera == w.snapshot_camera ? RtForwardOff::None : RtForwardOff::OtherView;
  if (w.snapshot_target || w.depth_camera == w.snapshot_camera) return RtForwardOff::None;
  return RtForwardOff::OtherView;
}

const char* RtForwardOffName(RtForwardOff o) {
  switch (o) {
    case RtForwardOff::None: return "traced";
    case RtForwardOff::NoConfig: return "shader not taken";
    case RtForwardOff::NoSnapshot: return "no snapshot";
    case RtForwardOff::NoCamera: return "no camera";
    case RtForwardOff::OtherView: return "another view";
    case RtForwardOff::Count: break;
  }
  return "?";
}

}  // namespace recomp::gfx
