// Ray-traced reflections (rt_reflection.h).

#import "runtime/gfx/rt_reflection.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"

#import <objc/runtime.h>

namespace recomp::gfx {

const char* const kRtReflectRayMsl = R"(
constant uint RT_INFO_OTHER_VIEW = 1u << 16, RT_INFO_CAPTURED = 1u << 17;  // kRtInfo*
// What the hits of reflection rays are lit with.
struct RtReflectLight {
  float4 light;     // xyz toward it (unit), w a shadow ray's length
  float4 colour;    // rgb (times its w; 0: no light)
  float4 ambient0, ambient1;  // a, b: a + b sat(0.5 - 0.5 N.up)
  float4 up;
  float wrap;          // the diffuse term sat((N.L - wrap) / (1 - wrap))
  float pixel;         // the angle between neighbouring pixels' rays
  float captured_gap;  // a shadow ray meets captured instances beyond this fraction of its path
  float pad;
  float4 offsets;      // a shadow ray's start off the hit: x per pixel footprint, y of the path, z of |hit|
};
// A reflection ray from o along unit r, `dist` from the camera: the
// closest hit, alpha-cut instances tested, no face culled, of static
// instances from tmin and captured ones beyond `gap`, to tmax; what it hit
// lit into rgb (the albedo at the level of the ray's cone, times the light
// (colour, wrapped diffuse, a shadow ray from the hit) plus the ambient;
// the hit's normal turned toward the ray) and its distance (INFINITY where
// it missed). What it was: 1 a hit, 2 a miss, 3 a hit on an instance
// without an albedo, 4 a back face of a static instance of the main view
// that culls, 5 of a captured one, 6 a hit on a back face of another
// view's instance (lit); rgb 0 but for 1 and 6.
uint RtReflectRay(instance_acceleration_structure scene, const device RtMaterial* materials,
                  const device RtInfo* info, float3 o, float3 r, float tmin, float tmax, float gap, float dist,
                  thread const RtReflectLight& lt, thread float3& rgb, thread float& hit_distance) {
  rgb = float3(0.0);
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  RtHit h = RtTraceClosest(scene, ray(o, r, tmin, tmax), fixed, materials, intersection_params());
  const float end = h.hit ? h.distance : tmax;
  if (gap < end) {
    const RtHit hc = RtTraceClosest(scene, ray(o, r, gap, end), moving, materials, intersection_params());
    if (hc.hit) h = hc;
  }
  hit_distance = h.distance;
  if (!h.hit) return 2u;
  // A back face of an instance whose draw culls them (MTLAccelerationStructure
  // InstanceOptionDisableTriangleCulling not given): single-sided geometry
  // seen from behind, the shader's own reflection; not another view's
  // instance (its front faces are not known: lit from either side).
  const uint fl = info[h.instance].flags;
  const bool other_back = !h.front && (fl & RT_INFO_OTHER_VIEW) != 0u;
  if (!h.front && (fl & 1u) == 0u && !other_back) return (fl & RT_INFO_CAPTURED) != 0u ? 5u : 4u;
  const device RtMaterial& m = materials[h.instance];
  if ((m.flags & RT_MAT_ALBEDO) == 0u) return 3u;
  float3 hn = RtHitNormal(h);
  if (dot(hn, r) > 0.0) hn = -hn;  // toward the ray (a face drawn from both sides)
  const float path = dist + h.distance;
  const float3 albedo = RtHitAlbedo(m, h, r, lt.pixel * path);
  const float3 hp = o + r * h.distance, l = lt.light.xyz;
  const float diffuse = saturate((dot(hn, l) - lt.wrap) / max(1.0 - lt.wrap, 1e-4));
  float lit = 0.0;
  if (diffuse > 0.0 && any(lt.colour.rgb > 0.0)) {
    const float hoff = path * (lt.offsets.x * lt.pixel + lt.offsets.y) + length(hp) * lt.offsets.z;
    const float3 hs = dot(hn, l) < 0.0 ? -hn : hn;
    lit = rt_shadowed(scene, hp + hs * hoff, l, lt.light.w, path * lt.captured_gap, materials, info) ? 0.0 : diffuse;
  }
  const float3 amb = lt.ambient0.rgb + lt.ambient1.rgb * saturate(0.5 - 0.5 * dot(hn, lt.up.xyz));
  rgb = albedo * (lt.colour.rgb * lit + amb);
  return other_back ? 6u : 1u;
}
)";

namespace {

// The kernel (after kRtMsl, kRtShadowMsl and kRtReflectRayMsl): a thread a
// pixel of the depth buffer.
constexpr const char* kRtReflectionMsl = R"(
struct RtReflectionParams {
  RtShadowParams geo;  // the pixel's point, the light (toward it), the ray's start
  float4 colour;       // the light's colour
  float4 ambient0, ambient1;
  float4 up;
  float scale, wrap, exponent_scale, blur_k;
  float max_blur, max_distance, sky_distance, wet;
  float wet_slope, wet_gain, wet_max_exponent;
  uint wet_on;
  uint spec_channel, gloss_channel;
  float graze, gap_pixels;
  uint checker, parity;   // the checkerboard: on, the half traced (x + y + parity even)
  float fill_depth, fill_cos;
  uint diag;  // a comparison: bit 0 no shadow ray at a hit (lit), bit 1 every instance opaque
  uint frame;        // the sparse rates' place in their blocks
  float sparse_cos;  // kRtReflectionSparseCos
  float sparse_spec; // kRtReflectionSparseSpec
  float wet_distance; // kRtReflectionWetDistance
  uint sparse_all;    // every pixel one of 2 x 2 (the checkerboard on)
  uint pad2[2];
};
// Texel of t at the place of depth pixel c (the depth `size`).
float4 rt_gbuf(texture2d<float, access::read> t, uint2 c, uint2 size) {
  const uint2 ts = uint2(t.get_width(), t.get_height());
  if (ts.x == size.x && ts.y == size.y) return t.read(c);
  return t.read(min(uint2((float2(c) + 0.5) * float2(ts) / float2(size)), ts - 1u));
}
constant uint RT_RATE_WET = 16u;  // rt_reflect_test's rate: wet ground (its rays wet_distance long at most)
// Whether pixel gid is reflected (its point's stored value v, unit normal n
// and Blinn-Phong exponent e out): not the sky, a normal, shiny (a
// specular intensity above 0, or wet) and sharp enough (blur at most
// max_blur). The surface first: without wetness it decides alone, and
// most pixels stop there (neither the depth nor the normal read). `rate`:
// how sparsely it is traced (rt_reflect_traced): 1 the checkerboard (or
// every pixel without it); with it, wet ground with little specular of its
// own (below sparse_spec: the wetness its reflection) 2, one pixel of
// 2 x 2, or 4, one of 4 x 4, where it is seen from above (the cosine to
// the view at least sparse_cos: the Fresnel term low).
template <typename D>
bool rt_reflect_test(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                     texture2d<float, access::read> stencil, constant RtReflectionParams& p, uint2 gid,
                     thread float& v, thread float3& n, thread float& e, thread uint& rate) {
  constant RtShadowParams& g = p.geo;
  rate = 1u;
  const float4 st = rt_gbuf(surface, gid, g.size);
  const float spec = st[p.spec_channel & 3u], gloss = st[p.gloss_channel & 3u];
  e = p.exponent_scale * gloss * gloss;
  // (A wetness of 0 makes w 0 and leaves e: the tests below on the surface
  // alone.)
  if ((p.wet_on == 0u || p.wet == 0.0) && !(spec > 0.0 && 1.0 - saturate(e * p.blur_k) <= p.max_blur)) return false;
  v = rt_depth(t, gid);
  if (rt_sky(g, v)) return false;
  n = rt_gbuf(normal, gid, g.size).xyz * 2.0 - 1.0;
  if (length_squared(n) < 1e-8) return false;
  n = normalize(n);
  float w = 0.0;
  if (p.wet_on != 0u && rt_gbuf(stencil, gid, g.size).r < 0.5 / 255.0) {
    w = saturate(p.wet_slope * dot(n, p.up.xyz)) * p.wet;
    e = max(e, mix(e, min(p.wet_gain * e, p.wet_max_exponent), w));
  }
  if (!(spec > 0.0 || w > 0.0)) return false;
  if (1.0 - saturate(e * p.blur_k) > p.max_blur) return false;
  if (p.checker != 0u && w > 0.0 && spec < p.sparse_spec) {
    const float3 q = rt_point(g, int2(gid), v);
    rate = abs(dot(q, n)) >= p.sparse_cos * length(q) ? 4u : 2u;
  }
  if (p.checker != 0u && p.sparse_all != 0u && rate == 1u) rate = 2u;
  if (w > 0.0) rate |= RT_RATE_WET;
  return true;
}
// The place in its block a sparse rate (2 or 4) traces this frame.
uint2 rt_reflect_place(constant RtReflectionParams& p, uint rate) {
  if (rate == 2u) {
    const uint2 at[4] = {uint2(0u, 0u), uint2(1u, 1u), uint2(1u, 0u), uint2(0u, 1u)};  // (Bayer's 2 x 2)
    return at[p.frame & 3u];
  }
  const uint2 at[16] = {uint2(0u, 0u), uint2(2u, 2u), uint2(2u, 0u), uint2(0u, 2u), uint2(1u, 1u), uint2(3u, 3u),
                        uint2(3u, 1u), uint2(1u, 3u), uint2(1u, 0u), uint2(3u, 2u), uint2(3u, 0u), uint2(1u, 2u),
                        uint2(0u, 1u), uint2(2u, 3u), uint2(2u, 1u), uint2(0u, 3u)};  // (Bayer's 4 x 4)
  return at[p.frame & 15u];
}
// Whether pixel gid of `rate` (RT_RATE_WET aside) is traced this frame (else filled): the
// checkerboard's half (x + y + parity even), or the frame's place in its
// 2 x 2 or 4 x 4 block (Bayer's order, each place once in 4 or 16 frames).
bool rt_reflect_traced(constant RtReflectionParams& p, uint2 gid, uint rate) {
  rate &= 7u;
  if (p.checker == 0u) return true;
  if (rate == 1u) return ((gid.x + gid.y + p.parity) & 1u) == 0u;
  return all((gid & (rate - 1u)) == rt_reflect_place(p, rate));
}

// Tracing with the intersector object, not an intersection query: on
// Apple family 9 GPUs (M3) its traversal runs in fixed-function hardware
// and the alpha test, an intersection function, goes through the reorder
// stage that groups calls from rays of different SIMD-groups (Apple tech
// talk 111375, "Explore GPU advancements in M3 and A17 Pro": "use the
// intersector object API whenever possible ... the intersection query API
// ... increases the amount of ray trace scratch memory ... as well as
// disables the reorder stage"). The material table travels in the ray's
// payload (kept small, as the talk asks): the function table is made once
// per pipeline and never written while frames use it.
struct RtAlphaPayload {
  const device RtMaterial* materials;
};
typedef intersector<triangle_data, instancing, world_space_data> rt_isect;
typedef intersection_function_table<triangle_data, instancing, world_space_data> rt_isect_table;
// A non-opaque instance's candidate (every alpha-cut one; the TLAS's
// instances name entry 0 of the table): kept as RtAlphaPass keeps it.
// RtTexelAlpha with the sampler's own filtering where the texture wraps or
// clamps on both axes (the common case): one sample, not four reads by
// hand (the same texels and weights, the weights at the sampler's
// precision); the other addressing modes by hand, as RtTexelAlpha does.
float rt_reflect_texel_alpha(const device RtMaterial& m, float2 uv) {
  const uint mu = (m.flags >> 26) & 7u, mv = (m.flags >> 29) & 7u;
  const bool wrap = mu <= 1u && mv <= 1u, edge = mu == 3u && mv == 3u;
  if (!wrap && !edge) return RtTexelAlpha(m, uv);
  const texture2d<float> t = m.alpha_texture;
  const float lvl = float(min(uint(max(m.max_level, 0.0)), t.get_num_mip_levels() - 1u));
  const uint ch = (m.flags >> 24) & 3u;
  const bool linear = (m.flags & RT_MAT_LINEAR) != 0u;
  const float2 x = clamp(uv, -1.0e6, 1.0e6);
  constexpr sampler wl(filter::linear, mip_filter::nearest, address::repeat);
  constexpr sampler wn(filter::nearest, mip_filter::nearest, address::repeat);
  constexpr sampler cl(filter::linear, mip_filter::nearest, address::clamp_to_edge);
  constexpr sampler cn(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
  const float4 s = wrap ? (linear ? t.sample(wl, x, level(lvl)) : t.sample(wn, x, level(lvl)))
                        : (linear ? t.sample(cl, x, level(lvl)) : t.sample(cn, x, level(lvl)));
  return s[ch];
}
// RtAlphaPass with rt_reflect_texel_alpha.
bool rt_reflect_alpha_pass(const device RtMaterial& m, const device void* prim, float2 b) {
  if ((m.flags & RT_MAT_CUT) == 0u) return true;
  const float3 w = float3(1.0 - b.x - b.y, b.x, b.y);
  float2 uv;
  uint alpha;
  if ((m.flags & RT_MAT_HIT_PRIM) != 0u) {
    const device RtHitPrim& hp = *(const device RtHitPrim*)prim;
    uv = rt_hit_uv(hp, w);
    alpha = hp.alpha;
  } else {
    const device RtCutPrim& cp = *(const device RtCutPrim*)prim;
    uv = float2(cp.uv[0]) * w.x + float2(cp.uv[1]) * w.y + float2(cp.uv[2]) * w.z;
    alpha = cp.alpha;
  }
  float v = 1.0;
  if ((m.flags & RT_MAT_TEXTURE) != 0u) v = rt_reflect_texel_alpha(m, uv);
  if ((m.flags & RT_MAT_VERTEX_ALPHA) != 0u) v *= dot(unpack_unorm4x8_to_float(alpha).xyz, w);
  if ((m.flags & RT_MAT_KILL) != 0u && m.kill_scale * v + m.kill_offset < 0.0) return false;
  const uint func = (m.flags >> 8) & 15u;
  if (func == 0u || func == 8u) return true;
  const int a = int(round(clamp(m.alpha_scale * v + m.alpha_offset, 0.0, 1.0) * 255.0));
  const int ref = int((m.flags >> 16) & 255u);
  switch (func) {
    case 1u: return false;
    case 2u: return a < ref;
    case 3u: return a == ref;
    case 4u: return a <= ref;
    case 5u: return a > ref;
    case 6u: return a != ref;
    case 7u: return a >= ref;
    default: return true;
  }
}
[[intersection(triangle, triangle_data, instancing, world_space_data)]]
bool rt_reflect_alpha_if(uint user [[user_instance_id]], uint geo [[geometry_id]], float2 b [[barycentric_coord]],
                         const device void* prim [[primitive_data]], ray_data RtAlphaPayload& pl [[payload]]) {
  // (Its slot: user ID plus geometry ID, an actor's BLAS a slot a part.)
  return rt_reflect_alpha_pass(pl.materials[user + geo], prim, b);
}
// rt_trace's RtHit, by the intersector: the closest kept hit, or (`any`)
// the first.
// (diag bit 1: every instance taken as opaque, RtReflectionParams::diag.)
RtHit rt_trace_hw(instance_acceleration_structure as, rt_isect_table table, ray r, uint mask,
                  const device RtMaterial* materials, bool any, uint diag) {
  rt_isect it;
  it.assume_geometry_type(geometry_type::triangle);
  it.accept_any_intersection(any);
  if ((diag & 2u) != 0u) it.force_opacity(forced_opacity::opaque);
  RtAlphaPayload pl;
  pl.materials = materials;
  const typename rt_isect::result_type x = it.intersect(r, as, mask, table, pl);
  RtHit h;
  h.hit = x.type == intersection_type::triangle;
  h.distance = INFINITY;
  h.instance = 0xFFFFFFFFu;
  h.primitive = 0xFFFFFFFFu;
  h.bary = float2(0.0);
  h.front = false;
  h.world_to_object = float4x3(1.0);
  h.prim = nullptr;
  if (h.hit) {
    h.distance = x.distance;
    h.instance = x.user_instance_id + x.geometry_id;  // its slot (RtMaterial, RtInfo)
    h.primitive = x.primitive_id;
    h.bary = x.triangle_barycentric_coord;
    h.front = x.triangle_front_facing;
    h.world_to_object = x.world_to_object_transform;
    h.prim = x.primitive_data;
  }
  return h;
}
// Whether h is a captured instance's hit nearer than `gap` (one the
// two-phase trace does not see).
bool rt_inside_gap(RtHit h, const device RtInfo* info, float gap) {
  return h.hit && h.distance < gap && (info[h.instance].flags & RT_INFO_CAPTURED) != 0u;
}
// rt_shadowed, by the intersector: one traversal of static and captured
// instances from o; only where what it met is a captured one within the
// gap (the hit on a person or a vehicle) the two phases apart.
bool rt_shadowed_hw(instance_acceleration_structure scene, rt_isect_table table, float3 o, float3 l, float tmax,
                    float gap, const device RtMaterial* materials, const device RtInfo* info, uint diag) {
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  const RtHit h = rt_trace_hw(scene, table, ray(o, l, 0.0, tmax), fixed | moving, materials, true, diag);
  if (!h.hit) return false;
  if (!rt_inside_gap(h, info, gap)) return true;
  return rt_trace_hw(scene, table, ray(o, l, 0.0, tmax), fixed, materials, true, diag).hit ||
         rt_trace_hw(scene, table, ray(o, l, gap, tmax), moving, materials, true, diag).hit;
}
// RtReflectRay (kRtReflectRayMsl, which forward pixel shaders trace with
// queries) by the intersector, from tmin 0: the same hits, lit the same.
// One traversal of static and captured instances: its closest hit is the
// two phases' (static from 0, captured beyond the gap) unless it is a
// captured one within the gap (a ray leaving a person or a vehicle), when
// the phases are traced apart. (No captured hit within the gap: none is
// nearer than the closest, so the closest of all is the closest of both.)
uint rt_reflect_ray_hw(instance_acceleration_structure scene, rt_isect_table table,
                       const device RtMaterial* materials, const device RtInfo* info, float3 o, float3 r,
                       float tmax, float gap, float dist, thread const RtReflectLight& lt, thread float3& rgb,
                       thread float& hit_distance, thread uint& hit_flags) {
  rgb = float3(0.0);
  hit_flags = 0u;  // (the hit instance's RtInfo flags; 0 with no hit)
  const uint diag = as_type<uint>(lt.pad);  // (RtReflectionParams::diag)
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  RtHit h = rt_trace_hw(scene, table, ray(o, r, 0.0, tmax), fixed | moving, materials, false, diag);
  if (rt_inside_gap(h, info, gap)) {
    h = rt_trace_hw(scene, table, ray(o, r, 0.0, tmax), fixed, materials, false, diag);
    const float end = h.hit ? h.distance : tmax;
    if (gap < end) {
      const RtHit hc = rt_trace_hw(scene, table, ray(o, r, gap, end), moving, materials, false, diag);
      if (hc.hit) h = hc;
    }
  }
  hit_distance = h.distance;
  if (!h.hit) return 2u;
  const uint fl = info[h.instance].flags;
  hit_flags = fl;
  const bool other_back = !h.front && (fl & RT_INFO_OTHER_VIEW) != 0u;
  if (!h.front && (fl & 1u) == 0u && !other_back) return (fl & RT_INFO_CAPTURED) != 0u ? 5u : 4u;
  const device RtMaterial& m = materials[h.instance];
  if ((m.flags & RT_MAT_ALBEDO) == 0u) return 3u;
  float3 hn = RtHitNormal(h);
  if (dot(hn, r) > 0.0) hn = -hn;
  const float path = dist + h.distance;
  const float3 albedo = RtHitAlbedo(m, h, r, lt.pixel * path);
  const float3 hp = o + r * h.distance, l = lt.light.xyz;
  const float diffuse = saturate((dot(hn, l) - lt.wrap) / max(1.0 - lt.wrap, 1e-4));
  float lit = 0.0;
  if (diffuse > 0.0 && any(lt.colour.rgb > 0.0)) {
    const float hoff = path * (lt.offsets.x * lt.pixel + lt.offsets.y) + length(hp) * lt.offsets.z;
    const float3 hs = dot(hn, l) < 0.0 ? -hn : hn;
    lit = (diag & 1u) == 0u && rt_shadowed_hw(scene, table, hp + hs * hoff, l, lt.light.w,
                                              path * lt.captured_gap, materials, info, diag)
              ? 0.0
              : diffuse;
  }
  const float3 amb = lt.ambient0.rgb + lt.ambient1.rgb * saturate(0.5 - 0.5 * dot(hn, lt.up.xyz));
  rgb = albedo * (lt.colour.rgb * lit + amb);
  return other_back ? 6u : 1u;
}
// A pixel's ray: its start (world space, and relative to the camera), its
// unit direction, length and the camera's distance to the pixel's point.
struct RtReflectRayStart {
  float3 o, o_rel, r;
  float tmax, dist;
};
// Pixel gid's ray (rt_reflect_test passed); false if it has none.
template <typename D>
bool rt_reflect_start(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                      texture2d<float, access::read> stencil, constant RtReflectionParams& p, uint2 gid,
                      thread RtReflectRayStart& st) {
  constant RtShadowParams& g = p.geo;
  float v, e;
  float3 n;
  uint rate;
  if (!rt_reflect_test(t, normal, surface, stencil, p, gid, v, n, e, rate)) return false;
  const int2 c = int2(gid);
  const float3 q = rt_point(g, c, v);
  const float dist = length(q);
  if (!(dist > 0.0)) return false;
  const float3 i = q / dist;
  // (reflect() is the same for -n; the side matters to the wet term only.)
  float3 r = reflect(i, n);
  // The ray's start, as a shadow ray's: along the depth's normal, turned
  // toward the camera (the side the surface is seen from).
  float3 dx, dy, ng = -q;
  if (rt_step(t, g, c, int2(1, 0), v, q, dx) && rt_step(t, g, c, int2(0, 1), v, q, dy)) {
    const float3 m = cross(dx, dy);
    if (length_squared(m) > 0.0) ng = m;
  }
  ng = normalize(ng);
  if (dot(ng, q) > 0.0) ng = -ng;
  // A ray the normal sends below the surface (or along it): straightened
  // to just above it.
  const float rn = dot(r, ng);
  if (rn < p.graze) r = normalize(r - ng * (rn - p.graze));
  const float3 world = g.eye.xyz + q;
  const float off = dist * (g.offset_pixels * g.pixel + g.offset_view) + length(world) * g.offset_world;
  st.o = world + ng * off;
  st.o_rel = q + ng * off;
  st.r = r;
  st.tmax = p.max_distance * exp(-5.0 * sqrt(2.0 / (e + 2.0)));
  // Wet ground's rays end sooner: past that, a miss, the lighting's own
  // reflection (its environment map), as for any miss.
  if ((rate & RT_RATE_WET) != 0u) st.tmax = min(st.tmax, p.wet_distance);
  st.dist = dist;
  return true;
}
// Pixel gid's reflection into out (rgb, a hit) and dist; what it was: 0
// not traced, 1 a hit, 2 a miss, 3 a hit on an instance without an albedo,
// 4 a back face of a static instance of the main view that culls, 5 of a
// captured one, 6 a hit on a back face of another view's instance.
template <typename D>
uint rt_reflect(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                texture2d<float, access::read> stencil, instance_acceleration_structure scene,
                rt_isect_table table, constant RtReflectionParams& p, const device RtMaterial* materials,
                const device RtInfo* info, uint2 gid, thread half4& out, thread float& dist_out) {
  constant RtShadowParams& g = p.geo;
  out = half4(0.0h);
  dist_out = 0.0;
  RtReflectRayStart st;
  if (!rt_reflect_start(t, normal, surface, stencil, p, gid, st)) return 0u;
  const float3 o = st.o, r = st.r;
  const float tmax = st.tmax, dist = st.dist;
  // Closest hit, no face culled: static instances from the start, captured
  // ones beyond a few pixel footprints; lit as the pass lights.
  RtReflectLight lt;
  lt.light = g.light;
  lt.colour = p.colour;
  lt.ambient0 = p.ambient0;
  lt.ambient1 = p.ambient1;
  lt.up = p.up;
  lt.wrap = p.wrap;
  lt.pixel = g.pixel;
  lt.captured_gap = g.captured_gap;
  lt.pad = as_type<float>(p.diag);
  lt.offsets = float4(g.offset_pixels, g.offset_view, g.offset_world, 0.0);
  float3 rgb;
  float hd;
  uint hf;
  const uint s = rt_reflect_ray_hw(scene, table, materials, info, o, r, tmax, dist * g.pixel * p.gap_pixels, dist,
                                   lt, rgb, hd, hf);
  dist_out = s == 2u ? p.sky_distance : hd;
  if (s == 1u || s == 6u) out = half4(half3(clamp(rgb * p.scale, 0.0, 65504.0)), 1.0h);
  return s;
}

// The pixel lists (RtReflector's, one set per output): a header of uints,
// then lists of pixels, a uint each (x | y << 16): A the pixels to
// reflect, B those of the checkerboard's other half (filled from their
// neighbours in A), D those of A the screen-space march did not settle
// (traced), C those of B no neighbour could fill (traced after; in D's
// room, listed once D is traced). RtListLayout gives where they start.
constant uint RT_LIST_FRESH = 2u;  // 1: the output's contents unknown (written whole once)
// A list's count (appended to), length (what a dispatch runs over) and
// indirect threadgroups (of
// RT_LIST_GROUP for that length, 1, 1).
constant uint RT_LIST_A_COUNT = 0u, RT_LIST_A_LENGTH = 1u, RT_LIST_A_ARGS = 4u;
constant uint RT_LIST_B_COUNT = 3u, RT_LIST_B_LENGTH = 7u, RT_LIST_B_ARGS = 8u;
constant uint RT_LIST_C_COUNT = 11u, RT_LIST_C_LENGTH = 15u, RT_LIST_C_ARGS = 12u;
constant uint RT_LIST_D_COUNT = 16u, RT_LIST_D_LENGTH = 17u, RT_LIST_D_ARGS = 20u;
constant uint RT_LIST_BLEND = 0x80000000u;  // a pixel of D: mix with what the march wrote (screen space)
constant uint RT_LIST_GROUP = 64u;
struct RtListLayout {
  uint a, b, c;  // where each list's pixels start (uints from the header's)
  uint pad;
};
// Which list a dispatch runs over: the index of its length, where its
// pixels start.
struct RtListOf {
  uint length, base;
};
uint rt_pack(uint2 gid) { return gid.x | (gid.y << 16); }
uint2 rt_unpack(uint c) { return uint2(c & 0xFFFFu, (c >> 16) & 0x7FFFu); }
// Appends `mine` pixels (0 or 1) of the SIMD-group to the list whose count
// is at `count`, pixels from `base`: one atomic, a prefix sum.
void rt_append(device uint* list, uint count, uint base, bool mine, uint2 gid) {
  const uint m = mine ? 1u : 0u, total = simd_sum(m);
  if (total == 0u) return;
  const uint before = simd_prefix_exclusive_sum(m);
  uint at = 0u;
  if (simd_is_first())
    at = atomic_fetch_add_explicit((device atomic_uint*)(list + count), total, memory_order_relaxed);
  at = simd_broadcast_first(at);
  if (mine) list[base + at + before] = rt_pack(gid);
}

// 1. The pixels to reflect (rt_reflect_test) listed, a SIMD-group's at
// once: in A those traced, in B those filled (the checkerboard's other
// half, the sparse rates' others); every other pixel written 0 where it is
// not (each listed one is written by what follows: the output holds 0
// wherever nothing is reflected).
template <typename D>
void rt_reflect_classify(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                         texture2d<float, access::read> stencil, constant RtReflectionParams& p, device uint* list,
                         constant RtListLayout& lay, texture2d<half, access::write> out,
                         texture2d<float, access::read_write> dist, texture2d<uint, access::write> traced,
                         device atomic_uint* counts, uint2 gid) {
  const bool in = gid.x < p.geo.size.x && gid.y < p.geo.size.y;
  float v, e;
  float3 n;
  uint rate = 1u;
  const bool shiny = in && rt_reflect_test(t, normal, surface, stencil, p, gid, v, n, e, rate);
  const bool fill = shiny && !rt_reflect_traced(p, gid, rate);
  rt_append(list, RT_LIST_A_COUNT, lay.a, shiny && !fill, gid);
  rt_append(list, RT_LIST_B_COUNT, lay.b, shiny && fill, gid);
  // (The rays of the checkerboard's rate, counted: RtReflectionCounts::dense.)
  const uint dense = simd_sum(shiny && !fill && (rate & 7u) == 1u ? 1u : 0u);
  if (simd_is_first() && dense != 0u) atomic_fetch_add_explicit(&counts[11], dense, memory_order_relaxed);
  // (Which pixels the fill may take from: this frame's traced ones.)
  if (in && p.checker != 0u) traced.write(uint4(shiny && !fill ? 1u : 0u), gid);
  // A pixel not reflected now: 0, if it is not already (a distance of 0
  // where the last frame's left nothing; a fresh output's are unknown).
  if (in && !shiny && (list[RT_LIST_FRESH] != 0u || dist.read(gid).r != 0.0)) {
    out.write(half4(0.0h), gid);
    dist.write(float4(0.0), gid);
  }
}
kernel void rt_reflect_classify_depth_cs(constant RtReflectionParams& p [[buffer(1)]],
                                         device uint* list [[buffer(5)]],
                                         constant RtListLayout& lay [[buffer(7)]],
                                         depth2d<float, access::read> depth [[texture(0)]],
                                         texture2d<float, access::read> normal [[texture(1)]],
                                         texture2d<float, access::read> surface [[texture(2)]],
                                         texture2d<float, access::read> stencil [[texture(3)]],
                                         texture2d<half, access::write> out [[texture(4)]],
                                         texture2d<float, access::read_write> dist [[texture(5)]],
                                         texture2d<uint, access::write> traced [[texture(7)]],
                                         device atomic_uint* counts [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  rt_reflect_classify(depth, normal, surface, stencil, p, list, lay, out, dist, traced, counts, gid);
}
kernel void rt_reflect_classify_color_cs(constant RtReflectionParams& p [[buffer(1)]],
                                         device uint* list [[buffer(5)]],
                                         constant RtListLayout& lay [[buffer(7)]],
                                         texture2d<float, access::read> depth [[texture(0)]],
                                         texture2d<float, access::read> normal [[texture(1)]],
                                         texture2d<float, access::read> surface [[texture(2)]],
                                         texture2d<float, access::read> stencil [[texture(3)]],
                                         texture2d<half, access::write> out [[texture(4)]],
                                         texture2d<float, access::read_write> dist [[texture(5)]],
                                         texture2d<uint, access::write> traced [[texture(7)]],
                                         device atomic_uint* counts [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  rt_reflect_classify(depth, normal, surface, stencil, p, list, lay, out, dist, traced, counts, gid);
}
void rt_list_close(device uint* list, uint count, uint length, uint args) {
  const uint n = list[count];
  list[length] = n;
  list[count] = 0u;
  list[args] = (n + RT_LIST_GROUP - 1u) / RT_LIST_GROUP;
  list[args + 1u] = 1u;
  list[args + 2u] = 1u;
}
// 3. A's and B's lengths and threadgroups; the counts reset, the output no
// longer fresh. (`which` 1: C's, after the fill; 2: D's, after the march.)
kernel void rt_reflect_args_cs(device uint* list [[buffer(0)]], constant uint& which [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
  if (i != 0u) return;
  if (which == 1u) {
    rt_list_close(list, RT_LIST_C_COUNT, RT_LIST_C_LENGTH, RT_LIST_C_ARGS);
    return;
  }
  if (which == 2u) {
    rt_list_close(list, RT_LIST_D_COUNT, RT_LIST_D_LENGTH, RT_LIST_D_ARGS);
    return;
  }
  rt_list_close(list, RT_LIST_A_COUNT, RT_LIST_A_LENGTH, RT_LIST_A_ARGS);
  rt_list_close(list, RT_LIST_B_COUNT, RT_LIST_B_LENGTH, RT_LIST_B_ARGS);
  list[RT_LIST_C_COUNT] = 0u;
  list[RT_LIST_D_COUNT] = 0u;
  list[RT_LIST_FRESH] = 0u;
}
// 4. A thread a pixel of list `of` (SIMD-groups full of rays, neighbours
// together): its reflection, and the SIMD-group's counts
// (RtReflectionCounts).
template <typename D>
void rt_reflect_listed(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                       texture2d<float, access::read> stencil, instance_acceleration_structure scene,
                       rt_isect_table table, constant RtReflectionParams& p, const device RtMaterial* materials,
                       const device RtInfo* info, const device uint* list, constant RtListOf& of,
                       texture2d<half, access::read_write> out, texture2d<float, access::read_write> dist,
                       device atomic_uint* counts, uint i) {
  const bool in = i < list[of.length];
  const uint packed = in ? list[of.base + i] : 0u;
  const uint2 gid = rt_unpack(packed);
  half4 o = half4(0.0h);
  float d = 0.0;
  uint s = 0u;
  if (in) s = rt_reflect(t, normal, surface, stencil, scene, table, p, materials, info, gid, o, d);
  // Flagged (D): the march's result (rgb, its weight w) mixed in: w of it,
  // 1 - w of the ray's, as the lighting would blend each over its own.
  if (in && (packed & RT_LIST_BLEND) != 0u) {
    const float4 m = float4(out.read(gid));
    const float a = m.a + (1.0 - m.a) * float(o.a);
    const float3 rgb = a > 0.0 ? (m.rgb * m.a + (1.0 - m.a) * float(o.a) * float3(o.rgb)) / a : float3(0.0);
    o = half4(half3(rgb), half(a));
    d = dist.read(gid).r;
  }
  const uint traced = simd_sum(s != 0u ? 1u : 0u), hits = simd_sum(s == 1u || s == 6u ? 1u : 0u),
             misses = simd_sum(s == 2u ? 1u : 0u), bare = simd_sum(s == 3u ? 1u : 0u),
             back = simd_sum(s == 4u || s == 5u ? 1u : 0u), back_captured = simd_sum(s == 5u ? 1u : 0u),
             other_back = simd_sum(s == 6u ? 1u : 0u);
  const float sum = simd_sum(s == 1u || s == 6u ? d : 0.0);
  if (simd_is_first() && traced != 0u) {
    atomic_fetch_add_explicit(&counts[0], traced, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], hits, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[2], misses, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[3], bare, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)(counts + 4), sum, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[5], back, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[6], back_captured, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[7], other_back, memory_order_relaxed);
  }
  if (in) {
    out.write(o, gid);
    dist.write(float4(d), gid);
  }
}
kernel void rt_reflect_trace_depth_cs(instance_acceleration_structure scene [[buffer(0)]],
                                      constant RtReflectionParams& p [[buffer(1)]],
                                      const device RtMaterial* materials [[buffer(2)]],
                                      device atomic_uint* counts [[buffer(3)]],
                                      const device RtInfo* info [[buffer(4)]],
                                      const device uint* list [[buffer(5)]],
                                      rt_isect_table table [[buffer(6)]],
                                      constant RtListOf& of [[buffer(8)]],
                                      depth2d<float, access::read> depth [[texture(0)]],
                                      texture2d<float, access::read> normal [[texture(1)]],
                                      texture2d<float, access::read> surface [[texture(2)]],
                                      texture2d<float, access::read> stencil [[texture(3)]],
                                      texture2d<half, access::read_write> out [[texture(4)]],
                                      texture2d<float, access::read_write> dist [[texture(5)]],
                                      uint i [[thread_position_in_grid]]) {
  rt_reflect_listed(depth, normal, surface, stencil, scene, table, p, materials, info, list, of, out, dist, counts,
                    i);
}
kernel void rt_reflect_trace_color_cs(instance_acceleration_structure scene [[buffer(0)]],
                                      constant RtReflectionParams& p [[buffer(1)]],
                                      const device RtMaterial* materials [[buffer(2)]],
                                      device atomic_uint* counts [[buffer(3)]],
                                      const device RtInfo* info [[buffer(4)]],
                                      const device uint* list [[buffer(5)]],
                                      rt_isect_table table [[buffer(6)]],
                                      constant RtListOf& of [[buffer(8)]],
                                      texture2d<float, access::read> depth [[texture(0)]],
                                      texture2d<float, access::read> normal [[texture(1)]],
                                      texture2d<float, access::read> surface [[texture(2)]],
                                      texture2d<float, access::read> stencil [[texture(3)]],
                                      texture2d<half, access::read_write> out [[texture(4)]],
                                      texture2d<float, access::read_write> dist [[texture(5)]],
                                      uint i [[thread_position_in_grid]]) {
  rt_reflect_listed(depth, normal, surface, stencil, scene, table, p, materials, info, list, of, out, dist, counts,
                    i);
}
// Screen-space reflections first (RtSsrParams): a pixel of list A marches
// its ray through the depth buffer; where it meets a surface that faces it
// on screen, the reflection is the previous frame's lit scene there (the
// point reprojected into it): no ray. Where it leaves the screen, finds
// nothing, meets a back face, or the point was off the previous frame, the
// pixel goes to list D and is traced. Near the screen's edges the two are
// blended (the pixel goes to D flagged: the trace mixes its result with
// what the march wrote). McGuire and Mara, "Efficient GPU Screen-Space Ray
// Tracing" (JCGT 3(4), 2014): the ray's ends projected, 1/w and the point
// over w stepped linearly across the screen, a crossing refined by halves.
struct RtSsrParams {
  float4x4 to_clip;   // camera-relative point to (x, y, v, 1) times its view depth (metres; scaled so)
  float4x4 to_prev;   // camera-relative point to the previous frame's clip (its camera, no jitter)
  float2 prev_size;   // the previous lit scene's size
  float2 prev_jitter; // its scene's offset, pixels (right, down)
  float thickness;    // a surface's assumed depth behind it, metres, at least
  float thickness_rel;// and as a fraction of its view depth
  float edge;         // the band along the screen's edges where the ray's result mixes in (pixels)
  float scale;        // the lit scene to the output's units (RayReflection::output_scale)
  uint steps, refine;
  float2 inv_depth;   // a stored depth v's 1 / view depth: x v + y (a perspective's: affine in v)
};
// The view depth's inverse at stored depth v (sky: 0, infinitely far).
float rt_ssr_k(constant RtShadowParams& g, constant RtSsrParams& s, float v) {
  return rt_sky(g, v) ? 0.0 : s.inv_depth.x * v + s.inv_depth.y;
}
// The pixel (x, y as rt_point takes them) a camera-relative point h (its
// to_clip) shows at.
float2 rt_ssr_pixel(constant RtShadowParams& g, float4 h) {
  const float2 ndc = h.xy / h.w;
  return float2((ndc.x + 1.0) * 0.5 * float(g.size.x) + g.jitter.x, (1.0 - ndc.y) * 0.5 * float(g.size.y) + g.jitter.y);
}
// The march: 1 a hit (rgb the previous frame's light there, its weight w:
// 1, less in the edge band; d its distance along the ray), 0 none.
template <typename D>
uint rt_ssr_march(D t, texture2d<float, access::read> normal, texture2d<float, access::sample> prev,
                  constant RtShadowParams& g, constant RtSsrParams& s, float3 o, float3 r, float tmax,
                  thread float3& rgb, thread float& w, thread float& d) {
  // The ray's end: no nearer the camera's plane than a tenth of a metre.
  const float4 h0 = s.to_clip * float4(o, 1.0);
  float len = tmax;
  const float dz = s.to_clip[0].w * r.x + s.to_clip[1].w * r.y + s.to_clip[2].w * r.z;  // d(view depth)/dt
  if (dz < 0.0) len = min(len, (h0.w - 0.1) / -dz);
  if (!(len > 0.0) || !(h0.w > 0.1)) return 0u;
  const float4 h1 = s.to_clip * float4(o + r * len, 1.0);
  const float k0 = 1.0 / h0.w, k1 = 1.0 / h1.w;
  const float2 p0 = rt_ssr_pixel(g, h0), p1 = rt_ssr_pixel(g, h1);
  const float span = max(abs(p1.x - p0.x), abs(p1.y - p0.y));
  if (!(span >= 1.0)) return 0u;
  // Steps of at least a pixel, the first a pixel and a half out (past the
  // pixel's own). The first step behind what the screen shows ends it: the
  // crossing refined by halves, a hit there if the ray is within the
  // surface's thickness of it, else nothing (it passed behind something:
  // what lies there is hidden, a ray knows).
  const float u0 = min(1.5 / span, 1.0), du = max((1.0 - u0) / float(max(s.steps, 1u)), 1.0 / span);
  float u_prev = 0.0;
  for (uint i = 0u; i < s.steps; i++) {
    const float u = min(u0 + float(i) * du, 1.0);
    const int2 c = int2(round(mix(p0, p1, u)));
    if (!rt_inside(g, c)) return 0u;  // off the screen
    // (Behind: the ray's 1 / depth at most the surface's.)
    const float v = rt_depth(t, uint2(c));
    if (mix(k0, k1, u) <= rt_ssr_k(g, s, v)) {
      float ua = u_prev, ub = u;
      int2 cb = c;
      float vb = v;
      for (uint j = 0u; j < s.refine; j++) {
        const float um = 0.5 * (ua + ub);
        const int2 cm = int2(round(mix(p0, p1, um)));
        const float vm = rt_inside(g, cm) ? rt_depth(t, uint2(cm)) : v;
        if (rt_inside(g, cm) && mix(k0, k1, um) <= rt_ssr_k(g, s, vm)) {
          ub = um;
          cb = cm;
          vb = vm;
        } else {
          ua = um;
        }
      }
      const float3 qb = rt_point(g, cb, vb);
      const float lb = (s.to_clip * float4(qb, 1.0)).w, lam = 1.0 / mix(k0, k1, ub);
      if (lam > lb + max(s.thickness, s.thickness_rel * lb)) return 0u;
      // A surface facing the ray (else what it met is hidden: traced).
      const float3 nb = rt_gbuf(normal, uint2(cb), g.size).xyz * 2.0 - 1.0;
      if (dot(nb, r) >= 0.0) return 0u;
      // The previous frame's light there.
      const float4 hp = s.to_prev * float4(qb, 1.0);
      if (!(hp.w > 0.0)) return 0u;
      const float2 pp = float2((hp.x / hp.w + 1.0) * 0.5 * s.prev_size.x + s.prev_jitter.x,
                               (1.0 - hp.y / hp.w) * 0.5 * s.prev_size.y + s.prev_jitter.y);
      if (any(pp < float2(0.0)) || any(pp > s.prev_size - 1.0)) return 0u;
      constexpr sampler lin(filter::linear, address::clamp_to_edge);
      rgb = prev.sample(lin, (pp + 0.5) / s.prev_size).rgb * s.scale;
      const float2 cf = float2(cb);
      const float e = min(min(cf.x, float(g.size.x) - 1.0 - cf.x), min(cf.y, float(g.size.y) - 1.0 - cf.y));
      w = saturate(e / max(s.edge, 1.0));
      d = length(qb - o);
      return w > 0.0 ? 1u : 0u;
    }
    u_prev = u;
    if (u >= 1.0) break;
  }
  return 0u;
}
// 4a. A thread a pixel of A: the march; what it does not settle to D.
// Counted (RtReflectionCounts::screen, screen_mixed).
template <typename D>
void rt_reflect_ssr(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                    texture2d<float, access::read> stencil, texture2d<float, access::sample> prev,
                    constant RtReflectionParams& p, constant RtSsrParams& s, device uint* list,
                    constant RtListLayout& lay, texture2d<half, access::read_write> out,
                    texture2d<float, access::read_write> dist, device atomic_uint* counts, uint i) {
  constant RtShadowParams& g = p.geo;
  const bool in = i < list[RT_LIST_A_LENGTH];
  const uint2 gid = in ? rt_unpack(list[lay.a + i]) : uint2(0u);
  bool done = false, mixed = false, trace = in;
  RtReflectRayStart st;
  if (in && rt_reflect_start(t, normal, surface, stencil, p, gid, st)) {
    float3 rgb;
    float w, d;
    if (rt_ssr_march(t, normal, prev, g, s, st.o_rel, st.r, st.tmax, rgb, w, d) != 0u) {
      out.write(half4(half3(clamp(rgb, 0.0, 65504.0)), half(w)), gid);
      dist.write(float4(d), gid);
      done = w >= 1.0;
      mixed = !done;
    }
  }
  trace = in && !done;
  // (D shares C's room: C is listed after D is traced.)
  const uint m = trace ? 1u : 0u, total = simd_sum(m);
  if (total != 0u) {
    const uint before = simd_prefix_exclusive_sum(m);
    uint at = 0u;
    if (simd_is_first())
      at = atomic_fetch_add_explicit((device atomic_uint*)(list + RT_LIST_D_COUNT), total, memory_order_relaxed);
    at = simd_broadcast_first(at);
    if (trace) list[lay.c + at + before] = rt_pack(gid) | (mixed ? RT_LIST_BLEND : 0u);
  }
  const uint sd = simd_sum(done ? 1u : 0u), sm = simd_sum(mixed ? 1u : 0u);
  if (simd_is_first() && sd + sm != 0u) {
    atomic_fetch_add_explicit(&counts[9], sd, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[10], sm, memory_order_relaxed);
  }
}
kernel void rt_reflect_ssr_depth_cs(constant RtReflectionParams& p [[buffer(1)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    device uint* list [[buffer(5)]],
                                    constant RtListLayout& lay [[buffer(7)]],
                                    constant RtSsrParams& s [[buffer(9)]],
                                    depth2d<float, access::read> depth [[texture(0)]],
                                    texture2d<float, access::read> normal [[texture(1)]],
                                    texture2d<float, access::read> surface [[texture(2)]],
                                    texture2d<float, access::read> stencil [[texture(3)]],
                                    texture2d<half, access::read_write> out [[texture(4)]],
                                    texture2d<float, access::read_write> dist [[texture(5)]],
                                    texture2d<float, access::sample> prev [[texture(6)]],
                                    uint i [[thread_position_in_grid]]) {
  rt_reflect_ssr(depth, normal, surface, stencil, prev, p, s, list, lay, out, dist, counts, i);
}
kernel void rt_reflect_ssr_color_cs(constant RtReflectionParams& p [[buffer(1)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    device uint* list [[buffer(5)]],
                                    constant RtListLayout& lay [[buffer(7)]],
                                    constant RtSsrParams& s [[buffer(9)]],
                                    texture2d<float, access::read> depth [[texture(0)]],
                                    texture2d<float, access::read> normal [[texture(1)]],
                                    texture2d<float, access::read> surface [[texture(2)]],
                                    texture2d<float, access::read> stencil [[texture(3)]],
                                    texture2d<half, access::read_write> out [[texture(4)]],
                                    texture2d<float, access::read_write> dist [[texture(5)]],
                                    texture2d<float, access::sample> prev [[texture(6)]],
                                    uint i [[thread_position_in_grid]]) {
  rt_reflect_ssr(depth, normal, surface, stencil, prev, p, s, list, lay, out, dist, counts, i);
}
// The lit scene kept for the next frame's march (RG11B10Float: a copy at
// two thirds the bytes).
kernel void rt_reflect_keep_cs(texture2d<float, access::read> lit [[texture(0)]],
                               texture2d<float, access::write> kept [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= kept.get_width() || gid.y >= kept.get_height()) return;
  kept.write(float4(max(lit.read(gid).rgb, 0.0), 1.0), gid);
}
// 5. A thread a pixel of B (the checkerboard's untraced half: its four
// neighbours are A's): the mean of its neighbours' reflections that were
// traced on the same surface (a hit distance written; its distance from the
// camera within kRtReflectionFillPixels footprints of its own; the
// G-buffer's normals within kRtReflectionFillCos), as the lighting would
// blend them (the colour of the hits weighted by their coverage, the
// coverage the mean); none: the pixel to C, traced after. Counted
// (RtReflectionCounts::filled).
template <typename D>
void rt_reflect_fill(D t, texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                     texture2d<float, access::read> stencil, constant RtReflectionParams& p, device uint* list,
                     constant RtListLayout& lay, texture2d<half, access::read_write> out,
                     texture2d<float, access::read_write> dist, texture2d<uint, access::read> traced,
                     device atomic_uint* counts, uint i) {
  constant RtShadowParams& g = p.geo;
  const bool in = i < list[RT_LIST_B_LENGTH];
  const uint2 gid = in ? rt_unpack(list[lay.b + i]) : uint2(0u);
  float v = 0.0, e;
  float3 n = float3(0.0);
  uint rate = 1u;
  bool filled = false, own = in;
  if (in && rt_reflect_test(t, normal, surface, stencil, p, gid, v, n, e, rate)) {
    const float3 q = rt_point(g, int2(gid), v);
    const float qd = length(q);
    // A neighbour on the same surface lies about a footprint away: the
    // pixels' angle over the cosine the surface is seen at (its normal's),
    // fill_depth of them, and 1% for the depth's precision.
    const float near = qd * (p.fill_depth * g.pixel / max(abs(dot(q, n)) / max(qd, 1e-6), 0.05) + 0.01);
    float4 rgb_a = float4(0.0);  // the hits' colour times their coverage, the coverage
    // (The distance: of the most neighbours' kind, the hits first, then
    // what stopped a ray without one (the mean of their distances), then
    // the misses (the sky's).)
    float cover = 0.0, k = 0.0, hit_d = 0.0, hit_k = 0.0, stop_d = 0.0, stop_k = 0.0, sky_k = 0.0;
    // The neighbours traced in its own pattern: the checkerboard's four
    // beside it; the 2 x 2 or 4 x 4 blocks' traced places within 1 or 2
    // pixels (1 to 4 of them).
    int2 cand[4];
    int nc = 0;
    const uint rt = rate & 7u;
    if (rt == 1u) {
      cand[0] = int2(-1, 0);
      cand[1] = int2(1, 0);
      cand[2] = int2(0, -1);
      cand[3] = int2(0, 1);
      nc = 4;
    } else {
      const uint2 o = rt_reflect_place(p, rt);
      const int per = int(rt), r = per / 2;
      int xs[2], ys[2], nx = 0, ny = 0;
      for (int d = -r; d <= r; d++) {
        if (((int(gid.x) + d - int(o.x)) & (per - 1)) == 0 && nx < 2) xs[nx++] = d;
        if (((int(gid.y) + d - int(o.y)) & (per - 1)) == 0 && ny < 2) ys[ny++] = d;
      }
      for (int a = 0; a < ny; a++)
        for (int b = 0; b < nx; b++) cand[nc++] = int2(xs[b], ys[a]);
    }
    for (int s = 0; s < nc; s++) {
      const int dx = cand[s].x, dy = cand[s].y;
      const int2 c = int2(gid) + cand[s];
      if ((dx == 0 && dy == 0) || !rt_inside(g, c)) continue;
      const uint2 u = uint2(c);
      if (traced.read(u).r == 0u) continue;  // (not traced this frame)
      const float d = dist.read(u).r;
      if (!(d > 0.0)) continue;  // (nothing to trace there after all)
      const float vn = rt_depth(t, u);
      if (rt_sky(g, vn)) continue;
      if (abs(length(rt_point(g, c, vn)) - qd) > near * float(max(abs(dx), abs(dy)))) continue;
      const float3 nn = rt_gbuf(normal, u, g.size).xyz * 2.0 - 1.0;
      if (dot(normalize(nn), n) < p.fill_cos) continue;
      const float4 o = float4(out.read(u));
      rgb_a += float4(o.rgb * o.a, o.a);
      cover += o.a;
      k += 1.0;
      if (o.a > 0.5) {
        hit_d += d;
        hit_k += 1.0;
      } else if (d < p.sky_distance) {
        stop_d += d;
        stop_k += 1.0;
      } else {
        sky_k += 1.0;
      }
    }
    if (k > 0.0) {
      const float a = cover / k;
      const float3 rgb = cover > 0.0 ? rgb_a.rgb / cover : float3(0.0);
      out.write(half4(half3(rgb), half(a)), gid);
      const float dd = hit_k >= stop_k && hit_k >= sky_k ? hit_d / hit_k
                       : stop_k >= sky_k                 ? stop_d / stop_k
                                                         : p.sky_distance;
      dist.write(float4(dd), gid);
      filled = true;
    }
    own = !filled;
  } else if (in) {
    own = false;  // (nothing to trace: its 0s stay)
  }
  rt_append(list, RT_LIST_C_COUNT, lay.c, own, gid);
  const uint f = simd_sum(filled ? 1u : 0u);
  if (simd_is_first() && f != 0u) atomic_fetch_add_explicit(&counts[8], f, memory_order_relaxed);
}
kernel void rt_reflect_fill_depth_cs(constant RtReflectionParams& p [[buffer(1)]],
                                     device atomic_uint* counts [[buffer(3)]],
                                     device uint* list [[buffer(5)]],
                                     constant RtListLayout& lay [[buffer(7)]],
                                     depth2d<float, access::read> depth [[texture(0)]],
                                     texture2d<float, access::read> normal [[texture(1)]],
                                     texture2d<float, access::read> surface [[texture(2)]],
                                     texture2d<float, access::read> stencil [[texture(3)]],
                                     texture2d<half, access::read_write> out [[texture(4)]],
                                     texture2d<float, access::read_write> dist [[texture(5)]],
                                     texture2d<uint, access::read> traced [[texture(7)]],
                                     uint i [[thread_position_in_grid]]) {
  rt_reflect_fill(depth, normal, surface, stencil, p, list, lay, out, dist, traced, counts, i);
}
kernel void rt_reflect_fill_color_cs(constant RtReflectionParams& p [[buffer(1)]],
                                     device atomic_uint* counts [[buffer(3)]],
                                     device uint* list [[buffer(5)]],
                                     constant RtListLayout& lay [[buffer(7)]],
                                     texture2d<float, access::read> depth [[texture(0)]],
                                     texture2d<float, access::read> normal [[texture(1)]],
                                     texture2d<float, access::read> surface [[texture(2)]],
                                     texture2d<float, access::read> stencil [[texture(3)]],
                                     texture2d<half, access::read_write> out [[texture(4)]],
                                     texture2d<float, access::read_write> dist [[texture(5)]],
                                     texture2d<uint, access::read> traced [[texture(7)]],
                                     uint i [[thread_position_in_grid]]) {
  rt_reflect_fill(depth, normal, surface, stencil, p, list, lay, out, dist, traced, counts, i);
}

// The reflections over a target (RtReflector::EncodeView).
kernel void rt_reflect_view_cs(texture2d<half, access::sample> refl [[texture(0)]],
                               texture2d<float, access::sample> dist [[texture(1)]],
                               texture2d<half, access::write> dst [[texture(2)]],
                               constant float& scale [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  constexpr sampler s(filter::nearest, address::clamp_to_edge);
  const float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
  const half4 c = refl.sample(s, uv);
  const float d = dist.sample(s, uv).r;
  const half3 rgb = c.a > 0.5h       ? half3(min(float3(c.rgb) / max(scale, 1e-6), 1.0))
                    : d >= 32768.0  ? half3(0.0h, 0.0h, 0.3h)   // a miss (the sky's distance)
                    : d > 0.0       ? half3(0.5h, 0.0h, 0.5h)   // no albedo, a back face
                                    : half3(0.0h);
  dst.write(half4(rgb, 1.0h), gid);
}
)";

float Len3(const float* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

id<MTLTexture> NewTexture(id<MTLDevice> dev, MTLPixelFormat pf, uint32_t w, uint32_t h) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                width:std::max(w, 1u)
                                                                               height:std::max(h, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev newTextureWithDescriptor:td];
}

}  // namespace

bool RtReflectionSameKey(const RtReflectionKey& a, const RtReflectionKey& b) {
  auto f = [](float x, float y) { return std::memcmp(&x, &y, 4) == 0; };
  const RayReflection &x = a.config, &y = b.config;
  bool same = x.output_sampler == y.output_sampler && x.depth_sampler == y.depth_sampler &&
              x.normal_sampler == y.normal_sampler && x.surface_sampler == y.surface_sampler &&
              x.spec_channel == y.spec_channel && x.gloss_channel == y.gloss_channel &&
              f(x.exponent_scale, y.exponent_scale) && x.blur_register == y.blur_register &&
              x.blur_component == y.blur_component && x.light_register == y.light_register &&
              x.toward_light == y.toward_light && x.colour_register == y.colour_register &&
              x.ambient_registers[0] == y.ambient_registers[0] && x.ambient_registers[1] == y.ambient_registers[1] &&
              f(x.up[0], y.up[0]) && f(x.up[1], y.up[1]) && f(x.up[2], y.up[2]) &&
              f(x.diffuse_wrap, y.diffuse_wrap) && f(x.output_scale, y.output_scale) &&
              f(x.max_distance, y.max_distance) && f(x.wet_slope, y.wet_slope) && f(x.wet_gain, y.wet_gain) &&
              f(x.wet_max_exponent, y.wet_max_exponent) && x.wet_register == y.wet_register &&
              x.wet_component == y.wet_component && x.stencil_sampler == y.stencil_sampler;
  return same && std::memcmp(&a.inputs, &b.inputs, sizeof a.inputs) == 0 &&
         std::memcmp(a.textures, b.textures, sizeof a.textures) == 0;
}

int RtReflectionEntry(RtReflectionCache* c, const RtReflectionKey& key) {
  const uint64_t now = ++c->uses;
  int oldest = 0;
  for (int k = 0; k < RtReflectionCache::kEntries; k++) {
    RtReflectionCache::Entry& e = c->entry[k];
    if (e.used && RtReflectionSameKey(e.key, key)) {
      e.last_use = now;
      return k;
    }
    if (!e.used || (c->entry[oldest].used && e.last_use < c->entry[oldest].last_use)) oldest = k;
  }
  RtReflectionCache::Entry& e = c->entry[oldest];
  e.used = true;
  e.key = key;
  e.memo = {};
  e.last_use = now;
  return oldest;
}

RtReflectionInputs RtReflectionInputsFor(const RayReflection& r, const float (*c)[4]) {
  RtReflectionInputs in;
  if (!c) return in;
  const float sign = r.toward_light ? 1.0f : -1.0f;
  if (r.light_register < 224)
    for (int k = 0; k < 3; k++) in.light[k] = sign * c[r.light_register][k];
  if (r.colour_register < 224)
    for (int k = 0; k < 3; k++) in.colour[k] = c[r.colour_register][k] * c[r.colour_register][3];
  for (int a = 0; a < 2; a++)
    if (r.ambient_registers[a] < 224)
      for (int k = 0; k < 3; k++) in.ambient[a][k] = c[r.ambient_registers[a]][k];
  if (r.blur_register < 224)
    in.blur_k = c[r.blur_register][r.blur_component & 3];
  else
    in.blur_k = r.exponent_scale > 0 ? 1.0f / r.exponent_scale : 0.0f;
  if (r.wet_slope > 0 && r.wet_register < 224) in.wet = c[r.wet_register][r.wet_component & 3];
  return in;
}

bool RtReflectionParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                           const float jitter[2], const RayReflection& r, const RtReflectionInputs& in, bool wet_on,
                           RtReflectionParams* out) {
  *out = RtReflectionParams{};
  float up[3] = {r.up[0], r.up[1], r.up[2]};
  const float ul = Len3(up);
  if (!(ul > 0) || !std::isfinite(ul)) {
    up[0] = up[1] = 0;
    up[2] = 1;
  } else {
    for (float& v : up) v /= ul;
  }
  const float ll = Len3(in.light);
  const bool light = ll > 0 && std::isfinite(ll);
  // (Without a light the hits get the ambient term: the ray toward "up" is
  // never traced, the colour is 0.)
  if (!RtShadowParamsFor(f, w, h, stored_reversed, depth_texture, jitter, light ? in.light : up, &out->geo))
    return false;
  for (int k = 0; k < 3; k++) {
    out->colour[k] = light ? in.colour[k] : 0.0f;
    out->ambient[0][k] = in.ambient[0][k];
    out->ambient[1][k] = in.ambient[1][k];
    out->up[k] = up[k];
  }
  out->scale = r.output_scale;
  out->wrap = std::min(r.diffuse_wrap, 0.999f);
  out->exponent_scale = r.exponent_scale;
  out->blur_k = in.blur_k;
  out->max_blur = kRtReflectionMaxBlur;
  out->max_distance = std::clamp(r.max_distance, 0.0f, kRtReflectionLongest);
  out->sky_distance = kRtReflectionSkyDistance;
  out->wet = in.wet;
  out->wet_slope = r.wet_slope;
  out->wet_gain = r.wet_gain;
  out->wet_max_exponent = r.wet_max_exponent;
  out->wet_on = wet_on && r.wet_slope > 0 ? 1 : 0;
  out->spec_channel = r.spec_channel & 3;
  out->gloss_channel = r.gloss_channel & 3;
  out->graze = kRtReflectionGraze;
  out->gap_pixels = kRtReflectionCapturedGapPixels;
  out->checker = 0;
  out->parity = 0;
  out->fill_depth = kRtReflectionFillPixels;
  out->fill_cos = kRtReflectionFillCos;
  out->sparse_cos = kRtReflectionSparseCos;
  out->sparse_spec = kRtReflectionSparseSpec;
  out->wet_distance = std::min(kRtReflectionWetDistance, out->max_distance);
  return true;
}

// ---- the far light bounced once (RtReflector::EncodeBounce)
//
// What the sun lights, lighting what it does not: at one pixel of each
// RT_BOUNCE_BLOCK x RT_BOUNCE_BLOCK block (turning each frame) a ray in a
// cosine-weighted direction around the surface's geometric normal, what it
// hits lit as a reflection's hit is (its albedo times the far light where
// a shadow ray from the hit reaches it; no ambient: the lighting has its
// own); and from the same ray the sky light's occlusion (how far it got,
// over RtBounceParams::ao_radius: 1 open; a miss open). Filtered over the
// blocks around on the same surface and blended with the block's last
// result where the camera has not moved off it; then, at half the depth's
// size, each texel from the blocks around its pixel on its plane. rgb the
// light arriving (before the surface's own albedo), a the occlusion: the
// lighting takes its flat ambient times a, plus rgb.
//
// What moves (captured instances: vehicles, people) takes its occlusion
// and its light with it, and the history stays where the screen's point
// was: a car's dark surround and the road that comes out from under it
// kept the car's occlusion for some fifteen frames, a dark trail behind it
// that grew with its speed. A sample on a captured instance, or whose ray
// met one, is marked (its point's w 2), a block's result carries the share
// of marked samples in its filter (the kept point's w, 1 + share), and
// where this share or the history's is RT_BOUNCE_MOVING_SHARE or more the
// frame's own result is taken at RT_BOUNCE_MOVING_BLEND instead of
// RtBounceParams::blend (between: in proportion).
const char* const kRtBounceMsl = R"(
constant uint RT_BOUNCE_BLOCK = 4u;
// The weight of the other blocks on a block's surface at which its result
// is taken whole (rt_bounce_filter_cs; a flat wall's 7 x 7 brings about 15).
constant float RT_BOUNCE_SUPPORT = 5.0;
constant float RT_BOUNCE_CAP = 1.5;  // a sample's greatest luminance
// The share of a block's filter weight on samples of what moves at which
// its history counts least, and this frame's share of the result then.
constant float RT_BOUNCE_MOVING_SHARE = 0.25;
constant float RT_BOUNCE_MOVING_BLEND = 0.6;
// The frame's share of a block's result: `blend`, more where the samples
// now (`share`) or those of the history read (`was`: its point's w - 1)
// are of what moves.
float rt_bounce_blend(float blend, float share, float was) {
  return mix(blend, max(blend, RT_BOUNCE_MOVING_BLEND), saturate(max(share, was) / RT_BOUNCE_MOVING_SHARE));
}
struct RtBounceParams {  // (RtBounceParams, rt_reflection.h: the same order)
  uint frame, stride;
  float distance, strength, blend, tolerance;
  uint2 blocks;
  float ao_radius, ao_strength;
  uint history, pad0;
  float4 prev[4];
  float4 eye_delta;
  uint2 size, pad1;
};
uint rt_bounce_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
template <typename D>
void rt_bounce_trace(D t, instance_acceleration_structure scene, rt_isect_table table,
                     constant RtReflectionParams& p, constant RtBounceParams& bp, const device RtMaterial* materials,
                     const device RtInfo* info, texture2d<half, access::write> out_rgb,
                     texture2d<float, access::write> out_q, texture2d<half, access::write> out_n, uint2 tid) {
  if (any(tid >= bp.blocks)) return;
  bool valid = false;
  constant RtShadowParams& g = p.geo;
  const uint2 gid = min(rt_light_sample(tid, bp.frame, bp.stride, RT_BOUNCE_BLOCK), g.size - 1u);
  const float v = rt_depth(t, gid);
  half4 rgb = half4(0.0h, 0.0h, 0.0h, 1.0h);  // (the sky: nothing, open)
  float3 q = float3(0.0), n = float3(0.0);
  // A pixel on cut-out geometry (leaves, grass, fences: alpha-tested
  // instances), found by a ray of a few centimetres through its point along
  // the view that meets only those: no bounce, no occlusion there (the
  // sky open). Its one ray in 16 pixels, whose rays meet leaf after leaf,
  // showed as dark blotches in the trees that no filter could smooth.
  bool cut = false;
  // What moves (a captured instance): the pixel's own surface, found by
  // the same short ray, or what its bounce ray meets below.
  bool moving = false;
  if (!rt_sky(g, v)) {
    const float3 qc = rt_point(g, int2(gid), v);
    const float dc = length(qc);
    const float3 vd = qc / max(dc, 1e-4);
    const float span = max(0.05, dc * 0.002);
    const float3 wc = g.eye.xyz + qc;
    cut = rt_trace_hw(scene, table, ray(wc - vd * span, vd, 0.0, 2.0 * span), RT_MASK_CUT | RT_MASK_CAPTURED_CUT,
                      materials, true, 0u).hit;
    if (!cut)
      moving = rt_trace_hw(scene, table, ray(wc - vd * span, vd, 0.0, 2.0 * span), RT_MASK_CAPTURED_OPAQUE,
                           materials, true, 0u).hit;
  }
  if (!rt_sky(g, v) && !cut) {
    const int2 c = int2(gid);
    q = rt_point(g, c, v);
    const float d = length(q);
    float3 dx, dy, ng = -q;
    if (rt_step(t, g, c, int2(1, 0), v, q, dx) && rt_step(t, g, c, int2(0, 1), v, q, dy)) {
      const float3 m = cross(dx, dy);
      if (length_squared(m) > 0.0) ng = m;
    }
    ng = normalize(ng);
    if (dot(ng, q) > 0.0) ng = -ng;
    n = ng;
    const float3 world = g.eye.xyz + q;
    const float off = d * (g.offset_pixels * g.pixel + g.offset_view) + length(world) * g.offset_world;
    const float3 o = world + ng * max(off, 0.02);
    const float3 a = abs(ng.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 tx = normalize(cross(a, ng)), ty = cross(ng, tx);
    const uint h = rt_bounce_hash(gid.x * 0x8da6b343u ^ gid.y * 0xd8163841u ^ bp.frame * 0x9e3779b9u);
    const float u1 = float(h & 0xFFFFu) / 65536.0, u2 = float(h >> 16) / 65536.0;
    const float r = sqrt(u1), phi = 6.28318530718 * u2;
    const float3 l = normalize(tx * (r * cos(phi)) + ty * (r * sin(phi)) + ng * sqrt(max(0.0, 1.0 - u1)));
    RtReflectLight lt;
    lt.light = g.light;
    lt.colour = p.colour;
    lt.ambient0 = float4(0.0);
    lt.ambient1 = float4(0.0);
    lt.up = p.up;
    lt.wrap = p.wrap;
    lt.pixel = g.pixel * 8.0;  // (a wide cone: the albedo's coarse levels)
    lt.captured_gap = g.captured_gap;
    lt.pad = as_type<float>(p.diag);
    lt.offsets = float4(g.offset_pixels, g.offset_view, g.offset_world, 0.0);
    float3 lit;
    float hd;
    uint hf;
    const uint s = rt_reflect_ray_hw(scene, table, materials, info, o, l, bp.distance, d * g.pixel * p.gap_pixels, d,
                                     lt, lit, hd, hf);
    if (s != 2u && (hf & RT_INFO_CAPTURED) != 0u) moving = true;
    // (A sample's light capped: one ray in sixteen pixels that met a bright
    // sunlit hit otherwise shows as a speck through the filter.)
    float3 b = (s == 1u || s == 6u) ? lit * bp.strength : float3(0.0);
    const float lum = dot(b, float3(0.2126, 0.7152, 0.0722));
    if (lum > RT_BOUNCE_CAP) b *= RT_BOUNCE_CAP / lum;
    // (A miss, or a hit past the radius: open. A hit on a back face (inside
    // a model) or of no albedo counts by its distance like any other.)
    const float open = s == 2u || !(hd < bp.ao_radius) ? 1.0 : saturate(hd / max(bp.ao_radius, 1e-3));
    rgb = half4(half3(b), half(mix(1.0, open, bp.ao_strength)));
    valid = true;
  }
  out_rgb.write(rgb, tid);
  out_q.write(float4(q, valid ? (moving ? 2.0 : 1.0) : 0.0), tid);
  out_n.write(half4(half3(n), 0.0h), tid);
}
kernel void rt_bounce_trace_depth_cs(instance_acceleration_structure scene [[buffer(0)]],
                                     constant RtReflectionParams& p [[buffer(1)]],
                                     const device RtMaterial* materials [[buffer(2)]],
                                     constant RtBounceParams& bp [[buffer(3)]],
                                     const device RtInfo* info [[buffer(4)]], rt_isect_table table [[buffer(6)]],
                                     depth2d<float, access::read> depth [[texture(0)]],
                                     texture2d<half, access::write> out_rgb [[texture(1)]],
                                     texture2d<float, access::write> out_q [[texture(2)]],
                                     texture2d<half, access::write> out_n [[texture(3)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_bounce_trace(depth, scene, table, p, bp, materials, info, out_rgb, out_q, out_n, tid);
}
kernel void rt_bounce_trace_color_cs(instance_acceleration_structure scene [[buffer(0)]],
                                     constant RtReflectionParams& p [[buffer(1)]],
                                     const device RtMaterial* materials [[buffer(2)]],
                                     constant RtBounceParams& bp [[buffer(3)]],
                                     const device RtInfo* info [[buffer(4)]], rt_isect_table table [[buffer(6)]],
                                     texture2d<float, access::read> depth [[texture(0)]],
                                     texture2d<half, access::write> out_rgb [[texture(1)]],
                                     texture2d<float, access::write> out_q [[texture(2)]],
                                     texture2d<half, access::write> out_n [[texture(3)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_bounce_trace(depth, scene, table, p, bp, materials, info, out_rgb, out_q, out_n, tid);
}
// A block's bounce: the samples of the 7 x 7 blocks around on its surface
// (by how far each lies off its plane, and how its normal agrees), then
// blended with its last result where that was for a point seen along a ray
// within `tolerance` radians of this one's and at about its distance (both
// relative to the camera: a camera that turned or moved on by more than a
// block takes the fresh one, so no pattern stays on the screen).
kernel void rt_bounce_filter_cs(constant RtBounceParams& bp [[buffer(0)]],
                                texture2d<half, access::read> rgb [[texture(0)]],
                                texture2d<float, access::read> qt [[texture(1)]],
                                texture2d<half, access::read> nt [[texture(2)]],
                                texture2d<half, access::read> hist [[texture(3)]],
                                texture2d<float, access::read> hist_q [[texture(4)]],
                                texture2d<half, access::write> out [[texture(5)]],
                                texture2d<float, access::write> out_q [[texture(6)]],
                                uint2 b [[thread_position_in_grid]]) {
  if (any(b >= bp.blocks)) return;
  if (qt.read(b).w < 0.5) {  // (the sky)
    out.write(half4(0.0h, 0.0h, 0.0h, 1.0h), b);
    out_q.write(float4(0.0), b);
    return;
  }
  const float3 q = qt.read(b).xyz, n = float3(nt.read(b).xyz);
  float4 sum = float4(0.0);
  float ws = 0.0, wm = 0.0;
  for (int dy = -3; dy <= 3; dy++)
    for (int dx = -3; dx <= 3; dx++) {
      const int2 s = int2(b) + int2(dx, dy);
      if (any(s < 0) || any(uint2(s) >= bp.blocks)) continue;
      const float4 qs = qt.read(uint2(s));
      if (qs.w < 0.5) continue;
      const half4 cs = rgb.read(uint2(s));
      const float3 dv = qs.xyz - q;
      const float len = length(dv);
      const float z = len > 0.0 ? saturate(1.0 - 4.0 * abs(dot(dv, n)) / len) : 1.0;
      const float nn = saturate(dot(n, float3(nt.read(uint2(s)).xyz)));
      const float w = z * z * nn * nn * exp(-float(dx * dx + dy * dy) / 8.0);
      sum += w * float4(cs);
      ws += w;
      if (qs.w > 1.5) wm += w;
    }
  float4 f = ws > 0.0 ? sum / ws : float4(0.0, 0.0, 0.0, 1.0);
  // Where few of the blocks around lie on this one's surface (leaves,
  // grass, small parts: no plane to average over), the samples cannot be
  // smoothed and show as grain: faded to none (no bounce, the sky open) by
  // how little weight the others bring (the centre's own is 1).
  f = mix(float4(0.0, 0.0, 0.0, 1.0), f, saturate((ws - 1.0) / RT_BOUNCE_SUPPORT));
  const float share = ws > 0.0 ? wm / ws : 0.0;  // (of what moves: rt_bounce_blend)
  // The last frame's result where this point was on its screen (the
  // camera's turn and move undone), kept where the point there was this
  // one (relative to the last camera: this one less eye_delta).
  if (bp.blend < 1.0 && bp.history != 0u) {
    const float4 c = q.x * bp.prev[0] + q.y * bp.prev[1] + q.z * bp.prev[2] + bp.prev[3];
    if (c.w > 0.0) {
      const float2 ndc = c.xy / c.w;
      const float2 px = float2((ndc.x * 0.5 + 0.5) * float(bp.size.x), (0.5 - ndc.y * 0.5) * float(bp.size.y));
      const int2 pb = int2(floor(px / float(RT_BOUNCE_BLOCK)));
      if (all(pb >= 0) && all(uint2(pb) < bp.blocks)) {
        const float4 hq = hist_q.read(uint2(pb));
        const float3 was = q + bp.eye_delta.xyz;
        if (hq.w > 0.5 && distance(hq.xyz, was) < 0.05 * length(q))
          f = mix(float4(hist.read(uint2(pb))), f, rt_bounce_blend(bp.blend, share, hq.w - 1.0));
      }
    }
  }
  out.write(half4(f), b);
  out_q.write(float4(q, 1.0 + share), b);
}
// The output's texel gid (full-size pixel 2 gid): the filtered blocks
// around it on its plane.
template <typename D>
void rt_bounce_up(D t, constant RtReflectionParams& p, constant RtBounceParams& bp,
                  texture2d<half, access::read> filtered, texture2d<float, access::read> qt,
                  texture2d<half, access::read> nt, texture2d<half, access::write> out, uint2 gid) {
  constant RtShadowParams& g = p.geo;
  if (any(gid * 2u >= g.size + 1u)) return;
  const uint2 c = min(gid * 2u, g.size - 1u);
  const float v = rt_depth(t, c);
  if (rt_sky(g, v)) {
    out.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
    return;
  }
  const float3 q = rt_point(g, int2(c), v);
  const int2 b0 = int2(c / RT_BOUNCE_BLOCK);
  float4 sum = float4(0.0);
  float ws = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      const int2 b = b0 + int2(dx, dy);
      if (any(b < 0) || any(uint2(b) >= bp.blocks)) continue;
      const float4 qb = qt.read(uint2(b));
      if (qb.w < 0.5) continue;
      // (A block of something at another distance, a branch seen through
      // leaves, does not colour this texel.)
      const float lq = length(q);
      if (abs(length(qb.xyz) - lq) > 0.03 * lq) continue;
      const half4 f = filtered.read(uint2(b));
      const float3 n = float3(nt.read(uint2(b)).xyz);
      const float3 dv = qb.xyz - q;
      const float len = length(dv);
      const float z = len > 0.0 ? saturate(1.0 - 5.0 * abs(dot(dv, n)) / len) : 1.0;
      const float w = (dx == 0 && dy == 0 ? 1.0 : (dx == 0 || dy == 0) ? 0.6 : 0.36) * z * z;
      sum += w * float4(f);
      ws += w;
    }
  out.write(half4(ws > 0.0 ? sum / ws : float4(0.0, 0.0, 0.0, 1.0)), gid);
}
kernel void rt_bounce_up_depth_cs(constant RtReflectionParams& p [[buffer(1)]],
                                  constant RtBounceParams& bp [[buffer(3)]],
                                  depth2d<float, access::read> depth [[texture(0)]],
                                  texture2d<half, access::read> filtered [[texture(1)]],
                                  texture2d<float, access::read> qt [[texture(2)]],
                                  texture2d<half, access::read> nt [[texture(3)]],
                                  texture2d<half, access::write> out [[texture(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  rt_bounce_up(depth, p, bp, filtered, qt, nt, out, gid);
}
kernel void rt_bounce_up_color_cs(constant RtReflectionParams& p [[buffer(1)]],
                                  constant RtBounceParams& bp [[buffer(3)]],
                                  texture2d<float, access::read> depth [[texture(0)]],
                                  texture2d<half, access::read> filtered [[texture(1)]],
                                  texture2d<float, access::read> qt [[texture(2)]],
                                  texture2d<half, access::read> nt [[texture(3)]],
                                  texture2d<half, access::write> out [[texture(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  rt_bounce_up(depth, p, bp, filtered, qt, nt, out, gid);
}
// ---- the same two passes with each block read once (EncodeBounce takes
// these unless RECOMP_RT_BOUNCE_TILED=0: then the two kernels above)
//
// rt_bounce_filter_cs reads the point, colour and normal of its 49 blocks
// for each block: up to 147 reads a thread, nearly all of them the ones its
// neighbours make. Here a threadgroup of RT_BOUNCE_GROUP x RT_BOUNCE_GROUP
// blocks first loads its blocks and the RT_BOUNCE_APRON around them (a tile
// of 14 x 14: at most 588 reads for 64 threads, about 9 a thread) into
// threadgroup memory, and each thread filters from the tile. Dispatched as
// whole groups (the edge's threads off the grid load too).
constant uint RT_BOUNCE_GROUP = 8u, RT_BOUNCE_APRON = 3u;
constant uint RT_BOUNCE_TILE = 14u;  // (RT_BOUNCE_GROUP + 2 RT_BOUNCE_APRON)
// A tap's weight by its place, exp(-(dx dx + dy dy) / 8) at the index
// dx dx + dy dy (0 to 18; the nearest float): no exp a tap.
constant float RT_BOUNCE_SPATIAL[19] = {1.0,        0.8824969,  0.7788008,  0.6872893,  0.60653067,
                                        0.53526145, 0.47236654, 0.416862,   0.36787945, 0.32465246,
                                        0.2865048,  0.2528396,  0.22313017, 0.19691168, 0.17377394,
                                        0.15335497, 0.13533528, 0.11943297, 0.10539922};
kernel void rt_bounce_filter_tile_cs(constant RtBounceParams& bp [[buffer(0)]],
                                     texture2d<half, access::read> rgb [[texture(0)]],
                                     texture2d<float, access::read> qt [[texture(1)]],
                                     texture2d<half, access::read> nt [[texture(2)]],
                                     texture2d<half, access::read> hist [[texture(3)]],
                                     texture2d<float, access::read> hist_q [[texture(4)]],
                                     texture2d<half, access::write> out [[texture(5)]],
                                     texture2d<float, access::write> out_q [[texture(6)]],
                                     uint2 b [[thread_position_in_grid]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint2 lt [[thread_position_in_threadgroup]],
                                     uint li [[thread_index_in_threadgroup]]) {
  threadgroup float4 tq[RT_BOUNCE_TILE * RT_BOUNCE_TILE];
  threadgroup half4 tc[RT_BOUNCE_TILE * RT_BOUNCE_TILE], tn[RT_BOUNCE_TILE * RT_BOUNCE_TILE];
  // Each thread's share of the tile (196 entries by 64 threads: 3 or 4
  // each). A block off the grid is kept as one of sky (w 0), so the taps'
  // one test of w below leaves out what the kernel above leaves out by its
  // two; the colour and normal of such a block are not read.
  const int2 origin = int2(tg * RT_BOUNCE_GROUP) - int(RT_BOUNCE_APRON);
  for (uint e = li; e < RT_BOUNCE_TILE * RT_BOUNCE_TILE; e += RT_BOUNCE_GROUP * RT_BOUNCE_GROUP) {
    const int2 s = origin + int2(e % RT_BOUNCE_TILE, e / RT_BOUNCE_TILE);
    float4 qs = float4(0.0);
    half4 cs = half4(0.0h), ns = half4(0.0h);
    if (all(s >= 0) && all(uint2(s) < bp.blocks)) {
      qs = qt.read(uint2(s));
      if (!(qs.w < 0.5)) {
        cs = rgb.read(uint2(s));
        ns = nt.read(uint2(s));
      }
    }
    tq[e] = qs;
    tc[e] = cs;
    tn[e] = ns;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (any(b >= bp.blocks)) return;
  const uint at = (lt.y + RT_BOUNCE_APRON) * RT_BOUNCE_TILE + lt.x + RT_BOUNCE_APRON;
  const float4 qc = tq[at];
  if (qc.w < 0.5) {  // (the sky)
    out.write(half4(0.0h, 0.0h, 0.0h, 1.0h), b);
    out_q.write(float4(0.0), b);
    return;
  }
  // The taps in the kernel above's order, each weight by its expression:
  // the same sums, but for the table's exp against the GPU's (a last bit).
  const float3 q = qc.xyz, n = float3(tn[at].xyz);
  float4 sum = float4(0.0);
  float ws = 0.0, wm = 0.0;
  for (int dy = -3; dy <= 3; dy++)
    for (int dx = -3; dx <= 3; dx++) {
      const uint e = uint(int(at) + dy * int(RT_BOUNCE_TILE) + dx);
      const float4 qs = tq[e];
      if (qs.w < 0.5) continue;
      const half4 cs = tc[e];
      const float3 dv = qs.xyz - q;
      const float len = length(dv);
      const float z = len > 0.0 ? saturate(1.0 - 4.0 * abs(dot(dv, n)) / len) : 1.0;
      const float nn = saturate(dot(n, float3(tn[e].xyz)));
      const float w = z * z * nn * nn * RT_BOUNCE_SPATIAL[dx * dx + dy * dy];
      sum += w * float4(cs);
      ws += w;
      if (qs.w > 1.5) wm += w;
    }
  // (From here as rt_bounce_filter_cs: the fade where few blocks bring
  // weight, then the last frame's result where this point was.)
  float4 f = ws > 0.0 ? sum / ws : float4(0.0, 0.0, 0.0, 1.0);
  f = mix(float4(0.0, 0.0, 0.0, 1.0), f, saturate((ws - 1.0) / RT_BOUNCE_SUPPORT));
  const float share = ws > 0.0 ? wm / ws : 0.0;  // (of what moves: rt_bounce_blend)
  if (bp.blend < 1.0 && bp.history != 0u) {
    const float4 c = q.x * bp.prev[0] + q.y * bp.prev[1] + q.z * bp.prev[2] + bp.prev[3];
    if (c.w > 0.0) {
      const float2 ndc = c.xy / c.w;
      const float2 px = float2((ndc.x * 0.5 + 0.5) * float(bp.size.x), (0.5 - ndc.y * 0.5) * float(bp.size.y));
      const int2 pb = int2(floor(px / float(RT_BOUNCE_BLOCK)));
      if (all(pb >= 0) && all(uint2(pb) < bp.blocks)) {
        const float4 hq = hist_q.read(uint2(pb));
        const float3 was = q + bp.eye_delta.xyz;
        if (hq.w > 0.5 && distance(hq.xyz, was) < 0.05 * length(q))
          f = mix(float4(hist.read(uint2(pb))), f, rt_bounce_blend(bp.blend, share, hq.w - 1.0));
      }
    }
  }
  out.write(half4(f), b);
  out_q.write(float4(q, 1.0 + share), b);
}
// The output's 2 x 2 texels of block bid: rt_bounce_up's texels gid whose
// pixel 2 gid lies in the block, so all four take the same 9 blocks, read
// here once for the four (27 reads a block for 108). Each texel's sum is
// made as rt_bounce_up makes it, tap by tap in its order. Only where the
// depth's size is the one the blocks were laid over (EncodeBounce checks
// g.size): then 2 gid is never clamped and its block is bid.
template <typename D>
void rt_bounce_up_block(D t, constant RtReflectionParams& p, constant RtBounceParams& bp,
                        texture2d<half, access::read> filtered, texture2d<float, access::read> qt,
                        texture2d<half, access::read> nt, texture2d<half, access::write> out, uint2 bid) {
  if (any(bid >= bp.blocks)) return;
  constant RtShadowParams& g = p.geo;
  const uint2 os = uint2(out.get_width(), out.get_height());
  // The four depths first: a sky texel is written at once, and a block of
  // nothing else reads no block (as a sky texel read none).
  float v4[4];
  bool live[4];
  bool some = false;
  for (uint k = 0u; k < 4u; k++) {
    const uint2 gid = bid * 2u + uint2(k & 1u, k >> 1u);
    live[k] = false;
    v4[k] = 0.0;
    // (The last block's texels may lie past the output's edge.)
    if (any(gid >= os) || any(gid * 2u >= g.size + 1u)) continue;
    v4[k] = rt_depth(t, min(gid * 2u, g.size - 1u));
    if (rt_sky(g, v4[k])) {
      out.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
      continue;
    }
    live[k] = true;
    some = true;
  }
  if (!some) return;
  // The blocks around, k = 3 (dy + 1) + dx + 1: whether one is there (on
  // the grid, not sky), its point, that point's distance, its filtered
  // result and its normal.
  bool on[9];
  float3 qb[9], nb[9];
  float lb[9];
  half4 fb[9];
  for (int k = 0; k < 9; k++) {
    const int2 b = int2(bid) + int2(k % 3 - 1, k / 3 - 1);
    on[k] = false;
    qb[k] = float3(0.0);
    nb[k] = float3(0.0);
    lb[k] = 0.0;
    fb[k] = half4(0.0h);
    if (any(b < 0) || any(uint2(b) >= bp.blocks)) continue;
    const float4 qs = qt.read(uint2(b));
    if (qs.w < 0.5) continue;
    on[k] = true;
    qb[k] = qs.xyz;
    lb[k] = length(qs.xyz);
    fb[k] = filtered.read(uint2(b));
    nb[k] = float3(nt.read(uint2(b)).xyz);
  }
  for (uint k = 0u; k < 4u; k++) {
    if (!live[k]) continue;
    const uint2 gid = bid * 2u + uint2(k & 1u, k >> 1u);
    const uint2 c = min(gid * 2u, g.size - 1u);
    const float3 q = rt_point(g, int2(c), v4[k]);
    const float lq = length(q);
    float4 sum = float4(0.0);
    float ws = 0.0;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        const int i = (dy + 1) * 3 + dx + 1;
        if (!on[i]) continue;
        // (A block of something at another distance: rt_bounce_up's test.)
        if (abs(lb[i] - lq) > 0.03 * lq) continue;
        const float3 dv = qb[i] - q;
        const float len = length(dv);
        const float z = len > 0.0 ? saturate(1.0 - 5.0 * abs(dot(dv, nb[i])) / len) : 1.0;
        const float w = (dx == 0 && dy == 0 ? 1.0 : (dx == 0 || dy == 0) ? 0.6 : 0.36) * z * z;
        sum += w * float4(fb[i]);
        ws += w;
      }
    out.write(half4(ws > 0.0 ? sum / ws : float4(0.0, 0.0, 0.0, 1.0)), gid);
  }
}
kernel void rt_bounce_up_block_depth_cs(constant RtReflectionParams& p [[buffer(1)]],
                                        constant RtBounceParams& bp [[buffer(3)]],
                                        depth2d<float, access::read> depth [[texture(0)]],
                                        texture2d<half, access::read> filtered [[texture(1)]],
                                        texture2d<float, access::read> qt [[texture(2)]],
                                        texture2d<half, access::read> nt [[texture(3)]],
                                        texture2d<half, access::write> out [[texture(4)]],
                                        uint2 bid [[thread_position_in_grid]]) {
  rt_bounce_up_block(depth, p, bp, filtered, qt, nt, out, bid);
}
kernel void rt_bounce_up_block_color_cs(constant RtReflectionParams& p [[buffer(1)]],
                                        constant RtBounceParams& bp [[buffer(3)]],
                                        texture2d<float, access::read> depth [[texture(0)]],
                                        texture2d<half, access::read> filtered [[texture(1)]],
                                        texture2d<float, access::read> qt [[texture(2)]],
                                        texture2d<half, access::read> nt [[texture(3)]],
                                        texture2d<half, access::write> out [[texture(4)]],
                                        uint2 bid [[thread_position_in_grid]]) {
  rt_bounce_up_block(depth, p, bp, filtered, qt, nt, out, bid);
}
)";

std::unique_ptr<RtReflector> RtReflector::Make(id<MTLDevice> dev) {
  if (!RtSupported(dev)) return nullptr;
  auto m = std::make_unique<RtReflector>();
  m->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib =
      [dev newLibraryWithSource:@((std::string(kRtMsl) + kRtShadowMsl + kRtReflectRayMsl + kRtReflectionMsl + kRtBounceMsl).c_str())
                        options:nil
                          error:&err];
  auto plain = [&](NSString* name) -> id<MTLComputePipelineState> {
    if (!lib || err) return nil;
    return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
  };
  // The tracing kernels, linked with the alpha test (an intersection
  // function) and each given its function table: entry 0, the one every
  // instance names (Apple's "Accelerating ray tracing using Metal" sample:
  // a table is specific to the pipeline that made it).
  id<MTLFunction> alpha = lib ? [lib newFunctionWithName:@"rt_reflect_alpha_if"] : nil;
  auto traced = [&](NSString* name, id<MTLIntersectionFunctionTable> __strong* table) -> id<MTLComputePipelineState> {
    if (!lib || !alpha || err) return nil;
    MTLLinkedFunctions* linked = [[MTLLinkedFunctions alloc] init];
    linked.functions = @[ alpha ];
    MTLComputePipelineDescriptor* pd = [[MTLComputePipelineDescriptor alloc] init];
    pd.computeFunction = [lib newFunctionWithName:name];
    pd.linkedFunctions = linked;
    pd.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;  // (kListGroup threads)
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithDescriptor:pd
                                                                        options:MTLPipelineOptionNone
                                                                     reflection:nil
                                                                          error:&err];
    if (!ps) return nil;
    MTLIntersectionFunctionTableDescriptor* td = [[MTLIntersectionFunctionTableDescriptor alloc] init];
    td.functionCount = 1;
    *table = [ps newIntersectionFunctionTableWithDescriptor:td];
    if (!*table) return nil;
    [*table setFunction:[ps functionHandleWithFunction:alpha] atIndex:0];
    return ps;
  };
  m->classify_pipe_[0] = plain(@"rt_reflect_classify_color_cs");
  m->classify_pipe_[1] = plain(@"rt_reflect_classify_depth_cs");
  m->trace_pipe_[0] = traced(@"rt_reflect_trace_color_cs", &m->table_[0]);
  m->trace_pipe_[1] = traced(@"rt_reflect_trace_depth_cs", &m->table_[1]);
  m->ssr_pipe_[0] = plain(@"rt_reflect_ssr_color_cs");
  m->ssr_pipe_[1] = plain(@"rt_reflect_ssr_depth_cs");
  m->keep_pipe_ = plain(@"rt_reflect_keep_cs");
  m->fill_pipe_[0] = plain(@"rt_reflect_fill_color_cs");
  m->fill_pipe_[1] = plain(@"rt_reflect_fill_depth_cs");
  m->args_pipe_ = plain(@"rt_reflect_args_cs");
  m->view_pipe_ = plain(@"rt_reflect_view_cs");
  bool ok = m->args_pipe_ && m->view_pipe_ && m->keep_pipe_;
  for (int k = 0; k < 2; k++)
    ok = ok && m->classify_pipe_[k] && m->trace_pipe_[k] && m->table_[k] && m->fill_pipe_[k] && m->ssr_pipe_[k];
  if (!ok) {
    std::fprintf(stderr, "gfx: ray-traced reflections: %s\n",
                 err ? err.localizedDescription.UTF8String : "an intersection function table");
    return nullptr;
  }
  // The bounce's passes (EncodeBounce): without them, no bounce.
  m->bounce_trace_[0] = traced(@"rt_bounce_trace_color_cs", &m->bounce_table_[0]);
  m->bounce_trace_[1] = traced(@"rt_bounce_trace_depth_cs", &m->bounce_table_[1]);
  m->bounce_filter_ = plain(@"rt_bounce_filter_cs");
  m->bounce_up_[0] = plain(@"rt_bounce_up_color_cs");
  m->bounce_up_[1] = plain(@"rt_bounce_up_depth_cs");
  if (!m->bounce_trace_[0] || !m->bounce_trace_[1] || !m->bounce_filter_ || !m->bounce_up_[0] || !m->bounce_up_[1]) {
    std::fprintf(stderr, "gfx: ray-traced bounce light: %s\n", err ? err.localizedDescription.UTF8String : "no passes");
    m->bounce_filter_ = nil;
    err = nil;
  }
  // The filter from a threadgroup's tile and the upsample by block (the
  // same results from fewer reads; kRtBounceMsl's last part), unless
  // RECOMP_RT_BOUNCE_TILED=0: then, or where one cannot be made, the
  // per-thread kernels above as before.
  const char* tiled = std::getenv("RECOMP_RT_BOUNCE_TILED");
  if (m->bounce_filter_ && !(tiled && tiled[0] == '0')) {
    m->bounce_filter_tile_ = plain(@"rt_bounce_filter_tile_cs");
    m->bounce_up_block_[0] = plain(@"rt_bounce_up_block_color_cs");
    m->bounce_up_block_[1] = plain(@"rt_bounce_up_block_depth_cs");
    if (!m->bounce_filter_tile_ || !m->bounce_up_block_[0] || !m->bounce_up_block_[1]) {
      std::fprintf(stderr, "gfx: ray-traced bounce light: the tiled passes: %s\n",
                   err ? err.localizedDescription.UTF8String : "not made");
      m->bounce_filter_tile_ = nil;
      m->bounce_up_block_[0] = m->bounce_up_block_[1] = nil;
      err = nil;
    }
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  m->miss_ = [dev newTextureWithDescriptor:td];
  if (!m->miss_) return nullptr;
  const uint16_t zero[4] = {};
  [m->miss_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:zero bytesPerRow:8];
  m->bounce_none_ = [dev newTextureWithDescriptor:td];
  if (!m->bounce_none_) return nullptr;
  const uint16_t open[4] = {0, 0, 0, 0x3C00};  // (half 1)
  [m->bounce_none_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:open bytesPerRow:8];
  m->flags_none_ = NewTexture(dev, MTLPixelFormatR8Uint, 1, 1);
  if (!m->flags_none_) return nullptr;
  const char* checker = std::getenv("RECOMP_RT_REFL_CHECKER");
  m->checker_ = !(checker && checker[0] == '0');
  m->checker_ab_ = checker && std::strcmp(checker, "ab") == 0;
  const char* diag = std::getenv("RECOMP_RT_REFL_DIAG");
  m->diag_ = diag && diag[0] == '1';
  const char* ssr = std::getenv("RECOMP_RT_REFL_SSR");
  // (Off unless asked for: measured in the game, the march settled the
  // short rays a trace does cheaply and the copy cost more than it saved.)
  m->ssr_ = ssr && (ssr[0] == '1' || std::strcmp(ssr, "ab") == 0);
  m->ssr_ab_ = ssr && std::strcmp(ssr, "ab") == 0;
  return m;
}

id<MTLTexture> RtReflector::NewOutput(uint32_t w, uint32_t h) const {
  return NewTexture(dev_, MTLPixelFormatRGBA16Float, w, h);
}

id<MTLTexture> RtReflector::NewDistance(uint32_t w, uint32_t h) const {
  return NewTexture(dev_, MTLPixelFormatR16Float, w, h);
}

id<MTLTexture> RtReflector::FlagsFor(id<MTLTexture> out) const {
  // (Kept with the output, as its lists: R8Uint, 1 where traced this frame.)
  static char key;
  id<MTLTexture> flags = objc_getAssociatedObject(out, &key);
  if (flags && flags.width == out.width && flags.height == out.height) return flags;
  flags = NewTexture(dev_, MTLPixelFormatR8Uint, uint32_t(out.width), uint32_t(out.height));
  if (flags) objc_setAssociatedObject(out, &key, flags, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return flags;
}

RtReflector::Layout RtReflector::LayoutFor(uint32_t w, uint32_t h) {
  // A: every pixel; B: every pixel too (the checkerboard fills half, but
  // the sparse rates of wet ground fill up to 15 of 16: with room for half
  // they ran into C and D's room); C and D (one room, used in turn): every
  // pixel.
  const uint32_t all = w * h;
  return Layout{kListHeader, kListHeader + all, kListHeader + all + all, 0};
}

id<MTLBuffer> RtReflector::ListFor(id<MTLTexture> out) const {
  // (Tied to the output's life: a new output, at whatever address, starts
  // with fresh lists.)
  static char key;
  id<MTLBuffer> list = objc_getAssociatedObject(out, &key);
  const Layout l = LayoutFor(uint32_t(out.width), uint32_t(out.height));
  const NSUInteger bytes = (l.c + (l.b - l.a)) * sizeof(uint32_t);
  if (list && list.length >= bytes) return list;
  list = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!list) return nil;
  // Counts and lengths 0, fresh, no threadgroups (x 1 x 1) for any list.
  uint32_t header[kListHeader] = {};
  header[2] = 1;
  for (uint32_t a : {4u, 8u, 12u, 20u}) header[a + 1] = header[a + 2] = 1;
  std::memcpy(list.contents, header, sizeof header);
  objc_setAssociatedObject(out, &key, list, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return list;
}

void RtReflector::Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> normal,
                         id<MTLTexture> surface, id<MTLTexture> stencil, id<MTLTexture> out,
                         id<MTLTexture> distance, const RtReflectionParams& p, id<MTLBuffer> counts,
                         size_t counts_offset, const RtSsrParams* ssr, id<MTLTexture> prev) const {
  if (!f || !f->tlas || !depth || !normal || !surface || !out || !distance || !counts) return;
  if (out.width > 65535 || out.height > 32767) return;  // (a pixel's coordinates: 16 and 15 bits)
  id<MTLBuffer> list = ListFor(out);
  if (!list) return;
  RtReflectionParams q = p;
  if (!stencil) q.wet_on = 0;
  const int k = q.geo.depth_texture ? 1 : 0;
  const Layout lay = LayoutFor(uint32_t(out.width), uint32_t(out.height));
  // (The header's: a list's length and indirect threadgroups.)
  struct Of {
    uint32_t length, base;
  };
  const Of of_a{1, lay.a}, of_c{15, lay.c}, of_d{17, lay.c};
  const NSUInteger args_a = 4 * sizeof(uint32_t), args_b = 8 * sizeof(uint32_t), args_c = 12 * sizeof(uint32_t),
                   args_d = 20 * sizeof(uint32_t);
  const MTLSize group = MTLSizeMake(kListGroup, 1, 1);
  const bool screen = ssr && prev;
  id<MTLTexture> flags = q.checker ? FlagsFor(out) : nil;
  if (q.checker && !flags) q.checker = 0;
  // Encoders by stage, each timed by name (gpu_timer.h), each with every
  // binding its kernels read. (Dispatches within one run in turn: each
  // sees the last one's writes, the indirect arguments included.)
  id<MTLComputeCommandEncoder> e = nil;
  auto begin = [&](const char* name) {
    e = TimedCompute(cmd, name);
    [e setBytes:&q length:sizeof q atIndex:1];
    [e setBuffer:counts offset:counts_offset atIndex:3];
    [e setBuffer:list offset:0 atIndex:5];
    [e setBytes:&lay length:sizeof lay atIndex:7];
    [e setTexture:depth atIndex:0];
    [e setTexture:normal atIndex:1];
    [e setTexture:surface atIndex:2];
    [e setTexture:stencil ? stencil : miss_ atIndex:3];
    [e setTexture:out atIndex:4];
    [e setTexture:distance atIndex:5];
    [e setTexture:flags ? flags : flags_none_ atIndex:7];
  };
  auto args = [&](uint32_t which) {
    [e setComputePipelineState:args_pipe_];
    [e setBuffer:list offset:0 atIndex:0];
    [e setBytes:&which length:sizeof which atIndex:1];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e setBytes:&q length:sizeof q atIndex:1];
  };
  auto trace = [&](const char* name, const Of& of, NSUInteger at) {
    begin(name);
    [e setComputePipelineState:trace_pipe_[k]];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
    [e setBuffer:f->info offset:f->info_offset atIndex:4];
    [e setIntersectionFunctionTable:table_[k] atBufferIndex:6];
    [e setBytes:&of length:sizeof of atIndex:8];
    RtUseResources(e, f);
    [e dispatchThreadgroupsWithIndirectBuffer:list indirectBufferOffset:at threadsPerThreadgroup:group];
    [e endEncoding];
  };
  // 1. List the pixels to reflect (A) and to fill (B), 0 written where
  // none is now but one was; 2. their numbers.
  begin("rt reflections");
  [e setComputePipelineState:classify_pipe_[k]];
  [e dispatchThreads:MTLSizeMake(q.geo.size[0], q.geo.size[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  args(0);
  if (screen) {
    // 4. Screen space first: A's march; what it leaves to D, traced.
    [e setComputePipelineState:ssr_pipe_[k]];
    [e setBytes:ssr length:sizeof *ssr atIndex:9];
    [e setTexture:prev atIndex:6];
    [e dispatchThreadgroupsWithIndirectBuffer:list indirectBufferOffset:args_a threadsPerThreadgroup:group];
    args(2);
    [e endEncoding];
    trace("rt reflections trace", of_d, args_d);
  } else {
    // 4. Trace A.
    [e endEncoding];
    trace("rt reflections trace", of_a, args_a);
  }
  if (q.checker) {
    // 5. Fill B from A; what it cannot to C, traced.
    begin("rt reflections fill");
    [e setComputePipelineState:fill_pipe_[k]];
    [e dispatchThreadgroupsWithIndirectBuffer:list indirectBufferOffset:args_b threadsPerThreadgroup:group];
    args(1);
    [e endEncoding];
    trace("rt reflections trace after", of_c, args_c);
  }
}

bool RtSsrParamsFor(const RtReflectionParams& p, const double prev_vp[16], const float prev_jitter[2],
                    uint32_t prev_w, uint32_t prev_h, float scale, RtSsrParams* out) {
  *out = RtSsrParams{};
  const RtShadowParams& g = p.geo;
  // The pixel's point h = A (x, y, v, 1) (camera-relative, homogeneous):
  // A's columns rx, ry, rv, rc; its inverse takes (q, 1) to (x, y, v, 1)
  // over h.w.
  double a[16], b[16];
  const float* cols[4] = {g.rx, g.ry, g.rv, g.rc};
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) a[r * 4 + c] = cols[c][r];
  if (!Invert4(a, b)) return false;
  // (The last row scaled to a unit gradient: the view depth in metres.)
  const double n = std::sqrt(b[12] * b[12] + b[13] * b[13] + b[14] * b[14]);
  if (!(n > 0) || !std::isfinite(n)) return false;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) out->to_clip[c * 4 + r] = float(b[r * 4 + c] / n);
  // The previous camera: clip = (eye + q, 1) M (rows).
  for (int c = 0; c < 4; c++) {
    for (int r = 0; r < 3; r++) out->to_prev[r * 4 + c] = float(prev_vp[r * 4 + c]);
    double t = prev_vp[12 + c];
    for (int r = 0; r < 3; r++) t += double(g.eye[r]) * prev_vp[r * 4 + c];
    out->to_prev[12 + c] = float(t);
  }
  out->prev_size[0] = float(prev_w);
  out->prev_size[1] = float(prev_h);
  out->prev_jitter[0] = prev_jitter ? prev_jitter[0] : 0.0f;
  out->prev_jitter[1] = prev_jitter ? prev_jitter[1] : 0.0f;
  out->thickness = kRtSsrThickness;
  out->thickness_rel = kRtSsrThicknessRel;
  out->edge = kRtSsrEdge;
  out->scale = scale;
  out->steps = kRtSsrSteps;
  out->refine = kRtSsrRefine;
  // 1 / view depth is affine in the stored depth (a perspective): from two
  // depths at the screen's centre.
  double k[2];
  const double vs[2] = {0.25, 0.75};
  for (int i = 0; i < 2; i++) {
    double h[4];
    for (int r = 0; r < 4; r++) h[r] = vs[i] * double(g.rv[r]) + double(g.rc[r]);
    if (!(std::fabs(h[3]) > 0)) return false;
    double lam = b[15];
    for (int c = 0; c < 3; c++) lam += b[12 + c] * h[c] / h[3];
    lam /= n;
    if (!(std::fabs(lam) > 0)) return false;
    k[i] = 1.0 / lam;
  }
  out->inv_depth[0] = float((k[1] - k[0]) / (vs[1] - vs[0]));
  out->inv_depth[1] = float(k[0] - out->inv_depth[0] * vs[0]);
  for (float v : out->to_clip)
    if (!std::isfinite(v)) return false;
  for (float v : out->to_prev)
    if (!std::isfinite(v)) return false;
  return prev_w && prev_h;
}

id<MTLTexture> RtReflector::NewKept(uint32_t w, uint32_t h) const {
  return NewTexture(dev_, MTLPixelFormatRG11B10Float, w, h);
}

id<MTLTexture> RtReflector::NewBounceOut(uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                width:std::max((w + 1) / 2, 1u)
                                                                               height:std::max((h + 1) / 2, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

void RtReflector::EncodeBounce(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth,
                               const RtReflectionParams& p, const RtBounceParams& bp_in, id<MTLTexture> out) const {
  if (!Bounce() || !f || !f->tlas || !depth || !out) return;
  const uint32_t w = uint32_t(depth.width), h = uint32_t(depth.height);
  if (out.width != (w + 1) / 2 || out.height != (h + 1) / 2) return;
  RtBounceParams bp = bp_in;
  bp.blocks[0] = (w + kRtBounceBlock - 1) / kRtBounceBlock;
  bp.blocks[1] = (h + kRtBounceBlock - 1) / kRtBounceBlock;
  bool fresh = false;
  if (!b_rgb_ || b_rgb_.width != bp.blocks[0] || b_rgb_.height != bp.blocks[1]) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:bp.blocks[0]
                                                                                 height:bp.blocks[1]
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    b_rgb_ = [dev_ newTextureWithDescriptor:td];
    b_n_ = [dev_ newTextureWithDescriptor:td];
    b_hist_[0] = [dev_ newTextureWithDescriptor:td];
    b_hist_[1] = [dev_ newTextureWithDescriptor:td];
    td.pixelFormat = MTLPixelFormatRGBA32Float;
    b_q_ = [dev_ newTextureWithDescriptor:td];
    b_hist_q_[0] = [dev_ newTextureWithDescriptor:td];
    b_hist_q_[1] = [dev_ newTextureWithDescriptor:td];
    fresh = true;
    if (!b_rgb_ || !b_n_ || !b_q_ || !b_hist_[0] || !b_hist_[1] || !b_hist_q_[0] || !b_hist_q_[1]) {
      b_rgb_ = nil;
      return;
    }
  }
  // (A history never written reads as anything: none taken that frame.)
  if (fresh) bp.blend = 1.0f;
  RtReflectionParams q = p;
  const int k = q.geo.depth_texture ? 1 : 0;
  const int now = b_turn_, was = 1 - b_turn_;
  b_turn_ = was;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt bounce trace");
  [e setComputePipelineState:bounce_trace_[k]];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBytes:&bp length:sizeof bp atIndex:3];
  [e setBuffer:f->info offset:f->info_offset atIndex:4];
  [e setIntersectionFunctionTable:bounce_table_[k] atBufferIndex:6];
  RtUseResources(e, f);
  [e setTexture:depth atIndex:0];
  [e setTexture:b_rgb_ atIndex:1];
  [e setTexture:b_q_ atIndex:2];
  [e setTexture:b_n_ atIndex:3];
  [e dispatchThreads:MTLSizeMake(bp.blocks[0], bp.blocks[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  // The filter from a threadgroup's tile and the upsample by block (Make:
  // RECOMP_RT_BOUNCE_TILED=0 leaves them nil, and the per-thread kernels
  // run as before). The block upsample only for a depth of the size the
  // blocks are laid over (the backend's always is): there every texel's
  // pixel lies in its thread's block.
  const bool tile = bounce_filter_tile_ != nil;
  const bool up_block = bounce_up_block_[k] != nil && q.geo.size[0] == w && q.geo.size[1] == h;
  e = TimedCompute(cmd, "rt bounce filter");
  [e setComputePipelineState:tile ? bounce_filter_tile_ : bounce_filter_];
  [e setBytes:&bp length:sizeof bp atIndex:0];
  [e setTexture:b_rgb_ atIndex:0];
  [e setTexture:b_q_ atIndex:1];
  [e setTexture:b_n_ atIndex:2];
  [e setTexture:b_hist_[was] atIndex:3];
  [e setTexture:b_hist_q_[was] atIndex:4];
  [e setTexture:b_hist_[now] atIndex:5];
  [e setTexture:b_hist_q_[now] atIndex:6];
  if (tile) {
    // (Whole groups: every thread of one loads a share of its tile.)
    [e dispatchThreadgroups:MTLSizeMake((bp.blocks[0] + kRtBounceGroup - 1) / kRtBounceGroup,
                                        (bp.blocks[1] + kRtBounceGroup - 1) / kRtBounceGroup, 1)
        threadsPerThreadgroup:MTLSizeMake(kRtBounceGroup, kRtBounceGroup, 1)];
  } else {
    [e dispatchThreads:MTLSizeMake(bp.blocks[0], bp.blocks[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  }
  // The upsample under its own name where passes are timed ('rt bounce up'
  // beside 'rt bounce filter' in the report, which had both under the
  // filter's); in the filter's encoder otherwise, as it was.
  if (CurrentGpuTimer().next) {
    [e endEncoding];
    e = TimedCompute(cmd, "rt bounce up");
  }
  [e setComputePipelineState:up_block ? bounce_up_block_[k] : bounce_up_[k]];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBytes:&bp length:sizeof bp atIndex:3];
  [e setTexture:depth atIndex:0];
  [e setTexture:b_hist_[now] atIndex:1];
  [e setTexture:b_q_ atIndex:2];
  [e setTexture:b_n_ atIndex:3];
  [e setTexture:out atIndex:4];
  if (up_block) {
    // (A thread a block: 8 x 8 blocks are the 16 x 16 texels of a group below.)
    [e dispatchThreads:MTLSizeMake(bp.blocks[0], bp.blocks[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  } else {
    [e dispatchThreads:MTLSizeMake(out.width, out.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  }
  [e endEncoding];
}

void RtReflector::EncodeKeep(id<MTLCommandBuffer> cmd, id<MTLTexture> lit, id<MTLTexture> kept) const {
  if (!lit || !kept) return;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt reflections keep");
  [e setComputePipelineState:keep_pipe_];
  [e setTexture:lit atIndex:0];
  [e setTexture:kept atIndex:1];
  [e dispatchThreads:MTLSizeMake(kept.width, kept.height, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
}

void RtReflector::EncodeView(id<MTLCommandBuffer> cmd, id<MTLTexture> out, id<MTLTexture> distance, float scale,
                             id<MTLTexture> target) const {
  if (!out || !distance || !target) return;
  const NSUInteger w = target.width, h = target.height;
  if (!view_tex_ || view_tex_.width != w || view_tex_.height != h || view_tex_.pixelFormat != target.pixelFormat) {
    // (The command buffers that used the one before hold it.)
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:target.pixelFormat
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderWrite;
    view_tex_ = [dev_ newTextureWithDescriptor:td];
    if (!view_tex_) return;
  }
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt reflection view");
  [e setComputePipelineState:view_pipe_];
  [e setTexture:out atIndex:0];
  [e setTexture:distance atIndex:1];
  [e setTexture:view_tex_ atIndex:2];
  [e setBytes:&scale length:sizeof scale atIndex:0];
  [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder];
  [bl copyFromTexture:view_tex_ toTexture:target];
  [bl endEncoding];
}

}  // namespace recomp::gfx
