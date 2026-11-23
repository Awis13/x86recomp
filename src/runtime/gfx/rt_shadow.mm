// Ray-traced shadow masks (rt_shadow.h).

#import "runtime/gfx/rt_shadow.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"

namespace recomp::gfx {

// The kernel (after kRtMsl): a thread a pixel of the depth buffer.
const char* const kRtShadowMsl = R"(
struct RtShadowParams {
  float4 rx, ry, rv, rc;  // the pixel's point relative to eye: h = x rx + y ry + v rv + rc, h.xyz / h.w
  float4 eye;             // the camera's position (world)
  float4 light;           // xyz toward the light (unit), w the ray's length
  uint2 size;
  float2 jitter;          // the scene's offset on screen (pixels, right and down)
  uint depth_texture;
  float sky;              // the far plane's stored value (0 or 1)
  float pixel;            // the angle between neighbouring pixels' rays
  float offset_pixels, offset_view, offset_world;
  float cone;             // half-angle of the light's cone (0: hard shadows, rays straight at it)
  uint diag;              // 0, or the blocker counts kept (RtShadowDiag)
  float captured_gap;     // captured instances block beyond this fraction of the distance
  uint frame;             // with a cone: the frame's number
  uint normals;           // 1: the surface's normal from the normal texture (xyz as unorm)
  float min_slope;        // the light step's N.L floor
};
struct RtLightParams {
  float4 position;        // xyz relative to eye; w 1 / radius^2
  float4 direction;       // xyz the way a spot light shines; w the cone's offset
  float cone_scale;
  uint cone;
  float lamp_gap;
  uint local;             // 1: a local light (position, not light)
  uint2 origin, extent;   // the pixels traced
};
struct RtShadowCascade {
  float4 view, splits;
  float4 select[4];
  float4 matrix[4];
  float range, max_texel;
  float2 atlas;
  uint on;
  float rounding_rel, rounding_abs;
  float shadow_gap;
};
struct RtShadowDiag {
  float4 point, normal, origin;
  float distance;
  uint instance, primitive, state;
};
constant uint RT_SHADOW_INFO_CAPTURED = 1u << 17;  // kRtInfoCaptured
constant float RT_CASCADE_MIN_SLOPE = 0.2;  // kRtCascadeMinSlope
// The shadow rays' alpha test: Metal's intersector (not an intersection
// query, which Apple advises against on Apple family 9: it adds scratch
// memory traffic and turns off the reorder stage) calls this intersection
// function for a non-opaque instance's candidate triangle (its primitive
// data, barycentrics, user ID); the material table comes in the payload.
struct RtShadowPayload {
  const device RtMaterial* materials;
};
// kRtMsl's RtTexelAlpha where the texel is bilinear at level 0, addressed
// the same way on both axes by WRAP, MIRROR or CLAMP: the four texels in
// one gather (the sampler's addressing as D3D's: repeat, mirrored_repeat,
// clamp_to_edge), weighted as there. Elsewhere RtTexelAlpha itself.
#define RT_SHADOW_GATHER(C)                                                                                    \
  float4 rt_shadow_gather_##C(texture2d<float> t, float2 at, uint mode) {                                      \
    constexpr sampler wrap(address::repeat), mirror(address::mirrored_repeat), clamp(address::clamp_to_edge);  \
    if (mode == 3u) return t.gather(clamp, at, int2(0), component::C);                                         \
    if (mode == 2u) return t.gather(mirror, at, int2(0), component::C);                                        \
    return t.gather(wrap, at, int2(0), component::C);                                                          \
  }
RT_SHADOW_GATHER(x)
RT_SHADOW_GATHER(y)
RT_SHADOW_GATHER(z)
RT_SHADOW_GATHER(w)
float rt_shadow_texel(const device RtMaterial& m, float2 uv) {
  const texture2d<float> t = m.alpha_texture;
  const uint ch = (m.flags >> 24) & 3u, mu = (m.flags >> 26) & 7u, mv = (m.flags >> 29) & 7u;
  const uint lvl = min(uint(max(m.max_level, 0.0)), t.get_num_mip_levels() - 1u);
  if ((m.flags & RT_MAT_LINEAR) == 0u || mu != mv || mu > 3u || lvl != 0u) return RtTexelAlpha(m, uv);
  const float2 size = float2(t.get_width(), t.get_height());
  const float2 c = uv * size - 0.5;
  if (!all(abs(c) < 65536.0)) return RtTexelAlpha(m, uv);
  const float2 f = floor(c), fr = c - f;
  // At the corner the four texels share: the footprint is exactly theirs.
  const float2 at = (f + 1.0) / size;
  float4 a;
  switch (ch) {
    case 0u: a = rt_shadow_gather_x(t, at, mu); break;
    case 1u: a = rt_shadow_gather_y(t, at, mu); break;
    case 2u: a = rt_shadow_gather_z(t, at, mu); break;
    default: a = rt_shadow_gather_w(t, at, mu); break;
  }
  // (x: texel (f.x, f.y + 1), y: (f.x + 1, f.y + 1), z: (f.x + 1, f.y), w: (f.x, f.y).)
  return mix(mix(a.w, a.z, fr.x), mix(a.x, a.y, fr.x), fr.y);
}
// kRtMsl's RtAlphaPass, the texel by rt_shadow_texel.
bool rt_shadow_alpha_pass(const device RtMaterial& m, const device void* prim, float2 b) {
  if ((m.flags & RT_MAT_CUT) == 0u) return true;
  const float3 w = float3(1.0 - b.x - b.y, b.x, b.y);
  float2 uv;
  uint alpha;
  if ((m.flags & RT_MAT_HIT_PRIM) != 0u) {
    const device RtHitPrim& p = *(const device RtHitPrim*)prim;
    uv = rt_hit_uv(p, w);
    alpha = p.alpha;
  } else {
    const device RtCutPrim& p = *(const device RtCutPrim*)prim;
    uv = float2(p.uv[0]) * w.x + float2(p.uv[1]) * w.y + float2(p.uv[2]) * w.z;
    alpha = p.alpha;
  }
  float v = 1.0;
  if ((m.flags & RT_MAT_TEXTURE) != 0u) v = rt_shadow_texel(m, uv);
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
[[intersection(triangle, triangle_data, instancing)]]
bool rt_shadow_alpha(float2 b [[barycentric_coord]], uint id [[user_instance_id]], uint geo [[geometry_id]],
                     const device void* prim [[primitive_data]], ray_data RtShadowPayload& pl [[payload]]) {
  // (The material's slot: the instance's user ID plus the geometry's, an
  // actor's BLAS having one a part: rt_scene.h's AddCaptured.)
  return rt_shadow_alpha_pass(pl.materials[id + geo], prim, b);
}
typedef intersection_function_table<triangle_data, instancing> RtShadowTable;
// Whether the depth's point `surface` (world) lies on a captured instance
// (a person, a vehicle): one met along the camera's line through it (unit
// `view`, from the camera) within `gap` either side.
bool rt_on_captured(instance_acceleration_structure scene, RtShadowTable table, float3 surface, float3 view, float gap,
                    thread RtShadowPayload& pl) {
  intersector<triangle_data, instancing> it;
  it.assume_geometry_type(geometry_type::triangle);
  it.accept_any_intersection(true);
  const uint moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  return gap > 0.0 &&
         it.intersect(ray(surface - view * gap, view, 0.0, 2.0 * gap), scene, moving, table, pl).type !=
             intersection_type::none;
}
// Whether the light along unit l is blocked from o to tmax (rt_shadowed's
// answer: static instances from o on, captured ones beyond `gap`), with one
// traversal of the scene for most rays: any instance from o on. Only where
// what it found is a captured instance within the gap is the depth's point
// (`surface`, seen along unit `view`) tested: on a captured instance
// itself, whose own triangles are not quite those the depth was drawn from,
// it is traced again as rt_shadowed does (`again`); on anything else (the
// ground at a person's feet, under a car) the captured instance blocks
// from o on, so that the shadow reaches what it stands on.
bool rt_shadow_blocked(instance_acceleration_structure scene, RtShadowTable table, float3 o, float3 l, float tmax,
                       float gap, float3 surface, float3 view, const device RtMaterial* materials,
                       const device RtInfo* info, thread bool& again, uint masks = 15u) {
  intersector<triangle_data, instancing> it;
  it.assume_geometry_type(geometry_type::triangle);
  it.accept_any_intersection(true);
  RtShadowPayload pl;
  pl.materials = materials;
  const uint fixed = masks & (RT_MASK_OPAQUE | RT_MASK_CUT), moving = masks & (RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT);
  const auto h = it.intersect(ray(o, l, 0.0, tmax), scene, fixed | moving, table, pl);
  if (h.type == intersection_type::none) return false;
  if ((info[h.user_instance_id + h.geometry_id].flags & RT_SHADOW_INFO_CAPTURED) == 0u || h.distance >= gap)
    return true;
  again = true;
  if (!rt_on_captured(scene, table, surface, view, gap, pl)) return true;
  if (it.intersect(ray(o, l, 0.0, tmax), scene, fixed, table, pl).type != intersection_type::none) return true;
  return gap < tmax && it.intersect(ray(o, l, gap, tmax), scene, moving, table, pl).type != intersection_type::none;
}
// The radical inverse of i in base b (a Halton sequence's coordinate).
float rt_halton(uint i, uint b) {
  float f = 1.0, r = 0.0;
  for (; i > 0u; i /= b) {
    f /= float(b);
    r += f * float(i % b);
  }
  return r;
}
// Two numbers in [0, 1) from a pixel (a 32-bit integer hash of each axis).
float2 rt_pixel_hash(uint2 c) {
  uint h = c.x * 0x8da6b343u ^ c.y * 0xd8163841u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return float2(float(h & 0xFFFFu), float(h >> 16)) / 65536.0;
}
float3 rt_cone_dir(float3 l, float cone, uint2 c, uint frame) {
  if (!(cone > 0.0)) return l;
  const float2 u = fract(float2(rt_halton(frame % 64u + 1u, 2u), rt_halton(frame % 64u + 1u, 3u)) + rt_pixel_hash(c));
  // Uniform over the cap: cos of the angle from l uniform in [cos(cone), 1].
  const float ca = 1.0 - u.x * (1.0 - cos(cone)), sa = sqrt(max(0.0, 1.0 - ca * ca));
  const float phi = 6.28318530718 * u.y;
  const float3 a = abs(l.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
  const float3 t = normalize(cross(l, a)), b = cross(l, t);
  return normalize(ca * l + sa * (cos(phi) * t + sin(phi) * b));
}
float rt_depth(depth2d<float, access::read> t, uint2 c) { return t.read(c); }
float rt_depth(texture2d<float, access::read> t, uint2 c) { return t.read(c).r; }
bool rt_sky(constant RtShadowParams& p, float v) { return isnan(v) || (p.sky < 0.5 ? v <= p.sky : v >= p.sky); }
// Pixel c's point (stored value v), relative to the camera: at D3D9's
// pixel centre (integral coordinates), less the jitter.
float3 rt_point(constant RtShadowParams& p, int2 c, float v) {
  const float x = 2.0 * (float(c.x) - p.jitter.x) / float(p.size.x) - 1.0;
  const float y = 1.0 - 2.0 * (float(c.y) - p.jitter.y) / float(p.size.y);
  const float4 h = x * p.rx + y * p.ry + v * p.rv + p.rc;
  return h.xyz / h.w;
}
bool rt_inside(constant RtShadowParams& p, int2 n) {
  return n.x >= 0 && n.y >= 0 && n.x < int(p.size.x) && n.y < int(p.size.y);
}
// The step from q (pixel c's point, stored value v) to the neighbour at
// c + o or from the one at c - o: the one on q's surface. A plane's stored
// depth is affine across the screen, so on each side the neighbour and the
// one past it predict v (2 v1 - v2); the side that predicts it better is on
// the same plane (at a crease or a depth edge the other is not). Where a
// side has no second neighbour, the one whose point is nearer q. False if
// neither neighbour is in the buffer and not sky.
template <typename D>
bool rt_step(D t, constant RtShadowParams& p, int2 c, int2 o, float v, float3 q, thread float3& out) {
  bool have = false;
  float best_err = INFINITY, best_len = INFINITY;
  for (int s = -1; s <= 1; s += 2) {
    const int2 n = c + s * o;
    if (!rt_inside(p, n)) continue;
    const float v1 = rt_depth(t, uint2(n));
    if (rt_sky(p, v1)) continue;
    float err = INFINITY;
    const int2 n2 = n + s * o;
    if (rt_inside(p, n2)) {
      const float v2 = rt_depth(t, uint2(n2));
      if (!rt_sky(p, v2)) err = abs(2.0 * v1 - v2 - v);
    }
    const float3 d = (rt_point(p, n, v1) - q) * float(s);
    const float l = length_squared(d);
    if (err < best_err || (err == best_err && l < best_len)) {
      best_err = err;
      best_len = l;
      out = d;
      have = true;
    }
  }
  return have;
}
// A pixel's shadow ray (rt_shadow_ray): its point q (relative to the
// camera), the normal it starts along (n, turned to the light's side), the
// start (relative to the camera), the light's direction l and the ray's
// end, the point's distance from the camera and the offset, and N.L
// before the normal was turned (below 0: the surface faces away).
struct RtShadowRay {
  float3 q, n, start, l;
  float tmax, dist, off, nl;
};
// Pixel gid's ray (depth stored v, not the sky's): 1 a ray to trace; 2
// outside a local light's reach, 3 within its lamp gap (lit, untraced).
template <typename D>
uint rt_shadow_ray(D t, texture2d<float, access::read> normals, constant RtShadowParams& p,
                   constant RtLightParams& lp, uint2 gid, float v, thread RtShadowRay& r) {
  const int2 c = int2(gid);
  const float3 q = rt_point(p, c, v);
  r.q = q;
  float3 l = p.light.xyz;
  if (lp.local != 0u) {
    // A local light: none where it does not reach (outside its sphere or
    // cone: lit, untraced).
    const float3 to = lp.position.xyz - q;
    const float d2 = dot(to, to);
    if (!(d2 * lp.position.w < 1.0) || !(d2 > 0.0)) return 2u;
    l = to * rsqrt(d2);
    if (lp.cone != 0u && !((dot(-l, lp.direction.xyz) - lp.direction.w) * lp.cone_scale > 0.0)) return 2u;
  }
  // The surface's geometric normal, from the depth of the neighbours on
  // the same surface. A G-buffer's normal is the shading one (a normal map
  // tilts it: at a grazing view or a low light it faces away from the
  // camera or the light where the surface does not), so it stands in only
  // where the depth gives none (no neighbour on an axis); else, facing the
  // camera.
  float3 n = -q;
  float3 dx, dy;
  bool geometric = false;
  if (rt_step(t, p, c, int2(1, 0), v, q, dx) && rt_step(t, p, c, int2(0, 1), v, q, dy)) {
    const float3 m = cross(dx, dy);
    if (length_squared(m) > 0.0) n = m, geometric = true;
  }
  if (!geometric && p.normals != 0u) {
    const float3 g = normals.read(gid).xyz * 2.0 - 1.0;
    if (length_squared(g) > 0.01) n = g;
  }
  n = normalize(n);
  if (dot(n, q) > 0.0) n = -n;  // toward the camera
  const float3 world = p.eye.xyz + q;
  const float dist = length(q);
  // The ray's start, `off` clear of the surface: the pixel's footprint's
  // share and the rounding terms. Seen from the light's side, mostly along
  // the light: off / N.L of it (a step along the ray does not move the
  // shadow's edges), and along the normal the rest where N.L falls below
  // min_slope. From the other side (the back of a leaf card; a solid's side
  // away from the light, whose rays start inside it and stay dark), along
  // the normal turned to the light.
  const float off = dist * (p.offset_pixels * p.pixel + p.offset_view) + length(world) * p.offset_world;
  const float nl = dot(n, l);
  float along_n = off, along_l = 0.0;
  if (nl >= 0.0) {
    along_n = off * saturate(1.0 - nl / p.min_slope);
    along_l = off / max(nl, p.min_slope);
  } else {
    n = -n;
  }
  float tmax = p.light.w;
  if (lp.local != 0u) along_l = min(along_l, 0.5 * length(lp.position.xyz - q));  // (short of the light)
  const float3 start = q + n * along_n + l * along_l;
  if (lp.local != 0u) {
    // From the ray's start to the light, less the lamp's gap: nothing in
    // between, lit (untraced, counted).
    const float3 to = lp.position.xyz - start;
    const float d = length(to);
    tmax = d - lp.lamp_gap;
    if (!(tmax > 0.0)) return 3u;
    l = to / d;
  }
  // A far light's soft shadows: one direction of its cone, another every
  // frame (p.cone, where a denoiser follows).
  if (lp.local == 0u) l = rt_cone_dir(l, p.cone, gid, p.frame);
  r.n = n;
  r.start = start;
  r.l = l;
  r.tmax = tmax;
  r.dist = dist;
  r.off = off;
  r.nl = nl;
  return 1u;
}
template <typename D>
half rt_shadow(D t, texture2d<float, access::read> normals, instance_acceleration_structure scene, RtShadowTable table,
               constant RtShadowParams& p, constant RtLightParams& lp, const device RtMaterial* materials,
               const device RtInfo* info, uint2 gid, thread bool& traced, thread bool& outside, thread bool& in_gap,
               thread bool& again, device RtShadowDiag* diag) {
  const float v = rt_depth(t, gid);
  const bool centre = p.diag != 0u && gid.x == p.size.x / 2u && gid.y == p.size.y / 2u;
  if (rt_sky(p, v)) {
    if (centre) {
      diag->point = float4(0.0, 0.0, 0.0, v);
      diag->state = 1u;
    }
    return 1.0h;
  }
  RtShadowRay sr;
  const uint kind = rt_shadow_ray(t, normals, p, lp, gid, v, sr);
  if (kind != 1u) {
    outside = kind == 2u;
    in_gap = kind == 3u;
    return 1.0h;
  }
  const float3 start = sr.start, l = sr.l, n = sr.n, world = p.eye.xyz + sr.q;
  const float tmax = sr.tmax, dist = sr.dist, off = sr.off;
  ray r(p.eye.xyz + start, l, 0.0, tmax);
  // No culling: a caster shadows from either side. Static instances from
  // the surface on; captured ones (which shadow themselves at a few
  // centimetres) beyond the gap.
  traced = true;
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  ray rc(r.origin, l, min(dist * p.captured_gap, tmax), tmax);
  if (p.diag == 0u)
    return rt_shadow_blocked(scene, table, r.origin, l, tmax, rc.min_distance, world, normalize(sr.q), materials, info,
                             again)
               ? 0.0h
               : 1.0h;
  // With diagnostics: what blocked it, counted by instance.
  RtHit h = rt_trace(scene, r, fixed, materials, intersection_params(), true);
  if (!h.hit) h = rt_trace(scene, rc, moving, materials, intersection_params(), true);
  if (h.hit) {
    device atomic_uint* counts = (device atomic_uint*)(diag + 1);
    atomic_fetch_add_explicit(&counts[min(h.instance, p.diag - 1u)], 1u, memory_order_relaxed);
  }
  if (centre) {
    diag->point = float4(world, v);
    diag->normal = float4(n, 0.0);
    diag->origin = float4(p.eye.xyz + start, off);
    diag->distance = h.hit ? h.distance : INFINITY;
    diag->instance = h.instance;
    diag->primitive = h.primitive;
    diag->state = h.hit ? 3u : 2u;
  }
  return h.hit ? 0.0h : 1.0h;
}
// The pixel's mask (thread `tid` of the grid over lp's pixels), and the
// simdgroup's rays, hits, pixels a local light does not reach and pixels
// within its lamp gap counted (counts[0] to counts[3]).
template <typename D>
void rt_shadow_pixel(D t, texture2d<float, access::read> normals, instance_acceleration_structure scene,
                     RtShadowTable table, constant RtShadowParams& p, constant RtLightParams& lp,
                     const device RtMaterial* materials, const device RtInfo* info,
                     texture2d<half, access::write> mask, device atomic_uint* counts, device RtShadowDiag* diag,
                     uint2 tid) {
  const uint2 gid = tid + lp.origin;
  const bool in = tid.x < lp.extent.x && tid.y < lp.extent.y && gid.x < p.size.x && gid.y < p.size.y;
  bool traced = false, outside = false, in_gap = false, again = false;
  half v = 1.0h;
  if (in) v = rt_shadow(t, normals, scene, table, p, lp, materials, info, gid, traced, outside, in_gap, again, diag);
  const uint rays = simd_sum(traced ? 1u : 0u), hits = simd_sum(traced && v < 0.5h ? 1u : 0u);
  const uint skipped = simd_sum(outside ? 1u : 0u), gap = simd_sum(in_gap ? 1u : 0u);
  if (simd_is_first()) {
    atomic_fetch_add_explicit(&counts[0], rays, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], hits, memory_order_relaxed);
    if (skipped) atomic_fetch_add_explicit(&counts[2], skipped, memory_order_relaxed);
    if (gap) atomic_fetch_add_explicit(&counts[3], gap, memory_order_relaxed);
  }
  if (in) mask.write(half4(v), gid);
}
kernel void rt_shadow_depth_cs(instance_acceleration_structure scene [[buffer(0)]],
                               constant RtShadowParams& p [[buffer(1)]],
                               const device RtMaterial* materials [[buffer(2)]],
                               device atomic_uint* counts [[buffer(3)]],
                               device RtShadowDiag* diag [[buffer(4)]],
                               constant RtLightParams& lp [[buffer(5)]],
                               RtShadowTable table [[buffer(6)]],
                               const device RtInfo* info [[buffer(7)]],
                               depth2d<float, access::read> depth [[texture(0)]],
                               texture2d<half, access::write> mask [[texture(1)]],
                               texture2d<float, access::read> normals [[texture(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  rt_shadow_pixel(depth, normals, scene, table, p, lp, materials, info, mask, counts, diag, gid);
}
kernel void rt_shadow_color_cs(instance_acceleration_structure scene [[buffer(0)]],
                               constant RtShadowParams& p [[buffer(1)]],
                               const device RtMaterial* materials [[buffer(2)]],
                               device atomic_uint* counts [[buffer(3)]],
                               device RtShadowDiag* diag [[buffer(4)]],
                               constant RtLightParams& lp [[buffer(5)]],
                               RtShadowTable table [[buffer(6)]],
                               const device RtInfo* info [[buffer(7)]],
                               texture2d<float, access::read> depth [[texture(0)]],
                               texture2d<half, access::write> mask [[texture(1)]],
                               texture2d<float, access::read> normals [[texture(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  rt_shadow_pixel(depth, normals, scene, table, p, lp, materials, info, mask, counts, diag, gid);
}
// A far light's mask at a lower resolution (RtShadowMasker::Encode): the
// top-left pixel of each step x step block (2, or kRtShadowDimStep for a
// dim light) traced into `half_mask` (counted as rt_shadow_pixel does),
// then rt_shadow_up_* fills the full mask from the four low-resolution
// texels around each pixel, weighted by how near their points are to its
// own (no shadow bleeding across a depth edge). The trace in two passes:
// rt_shadow_start_* makes each texel's ray (its start's offset from the
// point, as half floats; w < 0: none, the sky), rt_shadow_cast_* traces
// them, keeping no part of the normal's making alive across the ray (fewer
// registers: more threads in flight while the intersector works).
// Their `_turn` kernels (RtShadowMasker::EncodeSparse) trace the block's
// pixel turning from frame to frame instead (rt_light_sample, RtShadowTurn
// at buffer 8), for a temporal accumulation (gfx/rt_accumulate.h); the
// others are those passes as they were (Turn false: the top-left pixel).
struct RtShadowTurn {
  uint step, stride, frame, pad;
};
uint2 rt_light_sample(uint2 t, uint frame, uint stride, uint step);
// What the game's cascaded shadow map (RtShadowCascade, its 2 x 2 maxima
// and minima `most`) is sure of at world point w (dist from the camera, N.L
// of its surface facing the camera `nl`): 0 nothing; 1 lit by casters but
// those within `reach` of it along the light (what the map cannot
// resolve); 2 in the shadow of a caster at least shadow_gap away along the
// light over the whole footprint.
uint rt_cascade_sure(constant RtShadowCascade& c, texture2d<float, access::read> most, float3 w, float dist, float nl,
                     thread float& reach) {
  if (c.on == 0u || !(dist < c.range)) return 0u;
  // As the shader's lookup: the cascade by the view depth, the point in
  // light space, its texel.
  const float depth = dot(c.view.xyz, w);
  const float4 s4 = float4(1.0, select(float3(0.0), float3(1.0), -depth - c.splits.xyz >= 0.0));
  const float2 scale = float2(dot(s4, c.select[0]), dot(s4, c.select[1]));
  const float2 offset = float2(dot(s4, c.select[2]), dot(s4, c.select[3]));
  const float3 ls = w.x * c.matrix[0].xyw + w.y * c.matrix[1].xyw + w.z * c.matrix[2].xyw + c.matrix[3].xyw;
  const float2 x = (ls.xy * scale + offset) * c.atlas;
  // Its texels' world size: no larger than max_texel.
  const float2 per = abs(scale) * c.atlas * float2(length(float3(c.matrix[0].x, c.matrix[1].x, c.matrix[2].x)),
                                                   length(float3(c.matrix[0].y, c.matrix[1].y, c.matrix[2].y)));
  const float texel = 1.0 / min(per.x, per.y);
  if (!(texel <= c.max_texel)) return 0u;
  // The texels within 1.5 of the point's, inside the map and its cascade's
  // square.
  const float2 lo = x - 1.5, hi = x + 1.5;
  if (any(lo < 0.0) || any(hi >= c.atlas) || floor(lo.x / c.atlas.y) != floor(hi.x / c.atlas.y)) return 0u;
  const int2 b0 = int2(floor(lo * 0.5)), b1 = int2(floor(hi * 0.5));
  float m = -INFINITY, least = INFINITY;
  for (int by = b0.y; by <= b1.y; by++)
    for (int bx = b0.x; bx <= b1.x; bx++) {
      const float2 v = most.read(uint2(bx, by)).rg;
      m = max(m, v.r);
      least = min(least, v.g);
    }
  // Lit where z >= the map's value; the point's own surface rises toward
  // the light across the footprint (2 x 2 blocks: within 3.5 texels) by its
  // slope, and a texel more, and the map's values are rounded.
  const float zs = length(float3(c.matrix[0].w, c.matrix[1].w, c.matrix[2].w));
  if (!(zs > 0.0)) return 0u;
  // In shadow: every texel of the footprint holds a caster shadow_gap or
  // more nearer the light than the point (a building's, not the fine
  // gaps of a canopy near the surface the map's texels cannot resolve).
  if (c.shadow_gap > 0.0 &&
      least > ls.z + zs * c.shadow_gap + c.rounding_rel * max(abs(least), abs(ls.z)) + c.rounding_abs)
    return 2u;
  if (!(nl >= RT_CASCADE_MIN_SLOPE)) return 0u;
  const float tol = zs * (3.5 * texel * sqrt(max(0.0, 1.0 - nl * nl)) / nl + texel) +
                    c.rounding_rel * max(abs(m), abs(ls.z)) + c.rounding_abs;
  // And the map must see the point's own surface there (its nearest to
  // the light within the tolerance of the point): a map that does not (an
  // empty texel, another cascade's depth) says nothing.
  if (!(abs(m - ls.z) <= tol)) return 0u;
  // A caster lower than tol / zs over the surface may be missed: the short
  // ray's reach covers twice that height.
  reach = 2.0 * tol / zs / nl;
  return 1u;
}
template <bool Turn, typename D>
void rt_shadow_start(D t, texture2d<float, access::read> normals, constant RtShadowParams& p,
                     constant RtLightParams& lp, constant RtShadowCascade& cascade,
                     texture2d<float, access::read> most, texture2d<half, access::write> starts, uint step,
                     uint stride, uint frame, uint2 tid) {
  const uint2 hs = (p.size + step - 1u) / step;
  if (tid.x >= hs.x || tid.y >= hs.y) return;
  const uint2 gid = Turn ? min(rt_light_sample(tid, frame, stride, step), p.size - 1u) : min(tid * step, p.size - 1u);
  const float v = rt_depth(t, gid);
  RtShadowRay sr;
  if (rt_sky(p, v) || rt_shadow_ray(t, normals, p, lp, gid, v, sr) != 1u) {
    starts.write(half4(0.0h, 0.0h, 0.0h, -1.0h), tid);
    return;
  }
  // (w 1: a ray to the light; 2 + r: lit by the game's map but for a
  // short ray of length r; -2: in shadow by the map, no ray.)
  float reach = 0.0;
  const uint sure = rt_cascade_sure(cascade, most, p.eye.xyz + sr.q, sr.dist, sr.nl, reach);
  starts.write(half4(half3(sr.start - sr.q), sure == 1u ? half(2.0 + min(reach, 1000.0)) : sure == 2u ? -2.0h : 1.0h),
               tid);
}
// The cascade map's 2 x 2 maxima and minima (r, g).
template <typename M>
void rt_cascade_max(M map, texture2d<float, access::write> out, uint2 gid) {
  const uint2 size = uint2(map.get_width(), map.get_height());
  if (gid.x >= (size.x + 1u) / 2u || gid.y >= (size.y + 1u) / 2u) return;
  float m = -INFINITY, least = INFINITY;
  for (uint k = 0; k < 4u; k++) {
    const float v = rt_depth(map, min(gid * 2u + uint2(k & 1u, k >> 1u), size - 1u));
    m = max(m, v);
    least = min(least, v);
  }
  out.write(float4(m, least, 0.0, 0.0), gid);
}
kernel void rt_cascade_max_depth_cs(depth2d<float, access::read> map [[texture(0)]],
                                    texture2d<float, access::write> out [[texture(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  rt_cascade_max(map, out, gid);
}
kernel void rt_cascade_max_color_cs(texture2d<float, access::read> map [[texture(0)]],
                                    texture2d<float, access::write> out [[texture(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  rt_cascade_max(map, out, gid);
}
template <bool Turn, typename D>
void rt_shadow_cast(D t, instance_acceleration_structure scene, RtShadowTable table, constant RtShadowParams& p,
                    const device RtMaterial* materials, const device RtInfo* info,
                    texture2d<half, access::read> starts, texture2d<half, access::write> half_mask,
                    device atomic_uint* counts, uint step, uint stride, uint frame, uint2 tid) {
  const uint2 hs = (p.size + step - 1u) / step;
  const bool in = tid.x < hs.x && tid.y < hs.y;
  bool traced = false, again = false, sure = false;
  half m = 1.0h;
  if (in) {
    const half4 o = starts.read(tid);
    sure = o.w > 1.5h || o.w < -1.5h;
    if (o.w < -1.5h) m = 0.0h;  // in shadow by the map
    if (o.w > 0.0h) {
      const uint2 gid =
          Turn ? min(rt_light_sample(tid, frame, stride, step), p.size - 1u) : min(tid * step, p.size - 1u);
      const float3 q = rt_point(p, int2(gid), rt_depth(t, gid));
      const float3 l = rt_cone_dir(p.light.xyz, p.cone, gid, p.frame);
      const float tmax = sure ? min(p.light.w, float(o.w) - 2.0) : p.light.w;
      traced = true;
      const float gap = min(length(q) * p.captured_gap, tmax);
      m = rt_shadow_blocked(scene, table, p.eye.xyz + q + float3(o.xyz), l, tmax, gap, p.eye.xyz + q, normalize(q),
                            materials, info, again)
              ? 0.0h
              : 1.0h;
    }
  }
  const uint rays = simd_sum(traced && !sure ? 1u : 0u), hits = simd_sum(traced && m < 0.5h ? 1u : 0u);
  const uint retraced = simd_sum(again ? 1u : 0u), lit = simd_sum(sure ? 1u : 0u);
  if (simd_is_first()) {
    atomic_fetch_add_explicit(&counts[0], rays, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], hits, memory_order_relaxed);
    if (retraced) atomic_fetch_add_explicit(&counts[2], retraced, memory_order_relaxed);
    if (lit) atomic_fetch_add_explicit(&counts[3], lit, memory_order_relaxed);
  }
  if (in) half_mask.write(half4(m), tid);
}
kernel void rt_shadow_start_depth_cs(constant RtShadowParams& p [[buffer(1)]],
                                     constant RtLightParams& lp [[buffer(5)]],
                                     constant uint& step [[buffer(8)]],
                                     depth2d<float, access::read> depth [[texture(0)]],
                                     texture2d<half, access::write> starts [[texture(1)]],
                                     texture2d<float, access::read> normals [[texture(2)]],
                                     constant RtShadowCascade& cascade [[buffer(9)]],
                                     texture2d<float, access::read> most [[texture(3)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_shadow_start<false>(depth, normals, p, lp, cascade, most, starts, step, 0u, 0u, tid);
}
kernel void rt_shadow_start_color_cs(constant RtShadowParams& p [[buffer(1)]],
                                     constant RtLightParams& lp [[buffer(5)]],
                                     constant uint& step [[buffer(8)]],
                                     texture2d<float, access::read> depth [[texture(0)]],
                                     texture2d<half, access::write> starts [[texture(1)]],
                                     texture2d<float, access::read> normals [[texture(2)]],
                                     constant RtShadowCascade& cascade [[buffer(9)]],
                                     texture2d<float, access::read> most [[texture(3)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_shadow_start<false>(depth, normals, p, lp, cascade, most, starts, step, 0u, 0u, tid);
}
kernel void rt_shadow_cast_depth_cs(instance_acceleration_structure scene [[buffer(0)]],
                                    constant RtShadowParams& p [[buffer(1)]],
                                    const device RtMaterial* materials [[buffer(2)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    RtShadowTable table [[buffer(6)]],
                                    const device RtInfo* info [[buffer(7)]],
                                    constant uint& step [[buffer(8)]],
                                    depth2d<float, access::read> depth [[texture(0)]],
                                    texture2d<half, access::write> half_mask [[texture(1)]],
                                    texture2d<half, access::read> starts [[texture(3)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  rt_shadow_cast<false>(depth, scene, table, p, materials, info, starts, half_mask, counts, step, 0u, 0u, tid);
}
kernel void rt_shadow_cast_color_cs(instance_acceleration_structure scene [[buffer(0)]],
                                    constant RtShadowParams& p [[buffer(1)]],
                                    const device RtMaterial* materials [[buffer(2)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    RtShadowTable table [[buffer(6)]],
                                    const device RtInfo* info [[buffer(7)]],
                                    constant uint& step [[buffer(8)]],
                                    texture2d<float, access::read> depth [[texture(0)]],
                                    texture2d<half, access::write> half_mask [[texture(1)]],
                                    texture2d<half, access::read> starts [[texture(3)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  rt_shadow_cast<false>(depth, scene, table, p, materials, info, starts, half_mask, counts, step, 0u, 0u, tid);
}
// The same passes at the pixel turning within each block (RtShadowTurn);
// the starts' pass also writes each texel's sample (rt_shadow_sample_point).
// A texel's sample for a temporal accumulation (gfx/rt_accumulate.h): its
// point relative to the camera and its place in its block (x + 8 y); w -1
// for the sky's, and for a point too far for half floats (past 60000 world
// units: the accumulation then takes it for no sample).
template <typename D>
void rt_shadow_sample_point(D t, constant RtShadowParams& p, constant RtShadowTurn& turn,
                            texture2d<half, access::write> points, uint2 tid) {
  const uint2 hs = (p.size + turn.step - 1u) / turn.step;
  if (tid.x >= hs.x || tid.y >= hs.y) return;
  const uint2 gid = min(rt_light_sample(tid, turn.frame, turn.stride, turn.step), p.size - 1u),
              cell = gid - tid * turn.step;
  const float v = rt_depth(t, gid);
  const float3 q = rt_sky(p, v) ? float3(0.0) : rt_point(p, int2(gid), v);
  const bool ok = !rt_sky(p, v) && all(abs(q) < 60000.0);
  points.write(ok ? half4(half3(q), half(cell.x + 8u * cell.y)) : half4(0.0h, 0.0h, 0.0h, -1.0h), tid);
}
#define RT_SHADOW_TURN(KIND, DEPTH)                                                                                 \
  kernel void rt_shadow_start_turn_##KIND##_cs(                                                                     \
      constant RtShadowParams& p [[buffer(1)]], constant RtLightParams& lp [[buffer(5)]],                           \
      constant RtShadowTurn& turn [[buffer(8)]], DEPTH depth [[texture(0)]],                                        \
      texture2d<half, access::write> starts [[texture(1)]], texture2d<float, access::read> normals [[texture(2)]], \
      constant RtShadowCascade& cascade [[buffer(9)]], texture2d<float, access::read> most [[texture(3)]],          \
      texture2d<half, access::write> points [[texture(4)]], uint2 tid [[thread_position_in_grid]]) {               \
    rt_shadow_start<true>(depth, normals, p, lp, cascade, most, starts, turn.step, turn.stride, turn.frame, tid);  \
    rt_shadow_sample_point(depth, p, turn, points, tid);                                                           \
  }                                                                                                                 \
  kernel void rt_shadow_cast_turn_##KIND##_cs(                                                                      \
      instance_acceleration_structure scene [[buffer(0)]], constant RtShadowParams& p [[buffer(1)]],               \
      const device RtMaterial* materials [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],                  \
      RtShadowTable table [[buffer(6)]], const device RtInfo* info [[buffer(7)]],                                  \
      constant RtShadowTurn& turn [[buffer(8)]], DEPTH depth [[texture(0)]],                                        \
      texture2d<half, access::write> half_mask [[texture(1)]], texture2d<half, access::read> starts [[texture(3)]], \
      uint2 tid [[thread_position_in_grid]]) {                                                                      \
    rt_shadow_cast<true>(depth, scene, table, p, materials, info, starts, half_mask, counts, turn.step, turn.stride, \
                         turn.frame, tid);                                                                          \
  }
typedef depth2d<float, access::read> RtShadowDepthRead;
typedef texture2d<float, access::read> RtShadowColorRead;
RT_SHADOW_TURN(depth, RtShadowDepthRead)
RT_SHADOW_TURN(color, RtShadowColorRead)
// The full mask over one block of step x step pixels (thread `bid`): every
// pixel of the block lies between the same four low-resolution texels, so
// they and their points are read once for the block. Each texel's pixel is
// the one it was traced at (rt_light_sample with the turn's stride and
// frame; stride 0: its block's top-left), and the bilinear weights are by
// the distance to it.
template <typename D>
void rt_shadow_up(D t, constant RtShadowParams& p, texture2d<half, access::read> half_mask,
                  texture2d<half, access::write> mask, constant RtShadowTurn& turn, uint2 bid) {
  const uint step = turn.step;
  const uint2 hs = (p.size + step - 1u) / step;
  if (bid.x >= hs.x || bid.y >= hs.y) return;
  // The half texels whose pixels (step x, step y) surround each pixel,
  // bilinear weights times a falloff by the distance between the points.
  // Where the four agree (all lit or all in shadow: most of the screen) any
  // weights give their value: no points made.
  const int2 b = int2(bid);
  half m4[4];
  for (int k = 0; k < 4; k++) m4[k] = half_mask.read(uint2(clamp(b + int2(k & 1, k >> 1), int2(0), int2(hs) - 1))).r;
  const bool same = m4[0] == m4[1] && m4[0] == m4[2] && m4[0] == m4[3];
  float3 sp[4];
  bool sky[4];
  float2 at[4];
  if (!same)
    for (int k = 0; k < 4; k++) {
      // (Its block's, though clamped at the edge: the weights by place.)
      const uint2 h = uint2(max(b + int2(k & 1, k >> 1), int2(0)));
      const uint2 s = min(rt_light_sample(min(h, hs - 1u), turn.frame, turn.stride, step), p.size - 1u);
      at[k] = float2(h * step) + (float2(s) - float2(min(h, hs - 1u) * step));
      const float vs = rt_depth(t, s);
      sky[k] = rt_sky(p, vs);
      sp[k] = sky[k] ? float3(0.0) : rt_point(p, int2(s), vs);
    }
  for (uint j = 0; j < step; j++)
    for (uint i = 0; i < step; i++) {
      const uint2 gid = bid * step + uint2(i, j);
      if (gid.x >= p.size.x || gid.y >= p.size.y) continue;
      const float v = rt_depth(t, gid);
      if (rt_sky(p, v)) {
        mask.write(half4(1.0h), gid);
        continue;
      }
      if (same) {
        mask.write(half4(m4[0]), gid);
        continue;
      }
      const float2 g = float2(gid);
      const float3 q = rt_point(p, int2(gid), v);
      const float d = max(length(q), 1e-4);
      float sum = 0.0, wsum = 0.0;
      for (int k = 0; k < 4; k++) {
        const float2 f = saturate(1.0 - abs(g - at[k]) / float(step));
        const float bil = f.x * f.y + 1e-3;
        float near = 0.0;
        if (!sky[k]) {
          const float rel = length(sp[k] - q) / d;
          near = 1.0 / (1e-4 + rel * rel * 400.0);
        }
        const float wk = bil * near;
        sum += wk * float(m4[k]);
        wsum += wk;
      }
      // (No sample on the same surface: the nearest one's value.)
      mask.write(half4(wsum > 0.0 ? half(sum / wsum) : m4[0]), gid);
    }
}
kernel void rt_shadow_up_depth_cs(constant RtShadowParams& p [[buffer(1)]],
                                  constant RtShadowTurn& turn [[buffer(8)]],
                                  depth2d<float, access::read> depth [[texture(0)]],
                                  texture2d<half, access::read> half_mask [[texture(1)]],
                                  texture2d<half, access::write> mask [[texture(2)]],
                                  uint2 bid [[thread_position_in_grid]]) {
  rt_shadow_up(depth, p, half_mask, mask, turn, bid);
}
kernel void rt_shadow_up_color_cs(constant RtShadowParams& p [[buffer(1)]],
                                  constant RtShadowTurn& turn [[buffer(8)]],
                                  texture2d<float, access::read> depth [[texture(0)]],
                                  texture2d<half, access::read> half_mask [[texture(1)]],
                                  texture2d<half, access::write> mask [[texture(2)]],
                                  uint2 bid [[thread_position_in_grid]]) {
  rt_shadow_up(depth, p, half_mask, mask, turn, bid);
}
// The frame's local lights traced once per pixel (RtShadowMasker::
// EncodeLights), in two passes as the far light's half-resolution mask
// (fewer registers live across the rays). rt_lights_start: a thread for
// one pixel of every step x step (rt_light_sample), its point and normal as a
// far light's mask has them (rt_shadow_ray); its distance from the camera
// into `dist` (0: the sky), by which the light volumes' pixels weigh the
// texels around them (r_rt_shadow_rt); of the lights reaching it (within
// the sphere and the cone) the `cap` strongest by falloff, cone and N.L,
// their rays' starts (off the point as a local light's mask has them) and
// lights into the layers of `rays` (w the light's index plus 1; 0 after
// the last); every light's layer of `masks` lit, but 0 for a light the
// surface faces away from (no ray: one from there starts inside a solid
// and stays dark), and lit for one too faint there to matter (falloff
// times cone below 1/64: no ray). rt_lights_cast: those rays, 0 into the
// light's layer where one is blocked (rt_shadow_blocked).
struct RtLightsParams {
  uint count;  // lights in the list
  uint step;
  uint cap;
  uint frame;  // the sample's turn (rt_light_sample)
  uint2 low;   // the masks' size
  uint stride; // 0: every texel at its block's top-left pixel
  uint pad;
};
// The pixel texel t traces (RtLightSample's): one of its step x step
// block's, turning with the frame.
uint2 rt_light_sample(uint2 t, uint frame, uint stride, uint step) {
  if (stride == 0u) return t * step;
  uint h = t.x * 0x8da6b343u ^ t.y * 0xd8163841u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  const uint n = step * step;
  const uint cycle = frame / n;
  h += cycle + (cycle >> 1);  // (each cycle its order shifted: RtLightSample)
  const uint cell = (h % n + (frame % n) * stride) % n;
  return t * step + uint2(cell % step, cell / step);
}
struct RtLightGpu {
  float4 position;   // xyz relative to the camera; w 1 / radius^2
  float4 direction;  // xyz the way a spot light shines; w its cone's offset
  float cone_scale;
  uint cone;
  float lamp_gap;
  float pad;
};
constant uint RT_LIGHTS_MAX = 16u, RT_LIGHTS_CAP = 4u;
constant float RT_LIGHTS_FAINT = 1.0 / 64.0;
template <typename D>
void rt_lights_start(D t, texture2d<float, access::read> normals, constant RtShadowParams& p,
                     constant RtLightsParams& lp, constant RtLightGpu* lights,
                     texture2d_array<half, access::write> masks, texture2d<float, access::write> dist,
                     texture2d_array<half, access::write> rays, device atomic_uint* counts, uint2 tid) {
  const bool in = tid.x < lp.low.x && tid.y < lp.low.y;
  uint capped = 0u;
  if (in) {
    const uint2 gid = min(rt_light_sample(tid, lp.frame, lp.stride, lp.step), p.size - 1u);
    const float v = rt_depth(t, gid);
    const uint count = min(lp.count, RT_LIGHTS_MAX), cap = min(lp.cap, RT_LIGHTS_CAP);
    uint away = 0u;  // lights the surface faces away from
    uint picked = 0u;
    if (rt_sky(p, v)) {
      dist.write(float4(0.0), tid);
    } else {
      const int2 c = int2(gid);
      const float3 q = rt_point(p, c, v);
      const float d = length(q);
      dist.write(float4(d), tid);
      // The surface's geometric normal (as rt_shadow_ray's), facing the camera.
      float3 n = -q;
      float3 dx, dy;
      bool geometric = false;
      if (rt_step(t, p, c, int2(1, 0), v, q, dx) && rt_step(t, p, c, int2(0, 1), v, q, dy)) {
        const float3 m = cross(dx, dy);
        if (length_squared(m) > 0.0) n = m, geometric = true;
      }
      if (!geometric && p.normals != 0u) {
        const float3 g = normals.read(gid).xyz * 2.0 - 1.0;
        if (length_squared(g) > 0.01) n = g;
      }
      n = normalize(n);
      if (dot(n, q) > 0.0) n = -n;
      // The strongest `cap` (-1: none).
      float best[4];  // (RT_LIGHTS_CAP)
      uint which[4];
      for (uint k = 0u; k < RT_LIGHTS_CAP; k++) best[k] = -1.0, which[k] = 0u;
      uint reached = 0u;
      for (uint i = 0u; i < count; i++) {
        const float3 to = lights[i].position.xyz - q;
        const float d2 = dot(to, to), f = 1.0 - d2 * lights[i].position.w;
        if (!(f > 0.0) || !(d2 > 0.0)) continue;
        const float3 l = to * rsqrt(d2);
        float cw = 1.0;
        if (lights[i].cone != 0u) {
          cw = (dot(-l, lights[i].direction.xyz) - lights[i].direction.w) * lights[i].cone_scale;
          if (!(cw > 0.0)) continue;
          cw = min(cw, 1.0);
        }
        const float nl = dot(n, l);
        if (nl < 0.0) {
          away |= 1u << i;
          continue;
        }
        float w = f * f * cw;
        if (w < RT_LIGHTS_FAINT) continue;
        reached++;
        w *= max(nl, 0.05);
        uint at = i;
        for (uint k = 0u; k < cap; k++)
          if (w > best[k]) {
            const float bw = best[k];
            const uint bi = which[k];
            best[k] = w, which[k] = at;
            w = bw, at = bi;
          }
      }
      capped = reached > cap ? 1u : 0u;
      // The rays' starts, as rt_shadow_ray's for a local light.
      const float3 world = p.eye.xyz + q;
      const float off = d * (p.offset_pixels * p.pixel + p.offset_view) + length(world) * p.offset_world;
      for (uint k = 0u; k < cap; k++) {
        if (!(best[k] >= 0.0)) break;
        const uint i = which[k];
        const float3 lpos = lights[i].position.xyz;
        const float3 l = normalize(lpos - q);
        const float nl = dot(n, l);
        const float along_n = off * saturate(1.0 - nl / p.min_slope);
        const float along_l = min(off / max(nl, p.min_slope), 0.5 * length(lpos - q));
        const float3 o = n * along_n + l * along_l;
        if (!(length(lpos - q - o) - lights[i].lamp_gap > 0.0)) continue;  // (within the lamp's gap: lit)
        rays.write(half4(half3(o), half(i + 1u)), tid, picked++);
      }
    }
    if (picked < RT_LIGHTS_CAP) rays.write(half4(0.0h), tid, picked);
    for (uint i = 0u; i < count; i++) masks.write(half4((away & (1u << i)) != 0u ? 0.0h : 1.0h), tid, i);
  }
  const uint cp = simd_sum(capped);
  if (simd_is_first() && cp) atomic_fetch_add_explicit(&counts[2], cp, memory_order_relaxed);
}
template <typename D>
void rt_lights_cast(D t, instance_acceleration_structure scene, RtShadowTable table, constant RtShadowParams& p,
                    constant RtLightsParams& lp, constant RtLightGpu* lights, const device RtMaterial* materials,
                    const device RtInfo* info, texture2d_array<half, access::read> rays,
                    texture2d_array<half, access::write> masks, device atomic_uint* counts, uint2 tid) {
  const bool in = tid.x < lp.low.x && tid.y < lp.low.y;
  uint traced = 0u, hits = 0u, again_n = 0u;
  if (in) {
    const uint cap = min(lp.cap, RT_LIGHTS_CAP);
    half4 r = rays.read(tid, 0u);
    if (r.w > 0.5h) {
      const uint2 gid = min(rt_light_sample(tid, lp.frame, lp.stride, lp.step), p.size - 1u);
      const float3 q = rt_point(p, int2(gid), rt_depth(t, gid));
      const float gap = length(q) * p.captured_gap;
      for (uint k = 0u; k < cap; k++) {
        if (k > 0u) r = rays.read(tid, k);
        if (!(r.w > 0.5h)) break;
        const uint i = uint(r.w) - 1u;
        const float3 start = q + float3(r.xyz);
        const float3 to = lights[i].position.xyz - start;
        const float dl = length(to);
        const float tmax = dl - lights[i].lamp_gap;
        if (!(tmax > 0.0)) continue;
        bool again = false;
        traced++;
        if (rt_shadow_blocked(scene, table, p.eye.xyz + start, to / dl, tmax, min(gap, tmax), p.eye.xyz + q,
                              normalize(q), materials, info,
                              again)) {
          masks.write(half4(0.0h), tid, i);
          hits++;
        }
        again_n += again ? 1u : 0u;
      }
    }
  }
  const uint n = simd_sum(traced), h = simd_sum(hits), ag = simd_sum(again_n);
  if (simd_is_first()) {
    atomic_fetch_add_explicit(&counts[0], n, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], h, memory_order_relaxed);
    if (ag) atomic_fetch_add_explicit(&counts[3], ag, memory_order_relaxed);
  }
}
kernel void rt_lights_start_depth_cs(constant RtShadowParams& p [[buffer(1)]],
                                     device atomic_uint* counts [[buffer(3)]],
                                     constant RtLightsParams& lp [[buffer(5)]],
                                     constant RtLightGpu* lights [[buffer(8)]],
                                     depth2d<float, access::read> depth [[texture(0)]],
                                     texture2d_array<half, access::write> masks [[texture(1)]],
                                     texture2d<float, access::read> normals [[texture(2)]],
                                     texture2d<float, access::write> dist [[texture(3)]],
                                     texture2d_array<half, access::write> rays [[texture(4)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_lights_start(depth, normals, p, lp, lights, masks, dist, rays, counts, tid);
}
kernel void rt_lights_start_color_cs(constant RtShadowParams& p [[buffer(1)]],
                                     device atomic_uint* counts [[buffer(3)]],
                                     constant RtLightsParams& lp [[buffer(5)]],
                                     constant RtLightGpu* lights [[buffer(8)]],
                                     texture2d<float, access::read> depth [[texture(0)]],
                                     texture2d_array<half, access::write> masks [[texture(1)]],
                                     texture2d<float, access::read> normals [[texture(2)]],
                                     texture2d<float, access::write> dist [[texture(3)]],
                                     texture2d_array<half, access::write> rays [[texture(4)]],
                                     uint2 tid [[thread_position_in_grid]]) {
  rt_lights_start(depth, normals, p, lp, lights, masks, dist, rays, counts, tid);
}
kernel void rt_lights_cast_depth_cs(instance_acceleration_structure scene [[buffer(0)]],
                                    constant RtShadowParams& p [[buffer(1)]],
                                    const device RtMaterial* materials [[buffer(2)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    constant RtLightsParams& lp [[buffer(5)]],
                                    RtShadowTable table [[buffer(6)]],
                                    const device RtInfo* info [[buffer(7)]],
                                    constant RtLightGpu* lights [[buffer(8)]],
                                    depth2d<float, access::read> depth [[texture(0)]],
                                    texture2d_array<half, access::write> masks [[texture(1)]],
                                    texture2d_array<half, access::read> rays [[texture(4)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  rt_lights_cast(depth, scene, table, p, lp, lights, materials, info, rays, masks, counts, tid);
}
kernel void rt_lights_cast_color_cs(instance_acceleration_structure scene [[buffer(0)]],
                                    constant RtShadowParams& p [[buffer(1)]],
                                    const device RtMaterial* materials [[buffer(2)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    constant RtLightsParams& lp [[buffer(5)]],
                                    RtShadowTable table [[buffer(6)]],
                                    const device RtInfo* info [[buffer(7)]],
                                    constant RtLightGpu* lights [[buffer(8)]],
                                    texture2d<float, access::read> depth [[texture(0)]],
                                    texture2d_array<half, access::write> masks [[texture(1)]],
                                    texture2d_array<half, access::read> rays [[texture(4)]],
                                    uint2 tid [[thread_position_in_grid]]) {
  rt_lights_cast(depth, scene, table, p, lp, lights, materials, info, rays, masks, counts, tid);
}
// The mask over a target (RtShadowMasker::EncodeView).
kernel void rt_shadow_view_cs(texture2d<half, access::sample> mask [[texture(0)]],
                              texture2d<half, access::write> dst [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  constexpr sampler s(filter::nearest, address::clamp_to_edge);
  const half m = mask.sample(s, (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height())).r;
  dst.write(half4(m, m, m, 1.0h), gid);
}
)";

namespace {

void Put(float* out, const double* v) {
  for (int k = 0; k < 4; k++) out[k] = float(v[k]);
}

// The camera's rows (clip to world, RtFrame::inv_view_proj_exact where the
// scene gives it, else the float ones).
void CameraRows(const RtFrame& f, double r[4][4]) {
  bool exact = false;
  for (double v : f.inv_view_proj_exact) exact = exact || v != 0.0;
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 4; k++) r[i][k] = exact ? f.inv_view_proj_exact[i * 4 + k] : f.inv_view_proj[i * 4 + k];
}

// RtShadowParamsFor, with the camera's position in double (`eye_out`).
bool ShadowParams(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                  const float jitter[2], const float to_light[3], RtShadowParams* out, double eye_out[3]) {
  *out = RtShadowParams{};
  if (!w || !h) return false;
  const double lx = to_light[0], ly = to_light[1], lz = to_light[2];
  const double ll = std::sqrt(lx * lx + ly * ly + lz * lz);
  if (!(ll > 0.0) || !std::isfinite(ll)) return false;
  // The camera in double where the scene gives it (RtFrame::
  // inv_view_proj_exact), else the float one.
  double r[4][4];
  CameraRows(f, r);
  auto at = [&](double x, double y, double z, double* p) {  // clip (x, y, z, 1) to world
    double hv[4];
    for (int k = 0; k < 4; k++) hv[k] = x * r[0][k] + y * r[1][k] + z * r[2][k] + r[3][k];
    if (!(std::fabs(hv[3]) > 0.0)) return false;
    for (int k = 0; k < 3; k++) p[k] = hv[k] / hv[3];
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  };
  // The camera's position: the point clip space sends to w = 0 at the
  // centre (row 2, clip (0, 0, 1, 0)), whichever way depth runs; an
  // orthographic camera has none, the centre of its nearest depth's plane
  // stands for it.
  double eye[3];
  const double emag = std::sqrt(r[2][0] * r[2][0] + r[2][1] * r[2][1] + r[2][2] * r[2][2]);
  if (std::fabs(r[2][3]) > 1e-6 * emag) {
    for (int k = 0; k < 3; k++) eye[k] = r[2][k] / r[2][3];
  } else if (!at(0.0, 0.0, f.depth_reversed ? 1.0 : 0.0, eye)) {
    return false;
  }
  if (!std::isfinite(eye[0]) || !std::isfinite(eye[1]) || !std::isfinite(eye[2])) return false;
  // Rows relative to the camera: (r.xyz - eye r.w, r.w).
  double rel[4][4];
  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < 3; k++) rel[i][k] = r[i][k] - eye[k] * r[i][3];
    rel[i][3] = r[i][3];
  }
  // z = 1 - v where the buffer is stored reversed: x rx + y ry + (r2 + r3) - v r2.
  double rv[4], rc[4];
  for (int k = 0; k < 4; k++) {
    rv[k] = stored_reversed ? -rel[2][k] : rel[2][k];
    rc[k] = stored_reversed ? rel[2][k] + rel[3][k] : rel[3][k];
  }
  // The angle between neighbouring pixels' rays, at the centre (rows).
  double a[3], b[3];
  if (!at(0.0, 0.0, 0.5, a) || !at(0.0, 2.0 / double(h), 0.5, b)) return false;
  double da[3], db[3];
  double la = 0, lb = 0;
  for (int k = 0; k < 3; k++) {
    // (An orthographic camera's rays are parallel: measured from its
    // stand-in position, the angle a pixel's spacing makes from there.)
    da[k] = a[k] - eye[k];
    db[k] = b[k] - eye[k];
    la += da[k] * da[k];
    lb += db[k] * db[k];
  }
  la = std::sqrt(la);
  lb = std::sqrt(lb);
  double pixel = 0;
  if (la > 0 && lb > 0) {
    double d2 = 0;
    for (int k = 0; k < 3; k++) {
      const double e = da[k] / la - db[k] / lb;
      d2 += e * e;
    }
    pixel = 2.0 * std::asin(std::min(1.0, std::sqrt(d2) / 2.0));
  }
  Put(out->rx, rel[0]);
  Put(out->ry, rel[1]);
  Put(out->rv, rv);
  Put(out->rc, rc);
  for (int k = 0; k < 3; k++) out->eye[k] = float(eye[k]);
  out->light[0] = float(lx / ll);
  out->light[1] = float(ly / ll);
  out->light[2] = float(lz / ll);
  out->light[3] = std::numeric_limits<float>::infinity();
  out->size[0] = w;
  out->size[1] = h;
  out->jitter[0] = jitter ? jitter[0] : 0.0f;
  out->jitter[1] = jitter ? jitter[1] : 0.0f;
  out->depth_texture = depth_texture ? 1 : 0;
  const double far = f.depth_reversed ? 0.0 : 1.0;
  out->sky = float(stored_reversed ? 1.0 - far : far);
  out->pixel = float(pixel);
  out->offset_pixels = kRtShadowOffsetPixels;
  out->offset_view = kRtShadowOffsetView;
  out->offset_world = kRtShadowOffsetWorld;
  out->cone = 0.0f;  // hard
  out->min_slope = kRtShadowMinSlope;
  out->captured_gap = kRtShadowCapturedGap;
  for (int k = 0; k < 3; k++) eye_out[k] = eye[k];
  return true;
}

}  // namespace

bool RtShadowParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                       const float jitter[2], const float to_light[3], RtShadowParams* out) {
  double eye[3];
  return ShadowParams(f, w, h, stored_reversed, depth_texture, jitter, to_light, out, eye);
}

void RtShadowPoint(const RtShadowParams& p, double i, double j, double v, double out[3]) {
  const double x = 2.0 * (i - p.jitter[0]) / double(p.size[0]) - 1.0;
  const double y = 1.0 - 2.0 * (j - p.jitter[1]) / double(p.size[1]);
  double hv[4];
  for (int k = 0; k < 4; k++) hv[k] = x * p.rx[k] + y * p.ry[k] + v * p.rv[k] + p.rc[k];
  for (int k = 0; k < 3; k++) out[k] = p.eye[k] + hv[k] / hv[3];
}

bool RtShadowCascadeFor(const RayShadow& s, const float (*c)[4], RtShadowCascade* out) {
  *out = RtShadowCascade{};
  if (!c || s.cascade_sampler >= 16 || s.cascade_view_register >= 224 || s.cascade_split_register >= 224 ||
      s.cascade_matrix_register + 3 >= 224 || s.cascade_range_register >= 224)
    return false;
  for (uint32_t r : s.cascade_select_registers)
    if (r >= 224) return false;
  std::memcpy(out->view, c[s.cascade_view_register], 16);
  std::memcpy(out->splits, c[s.cascade_split_register], 16);
  for (int k = 0; k < 4; k++) {
    std::memcpy(out->select[k], c[s.cascade_select_registers[k]], 16);
    std::memcpy(out->matrix[k], c[s.cascade_matrix_register + k], 16);
  }
  out->range = c[s.cascade_range_register][3];
  out->max_texel = s.cascade_max_texel;
  out->shadow_gap = s.cascade_shadow_gap;
  out->on = 1;
  return true;
}

double RtCascadeTexel(const RtShadowCascade& c, int k) {
  const double s4[4] = {1.0, k >= 1 ? 1.0 : 0.0, k >= 2 ? 1.0 : 0.0, k >= 3 ? 1.0 : 0.0};
  double per[2];
  for (int a = 0; a < 2; a++) {
    double scale = 0.0, len = 0.0;
    for (int i = 0; i < 4; i++) scale += s4[i] * c.select[a][i];
    for (int i = 0; i < 3; i++) len += double(c.matrix[i][a]) * c.matrix[i][a];
    per[a] = std::fabs(scale) * c.atlas[a] * std::sqrt(len);
  }
  const double m = std::min(per[0], per[1]);
  return m > 0.0 ? 1.0 / m : INFINITY;
}

RtLocalLight RtLocalLightFor(const RayShadow& s, const float (*c)[4]) {
  RtLocalLight l;
  l.lamp_gap = s.lamp_gap;
  if (!c || s.position_register >= 224 || s.radius_register >= 224 || s.radius_component >= 4) return l;
  for (int k = 0; k < 3; k++) l.position[k] = c[s.position_register][k];
  l.inv_sq_radius = c[s.radius_register][s.radius_component];
  if (s.cone && s.direction_register < 224 && s.cone_offset_register < 224 && s.cone_scale_register < 224 &&
      s.cone_offset_component < 4 && s.cone_scale_component < 4) {
    l.cone = true;
    for (int k = 0; k < 3; k++) l.direction[k] = c[s.direction_register][k];
    l.cone_offset = c[s.cone_offset_register][s.cone_offset_component];
    l.cone_scale = c[s.cone_scale_register][s.cone_scale_component];
  }
  return l;
}

bool RtSameLocalLight(const RtLocalLight& a, const RtLocalLight& b) {
  return std::memcmp(a.position, b.position, sizeof a.position) == 0 &&
         std::memcmp(&a.inv_sq_radius, &b.inv_sq_radius, 4) == 0 && a.cone == b.cone &&
         std::memcmp(a.direction, b.direction, sizeof a.direction) == 0 &&
         std::memcmp(&a.cone_offset, &b.cone_offset, 4) == 0 && std::memcmp(&a.cone_scale, &b.cone_scale, 4) == 0 &&
         std::memcmp(&a.lamp_gap, &b.lamp_gap, 4) == 0;
}

double RtLocalLightRadius(const RtLocalLight& l) {
  const double inv = l.inv_sq_radius;
  if (!(inv > 0.0) || !std::isfinite(inv)) return 0.0;
  const double r = 1.0 / std::sqrt(inv);
  return std::isfinite(r) ? r : 0.0;
}

bool RtLocalLightBox(const RtLocalLight& l, double lo[3], double hi[3]) {
  const double radius = RtLocalLightRadius(l);
  if (radius <= 0.0) return false;
  for (int k = 0; k < 3; k++) {
    if (!std::isfinite(l.position[k])) return false;
    lo[k] = l.position[k] - radius;
    hi[k] = l.position[k] + radius;
  }
  if (!l.cone) return true;
  // Reached where dot(u, dir) > offset, u the unit vector from the light
  // (the kernel's (dot(-l, dir) - offset) scale > 0, its sign folded in).
  const double sign = l.cone_scale > 0.0f ? 1.0 : l.cone_scale < 0.0f ? -1.0 : 0.0;
  if (sign == 0.0 || !std::isfinite(l.cone_offset)) return false;
  double dir[3] = {sign * l.direction[0], sign * l.direction[1], sign * l.direction[2]};
  const double offset = sign * l.cone_offset;
  const double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (!std::isfinite(len)) return true;
  if (len == 0.0) return offset < 0.0;  // dot 0: everywhere or nowhere
  const double c = offset / len;
  if (c >= 1.0) return false;
  if (c <= 0.0) return true;  // half a space or more: the sphere's box
  const double half = std::acos(c);
  for (int k = 0; k < 3; k++) {
    // The cone's directions along axis k: the largest component cos(angle
    // to the axis less the half angle), 1 where the axis is inside it.
    const double a = std::acos(std::clamp(dir[k] / len, -1.0, 1.0)), b = M_PI - a;
    const double most = a <= half ? 1.0 : std::cos(a - half), least = b <= half ? -1.0 : -std::cos(b - half);
    lo[k] = l.position[k] + radius * std::min(0.0, least);
    hi[k] = l.position[k] + radius * std::max(0.0, most);
  }
  return true;
}

RtRect RtLocalLightBounds(const RtFrame& f, uint32_t w, uint32_t h, const float jitter[2], const RtLocalLight& l) {
  const RtRect all{0, 0, w, h};
  double lo[3], hi[3];
  if (!w || !h || !RtLocalLightBox(l, lo, hi)) return RtRect{};
  // World to clip: the rows' inverse (D3D's row vectors, p_clip = p_world vp).
  double r[4][4], vp[16];
  CameraRows(f, r);
  if (!Invert4(&r[0][0], vp)) return all;
  double c[8][4], wmax = 0.0;
  for (int corner = 0; corner < 8; corner++) {
    const double p[3] = {(corner & 1) ? hi[0] : lo[0], (corner & 2) ? hi[1] : lo[1], (corner & 4) ? hi[2] : lo[2]};
    for (int k = 0; k < 4; k++) c[corner][k] = p[0] * vp[k] + p[1] * vp[4 + k] + p[2] * vp[8 + k] + vp[12 + k];
    for (int k = 0; k < 4; k++)
      if (!std::isfinite(c[corner][k])) return all;
    wmax = std::max(wmax, std::fabs(c[corner][3]));
  }
  // The box clipped to w >= eps: its corners there, and where its edges
  // cross that plane.
  const double eps = std::max(wmax * 1e-6, 1e-12);
  double x0 = INFINITY, y0 = INFINITY, x1 = -INFINITY, y1 = -INFINITY;
  const double jx = jitter ? jitter[0] : 0.0, jy = jitter ? jitter[1] : 0.0;
  auto take = [&](const double* p) {
    // Pixel i at clip x = 2 (i - jitter) / w - 1 (D3D9's centres).
    const double i = (p[0] / p[3] + 1.0) * 0.5 * w + jx, j = (1.0 - p[1] / p[3]) * 0.5 * h + jy;
    x0 = std::min(x0, i), x1 = std::max(x1, i), y0 = std::min(y0, j), y1 = std::max(y1, j);
  };
  for (int a = 0; a < 8; a++) {
    if (c[a][3] >= eps) take(c[a]);
    for (int bit = 1; bit < 8; bit <<= 1) {
      const int b = a | bit;
      if (b == a) continue;
      const double wa = c[a][3], wb = c[b][3];
      if ((wa >= eps) == (wb >= eps)) continue;
      const double t = (eps - wa) / (wb - wa);
      double p[4];
      for (int k = 0; k < 4; k++) p[k] = c[a][k] + (c[b][k] - c[a][k]) * t;
      p[3] = eps;
      take(p);
    }
  }
  if (!(x0 <= x1) || !(y0 <= y1)) return RtRect{};  // wholly behind the camera
  // A pixel more on each side (its centre against the box's edge, the
  // depth's neighbours), within the buffer.
  auto clamp = [](double v, uint32_t n) { return uint32_t(std::clamp(v, 0.0, double(n))); };
  RtRect out{clamp(std::floor(x0) - 1.0, w), clamp(std::floor(y0) - 1.0, h), clamp(std::ceil(x1) + 2.0, w),
             clamp(std::ceil(y1) + 2.0, h)};
  return out.Empty() ? RtRect{} : out;
}

bool RtLightParamsFor(const RtFrame& f, uint32_t w, uint32_t h, bool stored_reversed, bool depth_texture,
                      const float jitter[2], const RtLocalLight& l, const RtRect& r, RtShadowParams* p,
                      RtLightParams* lp) {
  *lp = RtLightParams{};
  const double radius = RtLocalLightRadius(l);
  if (radius <= 0.0 || r.Empty() || r.x1 > w || r.y1 > h) return false;
  for (float v : l.position)
    if (!std::isfinite(v)) return false;
  const float up[3] = {0.0f, 0.0f, 1.0f};
  double eye[3];
  if (!ShadowParams(f, w, h, stored_reversed, depth_texture, jitter, up, p, eye)) return false;
  // Relative to the camera in double (as the points are).
  for (int k = 0; k < 3; k++) {
    lp->position[k] = float(double(l.position[k]) - eye[k]);
    lp->direction[k] = l.direction[k];
  }
  lp->position[3] = l.inv_sq_radius;
  lp->direction[3] = l.cone_offset;
  lp->cone_scale = l.cone_scale;
  lp->cone = l.cone ? 1 : 0;
  lp->lamp_gap = std::max(l.lamp_gap, 0.0f);
  lp->local = 1;
  lp->origin[0] = r.x0;
  lp->origin[1] = r.y0;
  lp->extent[0] = r.x1 - r.x0;
  lp->extent[1] = r.y1 - r.y0;
  return true;
}

bool RtLightsEye(const RtFrame& f, double eye[3]) {
  const float up[3] = {0.0f, 0.0f, 1.0f};
  RtShadowParams p;
  return ShadowParams(f, 1, 1, false, false, nullptr, up, &p, eye);
}

void RtLightSample(uint32_t tx, uint32_t ty, uint32_t frame, uint32_t stride, uint32_t step, uint32_t* px,
                   uint32_t* py) {
  *px = tx * step;
  *py = ty * step;
  if (!stride) return;
  uint32_t h = (tx * 0x8da6b343u) ^ (ty * 0xd8163841u);
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  const uint32_t n = step * step;
  const uint32_t cycle = frame / n;
  h += cycle + (cycle >> 1);
  const uint32_t cell = (h % n + (frame % n) * stride) % n;
  *px += cell % step;
  *py += cell / step;
}

uint32_t RtLightStride(uint32_t step) {
  const uint32_t n = step * step;
  if (n <= 1) return 0;
  auto gcd = [](uint32_t a, uint32_t b) {
    while (b) {
      const uint32_t t = a % b;
      a = b;
      b = t;
    }
    return a;
  };
  uint32_t k = n / 2 + 1;
  while (gcd(k, n) != 1) k++;
  return k;
}

bool RtLightGpuFor(const RtLocalLight& l, const double eye[3], RtLightGpu* out) {
  *out = RtLightGpu{};
  if (!(RtLocalLightRadius(l) > 0.0)) return false;
  for (int k = 0; k < 3; k++) {
    if (!std::isfinite(l.position[k])) return false;
    out->position[k] = float(double(l.position[k]) - eye[k]);
    out->direction[k] = l.direction[k];
  }
  out->position[3] = l.inv_sq_radius;
  out->direction[3] = l.cone_offset;
  out->cone_scale = l.cone_scale;
  out->cone = l.cone ? 1 : 0;
  out->lamp_gap = std::max(l.lamp_gap, 0.0f);
  return true;
}

RtLocalLight RtLightPredict(const RtLocalLight& now, const RtLocalLight& before) {
  RtLocalLight p = now;
  for (int k = 0; k < 3; k++) p.position[k] = float(2.0 * double(now.position[k]) - double(before.position[k]));
  return p;
}

bool RtLightMatch(const RtLocalLight& traced, bool moved, const RtLocalLight& drawn) {
  if (RtSameLocalLight(traced, drawn)) return true;
  if (!moved) return false;
  auto close = [](double x, double y) { return std::fabs(x - y) <= 1e-3 * std::max({std::fabs(x), std::fabs(y), 1e-6}); };
  if (traced.cone != drawn.cone || !close(traced.inv_sq_radius, drawn.inv_sq_radius) ||
      !close(traced.lamp_gap, drawn.lamp_gap))
    return false;
  double d2 = 0;
  for (int k = 0; k < 3; k++)
    d2 += (double(traced.position[k]) - drawn.position[k]) * (double(traced.position[k]) - drawn.position[k]);
  if (!(d2 <= double(kRtLightMatchOffset) * kRtLightMatchOffset)) return false;
  if (!traced.cone) return true;
  if (!close(traced.cone_offset, drawn.cone_offset) || !close(traced.cone_scale, drawn.cone_scale)) return false;
  double dot = 0, na = 0, nb = 0;
  for (int k = 0; k < 3; k++) {
    dot += double(traced.direction[k]) * drawn.direction[k];
    na += double(traced.direction[k]) * traced.direction[k];
    nb += double(drawn.direction[k]) * drawn.direction[k];
  }
  return dot >= 0.9994 * std::sqrt(na * nb);  // (cos 2 degrees)
}

bool RtNearLocalLight(const RtLocalLight& a, const RtLocalLight& b) {
  auto close = [](double x, double y) { return std::fabs(x - y) <= 1e-3 * std::max({std::fabs(x), std::fabs(y), 1e-6}); };
  if (a.cone != b.cone || !close(a.inv_sq_radius, b.inv_sq_radius) || !close(a.lamp_gap, b.lamp_gap)) return false;
  const double r = RtLocalLightRadius(a);
  if (!(r > 0.0)) return false;
  double d2 = 0;
  for (int k = 0; k < 3; k++) d2 += (double(a.position[k]) - b.position[k]) * (double(a.position[k]) - b.position[k]);
  if (!(d2 <= 0.01 * r * r)) return false;
  if (!a.cone) return true;
  if (!close(a.cone_offset, b.cone_offset) || !close(a.cone_scale, b.cone_scale)) return false;
  double dot = 0, na = 0, nb = 0;
  for (int k = 0; k < 3; k++) {
    dot += double(a.direction[k]) * b.direction[k];
    na += double(a.direction[k]) * a.direction[k];
    nb += double(b.direction[k]) * b.direction[k];
  }
  return dot >= 0.985 * std::sqrt(na * nb);  // (cos 10 degrees)
}

int RtLightMaskFind(RtLightMaskCache* c, const RtLocalLight& l) {
  int near = -1;
  double best = 0;
  for (int k = 0; k < RtLightMaskCache::kEntries; k++) {
    RtLightMaskCache::Entry& e = c->entry[k];
    if (!e.used) continue;
    if (RtSameLocalLight(e.light, l)) {
      e.last_use = ++c->uses;
      e.seen = c->frame;
      return k;
    }
    if (e.seen != c->frame && RtNearLocalLight(e.light, l)) {
      double d2 = 0;
      for (int i = 0; i < 3; i++) d2 += (double(e.light.position[i]) - l.position[i]) * (double(e.light.position[i]) - l.position[i]);
      if (near < 0 || d2 < best) near = k, best = d2;
    }
  }
  if (near >= 0) {
    RtLightMaskCache::Entry& e = c->entry[near];
    e.light = l;
    e.memo = {};
    e.last_use = ++c->uses;
    e.seen = e.moved = c->frame;
  }
  return near;
}

int RtLightMaskTake(RtLightMaskCache* c, const RtLocalLight& l) {
  int oldest = 0;
  for (int k = 0; k < RtLightMaskCache::kEntries; k++) {
    const RtLightMaskCache::Entry& e = c->entry[k];
    if (!e.used) {
      oldest = k;
      break;
    }
    if (e.last_use < c->entry[oldest].last_use) oldest = k;
  }
  RtLightMaskCache::Entry& e = c->entry[oldest];
  e = {};
  e.used = true;
  e.light = l;
  e.last_use = ++c->uses;
  e.born = e.seen = c->frame;
  return oldest;
}

RtLightStep RtLightMaskChoose(RtLightMaskCache* c, int e, uint32_t budget) {
  RtLightMaskCache::Entry* me = e >= 0 ? &c->entry[e] : nullptr;
  if (me) me->wanted = c->frame;
  if (me && me->moved == c->frame)
    return c->moving_traced < RtLightMaskCache::kMovingBudget ? RtLightStep::Trace : RtLightStep::Budget;
  const uint32_t traced = c->still_traced;
  // The reserved lights not traced yet this frame (but this one).
  uint32_t waiting = 0;
  for (int k = 0; k < RtLightMaskCache::kEntries; k++) {
    const RtLightMaskCache::Entry& x = c->entry[k];
    waiting += k != e && x.used && x.reserved && x.traced != c->frame ? 1 : 0;
  }
  if (traced < budget && ((me && me->reserved) || uint64_t(traced) + waiting < budget)) return RtLightStep::Trace;
  if (me && me->mask && c->frame - me->traced <= RtLightMaskCache::kMaxAge) return RtLightStep::Older;
  return RtLightStep::Budget;
}

void RtLightMaskTraced(RtLightMaskCache* c, int e, uint64_t area) {
  RtLightMaskCache::Entry& x = c->entry[e];
  x.mask = true;
  x.traced = c->frame;
  x.area = area;
  (x.moved == c->frame ? c->moving_traced : c->still_traced)++;
}

void RtLightMaskEndFrame(RtLightMaskCache* c, uint32_t budget) {
  int order[RtLightMaskCache::kEntries];
  int n = 0;
  for (int k = 0; k < RtLightMaskCache::kEntries; k++) {
    RtLightMaskCache::Entry& x = c->entry[k];
    x.memo = {};
    x.reserved = false;
    if (x.used && x.wanted == c->frame && x.born < c->frame && x.moved != c->frame) order[n++] = k;
  }
  // Without a mask first, then the oldest, then the most pixels; the
  // entry's number last (a stable order).
  auto key = [&](int k) {
    const RtLightMaskCache::Entry& x = c->entry[k];
    return std::make_tuple(x.mask ? 1 : 0, x.traced, ~x.area, k);
  };
  std::sort(order, order + n, [&](int a, int b) { return key(a) < key(b); });
  for (int i = 0; i < n && uint32_t(i) < budget; i++) c->entry[order[i]].reserved = true;
  c->moving_traced = c->still_traced = 0;
  c->frame++;
}

void RtShadowMasker::EncodeView(id<MTLCommandBuffer> cmd, id<MTLTexture> mask, id<MTLTexture> target) const {
  if (!mask || !target) return;
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
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt shadow view");
  [e setComputePipelineState:view_pipe_];
  [e setTexture:mask atIndex:0];
  [e setTexture:view_tex_ atIndex:1];
  [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder];
  [bl copyFromTexture:view_tex_ toTexture:target];
  [bl endEncoding];
}

uint64_t RtShadowWriter(uint64_t camera, bool clear, uint64_t vote, bool full_view) {
  if (clear) return 0;
  if (vote) return full_view ? vote : 0;
  return camera;
}

RtShadowWhite RtShadowWhyWhite(const RtShadowWant& w) {
  if (!w.depth) return RtShadowWhite::NoDepth;
  if (!w.main_size) return RtShadowWhite::WrongSize;
  if (w.light[0] == 0.0f && w.light[1] == 0.0f && w.light[2] == 0.0f) return RtShadowWhite::NoLight;
  if (w.dark) return RtShadowWhite::Dark;
  if (!w.scene_camera) return RtShadowWhite::NoCamera;
  if (w.depth_camera != w.scene_camera) return RtShadowWhite::OtherCamera;
  return RtShadowWhite::None;
}

const char* RtShadowWhiteName(RtShadowWhite w) {
  switch (w) {
    case RtShadowWhite::NoDepth: return "no depth";
    case RtShadowWhite::WrongSize: return "depth not the main view's size";
    case RtShadowWhite::NoLight: return "no light";
    case RtShadowWhite::NoCamera: return "no camera yet";
    case RtShadowWhite::OtherCamera: return "depth of another camera";
    case RtShadowWhite::NoTlas: return "nothing to trace";
    case RtShadowWhite::Budget: return "past the frame's traced lights";
    case RtShadowWhite::NoMask: return "no mask or parameters";
    case RtShadowWhite::Dark: return "the light's colour black";
    default: return "?";
  }
}

RtShadowStep RtShadowNext(RtShadowMemo* memo, const RtShadowWant& w) {
  const RtShadowWant& l = memo->last;
  if (memo->valid && l.depth == w.depth && l.epoch == w.epoch && l.depth_camera == w.depth_camera &&
      l.scene_camera == w.scene_camera && l.main_size == w.main_size && l.dark == w.dark &&
      std::memcmp(l.light, w.light, sizeof l.light) == 0)
    return RtShadowStep::Reuse;
  memo->valid = true;
  memo->last = w;
  memo->white = RtShadowWhyWhite(w) != RtShadowWhite::None;
  return memo->white ? RtShadowStep::White : RtShadowStep::Trace;
}

bool RtShadowSettlesClear(uint32_t clear_flags, bool reads_pass_depth) { return (clear_flags & 6) && reads_pass_depth; }

int RtShadowSampler(bool claimed, uint32_t mask_sampler, bool have_mask) {
  return claimed && have_mask && mask_sampler < 16 ? int(mask_sampler) : -1;
}

bool RtOwnMapDraw(RtOwnMaps* m, uint64_t hash) {
  m->draws++;
  if (m->count >= kRtOwnMapsSaid) return false;
  for (size_t k = 0; k < m->count; k++)
    if (m->said[k] == hash) return false;
  m->said[m->count++] = hash;
  return true;
}

bool RtSnapshotReuse(RtSnapshotMemo* memo, const RtShadowWant& w, uint64_t serial) {
  if (memo->valid && memo->depth == w.depth && memo->epoch == w.epoch && memo->depth_camera == w.depth_camera &&
      memo->scene_camera == w.scene_camera && memo->serial == serial)
    return true;
  memo->valid = true;
  memo->depth = w.depth;
  memo->epoch = w.epoch;
  memo->depth_camera = w.depth_camera;
  memo->scene_camera = w.scene_camera;
  memo->serial = serial;
  return false;
}

std::unique_ptr<RtShadowMasker> RtShadowMasker::Make(id<MTLDevice> dev) {
  if (!RtSupported(dev)) return nullptr;
  auto m = std::make_unique<RtShadowMasker>();
  m->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(kRtMsl) + kRtShadowMsl).c_str())
                                         options:nil
                                           error:&err];
  if (lib) {
    auto pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
      return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    };
    // The tracing kernels: the alpha test linked in as an intersection
    // function, and each pipeline's table naming it (every instance's and
    // geometry's table offset is 0). Apple's "Accelerating ray tracing
    // using Metal" sample's way.
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
    m->depth_pipe_ = tracing(@"rt_shadow_depth_cs", &m->depth_table_);
    if (m->depth_pipe_) m->color_pipe_ = tracing(@"rt_shadow_color_cs", &m->color_table_);
    if (m->color_pipe_) m->view_pipe_ = pipe(@"rt_shadow_view_cs");
    // The far light's half-resolution mask and its upsampling.
    if (m->view_pipe_) m->up_depth_pipe_ = pipe(@"rt_shadow_up_depth_cs");
    if (m->up_depth_pipe_) m->up_color_pipe_ = pipe(@"rt_shadow_up_color_cs");
    if (m->up_color_pipe_) m->start_depth_pipe_ = pipe(@"rt_shadow_start_depth_cs");
    if (m->start_depth_pipe_) m->start_color_pipe_ = pipe(@"rt_shadow_start_color_cs");
    if (m->start_color_pipe_) m->max_depth_pipe_ = pipe(@"rt_cascade_max_depth_cs");
    if (m->max_depth_pipe_) m->max_color_pipe_ = pipe(@"rt_cascade_max_color_cs");
    if (m->max_color_pipe_) m->cast_depth_pipe_ = tracing(@"rt_shadow_cast_depth_cs", &m->cast_depth_table_);
    if (m->cast_depth_pipe_) m->cast_color_pipe_ = tracing(@"rt_shadow_cast_color_cs", &m->cast_color_table_);
    // Those at the pixel turning within its block (EncodeSparse; none: it
    // returns nil).
    if (m->cast_color_pipe_) m->start_turn_depth_pipe_ = pipe(@"rt_shadow_start_turn_depth_cs");
    if (m->start_turn_depth_pipe_) m->start_turn_color_pipe_ = pipe(@"rt_shadow_start_turn_color_cs");
    if (m->start_turn_color_pipe_)
      m->cast_turn_depth_pipe_ = tracing(@"rt_shadow_cast_turn_depth_cs", &m->cast_turn_depth_table_);
    if (m->cast_turn_depth_pipe_)
      m->cast_turn_color_pipe_ = tracing(@"rt_shadow_cast_turn_color_cs", &m->cast_turn_color_table_);
    if (!m->cast_turn_color_pipe_ && m->cast_color_pipe_)
      std::fprintf(stderr, "gfx: ray-traced sun mask's turning samples: %s\n", err.localizedDescription.UTF8String);
    // The local lights' passes (none: the volumes' pixels trace their own).
    if (m->cast_color_pipe_) m->lights_start_depth_pipe_ = pipe(@"rt_lights_start_depth_cs");
    if (m->lights_start_depth_pipe_) m->lights_start_color_pipe_ = pipe(@"rt_lights_start_color_cs");
    if (m->lights_start_color_pipe_)
      m->lights_depth_pipe_ = tracing(@"rt_lights_cast_depth_cs", &m->lights_depth_table_);
    if (m->lights_depth_pipe_) m->lights_color_pipe_ = tracing(@"rt_lights_cast_color_cs", &m->lights_color_table_);
    if (!m->lights_color_pipe_ && m->cast_color_pipe_)
      std::fprintf(stderr, "gfx: ray-traced local lights' pass: %s\n", err.localizedDescription.UTF8String);
  }
  if (!m->depth_pipe_ || !m->color_pipe_ || !m->view_pipe_ || !m->up_color_pipe_ || !m->cast_color_pipe_) {
    std::fprintf(stderr, "gfx: ray-traced shadows: %s\n", err.localizedDescription.UTF8String);
    return nullptr;
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  m->white_ = [dev newTextureWithDescriptor:td];
  m->no_diag_ = [dev newBufferWithLength:sizeof(RtShadowDiag) options:MTLResourceStorageModePrivate];
  if (!m->white_ || !m->no_diag_) return nullptr;
  MTLTextureDescriptor* ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  ad.textureType = MTLTextureType2DArray;
  ad.arrayLength = 1;
  ad.storageMode = MTLStorageModeShared;
  ad.usage = MTLTextureUsageShaderRead;
  m->no_masks_ = [dev newTextureWithDescriptor:ad];
  MTLTextureDescriptor* dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  dd.storageMode = MTLStorageModeShared;
  dd.usage = MTLTextureUsageShaderRead;
  m->no_dist_ = [dev newTextureWithDescriptor:dd];
  if (!m->no_masks_ || !m->no_dist_) return nullptr;
  const float zero = 0.0f;
  [m->no_dist_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&zero bytesPerRow:4];
  const uint8_t one = 0xFF;
  [m->white_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&one bytesPerRow:1];
  return m;
}

id<MTLTexture> RtShadowMasker::NewMask(uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:std::max(w, 1u)
                                                                               height:std::max(h, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

id<MTLTexture> RtShadowMasker::NewLightMasks(uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:std::max(w, 1u)
                                                                               height:std::max(h, 1u)
                                                                            mipmapped:NO];
  td.textureType = MTLTextureType2DArray;
  td.arrayLength = kRtLightSlots;
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

id<MTLTexture> RtShadowMasker::NewLightDist(uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                width:std::max(w, 1u)
                                                                               height:std::max(h, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

void RtShadowMasker::EncodeLights(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth,
                                  id<MTLTexture> masks, id<MTLTexture> dist, const RtShadowParams& p,
                                  const RtLightsParams& lp, const RtLightGpu* lights, id<MTLBuffer> counts,
                                  size_t counts_offset, id<MTLTexture> normals) const {
  if (!f || !f->tlas || !depth || !masks || !dist || !counts || !lights || !lp.count || !lights_color_pipe_) return;
  if (lp.count > kRtLightSlots || masks.arrayLength < lp.count || masks.width < lp.low[0] ||
      masks.height < lp.low[1] || dist.width < lp.low[0] || dist.height < lp.low[1])
    return;
  RtShadowParams q = p;
  q.diag = 0;
  if (normals && (normals.width != depth.width || normals.height != depth.height)) normals = nil;
  q.normals = normals ? 1 : 0;
  // The rays' starts and lights (RGBA16Float, kRtLightsCap layers).
  if (!lights_rays_ || lights_rays_.width != lp.low[0] || lights_rays_.height != lp.low[1]) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:lp.low[0]
                                                                                 height:lp.low[1]
                                                                              mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.arrayLength = kRtLightsCap;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    lights_rays_ = [dev_ newTextureWithDescriptor:td];
    if (!lights_rays_) return;
  }
  const MTLSize grid = MTLSizeMake(lp.low[0], lp.low[1], 1), group = MTLSizeMake(8, 8, 1);
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt local lights starts");
  [e setComputePipelineState:q.depth_texture ? lights_start_depth_pipe_ : lights_start_color_pipe_];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setBytes:&lp length:sizeof lp atIndex:5];
  [e setBytes:lights length:sizeof(RtLightGpu) * lp.count atIndex:8];
  [e setTexture:depth atIndex:0];
  [e setTexture:masks atIndex:1];
  [e setTexture:normals ? normals : white_ atIndex:2];
  [e setTexture:dist atIndex:3];
  [e setTexture:lights_rays_ atIndex:4];
  [e dispatchThreads:grid threadsPerThreadgroup:group];
  [e endEncoding];
  e = TimedCompute(cmd, "rt local lights trace");
  [e setComputePipelineState:q.depth_texture ? lights_depth_pipe_ : lights_color_pipe_];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setBytes:&lp length:sizeof lp atIndex:5];
  [e setIntersectionFunctionTable:q.depth_texture ? lights_depth_table_ : lights_color_table_ atBufferIndex:6];
  [e setBuffer:f->info offset:f->info_offset atIndex:7];
  [e setBytes:lights length:sizeof(RtLightGpu) * lp.count atIndex:8];
  RtUseResources(e, f);
  [e setTexture:depth atIndex:0];
  [e setTexture:masks atIndex:1];
  [e setTexture:lights_rays_ atIndex:4];
  [e dispatchThreads:grid threadsPerThreadgroup:group];
  [e endEncoding];
}

bool RtShadowMasker::EncodeLow(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth,
                               const RtShadowParams& q, const RtLightParams& lp, id<MTLBuffer> counts,
                               size_t counts_offset, id<MTLTexture> normals, uint32_t step, bool turn, uint32_t stride,
                               uint32_t frame, const RtShadowCascade* cascade, id<MTLTexture> cascade_map,
                               id<MTLTexture> low, const char* trace_name) const {
  const uint32_t hw = (q.size[0] + step - 1) / step, hh = (q.size[1] + step - 1) / step;
  // (The turning trace its own starts' texture: the two alternate.)
  __strong id<MTLTexture>& starts_tex = turn ? sparse_starts_ : starts_;
  if (!starts_tex || starts_tex.width != hw || starts_tex.height != hh)
    starts_tex = NewLow(MTLPixelFormatRGBA16Float, hw, hh);
  if (turn && (!sparse_points_ || sparse_points_.width != hw || sparse_points_.height != hh))
    sparse_points_ = NewLow(MTLPixelFormatRGBA16Float, hw, hh);
  id<MTLTexture> starts = starts_tex;
  // The game's cascaded shadow map, its 2 x 2 maxima (RtShadowCascade).
  RtShadowCascade cas{};
  const MTLPixelFormat cf = cascade_map ? cascade_map.pixelFormat : MTLPixelFormatInvalid;
  const bool cdepth =
      cf == MTLPixelFormatDepth32Float || cf == MTLPixelFormatDepth16Unorm || cf == MTLPixelFormatDepth32Float_Stencil8;
  const bool cfloat = cf == MTLPixelFormatR32Float || cf == MTLPixelFormatR16Float || cf == MTLPixelFormatR16Unorm;
  if (cascade && cascade->on && (cdepth || cfloat) && cascade_map.textureType == MTLTextureType2D &&
      cascade_map.width >= cascade_map.height) {
    const uint32_t mw = uint32_t(cascade_map.width + 1) / 2, mh = uint32_t(cascade_map.height + 1) / 2;
    if (!cascade_max_ || cascade_max_.width != mw || cascade_max_.height != mh) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float
                                                                                    width:mw
                                                                                   height:mh
                                                                                mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      cascade_max_ = [dev_ newTextureWithDescriptor:td];
    }
    if (cascade_max_) {
      cas = *cascade;
      cas.atlas[0] = float(cascade_map.width);
      cas.atlas[1] = float(cascade_map.height);
      // (Half floats: 11 bits; 16-bit normalized: 1 / 65535.)
      const bool half_float = cf == MTLPixelFormatR16Float;
      const bool unorm16 = cf == MTLPixelFormatR16Unorm || cf == MTLPixelFormatDepth16Unorm;
      cas.rounding_rel = half_float ? 1.0f / 1024.0f : unorm16 ? 0.0f : 1.0f / 4194304.0f;
      cas.rounding_abs = unorm16 ? 1.0f / 65535.0f : 0.0f;
      id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt sun mask cascade maxima");
      [e setComputePipelineState:cdepth ? max_depth_pipe_ : max_color_pipe_];
      [e setTexture:cascade_map atIndex:0];
      [e setTexture:cascade_max_ atIndex:1];
      [e dispatchThreads:MTLSizeMake(mw, mh, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [e endEncoding];
    }
  }
  if (!low || !starts || (turn && !sparse_points_)) return false;
  // The rays' starts, then the rays (rt_shadow_start, rt_shadow_cast): at
  // the block's top-left pixel, or (turn) at the one turning with the frame.
  const struct {
    uint32_t step, stride, frame, pad;
  } t = {step, stride, frame, 0};
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt sun mask starts");
  [e setComputePipelineState:turn ? (q.depth_texture ? start_turn_depth_pipe_ : start_turn_color_pipe_)
                                  : (q.depth_texture ? start_depth_pipe_ : start_color_pipe_)];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBytes:&lp length:sizeof lp atIndex:5];
  if (turn)
    [e setBytes:&t length:sizeof t atIndex:8];
  else
    [e setBytes:&step length:sizeof step atIndex:8];
  [e setTexture:depth atIndex:0];
  [e setTexture:starts atIndex:1];
  [e setTexture:normals ? normals : white_ atIndex:2];
  [e setBytes:&cas length:sizeof cas atIndex:9];
  [e setTexture:cas.on ? cascade_max_ : white_ atIndex:3];
  if (turn) [e setTexture:sparse_points_ atIndex:4];
  [e dispatchThreads:MTLSizeMake(hw, hh, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  e = TimedCompute(cmd, trace_name);
  [e setComputePipelineState:turn ? (q.depth_texture ? cast_turn_depth_pipe_ : cast_turn_color_pipe_)
                                  : (q.depth_texture ? cast_depth_pipe_ : cast_color_pipe_)];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setIntersectionFunctionTable:turn ? (q.depth_texture ? cast_turn_depth_table_ : cast_turn_color_table_)
                                       : (q.depth_texture ? cast_depth_table_ : cast_color_table_)
                    atBufferIndex:6];
  [e setBuffer:f->info offset:f->info_offset atIndex:7];
  if (turn)
    [e setBytes:&t length:sizeof t atIndex:8];
  else
    [e setBytes:&step length:sizeof step atIndex:8];
  RtUseResources(e, f);
  [e setTexture:depth atIndex:0];
  [e setTexture:low atIndex:1];
  [e setTexture:starts atIndex:3];
  [e dispatchThreads:MTLSizeMake(hw, hh, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
  return true;
}

void RtShadowMasker::EncodeUp(id<MTLCommandBuffer> cmd, id<MTLTexture> depth, const RtShadowParams& q,
                              id<MTLTexture> low, id<MTLTexture> mask, uint32_t step, uint32_t stride,
                              uint32_t frame) const {
  const uint32_t hw = (q.size[0] + step - 1) / step, hh = (q.size[1] + step - 1) / step;
  const struct {
    uint32_t step, stride, frame, pad;
  } t = {step, stride, frame, 0};  // (RtShadowTurn)
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt sun mask upsample");
  [e setComputePipelineState:q.depth_texture ? up_depth_pipe_ : up_color_pipe_];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBytes:&t length:sizeof t atIndex:8];
  [e setTexture:depth atIndex:0];
  [e setTexture:low atIndex:1];
  [e setTexture:mask atIndex:2];
  [e dispatchThreads:MTLSizeMake(hw, hh, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];  // (a block a thread)
  [e endEncoding];
}

id<MTLTexture> RtShadowMasker::NewLow(MTLPixelFormat fmt, uint32_t w, uint32_t h) const {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                width:std::max(w, 1u)
                                                                               height:std::max(h, 1u)
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  return [dev_ newTextureWithDescriptor:td];
}

bool RtShadowMasker::PrepareSparse(uint32_t w, uint32_t h, uint32_t step) const {
  if (!SparseReady() || !w || !h || step < 1 || step > 8) return false;
  const uint32_t hw = (w + step - 1) / step, hh = (h + step - 1) / step;
  if (!sparse_ || sparse_.width != hw || sparse_.height != hh) sparse_ = NewMask(hw, hh);
  if (!sparse_starts_ || sparse_starts_.width != hw || sparse_starts_.height != hh)
    sparse_starts_ = NewLow(MTLPixelFormatRGBA16Float, hw, hh);
  if (!sparse_points_ || sparse_points_.width != hw || sparse_points_.height != hh)
    sparse_points_ = NewLow(MTLPixelFormatRGBA16Float, hw, hh);
  return sparse_ && sparse_starts_ && sparse_points_;
}

id<MTLTexture> RtShadowMasker::EncodeSparse(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth,
                                            const RtShadowParams& p, id<MTLBuffer> counts, size_t counts_offset,
                                            id<MTLTexture> normals, uint32_t step, uint32_t stride, uint32_t frame,
                                            const RtShadowCascade* cascade, id<MTLTexture> cascade_map,
                                            id<MTLTexture> mask) const {
  if (!f || !f->tlas || !depth || !counts || !cast_turn_color_pipe_ || step < 1 || step > 8) return nil;
  RtShadowParams q = p;
  q.diag = 0;
  if (normals && (normals.width != depth.width || normals.height != depth.height)) normals = nil;
  q.normals = normals ? 1 : 0;
  RtLightParams lp{};  // a far light: the whole buffer
  lp.extent[0] = q.size[0];
  lp.extent[1] = q.size[1];
  if (!lp.extent[0] || !lp.extent[1]) return nil;
  const uint32_t hw = (q.size[0] + step - 1) / step, hh = (q.size[1] + step - 1) / step;
  if (!sparse_ || sparse_.width != hw || sparse_.height != hh) sparse_ = NewMask(hw, hh);
  if (!EncodeLow(cmd, f, depth, q, lp, counts, counts_offset, normals, step, true, stride, frame, cascade, cascade_map,
                 sparse_, "rt sun mask trace (sparse)"))
    return nil;
  if (mask) EncodeUp(cmd, depth, q, sparse_, mask, step, stride, frame);
  return sparse_;
}

void RtShadowMasker::Encode(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> depth, id<MTLTexture> mask,
                            const RtShadowParams& p, id<MTLBuffer> counts, size_t counts_offset, id<MTLBuffer> diag,
                            const RtLightParams* light, id<MTLTexture> normals, uint32_t step,
                            const RtShadowCascade* cascade, id<MTLTexture> cascade_map) const {
  if (!f || !f->tlas || !depth || !mask || !counts) return;
  RtShadowParams q = p;
  if (!diag) q.diag = 0;
  // (The normals only at the depth's size.)
  if (normals && (normals.width != depth.width || normals.height != depth.height)) normals = nil;
  q.normals = normals ? 1 : 0;
  // A far light: the whole buffer.
  RtLightParams lp{};
  if (light) {
    lp = *light;
  } else {
    lp.extent[0] = q.size[0];
    lp.extent[1] = q.size[1];
  }
  if (!lp.extent[0] || !lp.extent[1]) return;
  // A far light without diagnostics: traced at one pixel of every step x
  // step (half the resolution, or a quarter for a dim light) and upsampled
  // by depth (RECOMP_RT_SHADOW_FULL=1: every pixel traced).
  const char* full_env = std::getenv("RECOMP_RT_SHADOW_FULL");
  const bool full = full_env && *full_env == '1';
  if (!light && !q.diag && !full) {
    step = step == kRtShadowDimStep ? kRtShadowDimStep : 2;
    const uint32_t hw = (q.size[0] + step - 1) / step, hh = (q.size[1] + step - 1) / step;
    __strong id<MTLTexture>& low = step == 2 ? half_ : dim_;
    if (!low || low.width != hw || low.height != hh) low = NewMask(hw, hh);
    if (EncodeLow(cmd, f, depth, q, lp, counts, counts_offset, normals, step, false, 0, 0, cascade, cascade_map, low,
                  step == 2 ? "rt sun mask trace (half)" : "rt sun mask trace (dim)")) {
      EncodeUp(cmd, depth, q, low, mask, step);
      return;
    }
  }
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt shadow mask");
  [e setComputePipelineState:q.depth_texture ? depth_pipe_ : color_pipe_];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setBuffer:q.diag ? diag : no_diag_ offset:0 atIndex:4];
  [e setBytes:&lp length:sizeof lp atIndex:5];
  [e setIntersectionFunctionTable:q.depth_texture ? depth_table_ : color_table_ atBufferIndex:6];
  [e setBuffer:f->info offset:f->info_offset atIndex:7];
  RtUseResources(e, f);
  [e setTexture:depth atIndex:0];
  [e setTexture:mask atIndex:1];
  [e setTexture:normals ? normals : white_ atIndex:2];
  [e dispatchThreads:MTLSizeMake(lp.extent[0], lp.extent[1], 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [e endEncoding];
}

}  // namespace recomp::gfx
