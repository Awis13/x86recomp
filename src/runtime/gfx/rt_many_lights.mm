// A frame's many local lights lit with ray-traced shadows (rt_many_lights.h).

#import "runtime/gfx/rt_many_lights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>

#include "runtime/gfx/gpu_timer.h"

namespace recomp::gfx {

const char* const kRtManyLightsMsl = R"(
struct RtManyLightGpu {
  float4 position;   // xyz relative to the camera; w 1 / radius^2
  float4 direction;  // xyz the way a spot shines; w its cone's offset
  float4 colour;     // rgb; w the cone's scale
  uint flags;        // model (bits 0-1), pixel class (bits 2-3)
  float lamp_gap;
  float reach;
  float pad;
};
struct RtManyModel {
  float falloff_power, falloff_bias, falloff_scale, wrap;
  float wrap_scale, spec_scale, exponent_scale;
  uint flags;  // 1 specular, 2 normal_low_bits
  uint factor_classes;
  uint pad0, pad1, pad2;
};
struct RtManyParams {
  uint2 size, blocks, tiles;
  uint frame, stride, count, textures;
  float rest_share;
  uint empty;  // 1: a group nothing lights is skipped (RECOMP_RT_MANY_EMPTY)
  RtManyModel models[4];
  uint classes[3][8];
};
// What a pixel's G-buffer holds, as the models read it.
struct RtManySurface {
  float3 q;       // the point, relative to the camera
  float3 i;       // from the camera to the point, unit
  float3 n_hi, n_lo;  // the normal, with and without the w channel's bits
  float3 r_hi, r_lo;  // the view's reflection about each
  float3 albedo;
  float spec, gloss2, factor;  // specular intensity, gloss^2, saturate(z^2 + 0.5)
  uint value;     // the class value
};
float3 rt_many_normal_hi(float4 t) {
  const float3 e = fract(t.w * float3(0.99609375, 7.96875, 63.75));
  float3 r = e + e;
  r.xy += e.yz * -0.25;
  return normalize(t.xyz * 256.0 + r - 127.99999);
}
float3 rt_many_normal_lo(float4 t) { return normalize(t.xyz * 2.0 - 0.99999); }
RtManySurface rt_many_surface(float3 q, uint2 gid, texture2d<float, access::read> albedo,
                              texture2d<float, access::read> normal, texture2d<float, access::read> surface,
                              texture2d<float, access::read> cls, constant RtManyParams& mp) {
  RtManySurface s;
  s.q = q;
  s.i = normalize(q + 1e-5);
  const float4 n = normal.read(gid);
  s.n_hi = rt_many_normal_hi(n);
  s.n_lo = rt_many_normal_lo(n);
  s.r_hi = s.i - 2.0 * dot(s.i, s.n_hi) * s.n_hi;
  s.r_lo = s.i - 2.0 * dot(s.i, s.n_lo) * s.n_lo;
  s.albedo = albedo.read(gid).rgb;
  // (No surface: no specular, factor 1.)
  const float3 sf = (mp.textures & 1u) != 0u ? surface.read(gid).xyz : float3(0.0, 0.0, 1.0);
  s.spec = sf.x;
  s.gloss2 = sf.y * sf.y;
  s.factor = saturate(sf.z * sf.z + 0.5);
  s.value = (mp.textures & 2u) != 0u ? uint(round(saturate(cls.read(gid).x) * 255.0)) : 0u;
  return s;
}
bool rt_many_in_class(constant RtManyParams& mp, uint c, uint v) {
  return c == 0u || ((mp.classes[c - 1u][v >> 5] >> (v & 31u)) & 1u) != 0u;
}
// Light l's at surface s, unshadowed (RayLightModel). (The cheap tests
// first: the sphere, the cone, the falloff, the class.)
float3 rt_many_light(thread const RtManyLightGpu& l, constant RtManyParams& mp, thread const RtManySurface& s) {
  const float3 to = l.position.xyz - s.q;
  const float a = saturate(1.0 - dot(to, to) * l.position.w);
  if (!(a > 0.0)) return float3(0.0);
  const float3 L = normalize(to + 1e-5);
  const float k = saturate((dot(L, -l.direction.xyz) - l.direction.w) * l.colour.w);
  if (!(k > 0.0)) return float3(0.0);
  constant RtManyModel& m = mp.models[l.flags & 3u];
  const float a2 = a * a;
  const float ap = m.falloff_power == 4.0 ? a2 * a2 : m.falloff_power == 2.0 ? a2 : pow(a, m.falloff_power);
  const float f = ap - m.falloff_bias;
  const uint c = (l.flags >> 2) & 3u;
  if (!(f >= 0.0) || !rt_many_in_class(mp, c, s.value)) return float3(0.0);
  const float3 col = l.colour.rgb * (f * m.falloff_scale * k);
  const bool hi = (m.flags & 2u) != 0u;
  float3 r = col * (saturate((dot(L, hi ? s.n_hi : s.n_lo) - m.wrap) * m.wrap_scale) * s.albedo);
  if ((m.flags & 1u) != 0u && s.spec > 0.0) {
    const float e = s.gloss2 * m.exponent_scale;
    const float sa = saturate(dot(L, hi ? s.r_hi : s.r_lo));
    r += col * (m.spec_scale * s.spec * (e > 0.0 ? pow(sa, e) : 1.0));
  }
  if (((m.factor_classes >> c) & 1u) != 0u) r *= s.factor;
  return r;
}
float rt_many_lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
uint rt_many_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Whether point q (relative to the camera; n its geometric normal, toward
// the camera; off its ray's offset, gap the captured instances') sees light
// l: a ray as the local lights' masks trace it (rt_lights_start,
// rt_lights_cast); none where the surface faces away (in shadow) or the
// start lies within the lamp's gap (lit).
bool rt_many_seen(instance_acceleration_structure scene, RtShadowTable table, constant RtShadowParams& p, float3 q,
                  float3 n, float off, float gap, thread const RtManyLightGpu& l, const device RtMaterial* materials,
                  const device RtInfo* info, thread uint& traced, thread uint& blocked,
                  device atomic_uint* dc = nullptr, bool centre = false) {
  const float3 lpos = l.position.xyz;
  const float3 lv = normalize(lpos - q);
  const float nl = dot(n, lv);
  if (!(nl >= 0.0)) return false;
  const float along_n = off * saturate(1.0 - nl / p.min_slope);
  const float along_l = min(off / max(nl, p.min_slope), 0.5 * length(lpos - q));
  const float3 start = q + n * along_n + lv * along_l;
  const float3 to = lpos - start;
  const float dl = length(to);
  const float tmax = min(dl - l.lamp_gap, l.reach);
  if (!(tmax > 0.0)) return true;
  bool again = false;
  traced++;
  const uint masks = as_type<uint>(l.pad) & 0xFFFFu;
  const bool b = rt_shadow_blocked(scene, table, p.eye.xyz + start, to / dl, tmax, min(gap, tmax), p.eye.xyz + q,
                                   normalize(q), materials, info, again, masks ? masks : 15u);
  blocked += b ? 1u : 0u;
  // (RECOMP_RT_LAMP_PROBE's diagnostics: what blocked it, by mesh.)
  if (dc != nullptr && b) {
    const uint mk = masks ? masks : 15u;
    RtHit h = rt_trace(scene, ray(p.eye.xyz + start, to / dl, 0.0, tmax), mk & 3u, materials, intersection_params(),
                       false);
    if (!h.hit) h = rt_trace(scene, ray(p.eye.xyz + start, to / dl, 0.0, tmax), mk & 12u, materials,
                             intersection_params(), false);
    if (h.hit) {
      const RtInfo in = info[h.instance];
      atomic_fetch_add_explicit(&dc[64u + (in.mesh & 4095u)], 1u, memory_order_relaxed);
      atomic_store_explicit(&dc[64u + 4096u + (in.mesh & 4095u)], in.triangles, memory_order_relaxed);
      atomic_fetch_add_explicit(&dc[64u + 8192u + min(uint(h.distance * 1000.0), 63u)], 1u, memory_order_relaxed);
      if (centre) {
        atomic_store_explicit(&dc[4], as_type<uint>(h.distance), memory_order_relaxed);
        atomic_store_explicit(&dc[5], in.draw, memory_order_relaxed);
        atomic_store_explicit(&dc[6], in.mesh, memory_order_relaxed);
        atomic_store_explicit(&dc[7], in.triangles, memory_order_relaxed);
        atomic_store_explicit(&dc[8], h.front ? 1u : 0u, memory_order_relaxed);
        atomic_store_explicit(&dc[9], in.flags, memory_order_relaxed);
        atomic_store_explicit(&dc[10], h.primitive, memory_order_relaxed);
        const float3 o = p.eye.xyz + start, dir = to / dl;
        atomic_store_explicit(&dc[11], as_type<uint>(o.x), memory_order_relaxed);
        atomic_store_explicit(&dc[12], as_type<uint>(o.y), memory_order_relaxed);
        atomic_store_explicit(&dc[13], as_type<uint>(o.z), memory_order_relaxed);
        atomic_store_explicit(&dc[14], as_type<uint>(dir.x), memory_order_relaxed);
        atomic_store_explicit(&dc[15], as_type<uint>(dir.y), memory_order_relaxed);
        atomic_store_explicit(&dc[16], as_type<uint>(dir.z), memory_order_relaxed);
        atomic_store_explicit(&dc[17], as_type<uint>(n.x), memory_order_relaxed);
        atomic_store_explicit(&dc[18], as_type<uint>(n.y), memory_order_relaxed);
        atomic_store_explicit(&dc[19], as_type<uint>(n.z), memory_order_relaxed);
        atomic_store_explicit(&dc[20], as_type<uint>(off), memory_order_relaxed);
        atomic_store_explicit(&dc[21], 1u, memory_order_relaxed);
      }
    }
  }
  return !b;
}
// Pixel c's geometric normal (as rt_shadow_ray's: from the depth of its
// neighbours on its surface; else the G-buffer's), facing the camera.
template <typename D>
float3 rt_many_geometric(D t, constant RtShadowParams& p, int2 c, float v, float3 q, float3 fallback) {
  float3 n = fallback;
  float3 dx, dy;
  if (rt_step(t, p, c, int2(1, 0), v, q, dx) && rt_step(t, p, c, int2(0, 1), v, q, dy)) {
    const float3 m = cross(dx, dy);
    if (length_squared(m) > 0.0) n = m;
  }
  n = normalize(n);
  return dot(n, q) > 0.0 ? -n : n;
}
// A threadgroup's (16 x 16 threads) lights: those of its tile's list whose
// sphere reaches the box around the group's points (each thread's, where
// `valid`), copied in (any order), with their places in the frame's list.
constant uint RT_MANY_GROUP = 256u;
struct RtManyShared {
  RtManyLightGpu lights[128];  // (kRtManyTileCap)
  ushort ids[128];
  atomic_uint n;
  atomic_uint lit;             // the shading's: some pixel of the group came out lit
  float4 lo[16], hi[16];       // per SIMD group
};
void rt_many_gather(threadgroup RtManyShared& sh, constant RtManyLightGpu* lights, constant ushort* index, uint first,
                    uint n_in, bool valid, float3 q, uint lid, uint sid, uint lane, uint nsimd) {
  float3 lo = valid ? q : float3(INFINITY), hi = valid ? q : float3(-INFINITY);
  lo = simd_min(lo);
  hi = simd_max(hi);
  if (lid == 0u) atomic_store_explicit(&sh.n, 0u, memory_order_relaxed);
  if (lane == 0u) sh.lo[sid] = float4(lo, 0.0), sh.hi[sid] = float4(hi, 0.0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float3 blo = sh.lo[0].xyz, bhi = sh.hi[0].xyz;
  for (uint k = 1u; k < nsimd; k++) blo = min(blo, sh.lo[k].xyz), bhi = max(bhi, sh.hi[k].xyz);
  if (all(blo <= bhi))
    for (uint k = lid; k < min(n_in, 128u); k += RT_MANY_GROUP) {
      const ushort i = index[first + k];
      const RtManyLightGpu l = lights[i];
      const float3 d = l.position.xyz - clamp(l.position.xyz, blo, bhi);
      if (!(dot(d, d) * l.position.w < 1.0)) continue;
      const uint at = atomic_fetch_add_explicit(&sh.n, 1u, memory_order_relaxed);
      sh.lights[at] = l;
      sh.ids[at] = i;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// A thread for one block (2 x 2 pixels), a group for a tile (32 x 32).
template <typename D>
void rt_many_trace(D t, texture2d<float, access::read> albedo, texture2d<float, access::read> normal,
                   texture2d<float, access::read> surface, texture2d<float, access::read> cls,
                   instance_acceleration_structure scene, RtShadowTable table, constant RtShadowParams& p,
                   constant RtManyParams& mp, constant RtManyLightGpu* lights, constant uint* heads,
                   constant ushort* index, const device RtMaterial* materials, const device RtInfo* info,
                   texture2d<uint, access::write> out, device atomic_uint* counts, device atomic_uint* retain_hits,
                   threadgroup RtManyShared& sh,
                   uint2 tid, uint2 group, uint lid, uint sid, uint lane, uint nsimd) {
  uint traced = 0u, blocked = 0u;
  const bool in = all(tid < mp.blocks);
  const uint2 gid = min(rt_light_sample(min(tid, mp.blocks - 1u), mp.frame, mp.stride, 2u), mp.size - 1u);
  const float v = rt_depth(t, gid);
  const bool valid = in && !rt_sky(p, v);
  const int2 c = int2(gid);
  const float3 q = valid ? rt_point(p, c, v) : float3(0.0);
  const uint ti = min(group.y, mp.tiles.y - 1u) * mp.tiles.x + min(group.x, mp.tiles.x - 1u);
  rt_many_gather(sh, lights, index, heads[2u * ti], heads[2u * ti + 1u], valid, q, lid, sid, lane, nsimd);
  uint2 res = uint2(0u);
  if (valid) {
    const float d = length(q);
    res.y = as_type<uint>(d);
    // (RECOMP_RT_LAMP_PROBE's 4th number, mp.empty bits 8-15, to measure: at
    // most that many of the group's lights, whichever they are.)
    const uint n_cap = (mp.empty >> 8) & 255u;
    const uint n_list = n_cap ? min(atomic_load_explicit(&sh.n, memory_order_relaxed), n_cap)
                              : atomic_load_explicit(&sh.n, memory_order_relaxed);
    // (mp.empty: a group whose gather kept no light picks none whatever its
    // surfaces hold, so it does not read them, four textures a thread. Its
    // samples are written all the same, no light: the shading reads them
    // from across the tile's border.)
    // Contact rays (mp.empty bit 4: RtManyLights::Encode's retain_hits;
    // bit 2: RECOMP_RT_LAMP_PROBE's counts too): from the point along its
    // surface's normal, 5 cm near the camera and 2 mm more a metre away. A
    // hit is a second surface lying on the visible one; on a retained
    // instance (kept from earlier frames, not drawn in this one) it is a
    // model the game no longer draws there, a far model in place of the
    // near ones, say: counted by its slot for RtScene::DropRetained.
    if ((mp.empty & 6u) != 0u) {
      const float3 nc = rt_many_geometric(t, p, c, v, q, -normalize(q));
      const float offc = d * (p.offset_pixels * p.pixel + p.offset_view) + length(p.eye.xyz + q) * p.offset_world;
      // (5 cm near the camera, 2 mm more per metre away: a far model's
      // surfaces lie further off the near one's.)
      const RtHit hc = rt_trace(scene, ray(p.eye.xyz + q + nc * offc, nc, 0.0, 0.05 + 0.002 * d), 3u, materials,
                                intersection_params(), false);
      if ((mp.empty & 4u) != 0u && hc.hit && (info[hc.instance].flags & (1u << 18)) != 0u)
        atomic_fetch_add_explicit(&retain_hits[min(hc.instance, 65535u)], 1u, memory_order_relaxed);
    }
    if ((mp.empty & 2u) != 0u) {
      const float3 nc = rt_many_geometric(t, p, c, v, q, -normalize(q));
      const float offc = d * (p.offset_pixels * p.pixel + p.offset_view) + length(p.eye.xyz + q) * p.offset_world;
      const RtHit hc = rt_trace(scene, ray(p.eye.xyz + q + nc * offc, nc, 0.0, 0.05 + 0.002 * d), 3u, materials,
                                intersection_params(), false);
      atomic_fetch_add_explicit(&counts[22], 1u, memory_order_relaxed);
      const uint band = d < 20.0 ? 0u : d < 60.0 ? 1u : d < 150.0 ? 2u : 3u;
      atomic_fetch_add_explicit(&counts[26u + band], 1u, memory_order_relaxed);
      if (hc.hit) atomic_fetch_add_explicit(&counts[30u + band], 1u, memory_order_relaxed);
      if (hc.hit) {
        const RtInfo ic = info[hc.instance];
        const bool kept = (ic.flags & (1u << 18)) != 0u;
        atomic_fetch_add_explicit(&counts[kept ? 24 : 23], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&counts[64u + 8192u + 64u + (ic.mesh & 4095u)], 1u, memory_order_relaxed);
        // (1 retained, 2 the main view's draw, 4 another view's: kRtInfoOtherView.)
        const uint kind = kept ? 1u : (ic.flags & (1u << 16)) != 0u ? 4u : 2u;
        atomic_fetch_or_explicit(&counts[64u + 8192u + 64u + 4096u + (ic.mesh & 4095u)], kind, memory_order_relaxed);
        if (kind == 4u) atomic_fetch_add_explicit(&counts[25], 1u, memory_order_relaxed);
        atomic_store_explicit(&counts[64u + 4096u + (ic.mesh & 4095u)], ic.triangles, memory_order_relaxed);
      }
    }
    if (n_list != 0u || (mp.empty & 1u) == 0u) {
      const RtManySurface s = rt_many_surface(q, gid, albedo, normal, surface, cls, mp);
      // The strongest, and one of the rest in proportion to its light
      // (weighted reservoir sampling; a light the strongest displaces joins
      // the rest).
      uint h = rt_many_hash(gid.x * 0x8da6b343u ^ gid.y * 0xd8163841u ^ mp.frame * 0x9e3779b9u);
      float total = 0.0, best = 0.0, rest = 0.0;
      uint b = 0xFFFFu, pick = 0xFFFFu;
      for (uint k = 0u; k < n_list; k++) {
        const RtManyLightGpu l = sh.lights[k];
        float w = rt_many_lum(rt_many_light(l, mp, s));
        if (!(w > 0.0)) continue;
        total += w;
        uint j = k;
        if (w > best) {
          const uint ob = b;
          const float obw = best;
          b = k, best = w;
          if (ob == 0xFFFFu) continue;
          j = ob, w = obw;
        }
        rest += w;
        h = h * 1664525u + 1013904223u;
        if (float(h >> 8) * (1.0 / 16777216.0) * rest < w) pick = j;
      }
      if (b != 0xFFFFu) {
        const float3 n = rt_many_geometric(t, p, c, v, q, s.n_lo);
        const float3 world = p.eye.xyz + q;
        const float off = d * (p.offset_pixels * p.pixel + p.offset_view) + length(world) * p.offset_world;
        const float gap = d * p.captured_gap;
        const RtManyLightGpu lb = sh.lights[b];
        const bool probe = (as_type<uint>(lb.pad) & 0x10000u) != 0u;
        const bool s1 = rt_many_seen(scene, table, p, q, n, off, gap, lb, materials, info, traced, blocked,
                                     probe ? counts : nullptr, probe && all(tid == mp.blocks / 2u));
        uint x = (uint(sh.ids[b]) + 1u) | (s1 ? 1u << 11 : 0u);
        if (pick != 0xFFFFu && rest >= mp.rest_share * total) {
          const RtManyLightGpu lp = sh.lights[pick];
          const bool s2 = rt_many_seen(scene, table, p, q, n, off, gap, lp, materials, info, traced, blocked);
          x |= ((uint(sh.ids[pick]) + 1u) << 12) | (s2 ? 1u << 23 : 0u);
        }
        res.x = x;
      }
    }
  }
  if (in) out.write(uint4(res, 0u, 0u), tid);
  const uint nt = simd_sum(traced), nb = simd_sum(blocked);
  if (simd_is_first() && nt) {
    atomic_fetch_add_explicit(&counts[0], nt, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], nb, memory_order_relaxed);
  }
}
#define RT_MANY_GROUP_ARGS                                                                                        \
  uint2 group [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],                         \
      uint sid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                       \
      uint nsimd [[simdgroups_per_threadgroup]]
#define RT_MANY_TRACE(KIND, DEPTH)                                                                                \
  kernel void rt_many_trace_##KIND##_cs(                                                                          \
      instance_acceleration_structure scene [[buffer(0)]], constant RtShadowParams& p [[buffer(1)]],              \
      const device RtMaterial* materials [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],                 \
      constant RtManyParams& mp [[buffer(4)]], RtShadowTable table [[buffer(5)]],                                 \
      const device RtInfo* info [[buffer(6)]], constant RtManyLightGpu* lights [[buffer(7)]],                     \
      constant uint* heads [[buffer(8)]], constant ushort* index [[buffer(9)]],                                   \
      device atomic_uint* retain_hits [[buffer(10)]], DEPTH depth [[texture(0)]],                                 \
      texture2d<float, access::read> albedo [[texture(1)]], texture2d<float, access::read> normal [[texture(2)]], \
      texture2d<float, access::read> surface [[texture(3)]], texture2d<float, access::read> cls [[texture(4)]],   \
      texture2d<uint, access::write> out [[texture(5)]], uint2 tid [[thread_position_in_grid]],                  \
      RT_MANY_GROUP_ARGS) {                                                                                       \
    threadgroup RtManyShared sh;                                                                                  \
    rt_many_trace(depth, albedo, normal, surface, cls, scene, table, p, mp, lights, heads, index, materials, info, \
                  out, counts, retain_hits, sh, tid, group, lid, sid, lane, nsimd);                               \
  }
typedef depth2d<float, access::read> RtManyDepthTex;
typedef texture2d<float, access::read> RtManyColorTex;
RT_MANY_TRACE(depth, RtManyDepthTex)
RT_MANY_TRACE(color, RtManyColorTex)
// A thread a pixel, a group for 16 x 16 of them: the pixel's light, its
// lights' (the group's) at the pixel, the strongest times its visibility
// and the rest times theirs, from the samples of its block and the three
// nearest on its surface (bilinear weights by place, times one by how far
// the sample's point lies off the pixel's plane for its distance from it),
// into `lit`.
// With mp.empty, a group none of whose pixels comes out lit (most of a
// day's frame) stops as soon as it knows: where its gather kept no light
// its surfaces are not read; `lit` is not written, and the group's flag (a
// texel of `flags`, by its first thread) is 0, by which the addition's draw
// leaves its pixels alone (rt_many_add_flag_fs). A group with some pixel
// lit: flag 1, every pixel written, as without it. Counted at counts[2] the
// groups nothing lit, at counts[3] those of them whose gather kept no light.
template <typename D>
void rt_many_shade(D t, texture2d<float, access::read> albedo, texture2d<float, access::read> normal,
                   texture2d<float, access::read> surface, texture2d<float, access::read> cls,
                   texture2d<uint, access::read> samples, constant RtShadowParams& p, constant RtManyParams& mp,
                   constant RtManyLightGpu* lights, constant uint* heads, constant ushort* index,
                   texture2d<half, access::write> lit, texture2d<uint, access::write> flags,
                   instance_acceleration_structure scene, RtShadowTable table, const device RtMaterial* materials,
                   const device RtInfo* info, device atomic_uint* counts, threadgroup RtManyShared& sh, uint2 gid,
                   uint2 group, uint lid, uint sid, uint lane, uint nsimd) {
  uint traced = 0u, blocked = 0u;
  const bool in = all(gid < mp.size);
  const uint2 g = min(gid, mp.size - 1u);
  const float v = rt_depth(t, g);
  const bool valid = in && !rt_sky(p, v);
  const float3 q = valid ? rt_point(p, int2(g), v) : float3(0.0);
  const uint2 tile = min(group / 2u, mp.tiles - 1u);
  const uint ti = tile.y * mp.tiles.x + tile.x;
  const bool skip = (mp.empty & 1u) != 0u;
  // (Before the gather's barriers: every thread finds it 0 after them.)
  if (skip && lid == 0u) atomic_store_explicit(&sh.lit, 0u, memory_order_relaxed);
  rt_many_gather(sh, lights, index, heads[2u * ti], heads[2u * ti + 1u], valid, q, lid, sid, lane, nsimd);
  // (The same for every thread of the group: all the gather's writes lie
  // before its last barrier.)
  const uint n_cap = (mp.empty >> 8) & 255u;
  const uint n_list = n_cap ? min(atomic_load_explicit(&sh.n, memory_order_relaxed), n_cap)
                            : atomic_load_explicit(&sh.n, memory_order_relaxed);
  float3 out = float3(0.0);
  if (valid && (n_list != 0u || !skip)) {
    const RtManySurface s = rt_many_surface(q, g, albedo, normal, surface, cls, mp);
    float3 total = float3(0.0), top = float3(0.0);
    float best = 0.0;
    uint b = 0xFFFFu;
    for (uint k = 0u; k < n_list; k++) {
      const RtManyLightGpu l = sh.lights[k];
      const float3 c = rt_many_light(l, mp, s);
      const float w = rt_many_lum(c);
      if (!(w > 0.0)) continue;
      total += c;
      if (w > best) best = w, b = k, top = c;
    }
    if (b != 0xFFFFu) {
      const int2 b0 = int2(g / 2u);
      const int2 o = int2((g.x & 1u) != 0u ? 1 : -1, (g.y & 1u) != 0u ? 1 : -1);
      const uint want = uint(sh.ids[b]) + 1u;
      const float d = length(q);
      // The four samples; where they are alike (the same lights, seen
      // alike) and lie at the pixel's distance (within 2%), theirs is the
      // pixel's (no plane to weigh them by).
      uint2 r[4];
      bool have[4];
      bool alike = true;
      for (uint k = 0u; k < 4u; k++) {
        const int2 at = b0 + int2((k & 1u) != 0u ? o.x : 0, (k & 2u) != 0u ? o.y : 0);
        have[k] = all(at >= 0) && all(uint2(at) < mp.blocks);
        r[k] = have[k] ? samples.read(uint2(at)).xy : uint2(0u);
        have[k] = have[k] && (r[k].x & 0x7FFu) != 0u;
        alike = alike && have[k] && r[k].x == r[0].x && abs(as_type<float>(r[k].y) - d) <= 0.02 * d;
      }
      float vis_top = 0.0, vis_rest = 0.0;
      if (alike && (r[0].x & 0x7FFu) == want) {
        vis_top = float((r[0].x >> 11) & 1u);
        vis_rest = ((r[0].x >> 12) & 0x7FFu) != 0u ? float((r[0].x >> 23) & 1u) : vis_top;
      } else {
        const float3 ng = rt_many_geometric(t, p, int2(g), v, q, s.n_lo);
        float a1 = 0.0, w1 = 0.0, ar = 0.0, wr = 0.0;
        for (uint k = 0u; k < 4u; k++) {
          if (!have[k]) continue;
          const int2 at = b0 + int2((k & 1u) != 0u ? o.x : 0, (k & 2u) != 0u ? o.y : 0);
          const uint first_l = r[k].x & 0x7FFu, second_l = (r[k].x >> 12) & 0x7FFu;
          const float v1 = float((r[k].x >> 11) & 1u), v2 = float((r[k].x >> 23) & 1u);
          const uint2 sp = min(rt_light_sample(uint2(at), mp.frame, mp.stride, 2u), mp.size - 1u);
          const float3 dv = rt_point(p, int2(sp), rt_depth(t, sp)) - q;
          const float len = length(dv);
          const float z = len > 0.0 ? saturate(1.0 - 5.0 * abs(dot(dv, ng)) / len) : 1.0;
          // (By place: the pixel's own block 9/16, beside 3/16, across 1/16.)
          const float w = (k == 0u ? 0.5625 : k == 3u ? 0.0625 : 0.1875) * z * z;
          if (!(w > 0.0)) continue;
          if (first_l == want) {
            a1 += w * v1, w1 += w;
            if (second_l != 0u) ar += w * v2, wr += w;
          } else if (second_l == want) {
            a1 += w * v2, w1 += w;
            ar += w * v1, wr += w;
          } else {
            ar += w * (second_l != 0u ? 0.5 * (v1 + v2) : v1), wr += w;
          }
        }
        if (w1 > 0.0 || wr > 0.0) {
          vis_top = w1 > 0.0 ? a1 / w1 : ar / wr;
          vis_rest = wr > 0.0 ? ar / wr : vis_top;
        } else {
          // No sample on the pixel's surface (all on another, at an
          // object's edge): its own ray to its strongest light, for all.
          const float3 world = p.eye.xyz + q;
          const float off = d * (p.offset_pixels * p.pixel + p.offset_view) + length(world) * p.offset_world;
          const RtManyLightGpu lb = sh.lights[b];
          vis_top = vis_rest =
              rt_many_seen(scene, table, p, q, ng, off, d * p.captured_gap, lb, materials, info, traced, blocked)
                  ? 1.0
                  : 0.0;
        }
      }
      out = top * vis_top + (total - top) * vis_rest;
    }
  }
  bool store = in;
  if (skip) {
    // Whether any pixel of the group is lit: its light's bits other than a
    // zero's (so a pixel that is not a number counts as lit and is written
    // as it was). Both conditions the same for every thread of the group,
    // as a barrier needs.
    bool any = false;
    if (n_list != 0u) {
      if (((as_type<uint>(out.x) | as_type<uint>(out.y) | as_type<uint>(out.z)) << 1) != 0u)
        atomic_store_explicit(&sh.lit, 1u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      any = atomic_load_explicit(&sh.lit, memory_order_relaxed) != 0u;
    }
    if (lid == 0u) {
      flags.write(uint4(any ? 1u : 0u, 0u, 0u, 0u), group);
      if (!any) {
        atomic_fetch_add_explicit(&counts[2], 1u, memory_order_relaxed);
        if (n_list == 0u) atomic_fetch_add_explicit(&counts[3], 1u, memory_order_relaxed);
      }
    }
    store = in && any;
  }
  if (store) lit.write(half4(half3(out), 0.0h), gid);
  const uint nt = simd_sum(traced), nb = simd_sum(blocked);
  if (simd_is_first() && nt) {
    atomic_fetch_add_explicit(&counts[0], nt, memory_order_relaxed);
    atomic_fetch_add_explicit(&counts[1], nb, memory_order_relaxed);
  }
}
#define RT_MANY_SHADE(KIND, DEPTH)                                                                                \
  kernel void rt_many_shade_##KIND##_cs(                                                                          \
      instance_acceleration_structure scene [[buffer(0)]], constant RtShadowParams& p [[buffer(1)]],              \
      const device RtMaterial* materials [[buffer(2)]], device atomic_uint* counts [[buffer(3)]],                 \
      constant RtManyParams& mp [[buffer(4)]], RtShadowTable table [[buffer(5)]],                                 \
      const device RtInfo* info [[buffer(6)]],                                                                    \
      constant RtManyLightGpu* lights [[buffer(7)]], constant uint* heads [[buffer(8)]],                          \
      constant ushort* index [[buffer(9)]], DEPTH depth [[texture(0)]],                                           \
      texture2d<float, access::read> albedo [[texture(1)]], texture2d<float, access::read> normal [[texture(2)]], \
      texture2d<float, access::read> surface [[texture(3)]], texture2d<float, access::read> cls [[texture(4)]],   \
      texture2d<uint, access::read> samples [[texture(5)]], texture2d<half, access::write> lit [[texture(6)]],    \
      texture2d<uint, access::write> flags [[texture(7)]], uint2 gid [[thread_position_in_grid]],                 \
      RT_MANY_GROUP_ARGS) {                                                                                       \
    threadgroup RtManyShared sh;                                                                                  \
    rt_many_shade(depth, albedo, normal, surface, cls, samples, p, mp, lights, heads, index, lit, flags, scene,   \
                  table, materials, info, counts, sh, gid, group, lid, sid, lane, nsimd);                         \
  }
RT_MANY_SHADE(depth, RtManyDepthTex)
RT_MANY_SHADE(color, RtManyColorTex)
// The light added into the target (a full-screen triangle).
struct RtManyVsOut {
  float4 pos [[position]];
};
vertex RtManyVsOut rt_many_vs(uint vid [[vertex_id]]) {
  const float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
  RtManyVsOut o;
  o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
fragment float4 rt_many_add_fs(RtManyVsOut in [[stage_in]], texture2d<half, access::read> lit [[texture(0)]]) {
  return float4(float3(lit.read(uint2(in.pos.xy)).rgb), 0.0);
}
// The same after a shading with mp.empty: nothing drawn where the pixel's
// group (16 x 16: kRtManyGroup) has flag 0 (nothing lit in it; its texels of
// `lit` are an earlier frame's and are not read). The flags at texture 30
// (kRtManyFlagsTexture).
fragment float4 rt_many_add_flag_fs(RtManyVsOut in [[stage_in]], texture2d<half, access::read> lit [[texture(0)]],
                                    texture2d<uint, access::read> flags [[texture(30)]]) {
  const uint2 at = uint2(in.pos.xy);
  if (flags.read(at / 16u).x == 0u) {
    discard_fragment();
    return float4(0.0);
  }
  return float4(float3(lit.read(at).rgb), 0.0);
}
)";
static_assert(kRtManyGroup == 16 && kRtManyFlagsTexture == 30,
              "kRtManyLightsMsl: rt_many_add_flag_fs's group and texture, the shading's 16 x 16 threads");

RtManyModel RtManyModelFor(const RayLightModel& m) {
  RtManyModel g{};
  g.falloff_power = m.falloff_power;
  g.falloff_bias = m.falloff_bias;
  g.falloff_scale = m.falloff_scale;
  g.wrap = m.wrap;
  g.wrap_scale = m.wrap_scale;
  g.spec_scale = m.spec_scale;
  g.exponent_scale = m.exponent_scale;
  g.flags = (m.specular ? 1u : 0u) | (m.normal_low_bits ? 2u : 0u);
  g.factor_classes = m.factor_classes;
  return g;
}

bool g_lamp_contact_probe = false;  // RECOMP_RT_LAMP_PROBE's masks bit 0x20000 (RtManyLights::Encode)
uint32_t g_lamp_cap_probe = 0;      // RECOMP_RT_LAMP_PROBE's 4th number (RtManyLights::Encode)
uint32_t g_lamp_probe_masks = 0;    // RECOMP_RT_LAMP_PROBE's 3rd number

void RtManyLightsBin(const RtFrame& f, uint32_t w, uint32_t h, const float jitter[2], std::span<const RtLight> lights,
                     const double eye[3], float lamp_gap, RtManyBins* out) {
  RtManyBins& b = *out;
  b.tiles[0] = (w + kRtManyTile - 1) / kRtManyTile;
  b.tiles[1] = (h + kRtManyTile - 1) / kRtManyTile;
  const size_t tiles = size_t(b.tiles[0]) * b.tiles[1];
  b.lights.clear();
  b.index.clear();
  b.heads.assign(tiles * 2, 0);
  b.dropped = b.off_screen = 0;
  // The lights by power, strongest first (a full tile drops the weakest).
  std::vector<uint32_t> order(lights.size());
  std::iota(order.begin(), order.end(), 0u);
  auto power = [&](const RtLight& l) {
    return (0.2126 * l.color[0] + 0.7152 * l.color[1] + 0.0722 * l.color[2]) * double(l.radius) * l.radius;
  };
  std::stable_sort(order.begin(), order.end(),
                   [&](uint32_t a, uint32_t c) { return power(lights[a]) > power(lights[c]); });
  std::vector<RtRect> rects;
  rects.reserve(order.size());
  // To measure (RECOMP_RT_LAMP_PROBE=<file>): the file's "reach gap" read
  // every 30 lists, a shadow ray's length from the lit point at most and the
  // gap at the light, for every light (a missing number: as before).
  static const char* probe = std::getenv("RECOMP_RT_LAMP_PROBE");
  static float probe_reach = 1e30f, probe_gap = -1.0f;
  static uint32_t probe_calls = 0, probe_masks = 0;
  static unsigned probe_cap = 0;
  if (probe && *probe && probe_calls++ % 30 == 0) {
    float r = 1e30f, gp = -1.0f;
    unsigned mk = 0, cap = 0;
    if (FILE* fp = std::fopen(probe, "r")) {
      if (std::fscanf(fp, "%f %f %u %u", &r, &gp, &mk, &cap) < 1) r = 1e30f;
      std::fclose(fp);
    }
    const float rr = r > 0 ? r : 1e30f;
    if (rr != probe_reach || gp != probe_gap || mk != probe_masks)
      std::fprintf(stderr, "gfx: lamp probe: reach %g gap %g masks %u\n", rr, gp, mk);
    probe_reach = rr;
    probe_gap = gp;
    probe_masks = mk;
    g_lamp_probe_masks = mk;
    g_lamp_contact_probe = (mk & 0x20000u) != 0;
    if (cap != probe_cap) std::fprintf(stderr, "gfx: lamp probe: lights a group at most %u\n", cap);
    probe_cap = cap;
    g_lamp_cap_probe = cap > 255 ? 255 : cap;
  }
  if (probe_gap >= 0) lamp_gap = probe_gap;
  for (uint32_t k : order) {
    const RtLight& l = lights[k];
    if (b.lights.size() >= kRtLightsMax) break;
    RtLocalLight ll;
    for (int i = 0; i < 3; i++) ll.position[i] = l.position[i];
    ll.inv_sq_radius = l.radius > 0 ? 1.0f / (l.radius * l.radius) : 0.0f;
    ll.cone = l.kind == RtLightKind::Spot;
    if (ll.cone) {
      for (int i = 0; i < 3; i++) ll.direction[i] = l.direction[i];
      ll.cone_offset = l.cone_offset;
      ll.cone_scale = l.cone_scale;
    }
    ll.lamp_gap = lamp_gap;
    if (!std::isfinite(ll.inv_sq_radius) || !(ll.inv_sq_radius > 0)) {
      b.off_screen++;
      continue;
    }
    const RtRect r = RtLocalLightBounds(f, w, h, jitter, ll);
    if (r.Empty()) {
      b.off_screen++;
      continue;
    }
    RtManyLightGpu g{};
    for (int i = 0; i < 3; i++) g.position[i] = float(double(l.position[i]) - eye[i]);
    g.position[3] = ll.inv_sq_radius;
    if (ll.cone) {
      for (int i = 0; i < 3; i++) g.direction[i] = l.direction[i];
      g.direction[3] = l.cone_offset;
      g.colour[3] = l.cone_scale;
    } else {
      g.direction[3] = -2.0f;
      g.colour[3] = 1.0f;
    }
    for (int i = 0; i < 3; i++) g.colour[i] = l.color[i];
    g.flags = RtLightModel(l.flags) | (RtLightClass(l.flags) << 2);
    // A vehicle's lamp the game does not shadow (a train's or a car's
    // cabin lights; headlights carry the game's shadow flag): no ray, lit as
    // the game lights it. A subway train's some fifty such lamps cost a
    // platform's frame a few milliseconds of rays for shadows no one sees
    // inside a lit carriage. RECOMP_RT_VEHICLE_LAMP_RAYS=1: traced as before.
    static const bool vehicle_rays = [] {
      const char* e = std::getenv("RECOMP_RT_VEHICLE_LAMP_RAYS");
      return e && *e == '1';
    }();
    const bool unshadowed_vehicle = !vehicle_rays && (l.flags & kRtLightVehicleLamp) && !(l.flags & kRtLightCastsShadow);
    g.lamp_gap = unshadowed_vehicle ? 1e9f : lamp_gap;
    g.reach = probe_reach;
    std::memcpy(&g.pad, &probe_masks, 4);
    b.lights.push_back(g);
    rects.push_back(r);
  }
  // Counts, then each tile's place, then the lists (in the lights' order).
  auto each_tile = [&](const RtRect& r, auto&& fn) {
    const uint32_t tx0 = r.x0 / kRtManyTile, tx1 = (r.x1 - 1) / kRtManyTile;
    const uint32_t ty0 = r.y0 / kRtManyTile, ty1 = (r.y1 - 1) / kRtManyTile;
    for (uint32_t ty = ty0; ty <= ty1 && ty < b.tiles[1]; ty++)
      for (uint32_t tx = tx0; tx <= tx1 && tx < b.tiles[0]; tx++) fn(size_t(ty) * b.tiles[0] + tx);
  };
  for (const RtRect& r : rects) each_tile(r, [&](size_t t) { b.heads[2 * t + 1]++; });
  uint32_t at = 0;
  for (size_t t = 0; t < tiles; t++) {
    const uint32_t n = b.heads[2 * t + 1];
    b.heads[2 * t] = at;
    b.heads[2 * t + 1] = 0;
    at += std::min(n, kRtManyTileCap);
  }
  b.index.resize(at);
  for (uint32_t i = 0; i < uint32_t(rects.size()); i++)
    each_tile(rects[i], [&](size_t t) {
      uint32_t& n = b.heads[2 * t + 1];
      if (n >= kRtManyTileCap) {
        b.dropped++;
        return;
      }
      b.index[b.heads[2 * t] + n++] = uint16_t(i);
    });
}

std::unique_ptr<RtManyLights> RtManyLights::Make(id<MTLDevice> dev, int empty) {
  if (!RtSupported(dev)) return nullptr;
  auto m = std::make_unique<RtManyLights>();
  m->dev_ = dev;
  // RECOMP_RT_MANY_EMPTY=0: the groups nothing lights shaded, written and
  // added like the others (as before their flags), to compare.
  if (empty < 0) {
    const char* e = std::getenv("RECOMP_RT_MANY_EMPTY");
    empty = e && *e == '0' ? 0 : 1;
    if (!empty) std::fprintf(stderr, "gfx: many lights: every group shaded and added (RECOMP_RT_MANY_EMPTY=0)\n");
  }
  m->empty_ = empty != 0;
  NSError* err = nil;
  m->lib_ = [dev newLibraryWithSource:@((std::string(kRtMsl) + kRtShadowMsl + kRtManyLightsMsl).c_str())
                              options:nil
                                error:&err];
  if (!m->lib_) {
    std::fprintf(stderr, "gfx: many lights: %s\n", err.localizedDescription.UTF8String);
    return nullptr;
  }
  // The trace: the alpha test linked in as an intersection function, the
  // pipeline's table naming it (as RtShadowMasker's).
  id<MTLFunction> alpha = [m->lib_ newFunctionWithName:@"rt_shadow_alpha"];
  auto tracing = [&](NSString* name, id<MTLIntersectionFunctionTable> __strong* table) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [m->lib_ newFunctionWithName:name];
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
  m->trace_depth_pipe_ = tracing(@"rt_many_trace_depth_cs", &m->trace_depth_table_);
  if (m->trace_depth_pipe_) m->trace_color_pipe_ = tracing(@"rt_many_trace_color_cs", &m->trace_color_table_);
  if (m->trace_color_pipe_) m->shade_depth_pipe_ = tracing(@"rt_many_shade_depth_cs", &m->shade_depth_table_);
  if (m->shade_depth_pipe_) m->shade_color_pipe_ = tracing(@"rt_many_shade_color_cs", &m->shade_color_table_);
  if (!m->shade_color_pipe_) {
    std::fprintf(stderr, "gfx: many lights' passes: %s\n", err ? err.localizedDescription.UTF8String : "no function");
    return nullptr;
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  m->white_ = [dev newTextureWithDescriptor:td];
  if (!m->white_) return nullptr;
  const uint32_t white = 0xFFFFFFFFu;
  [m->white_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&white bytesPerRow:4];
  return m;
}

id<MTLRenderPipelineState> RtManyLights::AddPipe(const AddKey& k) const {
  for (const auto& [f, p] : add_pipes_)
    if (f == k) return p;
  const MTLPixelFormat pf = k.colors[0];
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [lib_ newFunctionWithName:@"rt_many_vs"];
  pd.fragmentFunction = [lib_ newFunctionWithName:empty_ ? @"rt_many_add_flag_fs" : @"rt_many_add_fs"];
  pd.rasterSampleCount = std::max<NSUInteger>(k.samples, 1);
  for (int i = 1; i < k.targets && i < 4; i++) {
    pd.colorAttachments[NSUInteger(i)].pixelFormat = k.colors[i];
    pd.colorAttachments[NSUInteger(i)].writeMask = MTLColorWriteMaskNone;
  }
  pd.depthAttachmentPixelFormat = k.depth;
  pd.stencilAttachmentPixelFormat = k.stencil;
  MTLRenderPipelineColorAttachmentDescriptor* a = pd.colorAttachments[0];
  a.pixelFormat = pf;
  a.blendingEnabled = YES;
  a.rgbBlendOperation = a.alphaBlendOperation = MTLBlendOperationAdd;
  a.sourceRGBBlendFactor = a.destinationRGBBlendFactor = MTLBlendFactorOne;
  a.sourceAlphaBlendFactor = MTLBlendFactorZero;
  a.destinationAlphaBlendFactor = MTLBlendFactorOne;
  a.writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue;
  NSError* err = nil;
  id<MTLRenderPipelineState> pipe = [dev_ newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pipe) {
    std::fprintf(stderr, "gfx: many lights' addition into format %lu: %s\n", (unsigned long)pf,
                 err.localizedDescription.UTF8String);
    return nil;
  }
  add_pipes_.emplace_back(k, pipe);
  return pipe;
}

void RtManyLights::EncodeAddPass(id<MTLCommandBuffer> cmd, id<MTLTexture> target, uint32_t level) const {
  // (lit_ and flags_ are one shading's: with empty_ neither is read
  // without the other.)
  id<MTLRenderPipelineState> pipe =
      shaded_ && target && lit_ && (!empty_ || flags_) ? AddPipe(target.pixelFormat) : nil;
  if (!pipe) return;
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = target;
  rp.colorAttachments[0].level = level;
  rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> r = TimedRender(cmd, rp, "rt many lights add");
  [r setRenderPipelineState:pipe];
  [r setFragmentTexture:lit_ atIndex:0];
  if (empty_) [r setFragmentTexture:flags_ atIndex:kRtManyFlagsTexture];
  [r drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [r endEncoding];
}

bool RtManyLights::EncodeAddInto(id<MTLRenderCommandEncoder> e, const MTLPixelFormat colors[4], int targets,
                                 MTLPixelFormat depth, MTLPixelFormat stencil, NSUInteger samples) const {
  if (!shaded_ || !e || !lit_ || (empty_ && !flags_) || targets < 1) return false;
  AddKey k{{colors[0], targets > 1 ? colors[1] : MTLPixelFormatInvalid, targets > 2 ? colors[2] : MTLPixelFormatInvalid,
            targets > 3 ? colors[3] : MTLPixelFormatInvalid},
           std::min(targets, 4),
           depth,
           stencil,
           samples};
  id<MTLRenderPipelineState> pipe = AddPipe(k);
  if (!pipe) return false;
  [e setRenderPipelineState:pipe];
  [e setFragmentTexture:lit_ atIndex:0];
  // (An index no draw of the caller's reads: nothing of its to bind again.)
  if (empty_) [e setFragmentTexture:flags_ atIndex:kRtManyFlagsTexture];
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  return true;
}

void RtManyLights::Encode(id<MTLCommandBuffer> cmd, int slot, const RtFrame* f, const RtManyGBuffer& g,
                          const RtShadowParams& p, const RtManyParams& mp_in, const RtManyBins& bins,
                          id<MTLTexture> target, uint32_t level, id<MTLBuffer> counts, size_t counts_offset,
                          uint32_t parts, id<MTLBuffer> retain_hits) const {
  // (Nothing shaded yet this time: an addition before the shading below
  // would add an earlier frame's light where none is now.)
  shaded_ = false;
  if (!f || !f->tlas || !g.depth || !g.albedo || !g.normal || !target || !counts || slot < 0 || slot >= kSlots) return;
  const uint32_t w = uint32_t(g.depth.width), h = uint32_t(g.depth.height);
  auto same = [&](id<MTLTexture> t) { return !t || (t.width == w && t.height == h); };
  if (!same(g.albedo) || !same(g.normal) || !same(g.surface) || !same(g.cls)) return;
  if ((target.width >> level) != w || (target.height >> level) != h) return;
  if (bins.lights.empty() || bins.heads.size() != size_t(bins.tiles[0]) * bins.tiles[1] * 2) return;
  RtManyParams mp = mp_in;
  mp.size[0] = w, mp.size[1] = h;
  mp.blocks[0] = (w + kRtManyBlock - 1) / kRtManyBlock, mp.blocks[1] = (h + kRtManyBlock - 1) / kRtManyBlock;
  mp.tiles[0] = bins.tiles[0], mp.tiles[1] = bins.tiles[1];
  if (mp.tiles[0] != (w + kRtManyTile - 1) / kRtManyTile || mp.tiles[1] != (h + kRtManyTile - 1) / kRtManyTile) return;
  mp.count = uint32_t(bins.lights.size());
  mp.textures = (g.surface ? 1u : 0u) | (g.cls ? 2u : 0u);
  mp.empty = (empty_ ? 1u : 0u) | (g_lamp_contact_probe ? 2u : 0u) | (retain_hits ? 4u : 0u) | (g_lamp_cap_probe << 8);
  // The slot's buffers (grown as needed; the GPU is done with them).
  Slot& s = slots_[slot];
  auto fill = [&](id<MTLBuffer> __strong& buf, const void* data, size_t bytes) {
    const size_t want = std::max<size_t>(bytes, 16);
    if (!buf || buf.length < want) buf = [dev_ newBufferWithLength:want * 3 / 2 options:MTLResourceStorageModeShared];
    if (buf && bytes) std::memcpy(buf.contents, data, bytes);
    return buf != nil;
  };
  if (!fill(s.lights, bins.lights.data(), bins.lights.size() * sizeof(RtManyLightGpu)) ||
      !fill(s.heads, bins.heads.data(), bins.heads.size() * 4) ||
      !fill(s.index, bins.index.data(), bins.index.size() * 2))
    return;
  if (!samples_ || samples_.width != mp.blocks[0] || samples_.height != mp.blocks[1]) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Uint
                                                                                  width:mp.blocks[0]
                                                                                 height:mp.blocks[1]
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    samples_ = [dev_ newTextureWithDescriptor:td];
    if (!samples_) return;
  }
  const bool depth_tex = p.depth_texture != 0;
  RtShadowParams q = p;
  q.diag = 0;
  q.normals = 0;
  id<MTLTexture> surface = g.surface ? g.surface : white_, cls = g.cls ? g.cls : white_;

  // The trace: a group a tile (16 x 16 blocks of 2 x 2 pixels: 32 x 32).
  if (parts & 1u) {
    id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt many lights trace");
    [e setComputePipelineState:depth_tex ? trace_depth_pipe_ : trace_color_pipe_];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    [e setBytes:&q length:sizeof q atIndex:1];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
    [e setBuffer:counts offset:counts_offset atIndex:3];
    [e setBytes:&mp length:sizeof mp atIndex:4];
    [e setIntersectionFunctionTable:depth_tex ? trace_depth_table_ : trace_color_table_ atBufferIndex:5];
    [e setBuffer:f->info offset:f->info_offset atIndex:6];
    [e setBuffer:s.lights offset:0 atIndex:7];
    [e setBuffer:s.heads offset:0 atIndex:8];
    [e setBuffer:s.index offset:0 atIndex:9];
    // (Unused without retain_hits: the counters stand in.)
    [e setBuffer:retain_hits ? retain_hits : counts offset:retain_hits ? 0 : counts_offset atIndex:10];
    RtUseResources(e, f);
    [e setTexture:g.depth atIndex:0];
    [e setTexture:g.albedo atIndex:1];
    [e setTexture:g.normal atIndex:2];
    [e setTexture:surface atIndex:3];
    [e setTexture:cls atIndex:4];
    [e setTexture:samples_ atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(mp.tiles[0], mp.tiles[1], 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [e endEncoding];
  }
  if (!(parts & 2u)) return;
  // The shading (a group 16 x 16 pixels) into lit_, then added into the
  // target.
  if (!lit_ || lit_.width != w || lit_.height != h) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    lit_ = [dev_ newTextureWithDescriptor:td];
    if (!lit_) return;
  }
  // The groups' flags (a byte a group: 215 x 90 at 3440 x 1440; every one
  // written by each shading with empty_, none read or written without).
  const uint32_t gw = (w + kRtManyGroup - 1) / kRtManyGroup, gh = (h + kRtManyGroup - 1) / kRtManyGroup;
  if (!flags_ || flags_.width != gw || flags_.height != gh) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint
                                                                                  width:gw
                                                                                 height:gh
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    flags_ = [dev_ newTextureWithDescriptor:td];
    if (!flags_) return;
  }
  if (!(parts & kNoAdd) && !AddPipe(target.pixelFormat)) return;
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt many lights shade");
  [e setComputePipelineState:depth_tex ? shade_depth_pipe_ : shade_color_pipe_];
  [e setAccelerationStructure:f->tlas atBufferIndex:0];
  [e setBytes:&q length:sizeof q atIndex:1];
  [e setBuffer:f->materials offset:f->materials_offset atIndex:2];
  [e setBuffer:counts offset:counts_offset atIndex:3];
  [e setBytes:&mp length:sizeof mp atIndex:4];
  [e setIntersectionFunctionTable:depth_tex ? shade_depth_table_ : shade_color_table_ atBufferIndex:5];
  [e setBuffer:f->info offset:f->info_offset atIndex:6];
  RtUseResources(e, f);
  [e setBuffer:s.lights offset:0 atIndex:7];
  [e setBuffer:s.heads offset:0 atIndex:8];
  [e setBuffer:s.index offset:0 atIndex:9];
  [e setTexture:g.depth atIndex:0];
  [e setTexture:g.albedo atIndex:1];
  [e setTexture:g.normal atIndex:2];
  [e setTexture:surface atIndex:3];
  [e setTexture:cls atIndex:4];
  [e setTexture:samples_ atIndex:5];
  [e setTexture:lit_ atIndex:6];
  [e setTexture:flags_ atIndex:7];
  [e dispatchThreadgroups:MTLSizeMake(gw, gh, 1) threadsPerThreadgroup:MTLSizeMake(kRtManyGroup, kRtManyGroup, 1)];
  [e endEncoding];
  shaded_ = true;
  if (!(parts & kNoAdd)) EncodeAddPass(cmd, target, level);
}

}  // namespace recomp::gfx
