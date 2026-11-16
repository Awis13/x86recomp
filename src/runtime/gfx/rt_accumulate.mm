// Native-resolution accumulation of a sparsely traced term (rt_accumulate.h).

#import "runtime/gfx/rt_accumulate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"

namespace recomp::gfx {

const char* const kRtAccumulateMsl = R"(

#include <metal_stdlib>
using namespace metal;

struct RtAccParams {
  float4 rx, ry, rv, rc;  // a pixel's point relative to the camera: h = x rx + y ry + v rv + rc, h.xyz / h.w
  float4 prev[4];         // that point to the history's clip space
  float4 eye_delta;       // xyz: this camera's position less the history's
  float4 sky_value;
  uint2 size;
  float2 jitter;
  uint2 low;
  uint step, stride, frame;
  uint depth_texture;
  float sky, pixel;
  uint history, normals;
  float max_count, scale;
  float4 hdir[3];         // a history pixel's ray: (x hdir[0] + y hdir[1] + hdir[2]).xyz / .w (its clip x, y)
};
constant float RT_ACC_PLANE = 0.01;   // kRtAccumulatePlane
constant float RT_ACC_DEPTH = 0.01;   // kRtAccumulateDepth
constant float RT_ACC_SIGMA = 1.25;   // kRtAccumulateSigma
constant float RT_ACC_MOST = 4.0;     // kRtAccumulateMaxCount
// A group a tile of 8 x 8 blocks of step x step pixels, 8 x 8 threads
// going over its pixels 8 x 8 at a time (neighbouring threads, neighbouring
// pixels: their reads and writes together, every thread busy at any step);
// the tile's blocks' samples and a ring of blocks around them (10 x 10)
// shared, and each block's statistics of the 3 x 3 around it.
constant uint RT_ACC_GROUP = 8u, RT_ACC_RING = 10u;

// The term's type: float (one channel) or float3.
float rt_acc_get(half4 s, float) { return float(s.x); }
float3 rt_acc_get(half4 s, float3) { return float3(s.xyz); }
float rt_acc_get(float4 s, float) { return s.x; }
float3 rt_acc_get(float4 s, float3) { return s.xyz; }
half4 rt_acc_out(float v) { return half4(half(v)); }
half4 rt_acc_out(float3 v) { return half4(half3(v), 1.0h); }
float4 rt_acc_put(float v, float w) { return float4(v, 0.0, 0.0, w); }
float4 rt_acc_put(float3 v, float w) { return float4(v, w); }
float rt_acc_most(float v) { return v; }
float rt_acc_most(float3 v) { return max(max(v.x, v.y), v.z); }
bool rt_acc_same(float a, float b) { return a == b; }
bool rt_acc_same(float3 a, float3 b) { return all(a == b); }
bool rt_acc_near(float a, float b) { return abs(a - b) < 1.0 / 512.0; }
bool rt_acc_near(float3 a, float3 b) { return all(abs(a - b) < 1.0 / 512.0); }
// The history, an R32Uint a pixel: the distance's log2 in its low 16 bits
// (0: none), the sample count in the next 6 (0 to RT_ACC_MOST), a one
// channel term's value in the top 10 (unorm); three channels' value in an
// RGBA16Float beside it.
float rt_acc_log(float d) { return saturate((log2(max(d, 1e-30)) + 32.0) / 64.0) * 65535.0; }
uint rt_acc_hist(float log_d, float v, float n) {
  return max(uint(round(log_d)), 1u) | (uint(round(saturate(n / RT_ACC_MOST) * 63.0)) << 16) |
         (uint(round(saturate(v) * 1023.0)) << 22);
}
uint rt_acc_hist(float log_d, float3 v, float n) { return rt_acc_hist(log_d, 0.0, n); }
#define RT_ACC_STATS(T) (sizeof(T) == sizeof(float) ? 1u : 4u)
float rt_acc_count(uint g) { return float((g >> 16) & 63u) * (RT_ACC_MOST / 63.0); }
float rt_acc_value(uint g) { return float(g >> 22) * (1.0 / 1023.0); }
// The history's value at four texels (a gather, in its order: (x, y + 1),
// (x + 1, y + 1), (x + 1, y), (x, y) of i0), weighted.
float rt_acc_hist4(uint4 g, float4 w, texture2d<half, access::read> color, int2 i0, float) {
  return dot(w, float4(g >> 22) * (1.0 / 1023.0));
}
float3 rt_acc_hist4(uint4 g, float4 w, texture2d<half, access::read> color, int2 i0, float3) {
  const int2 m = int2(color.get_width(), color.get_height()) - 1;
  return w.x * float3(color.read(uint2(clamp(i0 + int2(0, 1), int2(0), m))).xyz) +
         w.y * float3(color.read(uint2(clamp(i0 + int2(1, 1), int2(0), m))).xyz) +
         w.z * float3(color.read(uint2(clamp(i0 + int2(1, 0), int2(0), m))).xyz) +
         w.w * float3(color.read(uint2(clamp(i0, int2(0), m))).xyz);
}
float rt_acc_hist1(uint g, texture2d<half, access::read> color, uint2 at, float) { return rt_acc_value(g); }
float3 rt_acc_hist1(uint g, texture2d<half, access::read> color, uint2 at, float3) { return float3(color.read(at).xyz); }
void rt_acc_store(texture2d<half, access::write> color, uint2 at, float v) {}
void rt_acc_store(texture2d<half, access::write> color, uint2 at, float3 v) { color.write(half4(half3(v), 1.0h), at); }
// A block's statistics in threadgroup memory: one half4 for one channel
// (lo, hi, mean, spread), four for three (RT_ACC_STATS).
void rt_acc_stats_put(threadgroup half4* s, float lo, float hi, float mu, float sd) { s[0] = half4(lo, hi, mu, sd); }
void rt_acc_stats_put(threadgroup half4* s, float3 lo, float3 hi, float3 mu, float3 sd) {
  s[0] = half4(half3(lo), 0.0h);
  s[1] = half4(half3(hi), 0.0h);
  s[2] = half4(half3(mu), 0.0h);
  s[3] = half4(half3(sd), 0.0h);
}
void rt_acc_stats_get(const threadgroup half4* s, thread float& lo, thread float& hi) {
  const half4 v = s[0];
  lo = v.x, hi = v.y;
}
void rt_acc_stats_get(const threadgroup half4* s, thread float3& lo, thread float3& hi) {
  lo = float3(s[0].xyz), hi = float3(s[1].xyz);
}
void rt_acc_stats_mu(const threadgroup half4* s, thread float& mu, thread float& sd) { mu = s[0].z, sd = s[0].w; }
void rt_acc_stats_mu(const threadgroup half4* s, thread float3& mu, thread float3& sd) {
  mu = float3(s[2].xyz), sd = float3(s[3].xyz);
}

float rt_acc_depth(depth2d<float, access::read> t, uint2 c) { return t.read(c); }
float rt_acc_depth(texture2d<float, access::read> t, uint2 c) { return t.read(c).r; }
bool rt_acc_sky(constant RtAccParams& p, float v) { return isnan(v) || (p.sky < 0.5 ? v <= p.sky : v >= p.sky); }
// Pixel c's homogeneous point relative to the camera (rt_shadow.h's
// rt_point before its divide: D3D9's centre less the jitter).
float4 rt_acc_h(constant RtAccParams& p, int2 c, float v) {
  const float x = 2.0 * (float(c.x) - p.jitter.x) / float(p.size.x) - 1.0;
  const float y = 1.0 - 2.0 * (float(c.y) - p.jitter.y) / float(p.size.y);
  return x * p.rx + y * p.ry + v * p.rv + p.rc;
}
float3 rt_acc_point(constant RtAccParams& p, int2 c, float v) {
  const float4 h = rt_acc_h(p, c, v);
  return h.xyz / h.w;
}
// The step from pixel c (homogeneous point h, stored v) to its neighbour
// c + o or from c - o, whichever's stored depth is nearer v (on the same
// surface); false where neither is in the buffer and not the sky.
template <typename D>
bool rt_acc_side(D t, constant RtAccParams& p, int2 c, int2 o, float v, float4 h, float3 q, thread float3& d) {
  float best = INFINITY;
  for (int s = -1; s <= 1; s += 2) {
    const int2 nb = c + s * o;
    if (nb.x < 0 || nb.y < 0 || nb.x >= int(p.size.x) || nb.y >= int(p.size.y)) continue;
    const float vn = rt_acc_depth(t, uint2(nb));
    if (rt_acc_sky(p, vn) || !(abs(vn - v) < best)) continue;
    best = abs(vn - v);
    // (The neighbour's homogeneous point: h moved by a pixel and the depth.)
    const float4 hn = h + float(s) * (o.x != 0 ? (2.0 / float(p.size.x)) * p.rx : (-2.0 / float(p.size.y)) * p.ry) +
                      (vn - v) * p.rv;
    d = (hn.xyz / hn.w - q) * float(s);
  }
  return best < INFINITY;
}
// Pixel c's surface normal (point q, homogeneous h, stored v): the normal
// texture's where given, else the geometric one from the depth (on each
// axis the neighbour on the same surface), else toward the camera; unit,
// turned toward the camera.
template <typename D>
float3 rt_acc_normal(D t, texture2d<float, access::read> normals, constant RtAccParams& p, int2 c, float v, float4 h,
                     float3 q) {
  float3 n = float3(0.0);
  if (p.normals != 0u) n = normals.read(uint2(c)).xyz * 2.0 - 1.0;
  if (!(length_squared(n) > 0.01)) {
    float3 dx, dy;
    n = rt_acc_side(t, p, c, int2(1, 0), v, h, q, dx) && rt_acc_side(t, p, c, int2(0, 1), v, h, q, dy) ? cross(dx, dy)
                                                                                                       : -q;
    if (!(length_squared(n) > 0.0)) n = -q;
  }
  n = normalize(n);
  return dot(n, q) > 0.0 ? -n : n;
}
// The history at point q (relative to the camera): reprojected by the
// cameras, read bilinearly at the history's pixel centres (one gather),
// each tap kept only where its distance from the history's camera is within
// `tol` of the point's (compared as logarithms); with `plane`, also only
// where its point (along its pixel's ray) lies within `plane` of the plane
// through q with unit normal n (another surface met at the same distance,
// a crease: none). w: the kept taps' weight (0: none).
template <typename T>
struct RtAccHist {
  T v;
  float n, w;
};
template <typename T>
RtAccHist<T> rt_acc_history(constant RtAccParams& p, texture2d<uint, access::sample> hist,
                            texture2d<half, access::read> color, float3 q, float tol, float3 n, float plane) {
  RtAccHist<T> h;
  h.v = T(0.0);
  h.n = h.w = 0.0;
  const float4 c = q.x * p.prev[0] + q.y * p.prev[1] + q.z * p.prev[2] + p.prev[3];
  if (!(c.w > 0.0)) return h;
  // (The point is where the jittered depth was drawn: the pixel's centre is
  // the jitter away from it, there as here.)
  const float2 size = float2(p.size);
  const float2 at = float2((c.x / c.w + 1.0) * 0.5, (1.0 - c.y / c.w) * 0.5) * size + p.jitter;
  if (!(at.x > -1.0 && at.y > -1.0 && at.x < size.x && at.y < size.y)) return h;
  const float2 f = floor(at), fr = at - f;
  constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
  const uint4 g = hist.gather(s, (f + 1.0) / size);  // (the corner the four texels share)
  float4 w = float4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
  // (Taps past the edges: none.)
  if (f.x < 0.0) w.xw = 0.0;
  if (f.x + 1.0 >= size.x) w.yz = 0.0;
  if (f.y < 0.0) w.zw = 0.0;
  if (f.y + 1.0 >= size.y) w.xy = 0.0;
  const float3 qh = q + p.eye_delta.xyz;  // (relative to the history's camera)
  const float lp = rt_acc_log(length(qh)), lt = log2(1.0 + tol) * (65535.0 / 64.0);
  const float4 lg = float4(g & 0xFFFFu);
  w *= select(float4(0.0), float4(1.0), lg > 0.0 && abs(lg - lp) <= lt);
  if (plane > 0.0) {
    // (Gather's order: (x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y).)
    const float2 o[4] = {float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0)};
    for (int k = 0; k < 4; k++) {
      if (!(w[k] > 0.0)) continue;
      const float2 xy = float2(2.0 * (f.x + o[k].x) / size.x - 1.0, 1.0 - 2.0 * (f.y + o[k].y) / size.y);
      const float4 r = xy.x * p.hdir[0] + xy.y * p.hdir[1] + p.hdir[2];
      const float d = exp2(lg[k] * (64.0 / 65535.0) - 32.0);
      if (!(abs(dot(normalize(r.xyz / r.w) * d - qh, n)) <= plane)) w[k] = 0.0;
    }
  }
  const float wsum = w.x + w.y + w.z + w.w;
  if (wsum > 1e-3) {
    w /= wsum;
    h.v = rt_acc_hist4(g, w, color, int2(f), T(0.0));
    // (Taps of under half the weight: fewer samples' worth.)
    h.n = dot(w, float4((g >> 16) & 63u) * (RT_ACC_MOST / 63.0)) * saturate(2.0 * wsum);
    h.w = wsum;
  }
  return h;
}
// The history around a sample (its point q): the 3 x 3 pixels around where
// it lands, those at its distance (within 3%: room for a slant) holding half
// a sample or more: their range, and the value and samples of the middle
// one (count 0: not among them). False: fewer than 5 of them (a history too
// thin yet to say what lies outside it).
template <typename T>
bool rt_acc_around(constant RtAccParams& p, texture2d<uint, access::sample> hist, texture2d<half, access::read> color,
                   float3 q, thread T& lo, thread T& hi, thread T& mid, thread float& mid_n) {
  const float4 c = q.x * p.prev[0] + q.y * p.prev[1] + q.z * p.prev[2] + p.prev[3];
  if (!(c.w > 0.0)) return false;
  const float2 at =
      float2((c.x / c.w + 1.0) * 0.5, (1.0 - c.y / c.w) * 0.5) * float2(p.size) + p.jitter;
  const int2 p0 = int2(floor(at + 0.5));
  const float lp = rt_acc_log(length(q + p.eye_delta.xyz)), lt = log2(1.03) * (65535.0 / 64.0);
  int taps = 0;
  lo = T(INFINITY), hi = T(-INFINITY), mid = T(0.0), mid_n = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      const int2 x = p0 + int2(dx, dy);
      if (x.x < 0 || x.y < 0 || x.x >= int(p.size.x) || x.y >= int(p.size.y)) continue;
      const uint g = hist.read(uint2(x)).x;
      const float lg = float(g & 0xFFFFu), n = rt_acc_count(g);
      if (lg == 0.0 || abs(lg - lp) > lt || n < 0.5) continue;
      const T v = rt_acc_hist1(g, color, uint2(x), T(0.0));
      lo = min(lo, v);
      hi = max(hi, v);
      taps++;
      if (dx == 0 && dy == 0) mid = v, mid_n = n;
    }
  return taps >= 5;
}

// The temporal pass: a group a tile.
//  1. The samples of the tile's blocks and a ring around (10 x 10 blocks)
//     into threadgroup memory: each one's point, value and place in its
//     block (`points`, the trace's), and two changes against the history
//     (A-SVGF's temporal gradient, robust to the jitter): how far it lies
//     outside the range of the history's 3 x 3 pixels around it (a value
//     none of them held: a moved shadow), and how far from the history of
//     its own pixel (only counted where every sample around agrees: what a
//     thin or partly covered caster that left behind). Then each block's
//     statistics of the samples of the 3 x 3 blocks around it (range, mean
//     and spread, every surface's) and its change (the largest of the first
//     kind; of the second too where they all agree).
//  2. Each pixel of the tile: its point; where the samples around all agree
//     (most of the screen) that value, else its normal and the samples of
//     the blocks on its side on its surface (the fill, its own sample); the
//     history (off the uniform path also by its plane), the clamp where a
//     change is seen, the running mean, the output (what the history keeps
//     too while it holds under a sample); the tile listed for the filter
//     where one of its pixels has less than one sample of its own (but
//     where all agree).
template <typename T, typename D>
void rt_acc_temporal(D t, texture2d<float, access::read> normals, constant RtAccParams& p,
                     texture2d<half, access::read> sparse, texture2d<half, access::read> points,
                     texture2d<uint, access::sample> hist_in, texture2d<half, access::read> color_in,
                     texture2d<uint, access::write> hist_out, texture2d<half, access::write> color_out,
                     texture2d<half, access::write> out, device atomic_uint* counts, device atomic_uint* tiles,
                     device uint* next_tiles, device uint* list, threadgroup half4* tq, threadgroup half4* tv,
                     threadgroup half4* ts, threadgroup uchar* tn, threadgroup half* tx, threadgroup half* tc,
                     threadgroup atomic_uint& rough_group, uint2 tg, uint2 lt, uint li) {
  const T zero = T(0.0);
  if (li == 0u) {
    atomic_store_explicit(&rough_group, 0u, memory_order_relaxed);
    if (tg.x == 0u && tg.y == 0u) next_tiles[0] = 0u;  // (the next frame's list starts empty)
  }
  const int2 origin = int2(tg * RT_ACC_GROUP) - 1;
  for (uint e = li; e < RT_ACC_RING * RT_ACC_RING; e += RT_ACC_GROUP * RT_ACC_GROUP) {
    const int2 bk = origin + int2(e % RT_ACC_RING, e / RT_ACC_RING);
    half4 a = half4(0.0h, 0.0h, 0.0h, -1.0h), b = half4(0.0h);
    float ex = 0.0;
    uchar cell = 0;
    if (bk.x >= 0 && bk.y >= 0 && bk.x < int(p.low.x) && bk.y < int(p.low.y)) {
      const half4 pt = points.read(uint2(bk));
      if (pt.w >= 0.0h) {
        b = sparse.read(uint2(bk));
        cell = uchar(pt.w);
        float fp = 0.0;
        T lo, hi, mid;
        float mid_n;
        if (p.history != 0u && rt_acc_around<T>(p, hist_in, color_in, float3(pt.xyz), lo, hi, mid, mid_n)) {
          const T v = rt_acc_get(b, zero);
          const float scale = p.scale > 0.0 ? p.scale : max(max(rt_acc_most(v), rt_acc_most(hi)), 1e-4);
          const float outside = max(rt_acc_most(lo - v), rt_acc_most(v - hi)) / scale;
          fp = saturate((outside - 0.2) * (1.0 / 0.3));
          if (mid_n >= 1.0) ex = saturate((rt_acc_most(abs(v - mid)) / scale - 0.3) * 2.5);
        }
        a = half4(pt.xyz, half(fp));
      }
    }
    tq[e] = a;
    tv[e] = b;
    tn[e] = cell;
    tx[e] = half(ex);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    T lo = T(INFINITY), hi = T(-INFINITY), m1 = zero, m2 = zero;
    float cnt = 0.0, most_change = 0.0, most_ex = 0.0;
    for (uint sy = 0u; sy < 3u; sy++)
      for (uint sx = 0u; sx < 3u; sx++) {
        const uint e = (lt.y + sy) * RT_ACC_RING + lt.x + sx;
        if (tq[e].w < 0.0h) continue;
        most_change = max(most_change, float(tq[e].w));
        most_ex = max(most_ex, float(tx[e]));
        const T val = rt_acc_get(tv[e], zero);
        lo = min(lo, val);
        hi = max(hi, val);
        m1 += val;
        m2 += val * val;
        cnt += 1.0;
      }
    const T mu = cnt > 0.0 ? m1 / cnt : zero;
    if (cnt > 0.0 && rt_acc_same(lo, hi)) most_change = max(most_change, most_ex);
    // (No sample: lo above hi, never uniform.)
    rt_acc_stats_put(ts + (lt.y * RT_ACC_GROUP + lt.x) * RT_ACC_STATS(T), cnt > 0.0 ? lo : T(1.0),
                     cnt > 0.0 ? hi : zero, mu, sqrt(max(cnt > 0.0 ? m2 / cnt - mu * mu : zero, zero)));
    tc[lt.y * RT_ACC_GROUP + lt.x] = half(most_change);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint n_uniform = 0u, n_fresh = 0u, n_hole = 0u, n_changed = 0u;
  bool rough = false;
  // (Integer division is slow on the GPU: the block by float, exact for
  // these sizes.)
  const float stepf = float(p.step), inv_step = 1.0 / stepf;
  const uint tw = RT_ACC_GROUP * p.step;
  for (uint y = lt.y; y < tw; y += RT_ACC_GROUP)
    for (uint x = lt.x; x < tw; x += RT_ACC_GROUP) {
      const uint2 gid = tg * tw + uint2(x, y);
      if (gid.x >= p.size.x || gid.y >= p.size.y) continue;
      const uint2 bl = uint2((float2(x, y) + 0.5) * inv_step);
      const float2 ij = float2(uint2(x, y) - bl * p.step);
      const float v = rt_acc_depth(t, gid);
      if (rt_acc_sky(p, v)) {
        const T sv = rt_acc_get(p.sky_value, zero);
        out.write(rt_acc_out(sv), gid);
        hist_out.write(uint4(rt_acc_hist(0.0, sv, 0.0) & ~0xFFFFu), gid);
        rt_acc_store(color_out, gid, sv);
        continue;
      }
      const int2 c = int2(gid);
      const float4 h4 = rt_acc_h(p, c, v);
      const float3 q = h4.xyz / h4.w;
      const float d = length(q);
      const threadgroup half4* sb = ts + (bl.y * RT_ACC_GROUP + bl.x) * RT_ACC_STATS(T);
      T lo, hi;
      rt_acc_stats_get(sb, lo, hi);
      const bool stats = rt_acc_most(lo - hi) <= 0.0;  // (some sample)
      const bool agree = stats && rt_acc_same(lo, hi);
      // The change seen among the samples of the 3 x 3 blocks around (the
      // block's).
      const float change = float(tc[bl.y * RT_ACC_GROUP + bl.x]);
      T fill = lo, esum = zero;
      float ew = 0.0;
      bool hole = false;
      // The history's distance tolerance: kRtAccumulateDepth, plus 1.5
      // pixels' change of the distance across a surface seen at a slant
      // (where every sample agrees, no normal is made: a slant of 4).
      float tol = RT_ACC_DEPTH + 6.0 * p.pixel, plane = 0.0;
      float3 n = float3(0.0);
      if (agree) {
        // Every sample around the same (most of the screen): that value,
        // the pixel's own sample that value.
        const uint e = (bl.y + 1u) * RT_ACC_RING + bl.x + 1u, cell = tn[e];
        if (tq[e].w >= 0.0h && float(cell & 7u) == ij.x && float(cell >> 3) == ij.y) esum = lo, ew = 1.0;
      } else {
        // Else the samples on the pixel's surface (within kRtAccumulatePlane
        // of its distance of its plane) of the blocks on its side (both
        // sides for the middle row or column of an odd step): the fill (a
        // tent over the samples' grid), the pixel's own sample.
        n = rt_acc_normal(t, normals, p, c, v, h4, q);
        const float cs = max(abs(dot(n, q)) / d, 0.05), slant = 1.5 * p.pixel * sqrt(max(1.0 - cs * cs, 0.0)) / cs;
        tol = RT_ACC_DEPTH + slant;
        // (A tap off the plane by more than its rounding and the jitter's
        // half pixel across the slant: another surface.)
        plane = d * (0.003 + slant);
        T fsum = zero, any = zero;
        float tsum = 0.0, on = 0.0;
        const float2 mid2 = 2.0 * ij + 1.0;
        const int2 from = select(int2(0), int2(-1), mid2 <= stepf), to = select(int2(0), int2(1), mid2 >= stepf);
        for (int oy = from.y; oy <= to.y; oy++)
          for (int ox = from.x; ox <= to.x; ox++) {
            const uint e = uint(int(bl.y) + 1 + oy) * RT_ACC_RING + uint(int(bl.x) + 1 + ox);
            const float4 a = float4(tq[e]);
            if (a.w < 0.0 || abs(dot(a.xyz - q, n)) > RT_ACC_PLANE * d) continue;
            const uint cell = tn[e];
            const float2 dp = float2(ox, oy) * stepf + float2(cell & 7u, cell >> 3) - ij;
            const float2 tw2 = saturate(1.0 - abs(dp) * inv_step);
            const T val = rt_acc_get(tv[e], zero);
            any += val;
            on += 1.0;
            if (tw2.x * tw2.y > 0.0) {
              fsum += tw2.x * tw2.y * val;
              tsum += tw2.x * tw2.y;
            }
            if (dp.x == 0.0 && dp.y == 0.0) {
              esum = val;
              ew = 1.0;
            }
          }
        hole = !(tsum > 0.0);
        fill = hole ? (on > 0.0 ? any / on : zero) : fsum / tsum;
        if (hole && on == 0.0) {
          // (No sample on its surface: the block's own, whatever surface.)
          const uint e = (bl.y + 1u) * RT_ACC_RING + bl.x + 1u;
          fill = tq[e].w < 0.0h ? rt_acc_get(p.sky_value, zero) : rt_acc_get(tv[e], zero);
        }
      }
      T hv = zero;
      float hn = 0.0;
      bool have = false;
      if (p.history != 0u) {
        const RtAccHist<T> h = rt_acc_history<T>(p, hist_in, color_in, q, tol, n, plane);
        if (h.w > 0.0) {
          hv = h.v;
          hn = h.n;
          have = true;
        }
      }
      // A change seen around: the history clamped to the samples' mean +-
      // k sigma within their range, and as much of it dropped.
      if (have && change > 0.0 && stats) {
        T mu, sd;
        rt_acc_stats_mu(sb, mu, sd);
        hv = mix(hv, clamp(hv, max(lo, mu - RT_ACC_SIGMA * sd), min(hi, mu + RT_ACC_SIGMA * sd)), change);
        hn *= 1.0 - change;
      }
      // The running mean; the output the fill in proportion to what the
      // pixel lacks of a sample of its own, and while it lacks one, what
      // the history keeps (the filter reads it).
      const float wsum = hn + ew;
      const T val = wsum > 1e-4 ? (hv * hn + esum) / wsum : fill;
      T o = mix(fill, val, saturate(wsum));
      const T keep = val;
      const bool uniform = agree && (!have || rt_acc_near(hv, lo));
      if (uniform) o = lo;
      rough = rough || (!uniform && wsum < 1.0);
      n_uniform += uniform ? 1u : 0u;
      n_fresh += have ? 0u : 1u;
      n_hole += hole ? 1u : 0u;
      n_changed += change >= 0.5 ? 1u : 0u;
      hist_out.write(uint4(rt_acc_hist(rt_acc_log(d), keep, min(wsum, p.max_count))), gid);
      rt_acc_store(color_out, gid, keep);
      out.write(rt_acc_out(o), gid);
    }
  if (rough) atomic_store_explicit(&rough_group, 1u, memory_order_relaxed);
  const uint nu = simd_sum(n_uniform), nf = simd_sum(n_fresh), nh = simd_sum(n_hole), nc = simd_sum(n_changed);
  if (simd_is_first()) {
    if (nu) atomic_fetch_add_explicit(&counts[0], nu, memory_order_relaxed);
    if (nf) atomic_fetch_add_explicit(&counts[1], nf, memory_order_relaxed);
    if (nh) atomic_fetch_add_explicit(&counts[2], nh, memory_order_relaxed);
    if (nc) atomic_fetch_add_explicit(&counts[4], nc, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (li == 0u && atomic_load_explicit(&rough_group, memory_order_relaxed) != 0u) {
    const uint k = atomic_fetch_add_explicit(&tiles[0], 1u, memory_order_relaxed);
    const uint most = ((p.low.x + RT_ACC_GROUP - 1u) / RT_ACC_GROUP) * ((p.low.y + RT_ACC_GROUP - 1u) / RT_ACC_GROUP);
    if (k < most) list[k] = tg.x | (tg.y << 16);
  }
}

// The filter: a group a listed tile (its threadgroups dispatched by the
// list's count), its threads over its pixels as the temporal pass's; an
// edge-aware a-trous step over the pixels with less than one sample of
// their own (what the temporal pass showed: the fill in proportion): 3 x 3
// taps (1 2 1) one pixel apart, 5 x 5 (1 4 6 4 1) under half a sample (the
// fill's step-sized structure hidden), each by its plane distance and its
// normal (where the normal texture gives them) and its own samples (a
// settled neighbour counts more; the pixel itself the less the fewer it
// holds).
template <typename T, typename D>
void rt_acc_filter(D t, texture2d<float, access::read> normals, constant RtAccParams& p,
                   texture2d<uint, access::read> hist, texture2d<half, access::read> color,
                   texture2d<half, access::write> out, device atomic_uint* counts, const device uint* list, uint g,
                   uint2 lt, uint li) {
  const uint most = ((p.low.x + RT_ACC_GROUP - 1u) / RT_ACC_GROUP) * ((p.low.y + RT_ACC_GROUP - 1u) / RT_ACC_GROUP);
  if (g >= most) return;
  if (li == 0u) atomic_fetch_add_explicit(&counts[3], 1u, memory_order_relaxed);
  const uint packed = list[g];
  const uint2 tg = uint2(packed & 0xFFFFu, packed >> 16);
  const T zero = T(0.0);
  const uint tw = RT_ACC_GROUP * p.step;
  for (uint y = lt.y; y < tw; y += RT_ACC_GROUP)
    for (uint x = lt.x; x < tw; x += RT_ACC_GROUP) {
      const uint2 gid = tg * tw + uint2(x, y);
      if (gid.x >= p.size.x || gid.y >= p.size.y) continue;
      const uint hc = hist.read(gid).x;
      const float cn = rt_acc_count(hc);
      if (cn >= 1.0) continue;
      const float v = rt_acc_depth(t, gid);
      if (rt_acc_sky(p, v)) continue;
      const float4 h4 = rt_acc_h(p, int2(gid), v);
      const float3 q = h4.xyz / h4.w;
      const float d = length(q);
      const float3 n = rt_acc_normal(t, normals, p, int2(gid), v, h4, q);
      const int r = cn < 0.5 ? 2 : 1;
      T sum = zero;
      float wsum = 0.0;
      for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++) {
          const int2 at = int2(gid) + int2(dx, dy);
          if (at.x < 0 || at.y < 0 || at.x >= int(p.size.x) || at.y >= int(p.size.y)) continue;
          const uint hn = hist.read(uint2(at)).x;
          // (Binomial weights: 1 2 1, or 1 4 6 4 1.)
          const float bx = r == 1 ? (dx == 0 ? 2.0 : 1.0) : (dx == 0 ? 6.0 : abs(dx) == 1 ? 4.0 : 1.0);
          const float by = r == 1 ? (dy == 0 ? 2.0 : 1.0) : (dy == 0 ? 6.0 : abs(dy) == 1 ? 4.0 : 1.0);
          float w = bx * by;
          if (dx != 0 || dy != 0) {
            const float vn = rt_acc_depth(t, uint2(at));
            if (rt_acc_sky(p, vn)) continue;
            const float pl = abs(dot(rt_acc_point(p, at, vn) - q, n)) / (RT_ACC_PLANE * d);
            w *= exp2(-pl * pl * 1.442695);
            if (p.normals != 0u) {
              const float3 m = normals.read(uint2(at)).xyz * 2.0 - 1.0;
              const float nn = saturate(abs(dot(n, m)) * rsqrt(max(dot(m, m), 1e-8)));
              const float n2 = nn * nn, n4 = n2 * n2;
              w *= n4 * n4;
            }
          }
          w *= 0.25 + saturate(rt_acc_count(hn));
          sum += w * rt_acc_hist1(hn, color, uint2(at), zero);
          wsum += w;
        }
      out.write(rt_acc_out(sum / wsum), gid);
    }
}

#define RT_ACC_KERNELS(KIND, DEPTH, CH, T, STATS)                                                                  \
  kernel void rt_acc_temporal_##KIND##CH##_cs(                                                                     \
      constant RtAccParams& p [[buffer(0)]], device atomic_uint* counts [[buffer(1)]],                             \
      device atomic_uint* tiles [[buffer(2)]], device uint* next_tiles [[buffer(3)]],                              \
      device uint* list [[buffer(4)]], DEPTH depth [[texture(0)]],                                                 \
      texture2d<float, access::read> normals [[texture(1)]], texture2d<half, access::read> sparse [[texture(2)]],  \
      texture2d<uint, access::sample> hist_in [[texture(3)]], texture2d<half, access::read> color_in [[texture(4)]], \
      texture2d<uint, access::write> hist_out [[texture(5)]],                                                      \
      texture2d<half, access::write> color_out [[texture(6)]], texture2d<half, access::write> out [[texture(7)]],  \
      texture2d<half, access::read> points [[texture(8)]], uint2 tg [[threadgroup_position_in_grid]],              \
      uint2 lt [[thread_position_in_threadgroup]], uint li [[thread_index_in_threadgroup]]) {                      \
    threadgroup half4 tq[RT_ACC_RING * RT_ACC_RING], tv[RT_ACC_RING * RT_ACC_RING];                                \
    threadgroup half4 ts[RT_ACC_GROUP * RT_ACC_GROUP * STATS];                                                     \
    threadgroup uchar tn[RT_ACC_RING * RT_ACC_RING];                                                               \
    threadgroup half tx[RT_ACC_RING * RT_ACC_RING], tc[RT_ACC_GROUP * RT_ACC_GROUP];                               \
    threadgroup atomic_uint rough_group;                                                                           \
    rt_acc_temporal<T>(depth, normals, p, sparse, points, hist_in, color_in, hist_out, color_out, out, counts,     \
                       tiles, next_tiles, list, tq, tv, ts, tn, tx, tc, rough_group, tg, lt, li);                  \
  }                                                                                                                \
  kernel void rt_acc_filter_##KIND##CH##_cs(                                                                       \
      constant RtAccParams& p [[buffer(0)]], device atomic_uint* counts [[buffer(1)]],                             \
      const device uint* list [[buffer(4)]], DEPTH depth [[texture(0)]],                                           \
      texture2d<float, access::read> normals [[texture(1)]], texture2d<uint, access::read> hist [[texture(5)]],    \
      texture2d<half, access::read> color [[texture(6)]], texture2d<half, access::write> out [[texture(7)]],        \
      uint2 tg [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]],                      \
      uint li [[thread_index_in_threadgroup]]) {                                                                   \
    rt_acc_filter<T>(depth, normals, p, hist, color, out, counts, list, tg.x, lt, li);                             \
  }
typedef depth2d<float, access::read> RtAccDepthRead;
typedef texture2d<float, access::read> RtAccColorRead;
RT_ACC_KERNELS(depth, RtAccDepthRead, 1, float, 1)
RT_ACC_KERNELS(color, RtAccColorRead, 1, float, 1)
RT_ACC_KERNELS(depth, RtAccDepthRead, 3, float3, 4)
RT_ACC_KERNELS(color, RtAccColorRead, 3, float3, 4)
)";

namespace {

void Put(float* out, const double* v) {
  for (int k = 0; k < 4; k++) out[k] = float(v[k]);
}

// (x, y, z, 1) of clip space through the rows `r` (clip to world): the
// homogeneous world point.
void ClipTo(const double* r, double x, double y, double z, double h[4]) {
  for (int k = 0; k < 4; k++) h[k] = x * r[k] + y * r[4 + k] + z * r[8 + k] + r[12 + k];
}

}  // namespace

bool RtAccumulateCameraOf(const RtFrame& f, RtAccumulateCamera* out) {
  *out = RtAccumulateCamera{};
  bool exact = false;
  for (double v : f.inv_view_proj_exact) exact = exact || v != 0.0;
  for (int k = 0; k < 16; k++) out->inv[k] = exact ? f.inv_view_proj_exact[k] : double(f.inv_view_proj[k]);
  double vp[16];
  if (!Invert4(out->inv, vp) || !RtLightsEye(f, out->eye)) return false;
  out->valid = true;
  return true;
}

bool RtAccumulateReproject(const RtAccumulateCamera& prev, const RtAccumulateCamera& cur, float rows[4][4],
                           float eye_delta[3]) {
  if (!prev.valid || !cur.valid) return false;
  double vp[16];
  if (!Invert4(prev.inv, vp)) return false;
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 3; i++) rows[i][k] = float(vp[i * 4 + k]);
    rows[3][k] = float(cur.eye[0] * vp[k] + cur.eye[1] * vp[4 + k] + cur.eye[2] * vp[8 + k] + vp[12 + k]);
  }
  for (int k = 0; k < 3; k++) eye_delta[k] = float(cur.eye[k] - prev.eye[k]);
  return true;
}

bool RtAccumulateCut(const RtAccumulateCamera& prev, const RtAccumulateCamera& cur) {
  if (!prev.valid || !cur.valid) return true;
  double vp[16];
  if (!Invert4(prev.inv, vp)) return true;
  // The distances of the centre's points at depth 0 and 1 (the near and far
  // planes, whichever way depth runs; one may be at infinity).
  auto dist = [&](const double h[4]) -> double {
    if (!(std::fabs(h[3]) > 0.0)) return std::numeric_limits<double>::infinity();
    double s = 0;
    for (int k = 0; k < 3; k++) s += (h[k] / h[3] - cur.eye[k]) * (h[k] / h[3] - cur.eye[k]);
    return std::sqrt(s);
  };
  double h0[4], h1[4];
  ClipTo(cur.inv, 0, 0, 0, h0);
  ClipTo(cur.inv, 0, 0, 1, h1);
  const double a = dist(h0), b = dist(h1), lo = std::min(a, b), hi = std::max(a, b);
  if (!(lo > 0.0) || !std::isfinite(lo)) return true;
  const double mid = std::isfinite(hi) && hi > lo ? std::sqrt(lo * hi) : 10.0 * lo;
  // Where (x, y, ndc) of the last camera lands for the homogeneous world
  // point `w` (false: behind it).
  auto moved = [&](const double w[4], double x, double y) {
    double c[4] = {};
    for (int k = 0; k < 4; k++)
      for (int i = 0; i < 4; i++) c[k] += w[i] * vp[i * 4 + k];
    if (!(c[3] > 1e-12)) return true;
    const double dx = c[0] / c[3] - x, dy = c[1] / c[3] - y;
    return dx * dx + dy * dy > (2.0 / 3.0) * (2.0 / 3.0);
  };
  static constexpr double kPts[5][2] = {{0, 0}, {-0.8, -0.8}, {0.8, -0.8}, {-0.8, 0.8}, {0.8, 0.8}};
  for (const auto& p : kPts) {
    // The ray through the point (its direction from the camera's position).
    double hm[4], d[3];
    ClipTo(cur.inv, p[0], p[1], 0.5, hm);
    if (!(std::fabs(hm[3]) > 0.0)) return true;
    double l = 0;
    for (int k = 0; k < 3; k++) l += (d[k] = hm[k] / hm[3] - cur.eye[k]) * d[k];
    l = std::sqrt(l);
    if (!(l > 0.0)) return true;
    const double far_dir[4] = {d[0] / l, d[1] / l, d[2] / l, 0.0};
    const double at_mid[4] = {cur.eye[0] + d[0] / l * mid, cur.eye[1] + d[1] / l * mid, cur.eye[2] + d[2] / l * mid,
                              1.0};
    if (moved(far_dir, p[0], p[1]) || moved(at_mid, p[0], p[1])) return true;
  }
  return false;
}

std::unique_ptr<RtAccumulator> RtAccumulator::Make(id<MTLDevice> dev) {
  if (!dev) return nullptr;
  auto a = std::make_unique<RtAccumulator>();
  a->dev_ = dev;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(kRtAccumulateMsl) options:nil error:&err];
  bool ok = lib != nil;
  auto pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    return fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
  };
  for (int ch = 0; ch < 2 && ok; ch++)
    for (int dk = 0; dk < 2 && ok; dk++) {
      NSString* kind = [NSString stringWithFormat:@"%@%d", dk ? @"color" : @"depth", ch ? 3 : 1];
      a->temporal_[ch][dk] = pipe([NSString stringWithFormat:@"rt_acc_temporal_%@_cs", kind]);
      a->filter_[ch][dk] = pipe([NSString stringWithFormat:@"rt_acc_filter_%@_cs", kind]);
      ok = a->temporal_[ch][dk] && a->filter_[ch][dk];
    }
  if (!ok) {
    std::fprintf(stderr, "gfx: the ray-traced terms' accumulation: %s\n",
                 err ? err.localizedDescription.UTF8String : "no pipeline");
    return nullptr;
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageShaderRead;
  a->no_normals_ = [dev newTextureWithDescriptor:td];
  a->no_counts_ = [dev newBufferWithLength:sizeof(RtAccumulateCounts) options:MTLResourceStorageModePrivate];
  if (!a->no_normals_ || !a->no_counts_) return nullptr;
  return a;
}

uint32_t RtAccumulateTiles(uint32_t w, uint32_t h, uint32_t step) {
  if (!step) return 0;
  const uint32_t lw = (w + step - 1) / step, lh = (h + step - 1) / step;
  return ((lw + kRtAccumulateGroup - 1) / kRtAccumulateGroup) * ((lh + kRtAccumulateGroup - 1) / kRtAccumulateGroup);
}

bool RtAccumulator::Begin(RtAccumulateHistory* h, const RtFrame& f, const RtShadowParams& p,
                          const RtAccumulateTerm& term, uint32_t step, uint32_t stride, uint32_t frame,
                          const float light[3], bool reset, RtAccumulateFrame* out, std::vector<id>* retired) const {
  *out = RtAccumulateFrame{};
  if (!h || (term.channels != 1 && term.channels != 3) || step < 1 || step > 8) return false;
  const uint32_t w = p.size[0], hh = p.size[1];
  if (!w || !hh || w > 65535u * step || hh > 65535u * step) return false;
  // A second frame of the same number: not accumulated again (its samples
  // would count twice; another light's would take the history over).
  if (h->have && h->frame == frame) {
    out->again = true;
    return false;
  }
  RtAccumulateCamera cam;
  if (!RtAccumulateCameraOf(f, &cam)) {
    h->have = false;
    return false;
  }
  auto make = [&](MTLPixelFormat fmt, uint32_t tw, uint32_t th) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                  width:std::max(tw, 1u)
                                                                                 height:std::max(th, 1u)
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [dev_ newTextureWithDescriptor:td];
  };
  if (h->w != w || h->h != hh || h->channels != term.channels || !h->hist[0] || !h->hist[1] || !h->color[0] ||
      !h->color[1] || !h->tiles) {
    for (int k = 0; k < 2; k++) {
      if (retired && h->hist[k]) retired->push_back(h->hist[k]);
      if (retired && h->color[k]) retired->push_back(h->color[k]);
      h->hist[k] = make(MTLPixelFormatR32Uint, w, hh);
      // (One channel: the value is in hist; a texel stands in.)
      h->color[k] = term.channels == 3 ? make(MTLPixelFormatRGBA16Float, w, hh) : make(MTLPixelFormatRGBA16Float, 1, 1);
    }
    if (retired && h->tiles) retired->push_back(h->tiles);
    // Two lists of tiles (each frame fills one, empties the other), room
    // for every tile at any step: their threadgroup counts (indirect
    // dispatch arguments, 16 bytes apart), then the lists.
    const uint32_t most = RtAccumulateTiles(w, hh, 1);
    h->tiles = [dev_ newBufferWithLength:32 + 8 * size_t(most) options:MTLResourceStorageModeShared];
    if (h->tiles) {
      std::memset(h->tiles.contents, 0, h->tiles.length);
      auto* args = static_cast<uint32_t*>(h->tiles.contents);
      args[1] = args[2] = args[5] = args[6] = 1;
    }
    h->tiles_most = most;
    h->w = w;
    h->h = hh;
    h->channels = term.channels;
    h->have = false;
  }
  if (!h->hist[0] || !h->hist[1] || !h->color[0] || !h->color[1] || !h->tiles) {
    h->w = 0;  // (made again next time)
    h->have = false;
    return false;
  }
  // The light turned by more than about a degree (a jump of the sun, the
  // moon taking over): its shadows elsewhere.
  bool turned = false;
  if (light) {
    double a = 0, b = 0, ab = 0;
    for (int k = 0; k < 3; k++) {
      a += double(light[k]) * light[k];
      b += double(h->light[k]) * h->light[k];
      ab += double(light[k]) * h->light[k];
    }
    turned = !(a > 0 && b > 0 && ab >= kRtAccumulateLightCos * std::sqrt(a * b));
    for (int k = 0; k < 3; k++) h->light[k] = light[k];
  }
  double vp[16];
  const bool fresh = reset || !h->have || frame != h->frame + 1 || step != h->step || turned ||
                     RtAccumulateCut(h->camera, cam) || !Invert4(h->camera.inv, vp);
  RtAccumulateParams& q = out->p;
  std::memcpy(q.rx, p.rx, sizeof q.rx);
  std::memcpy(q.ry, p.ry, sizeof q.ry);
  std::memcpy(q.rv, p.rv, sizeof q.rv);
  std::memcpy(q.rc, p.rc, sizeof q.rc);
  for (int k = 0; k < 3; k++) q.sky_value[k] = term.sky[k];
  q.size[0] = w;
  q.size[1] = hh;
  q.jitter[0] = p.jitter[0];
  q.jitter[1] = p.jitter[1];
  q.low[0] = (w + step - 1) / step;
  q.low[1] = (hh + step - 1) / step;
  q.step = step;
  q.stride = stride;
  q.frame = frame;
  q.depth_texture = p.depth_texture;
  q.sky = p.sky;
  q.pixel = p.pixel;
  q.history = fresh ? 0 : 1;
  q.max_count = kRtAccumulateMaxCount;
  q.scale = term.scale;
  if (!fresh) {
    // The point relative to this camera, in the history's clip space:
    // (q + eye) vp, the eye's part in double.
    double rows[4][4];
    for (int k = 0; k < 4; k++) {
      for (int i = 0; i < 3; i++) rows[i][k] = vp[i * 4 + k];
      rows[3][k] = cam.eye[0] * vp[k] + cam.eye[1] * vp[4 + k] + cam.eye[2] * vp[8 + k] + vp[12 + k];
    }
    for (int i = 0; i < 4; i++) Put(q.prev[i], rows[i]);
    for (int k = 0; k < 3; k++) q.eye_delta[k] = float(cam.eye[k] - h->camera.eye[k]);
    // A history pixel's ray (relative to that camera's position): clip (x,
    // y, 0.5, 1) through its rows less its position.
    double rel[4][4];
    for (int i = 0; i < 4; i++) {
      for (int k = 0; k < 3; k++) rel[i][k] = h->camera.inv[i * 4 + k] - h->camera.eye[k] * h->camera.inv[i * 4 + 3];
      rel[i][3] = h->camera.inv[i * 4 + 3];
    }
    double r2[4];
    for (int k = 0; k < 4; k++) r2[k] = 0.5 * rel[2][k] + rel[3][k];
    Put(q.hdir[0], rel[0]);
    Put(q.hdir[1], rel[1]);
    Put(q.hdir[2], r2);
  }
  out->channels = term.channels;
  out->reset = fresh;
  out->hist_in = h->hist[h->cur];
  out->color_in = h->color[h->cur];
  out->hist_out = h->hist[1 - h->cur];
  out->color_out = h->color[1 - h->cur];
  out->tiles = h->tiles;
  // (This frame's list: the one the frame before emptied.)
  const size_t list = 1 - size_t(h->cur);
  out->tiles_args = 16 * list;
  out->tiles_next = 16 * (1 - list);
  out->tiles_list = 32 + 4 * size_t(h->tiles_most) * list;
  h->cur = 1 - h->cur;
  h->camera = cam;
  h->have = true;
  h->frame = frame;
  h->step = step;
  h->frames++;
  h->resets += fresh ? 1 : 0;
  return true;
}

void RtAccumulator::Encode(id<MTLCommandBuffer> cmd, const RtAccumulateFrame& fr, id<MTLTexture> sparse,
                           id<MTLTexture> points, id<MTLTexture> depth, id<MTLTexture> normals, id<MTLTexture> out,
                           id<MTLBuffer> counts, size_t counts_offset) const {
  if (!cmd || !sparse || !points || !depth || !out || !fr.hist_in || !fr.hist_out || !fr.color_in || !fr.color_out ||
      !fr.tiles)
    return;
  RtAccumulateParams p = fr.p;
  if (sparse.width < p.low[0] || sparse.height < p.low[1] || points.width < p.low[0] || points.height < p.low[1] ||
      depth.width != p.size[0] || depth.height != p.size[1] || out.width != p.size[0] || out.height != p.size[1])
    return;
  if (normals && (normals.width != depth.width || normals.height != depth.height)) normals = nil;
  p.normals = normals ? 1 : 0;
  const int ch = fr.channels == 3 ? 1 : 0, dk = p.depth_texture ? 0 : 1;
  id<MTLBuffer> cb = counts ? counts : no_counts_;
  const size_t co = counts ? counts_offset : 0;
  const MTLSize group = MTLSizeMake(kRtAccumulateGroup, kRtAccumulateGroup, 1);
  // The pixels (a group a tile of 8 x 8 blocks), then the filter over the
  // tiles they listed (as many threadgroups as they counted).
  id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt accumulate temporal");
  [e setComputePipelineState:temporal_[ch][dk]];
  [e setBytes:&p length:sizeof p atIndex:0];
  [e setBuffer:cb offset:co atIndex:1];
  [e setBuffer:fr.tiles offset:fr.tiles_args atIndex:2];
  [e setBuffer:fr.tiles offset:fr.tiles_next atIndex:3];
  [e setBuffer:fr.tiles offset:fr.tiles_list atIndex:4];
  [e setTexture:depth atIndex:0];
  [e setTexture:normals ? normals : no_normals_ atIndex:1];
  [e setTexture:sparse atIndex:2];
  [e setTexture:fr.hist_in atIndex:3];
  [e setTexture:fr.color_in atIndex:4];
  [e setTexture:fr.hist_out atIndex:5];
  [e setTexture:fr.color_out atIndex:6];
  [e setTexture:out atIndex:7];
  [e setTexture:points atIndex:8];
  [e dispatchThreadgroups:MTLSizeMake((p.low[0] + kRtAccumulateGroup - 1) / kRtAccumulateGroup,
                                      (p.low[1] + kRtAccumulateGroup - 1) / kRtAccumulateGroup, 1)
      threadsPerThreadgroup:group];
  [e endEncoding];
  e = TimedCompute(cmd, "rt accumulate filter");
  [e setComputePipelineState:filter_[ch][dk]];
  [e setBytes:&p length:sizeof p atIndex:0];
  [e setBuffer:cb offset:co atIndex:1];
  [e setBuffer:fr.tiles offset:fr.tiles_list atIndex:4];
  [e setTexture:depth atIndex:0];
  [e setTexture:normals ? normals : no_normals_ atIndex:1];
  [e setTexture:fr.hist_out atIndex:5];
  [e setTexture:fr.color_out atIndex:6];
  [e setTexture:out atIndex:7];
  [e dispatchThreadgroupsWithIndirectBuffer:fr.tiles indirectBufferOffset:fr.tiles_args threadsPerThreadgroup:group];
  [e endEncoding];
}

}  // namespace recomp::gfx
