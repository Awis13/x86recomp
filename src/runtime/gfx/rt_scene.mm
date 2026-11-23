// Ray tracing of the scene on Metal (rt_scene.h).

#import "runtime/gfx/rt_scene.h"

#include <TargetConditionals.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"

namespace recomp::gfx {

// ---- meshes

float HalfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  if (exp == 0) {  // zero or subnormal: man * 2^-24
    const float v = std::ldexp(float(man), -24);
    return sign ? -v : v;
  }
  const uint32_t bits = exp == 31 ? sign | 0x7F800000u | (man << 13) : sign | ((exp + 112) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

namespace {

// Primitives whose index count (3 a triangle of a list) fits 32 bits.
constexpr uint32_t kMaxPrimitives = (UINT32_MAX - 2) / 3;

// Bytes a position of D3DDECLTYPE `type` takes (0: not gathered).
uint32_t PositionBytes(uint32_t type) {
  switch (type) {
    case 2: return 12;  // FLOAT3
    case 3: return 16;  // FLOAT4
    case 16: return 8;  // FLOAT16_4
    default: return 0;
  }
}

void ReadPosition(const uint8_t* p, uint32_t type, float* o) {
  if (type == 16) {
    uint16_t h[3];
    std::memcpy(h, p, 6);
    for (int k = 0; k < 3; k++) o[k] = HalfToFloat(h[k]);
  } else {
    std::memcpy(o, p, 12);
  }
}

// Indices the primitives read: 3 per triangle of a list, primitives + 2 of a
// strip or fan; 0 for other types.
uint32_t IndexCount(const RtGeometry& g) {
  if (!g.primitives || g.primitives > kMaxPrimitives) return 0;
  switch (g.prim) {
    case 4: return g.primitives * 3;
    case 5: case 6: return g.primitives + 2;
    default: return 0;
  }
}

// Vertex number (before base_vertex) of the draw's i-th index.
uint32_t IndexAt(const RtGeometry& g, uint32_t i) {
  if (!g.indices) return i;
  if (g.index_size == 4) {
    uint32_t v;
    std::memcpy(&v, static_cast<const uint8_t*>(g.indices) + size_t(i) * 4, 4);
    return v;
  }
  uint16_t v;
  std::memcpy(&v, static_cast<const uint8_t*>(g.indices) + size_t(i) * 2, 2);
  return v;
}

// The draw's index numbers of triangle t's corners.
void Corners(const RtGeometry& g, uint32_t t, uint32_t c[3]) {
  if (g.prim == 4) {
    c[0] = 3 * t, c[1] = 3 * t + 1, c[2] = 3 * t + 2;
  } else if (g.prim == 5) {
    // D3D's strip: every other triangle's first two swapped, so all wind alike.
    c[0] = (t & 1) ? t + 1 : t, c[1] = (t & 1) ? t : t + 1, c[2] = t + 2;
  } else {
    c[0] = 0, c[1] = t + 1, c[2] = t + 2;
  }
}

}  // namespace

bool RtPositionType(uint32_t decl_type) { return PositionBytes(decl_type) != 0; }

uint32_t RtAttributeBytes(uint32_t type) {
  static constexpr uint8_t kBytes[17] = {4, 8, 12, 16, 4, 4, 4, 8, 4, 4, 8, 4, 8, 4, 4, 4, 8};
  return type < 17 ? kBytes[type] : 0;
}

void RtFetch(const uint8_t* p, uint32_t type, float o[4]) {
  // As the Metal prelude's r_fetch fills an input register.
  uint32_t w[4] = {};
  std::memcpy(w, p, RtAttributeBytes(type));
  auto f = [&](int i) {
    float x;
    std::memcpy(&x, &w[i], 4);
    return x;
  };
  auto lo = [](uint32_t x) { return float(int16_t(x & 0xFFFF)); };
  auto hi = [](uint32_t x) { return float(int16_t(x >> 16)); };
  auto b = [](uint32_t x, int k) { return float((x >> (8 * k)) & 0xFF); };
  o[0] = o[1] = o[2] = 0.0f;
  o[3] = 1.0f;
  switch (type) {
    case 3: o[3] = f(3); [[fallthrough]];
    case 2: o[2] = f(2); [[fallthrough]];
    case 1: o[1] = f(1); [[fallthrough]];
    case 0: o[0] = f(0); break;
    case 4:  // D3DCOLOR: BGRA bytes
      o[0] = b(w[0], 2) / 255.0f, o[1] = b(w[0], 1) / 255.0f, o[2] = b(w[0], 0) / 255.0f, o[3] = b(w[0], 3) / 255.0f;
      break;
    case 5: case 8:
      for (int k = 0; k < 4; k++) o[k] = type == 8 ? b(w[0], k) / 255.0f : b(w[0], k);
      break;
    case 6: o[0] = lo(w[0]), o[1] = hi(w[0]); break;
    case 7: o[0] = lo(w[0]), o[1] = hi(w[0]), o[2] = lo(w[1]), o[3] = hi(w[1]); break;
    case 9: o[0] = std::max(lo(w[0]) / 32767.0f, -1.0f), o[1] = std::max(hi(w[0]) / 32767.0f, -1.0f); break;
    case 10:
      o[0] = lo(w[0]), o[1] = hi(w[0]), o[2] = lo(w[1]), o[3] = hi(w[1]);
      for (int k = 0; k < 4; k++) o[k] = std::max(o[k] / 32767.0f, -1.0f);
      break;
    case 11: o[0] = float(w[0] & 0xFFFF) / 65535.0f, o[1] = float(w[0] >> 16) / 65535.0f; break;
    case 12:
      o[0] = float(w[0] & 0xFFFF) / 65535.0f, o[1] = float(w[0] >> 16) / 65535.0f;
      o[2] = float(w[1] & 0xFFFF) / 65535.0f, o[3] = float(w[1] >> 16) / 65535.0f;
      break;
    case 13: o[0] = float(w[0] & 0x3FF), o[1] = float((w[0] >> 10) & 0x3FF), o[2] = float((w[0] >> 20) & 0x3FF); break;
    case 14:
      for (int k = 0; k < 3; k++) {
        const int32_t v = int32_t(w[0] << (22 - 10 * k)) >> 22;
        o[k] = std::max(float(v) / 511.0f, -1.0f);
      }
      break;
    case 15: o[0] = HalfToFloat(uint16_t(w[0])), o[1] = HalfToFloat(uint16_t(w[0] >> 16)); break;
    case 16:
      o[0] = HalfToFloat(uint16_t(w[0])), o[1] = HalfToFloat(uint16_t(w[0] >> 16));
      o[2] = HalfToFloat(uint16_t(w[1])), o[3] = HalfToFloat(uint16_t(w[1] >> 16));
      break;
    default: break;
  }
}

bool RtReadLane(const RtAttribute& a, uint32_t v, uint32_t lane, float* out) {
  const uint32_t bytes = RtAttributeBytes(a.type);
  if (!bytes || !a.data || lane > 3) return false;
  const uint64_t at = uint64_t(v) * a.stride;
  if (at + bytes > a.size) return false;
  float f[4];
  RtFetch(a.data + at, a.type, f);
  *out = f[lane];
  return true;
}

bool RtCutTriangle(const RtGeometry& g, uint32_t first, const uint32_t tri[3], RtCutPrim* out) {
  out->alpha = 0;
  for (int k = 0; k < 3; k++) {
    const uint32_t v = first + tri[k];
    float u = 0.0f, w = 0.0f, a = 1.0f;
    if (g.uv.data && (!RtReadLane(g.uv, v, g.uv.lane[0], &u) || !RtReadLane(g.uv, v, g.uv.lane[1], &w))) return false;
    if (g.alpha.data && !RtReadLane(g.alpha, v, g.alpha.lane[0], &a)) return false;
    out->uv[k][0] = u;
    out->uv[k][1] = w;
    const float c = std::isnan(a) ? 0.0f : std::clamp(a, 0.0f, 1.0f);
    out->alpha |= uint32_t(std::lround(c * 255.0f)) << (8 * k);
  }
  return true;
}

uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) return uint16_t(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));  // inf, NaN
  if (ax >= 0x477FF000u) return uint16_t(sign | 0x7C00u);  // 65520 and up round to infinity
  if (ax < 0x38800000u) {  // below 2^-14: a subnormal half (units of 2^-24)
    if (ax <= 0x33000000u) return uint16_t(sign);  // 2^-25 and below round to 0
    const uint32_t e = ax >> 23, m = (ax & 0x7FFFFFu) | 0x800000u, shift = 126 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return uint16_t(sign | r);
  }
  uint32_t r = (ax - 0x38000000u) >> 13;  // the exponent rebiased (127 to 15), 10 bits of mantissa
  const uint32_t rem = ax & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
  return uint16_t(sign | r);
}

uint32_t RtOctNormal(const float n[3]) {
  double x = n[0], y = n[1], z = n[2];
  const double s = std::fabs(x) + std::fabs(y) + std::fabs(z);
  if (!(s > 0) || !std::isfinite(s)) return 0;
  x /= s, y /= s, z /= s;
  if (z < 0) {
    const double ox = (1 - std::fabs(y)) * (x >= 0 ? 1 : -1), oy = (1 - std::fabs(x)) * (y >= 0 ? 1 : -1);
    x = ox, y = oy;
  }
  // As MSL's pack_float_to_snorm2x16: round(clamp(v, -1, 1) * 32767).
  auto snorm = [](double v) { return uint32_t(uint16_t(int16_t(std::lround(std::clamp(v, -1.0, 1.0) * 32767.0)))); };
  return snorm(x) | snorm(y) << 16;
}

void RtOctDecode(uint32_t v, float out[3]) {
  auto f = [](uint32_t b) { return std::max(float(int16_t(uint16_t(b))) / 32767.0f, -1.0f); };
  float x = f(v), y = f(v >> 16), z = 1.0f - std::fabs(x) - std::fabs(y);
  const float t = std::max(-z, 0.0f);
  x += x >= 0 ? -t : t;
  y += y >= 0 ? -t : t;
  const float l = std::sqrt(x * x + y * y + z * z);
  out[0] = x / l, out[1] = y / l, out[2] = z / l;
}

void RtHitTriangle(const RtCutPrim& cut, const float a[3], const float b[3], const float c[3], RtHitPrim* out) {
  *out = RtHitPrim{};
  const float u1[2] = {cut.uv[1][0] - cut.uv[0][0], cut.uv[1][1] - cut.uv[0][1]};
  const float u2[2] = {cut.uv[2][0] - cut.uv[0][0], cut.uv[2][1] - cut.uv[0][1]};
  out->uv0[0] = cut.uv[0][0];
  out->uv0[1] = cut.uv[0][1];
  auto h = [](float d) { return uint32_t(FloatToHalf(std::isnan(d) ? 0.0f : std::clamp(d, -65504.0f, 65504.0f))); };
  out->duv[0] = h(u1[0]) | h(u1[1]) << 16;
  out->duv[1] = h(u2[0]) | h(u2[1]) << 16;
  out->alpha = cut.alpha;
  const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  out->normal = RtOctNormal(n);
  // Twice each area: the UVs' (a 2D cross product) and the triangle's.
  const double ua = std::fabs(double(u1[0]) * u2[1] - double(u1[1]) * u2[0]);
  const double wa = std::sqrt(double(n[0]) * n[0] + double(n[1]) * n[1] + double(n[2]) * n[2]);
  out->lod = ua > 0 && wa > 0 && std::isfinite(ua / wa) ? float(0.5 * std::log2(ua / wa)) : 0.0f;
}

bool RtIndexRange(const RtGeometry& g, uint32_t* first, uint32_t* count) {
  const uint32_t n = IndexCount(g);
  if (!n) return false;
  uint32_t lo = 0, hi = n - 1;
  if (g.indices) {
    lo = UINT32_MAX, hi = 0;
    for (uint32_t i = 0; i < n; i++) {
      const uint32_t v = IndexAt(g, i);
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  const int64_t a = int64_t(g.base_vertex) + lo, b = int64_t(g.base_vertex) + hi;
  if (a < 0 || b >= int64_t(UINT32_MAX)) return false;
  *first = uint32_t(a);
  *count = uint32_t(b - a + 1);
  return true;
}

RtIndexRangeMemo::RtIndexRangeMemo(size_t slots) {
  size_t n = 1;
  while (n < slots) n <<= 1;
  slots_.resize(n);
}

bool RtIndexRangeMemo::Range(const RtGeometry& g, uint64_t buffer, uint64_t version, uint32_t offset, uint32_t* first,
                             uint32_t* count) {
  if (!g.indices || !version) return RtIndexRange(g, first, count);
  uint64_t h = buffer * 0x9E3779B97F4A7C15ull ^ (uint64_t(offset) << 20) ^ g.primitives ^
               (uint64_t(uint32_t(g.base_vertex)) << 40);
  h ^= h >> 29;
  Slot& s = slots_[size_t(h) & (slots_.size() - 1)];
  if (s.set && s.buffer == buffer && s.version == version && s.offset == offset && s.primitives == g.primitives &&
      s.prim == g.prim && s.index_size == g.index_size && s.base_vertex == g.base_vertex) {
    hits_++;
    if (s.ok) {
      *first = s.first;
      *count = s.count;
    }
    return s.ok;
  }
  misses_++;
  Slot n;
  n.buffer = buffer;
  n.version = version;
  n.offset = offset;
  n.primitives = g.primitives;
  n.prim = g.prim;
  n.index_size = g.index_size;
  n.base_vertex = g.base_vertex;
  n.ok = RtIndexRange(g, &n.first, &n.count);
  n.set = true;
  s = n;
  if (n.ok) {
    *first = n.first;
    *count = n.count;
  }
  return n.ok;
}

bool RtVertexRange(const RtGeometry& g, uint32_t* first, uint32_t* count) {
  const uint32_t bytes = PositionBytes(g.type);
  if (!bytes || !g.stride || !g.positions) return false;
  uint32_t a = 0, n = 0;
  if (!RtIndexRange(g, &a, &n)) return false;
  if ((uint64_t(a) + n - 1) * g.stride + bytes > g.size) return false;
  *first = a;
  *count = n;
  return true;
}

uint32_t RtCaptureStart(uint32_t first, int32_t base_vertex, bool in_place) {
  return in_place ? first : uint32_t(int64_t(first) - base_vertex);
}

void RtTriangles(const RtGeometry& g, uint32_t first, uint32_t* out) {
  const int64_t bias = int64_t(g.base_vertex) - int64_t(first);
  for (uint32_t t = 0; t < g.primitives; t++) {
    uint32_t c[3];
    Corners(g, t, c);
    for (int k = 0; k < 3; k++) out[size_t(t) * 3 + k] = uint32_t(int64_t(IndexAt(g, c[k])) + bias);
  }
}

bool RtGather(const RtGeometry& g, RtGathered* o) {
  uint32_t first = 0, count = 0;
  if (!RtVertexRange(g, &first, &count)) return false;
  const size_t n = size_t(g.primitives) * 3;
  o->indices.resize(n);
  RtTriangles(g, first, o->indices.data());
  o->used.clear();
  o->first = first;
  if (count <= std::max<size_t>(n * 4, 65536)) {
    // A table over the range, in the order of first use.
    o->remap.assign(count, UINT32_MAX);
    for (uint32_t& i : o->indices) {
      uint32_t& r = o->remap[i];
      if (r == UINT32_MAX) {
        r = uint32_t(o->used.size());
        o->used.push_back(i);
      }
      i = r;
    }
  } else {
    // A range far wider than the triangles use (indices into a large
    // shared buffer): the used vertices sorted.
    o->used.assign(o->indices.begin(), o->indices.end());
    std::sort(o->used.begin(), o->used.end());
    o->used.erase(std::unique(o->used.begin(), o->used.end()), o->used.end());
    for (uint32_t& i : o->indices)
      i = uint32_t(std::lower_bound(o->used.begin(), o->used.end(), i) - o->used.begin());
  }
  o->positions.resize(o->used.size() * 4);
  for (size_t v = 0; v < o->used.size(); v++) {
    ReadPosition(g.positions + size_t(first + o->used[v]) * g.stride, g.type, &o->positions[v * 4]);
    o->positions[v * 4 + 3] = 0.0f;
  }
  return true;
}

size_t RtMeshKeyHash::operator()(const RtMeshKey& k) const {
  uint64_t w[sizeof(RtMeshKey) / 8];
  static_assert(sizeof(RtMeshKey) % 8 == 0);
  std::memcpy(w, &k, sizeof w);
  uint64_t h = 0x9E3779B97F4A7C15ull;
  for (uint64_t x : w) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 31;
  }
  return size_t(h);
}

RtView RtViewOf(const RtDrawPass& p, bool captured, bool other_passes, bool other_color) {
  if (!p.depth || !p.test || !p.write) return RtView::None;
  if (p.color && p.main_size) return RtView::Main;
  return other_passes && !captured && (!p.color || p.depth_color || other_color) ? RtView::Other : RtView::None;
}

const char* RtSkipName(RtSkip s) {
  switch (s) {
    case RtSkip::NotTriangles: return "not triangles";
    case RtSkip::NotScene: return "not scene";
    case RtSkip::Blended: return "blended";
    case RtSkip::AlphaSource: return "alpha source";
    case RtSkip::NoMatrices: return "no matrices";
    case RtSkip::NoCapture: return "no capture variant";
    case RtSkip::NoPosition: return "no position";
    case RtSkip::ConstantStream: return "constant positions";
    case RtSkip::PositionType: return "position type";
    case RtSkip::UserMemory: return "user memory";
    case RtSkip::Dynamic: return "dynamic";
    case RtSkip::Filtered: return "filtered";
    case RtSkip::NotAffine: return "not affine";
    case RtSkip::OtherCamera: return "other camera";
    case RtSkip::OtherMirrored: return "other view mirrored";
    case RtSkip::BadCamera: return "bad camera";
    case RtSkip::BadIndices: return "bad indices";
    case RtSkip::TooLarge: return "too large";
    case RtSkip::Rebuilt: return "rebuilt often";
    case RtSkip::BuildBudget: return "build budget";
    case RtSkip::CaptureBudget: return "capture budget";
    case RtSkip::NoMemory: return "no memory";
    case RtSkip::Count: break;
  }
  return "?";
}

const char* const kRtMsl = R"(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#define RT_MATERIAL_VERSION 2
constant uint RT_MAT_CUT = 1u, RT_MAT_TEXTURE = 2u, RT_MAT_VERTEX_ALPHA = 4u, RT_MAT_KILL = 8u, RT_MAT_LINEAR = 16u,
              RT_MAT_BORDER_ALPHA = 32u, RT_MAT_HIT_PRIM = 64u, RT_MAT_ALBEDO = 128u;
constant uint RT_MASK_OPAQUE = 1u, RT_MASK_CUT = 2u, RT_MASK_CAPTURED_OPAQUE = 4u, RT_MASK_CAPTURED_CUT = 8u;
struct RtInfo { uint draw, mesh, triangles, flags; };
struct RtMaterial {
  texture2d<float> alpha_texture;
  float alpha_scale, alpha_offset, kill_scale, kill_offset;
  float max_level;
  uint flags;  // RT_MAT_*; func 8-11, ref 16-23, channel 24-25, address U 26-28, V 29-31
  texture2d<float> albedo_texture;
  float albedo_max_level;
  uint albedo_flags;   // address U 0-2, V 3-5
  uint albedo_border;  // D3DCOLOR
  uint reserved[3];
};
struct RtCutPrim { packed_float2 uv[3]; uint alpha; };
struct RtHitPrim {
  packed_float2 uv0;
  uint duv[2];  // half2: uv1 - uv0, uv2 - uv0
  uint alpha;
  uint normal;  // octahedral, snorm16 x 2
  float lod;
};
// An octahedrally encoded direction (RtOctNormal), unit length.
float3 rt_oct_decode(uint v) {
  const float2 e = unpack_snorm2x16_to_float(v);
  float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
  const float t = max(-n.z, 0.0);
  n.x += n.x >= 0.0 ? -t : t;
  n.y += n.y >= 0.0 ? -t : t;
  return normalize(n);
}
// The UV at barycentric weights w.
float2 rt_hit_uv(const device RtHitPrim& p, float3 w) {
  return float2(p.uv0) + float2(as_type<half2>(p.duv[0])) * w.y + float2(as_type<half2>(p.duv[1])) * w.z;
}
// Texel number i of n under D3DTADDRESS `mode`; `out`: outside (BORDER).
int rt_address(int i, int n, uint mode, thread bool& out) {
  switch (mode) {
    case 2u: {  // MIRROR
      const int p = 2 * n;
      int m = i % p;
      if (m < 0) m += p;
      return m < n ? m : p - 1 - m;
    }
    case 3u: return clamp(i, 0, n - 1);  // CLAMP
    case 4u:  // BORDER
      out = out || i < 0 || i >= n;
      return clamp(i, 0, n - 1);
    case 5u: return min(i < 0 ? -1 - i : i, n - 1);  // MIRRORONCE
    default: {  // WRAP
      const int m = i % n;
      return m < 0 ? m + n : m;
    }
  }
}
// Texel p of level lvl, channel ch: sampled at its centre, nearest.
float rt_texel(texture2d<float> t, int2 p, uint lvl, uint ch, float2 size) {
  constexpr sampler s(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
  return t.sample(s, (float2(p) + 0.5) / size, level(float(lvl)))[ch];
}
// The material's texture's channel at uv: one level, addressed by hand.
float RtTexelAlpha(const device RtMaterial& m, float2 uv) {
  const texture2d<float> t = m.alpha_texture;
  const uint lvl = min(uint(max(m.max_level, 0.0)), t.get_num_mip_levels() - 1u);
  const int w = int(t.get_width(lvl)), h = int(t.get_height(lvl));
  const float2 size = float2(w, h);
  const uint ch = (m.flags >> 24) & 3u, mu = (m.flags >> 26) & 7u, mv = (m.flags >> 29) & 7u;
  const float border = (m.flags & RT_MAT_BORDER_ALPHA) != 0u ? 1.0 : 0.0;
  const float2 x = clamp(uv * size, -1.0e8, 1.0e8);
  if ((m.flags & RT_MAT_LINEAR) == 0u) {
    bool out = false;
    const int2 p = int2(rt_address(int(floor(x.x)), w, mu, out), rt_address(int(floor(x.y)), h, mv, out));
    return out ? border : rt_texel(t, p, lvl, ch, size);
  }
  const float2 c = x - 0.5, f = floor(c), fr = c - f;
  float a[4];
  for (int k = 0; k < 4; k++) {
    bool out = false;
    const int2 p = int2(rt_address(int(f.x) + (k & 1), w, mu, out), rt_address(int(f.y) + (k >> 1), h, mv, out));
    a[k] = out ? border : rt_texel(t, p, lvl, ch, size);
  }
  return mix(mix(a[0], a[1], fr.x), mix(a[2], a[3], fr.x), fr.y);
}
// Whether a candidate hit (primitive data `prim`, barycentrics b) on an
// instance of material m is kept (RtMaterial in rt_scene.h).
bool RtAlphaPass(const device RtMaterial& m, const device void* prim, float2 b) {
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
  if ((m.flags & RT_MAT_TEXTURE) != 0u) v = RtTexelAlpha(m, uv);
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
struct RtHit {
  bool hit;
  float distance;
  uint instance, primitive;  // user ID + geometry ID (its RtInfo, RtMaterial), triangle
  float2 bary;
  bool front;  // D3D's front face
  float4x3 world_to_object;
  const device void* prim;  // primitive data
};
// The intersection query loop: candidates (non-opaque instances) committed
// if RtAlphaPass keeps them; `any`: the first kept ends it.
RtHit rt_trace(instance_acceleration_structure as, ray r, uint mask, const device RtMaterial* table,
               intersection_params params, bool any) {
  params.assume_geometry_type(geometry_type::triangle);
  params.accept_any_intersection(any);
  intersection_query<instancing, triangle_data> q;
  q.reset(r, as, mask, params);
  while (q.next())
    if (RtAlphaPass(table[q.get_candidate_user_instance_id() + q.get_candidate_geometry_id()],
                    q.get_candidate_primitive_data(), q.get_candidate_triangle_barycentric_coord()))
      q.commit_triangle_intersection();
  RtHit h;
  h.hit = q.get_committed_intersection_type() == intersection_type::triangle;
  h.distance = INFINITY;
  h.instance = 0xFFFFFFFFu;
  h.primitive = 0xFFFFFFFFu;
  h.bary = float2(0.0);
  h.front = false;
  h.world_to_object = float4x3(1.0);
  h.prim = nullptr;
  if (h.hit) {
    h.distance = q.get_committed_distance();
    h.instance = q.get_committed_user_instance_id() + q.get_committed_geometry_id();  // its slot (RtMaterial, RtInfo)
    h.primitive = q.get_committed_primitive_id();
    h.bary = q.get_committed_triangle_barycentric_coord();
    h.front = q.is_committed_triangle_front_facing();
    h.world_to_object = q.get_committed_world_to_object_transform();
    h.prim = q.get_committed_primitive_data();
  }
  return h;
}
RtHit RtTraceClosest(instance_acceleration_structure as, ray r, uint mask, const device RtMaterial* table,
                     intersection_params params) {
  return rt_trace(as, r, mask, table, params, false);
}
bool RtTraceAny(instance_acceleration_structure as, ray r, uint mask, const device RtMaterial* table,
                intersection_params params) {
  return rt_trace(as, r, mask, table, params, true).hit;
}
// Whether the light along unit l is blocked from o (a ray start off the
// surface), to tmax: static instances from o on, captured ones beyond
// `gap` (their triangles are not the ones drawn: kRtShadowCapturedGap,
// gfx/rt_shadow.h).
bool rt_shadowed(instance_acceleration_structure scene, float3 o, float3 l, float tmax, float gap,
                 const device RtMaterial* materials) {
  const uint fixed = RT_MASK_OPAQUE | RT_MASK_CUT, moving = RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  return RtTraceAny(scene, ray(o, l, 0.0, tmax), fixed, materials, intersection_params()) ||
         RtTraceAny(scene, ray(o, l, gap, tmax), moving, materials, intersection_params());
}
// The same answer with the instances' info (RtInfo: which are captured):
// one traversal of static and captured instances from o, the first kept
// hit ending it; a static one, or a captured one beyond the gap, blocks.
// Only where it met a captured instance within the gap (a point on or
// next to a person or vehicle, which must not shadow itself) are the two
// phases above traced. As the sun mask's kernel does (gfx/rt_shadow.h,
// rt_shadow_blocked).
constant uint RT_INFO_IS_CAPTURED = 1u << 17;  // rt_scene.h's kRtInfoCaptured
bool rt_shadowed(instance_acceleration_structure scene, float3 o, float3 l, float tmax, float gap,
                 const device RtMaterial* materials, const device RtInfo* info) {
  const uint all = RT_MASK_OPAQUE | RT_MASK_CUT | RT_MASK_CAPTURED_OPAQUE | RT_MASK_CAPTURED_CUT;
  const RtHit h = rt_trace(scene, ray(o, l, 0.0, tmax), all, materials, intersection_params(), true);
  if (!h.hit) return false;
  if ((info[h.instance].flags & RT_INFO_IS_CAPTURED) == 0u || h.distance >= gap) return true;
  return rt_shadowed(scene, o, l, tmax, gap, materials);
}
// The hit triangle's world-space unit normal (RtHitPrim's, by the
// transpose of world-to-object), and how many times larger its area is in
// the world than in its BLAS: |det(object to world)| |n world| / |n| (n
// unit).
float3 rt_hit_normal(RtHit h, thread float& area_scale) {
  const device RtHitPrim& p = *(const device RtHitPrim*)h.prim;
  const float3 no = rt_oct_decode(p.normal);
  const float4x3 w = h.world_to_object;
  const float3x3 m = float3x3(w[0], w[1], w[2]);
  const float3 nw = no * m;
  const float lw = length(nw), d = abs(determinant(m));
  area_scale = d > 0.0 ? lw / d : 1.0;
  return lw > 0.0 ? nw / lw : float3(0.0, 0.0, 1.0);
}
float3 RtHitNormal(RtHit h) {
  float s;
  return rt_hit_normal(h, s);
}
// The level of a w x h texture at the hit of a ray along `dir` whose cone
// is `cone` wide there (ray cones: the triangle's constant in world space,
// the texture's size, the width over the cosine of the incidence).
float RtHitLod(RtHit h, float3 dir, float cone, float2 size) {
  float s;
  const float3 n = rt_hit_normal(h, s);
  const float lod = ((const device RtHitPrim*)h.prim)->lod - 0.5 * log2(max(s, 1e-30));
  const float c = max(abs(dot(normalize(dir), n)), 1e-3);
  return lod + 0.5 * log2(size.x * size.y) + log2(max(cone, 1e-30) / c);
}
// A UV coordinate under D3DTADDRESS `mode`; `out`: outside (BORDER).
float rt_address_uv(float u, uint mode, thread bool& out) {
  switch (mode) {
    case 2u: {  // MIRROR
      const float m = u - 2.0 * floor(u * 0.5);
      return m <= 1.0 ? m : 2.0 - m;
    }
    case 3u: return saturate(u);  // CLAMP
    case 4u:  // BORDER
      out = out || u < 0.0 || u > 1.0;
      return saturate(u);
    case 5u: return saturate(abs(u));  // MIRRORONCE
    default: return u - floor(u);  // WRAP
  }
}
// The material's albedo at the hit of a ray along `dir`, its cone `cone`
// wide there: trilinear, at the cone's level, no finer than
// albedo_max_level; the border colour outside a BORDER texture; white
// without one. WRAP on both axes by the sampler (no seam), the other modes
// by hand.
float3 RtHitAlbedo(const device RtMaterial& m, RtHit h, float3 dir, float cone) {
  if ((m.flags & RT_MAT_ALBEDO) == 0u || (m.flags & RT_MAT_HIT_PRIM) == 0u) return float3(1.0);
  const float3 w = float3(1.0 - h.bary.x - h.bary.y, h.bary.x, h.bary.y);
  float2 uv = clamp(rt_hit_uv(*(const device RtHitPrim*)h.prim, w), -1.0e8, 1.0e8);
  const uint mu = m.albedo_flags & 7u, mv = (m.albedo_flags >> 3) & 7u;
  const bool wrap = (mu == 0u || mu == 1u) && (mv == 0u || mv == 1u);
  if (!wrap) {
    bool out = false;
    uv = float2(rt_address_uv(uv.x, mu, out), rt_address_uv(uv.y, mv, out));
    if (out) return unpack_unorm4x8_to_float(m.albedo_border).zyx;  // D3DCOLOR: B, G, R, A from the low byte
  }
  const texture2d<float> t = m.albedo_texture;
  const float top = float(t.get_num_mip_levels() - 1u);
  const float lod = clamp(RtHitLod(h, dir, cone, float2(t.get_width(), t.get_height())),
                          min(max(m.albedo_max_level, 0.0), top), top);
  constexpr sampler repeat(filter::linear, mip_filter::linear, address::repeat);
  constexpr sampler edge(filter::linear, mip_filter::linear, address::clamp_to_edge);
  return (wrap ? t.sample(repeat, uv, level(lod)) : t.sample(edge, uv, level(lod))).rgb;
}
)";

bool RtAlphaPasses(const RtMaterial& m, float p) {
  if ((m.flags & kRtMatKill) && m.kill_scale * p + m.kill_offset < 0.0f) return false;
  const uint32_t func = (m.flags >> 8) & 15u;
  if (func == 0 || func == 8) return true;
  const float x = m.alpha_scale * p + m.alpha_offset;
  const float v = std::isnan(x) ? 0.0f : std::clamp(x, 0.0f, 1.0f);
  const int a = int(std::round(v * 255.0f)), ref = int((m.flags >> 16) & 255u);
  switch (func) {
    case 1: return false;
    case 2: return a < ref;
    case 3: return a == ref;
    case 4: return a <= ref;
    case 5: return a > ref;
    case 6: return a != ref;
    case 7: return a >= ref;
    default: return true;
  }
}

bool RtCutAlwaysPasses(const RtMaterial& m, float lo) {
  const uint32_t func = (m.flags >> 8) & 15u;
  if (func == 3 || func == 6) return false;
  return RtAlphaPasses(m, lo) && RtAlphaPasses(m, 1.0f);
}

void RtUseResources(id<MTLComputeCommandEncoder> e, const RtFrame* f) {
  if (!f) return;
  if (f->blas_count && !f->blas_resident) [e useResources:f->blas count:f->blas_count usage:MTLResourceUsageRead];
  if (f->texture_count) [e useResources:f->textures count:f->texture_count usage:MTLResourceUsageRead];
}

void RtUseResources(id<MTLRenderCommandEncoder> e, const RtFrame* f, MTLRenderStages stages) {
  if (!f) return;
  if (f->blas_count && !f->blas_resident)
    [e useResources:f->blas count:f->blas_count usage:MTLResourceUsageRead stages:stages];
  if (f->texture_count)
    [e useResources:f->textures count:f->texture_count usage:MTLResourceUsageRead stages:stages];
}

// A mesh's BLAS.
// A large static mesh's other pieces (RtBlas::pieces).
struct RtBlasPiece {
  id<MTLAccelerationStructure> as = nil;
  MTLResourceID id{};
};

struct RtBlas {
  using Piece = RtBlasPiece;
  id<MTLAccelerationStructure> as = nil;
  MTLResourceID id{};
  // A large static mesh's other pieces (RtScene::Pieces): its triangles
  // split by place, `as` the first piece's, each piece a BLAS of its own
  // and an instance of its own (the same matrix, mask and material), so
  // that a ray enters only the pieces near it, not every mesh whose bounds
  // hold it (a station's floors, walls and ceilings, each as large as the
  // station).
  std::vector<RtBlasPiece> pieces;
  RtMeshKey key;
  uint64_t bytes = 0;
  uint64_t last_serial = 0;  // the last frame whose work used it: its build, a TLAS
  uint64_t last_frame = 0;   // the last presented frame that added an instance of it
  uint64_t mark = 0;         // the last TLAS that listed it (RtFrame::blas)
  uint32_t mesh = 0, triangles = 0;
  // A captured draw's (AddCaptured): rebuilt every frame, not cached; ready
  // once its conversion is handed over (TakeCaptures), with the instance
  // options the frame's camera gave (`opts`, from D3DCULL `cull`).
  bool captured = false, ready = false;
  uint32_t cull = 0, opts = 0;
  // A captured draw's with AddCaptured's key (in `key`): refittable, kept
  // for that key's later draws (RtScene::kCaptureRefits); refits since its
  // last build, its primitive data's stride, its scratch sizes.
  bool keyed = false;
  uint32_t refits = 0, stride = 0, nth = 0;
  uint64_t build_scratch = 0, refit_scratch = 0;
  bool prim_data = false;  // a geometry has primitive data (a refit copies it anew)
  // An actor's (AddCaptured's `actor`, else 0): its parts, a geometry
  // each in draw order: their keys, triangles << 32 | primitive data
  // stride (what a refit wants the same), materials (kRtMatHitPrim set
  // where it applies) and their textures (the scene's HeldTex: alpha,
  // albedo), how many are alpha-cut, its instance's mask; the geometries
  // until its BLAS is made (FinishActor).
  uint64_t actor = 0;
  uint32_t parts = 0, cut_parts = 0, mask = 0;
  std::vector<RtMeshKey> part_keys;
  std::vector<uint64_t> part_shape;
  std::vector<RtMaterial> part_mat;
  std::vector<void*> part_tex, part_albedo;
  NSMutableArray* part_geos = nil;
  bool cut = false;  // alpha-cut: RtCutPrim primitive data (RtMeshKey::cut)
  uint8_t alpha_lo = 0;  // with `cut`: the least vertex alpha of its primitive data (0-255)
  // Hit data (RtSceneOptions::hit_data, drawn with UVs): RtHitPrim primitive
  // data; `uvs`: its UVs read (else zeros: no albedo).
  bool hit = false, uvs = true;
  float lo[3] = {}, hi[3] = {};  // object-space bounds of a static mesh's vertices (RtInstanceSummary)
  // A static mesh's: RtMeshKeyHash of `key` and of its geometry (the key
  // without what alpha-cuts it), made once for every lookup of it.
  size_t key_hash = 0, geo_hash = 0;
  uint32_t retained = 0;  // retained instances naming it
  bool retired = false;   // out of the cache (Retire, DropUnbuilt): its retained instances go
};

// BLAS builds handed to the encoding side.
struct RtBuildBatch {
  struct Job {
    MTLPrimitiveAccelerationStructureDescriptor* desc = nil;
    id<MTLAccelerationStructure> as = nil;
    __unsafe_unretained id<MTLBuffer> scratch = nil;  // the frame slot's
    size_t scratch_off = 0;
    RtBlas* blas = nullptr;
    // A refit of `as` in place (it holds a build of the same triangles,
    // made with MTLAccelerationStructureUsageRefit), its vertices and
    // primitive data from `desc`; else a build.
    bool refit = false;
  };
  std::vector<Job> jobs;
};

// Captured draws handed to the encoding side: their positions converted to
// world space (a dispatch per range of a capture buffer), then their BLASes
// built.
struct RtCaptureBatch {
  // Vertices [first, first + count) of a frame slot's capture buffer `buf`
  // (captures at 16 bytes a vertex from 0, world positions at 12 from
  // capacity * 16).
  struct Range {
    __unsafe_unretained id<MTLBuffer> buf;
    uint32_t first, count, capacity;
  };
  // The triangles' normals (the debug view's, RtDebug::Normals; with `hit`
  // RtHitPrim's, their LOD constants too): from the positions at voff of
  // vbuf and the indices at ioff of ibuf, to noff of ibuf (the debug view's
  // after an alpha-cut triangle's RtCutPrim; the RtHitPrims), `stride` bytes
  // apart.
  struct Normals {
    __unsafe_unretained id<MTLBuffer> vbuf;
    size_t voff;
    __unsafe_unretained id<MTLBuffer> ibuf;
    size_t ioff, noff;
    uint32_t triangles, stride, hit;
  };
  std::vector<Range> ranges;
  std::vector<RtBuildBatch::Job> jobs;
  std::vector<Normals> normals;
  // World position = (clip - t) . (n0, n1, n2): n0, n1, n2, t (float4s over
  // the clip position's components).
  float convert[16] = {};
};

namespace {

// The debug view (RtDebug): one ray per pixel from the frame's camera
// (after kRtMsl).
constexpr const char* kRtDebugMsl = R"(
struct RtDebugParams {
  float4x4 inv_vp;  // clip space to world: the rows of D3D's matrix as columns
  uint2 size;
  uint mode;        // 1 normals, 2 instances
  uint reversed;    // depth 1 nearest (RtFrame::depth_reversed)
};
uint rt_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
kernel void rt_debug_cs(instance_acceleration_structure scene [[buffer(0)]],
                        constant RtDebugParams& p [[buffer(1)]],
                        const device RtInfo* info [[buffer(2)]],
                        const device RtMaterial* materials [[buffer(3)]],
                        texture2d<half, access::read> src [[texture(0)]],
                        texture2d<half, access::write> dst [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.size.x || gid.y >= p.size.y) return;
  const float2 ndc = float2((float(gid.x) + 0.5) / float(p.size.x) * 2.0 - 1.0,
                            1.0 - (float(gid.y) + 0.5) / float(p.size.y) * 2.0);
  // The ray from the camera's position (the point clip space sends to w = 0
  // at the centre: whichever way depth runs) through the pixel at half
  // depth; an orthographic camera has none, its rays start on the nearest
  // depth's plane (1 where depth is reversed).
  const float4 e = p.inv_vp * float4(0.0, 0.0, 1.0, 0.0);
  const float4 a = p.inv_vp * float4(ndc, p.reversed != 0 ? 1.0 : 0.0, 1.0), b = p.inv_vp * float4(ndc, 0.5, 1.0);
  const float3 pb = b.xyz / b.w;
  const bool persp = abs(e.w) > 1e-6 * length(e.xyz);
  const float3 o = persp ? e.xyz / e.w : a.xyz / a.w;
  const float3 d = normalize(pb - (persp ? o : a.xyz / a.w));
  ray r(o, d, 0.0, INFINITY);
  const RtHit hit = RtTraceClosest(scene, r, 0xFF, materials, intersection_params());
  half4 c = src.read(gid);
  if (hit.hit) {
    float3 rgb;
    if (p.mode == 1) {
      // The triangle's normal kept in its BLAS (object space; after an
      // alpha-cut triangle's RtCutPrim, or in RtHitPrim) to world space: by
      // the transpose of world-to-object, facing the camera.
      const float4x3 m = hit.world_to_object;
      const uint flags = materials[hit.instance].flags;
      const uint at = (flags & RT_MAT_CUT) != 0u ? 28u : 0u;
      const float3 no = (flags & RT_MAT_HIT_PRIM) != 0u
                            ? rt_oct_decode(((const device RtHitPrim*)hit.prim)->normal)
                            : float3(*(const device packed_float3*)((const device uchar*)hit.prim + at));
      float3 n = normalize(no * float3x3(m[0], m[1], m[2]));
      if (dot(n, d) > 0.0) n = -n;
      rgb = n * 0.5 + 0.5;
    } else {
      const uint h = rt_hash(info[hit.instance].draw * 0x9E3779B9u + 1u);
      rgb = float3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.75 + 0.25;
      const float2 uv = hit.bary;
      if (min(min(uv.x, uv.y), 1.0 - uv.x - uv.y) < 0.02) rgb *= 0.6;  // triangle edges
    }
    c = half4(half3(rgb), 1.0h);
  } else {
    c.rgb *= 0.3h;
  }
  dst.write(c, gid);
}
)";

// Captured draws: their clip-space positions to world space, and (the
// debug view's) their triangles' normals.
constexpr const char* kRtCaptureMsl = R"(
#include <metal_stdlib>
using namespace metal;
struct RtConvert {
  float4 n0, n1, n2, t;  // world = (clip - t) . (n0, n1, n2)
  uint count;
};
kernel void rt_convert_cs(const device float4* cap [[buffer(0)]], device packed_float3* out [[buffer(1)]],
                          constant RtConvert& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  // (The camera's part, t, taken off first: what is left is near the
  // camera's size, not the world's.)
  const float4 c = cap[i] - p.t;
  const float3 w = float3(dot(c, p.n0), dot(c, p.n1), dot(c, p.n2));
  // A vertex a zero bone matrix put at the world's origin (a part the
  // game hides: GTA IV's people and vehicles do so): its triangles have no
  // area there, but they stretched their BLAS's bounds from the person to
  // the origin, hundreds of metres, a box rays all over the view entered.
  // NaN: Metal leaves such a triangle out of the BLAS (in a build, and in a
  // refit, where one made inactive stays so until the next build).
  out[i] = length_squared(w) < 1e-4 ? packed_float3(NAN) : packed_float3(w);
}
// rt_scene.h's RtHitPrim, RtOctNormal.
struct RtHitPrim {
  packed_float2 uv0;
  uint duv[2];
  uint alpha;
  uint normal;
  float lod;
};
uint rt_oct_encode(float3 n) {
  const float s = abs(n.x) + abs(n.y) + abs(n.z);
  if (!(s > 0.0)) return 0u;
  n /= s;
  float2 e = n.xy;
  if (n.z < 0.0) e = (1.0 - abs(n.yx)) * select(float2(-1.0), float2(1.0), n.xy >= 0.0);
  return pack_float_to_snorm2x16(e);
}
kernel void rt_normals_cs(const device packed_float3* v [[buffer(0)]], const device uint* idx [[buffer(1)]],
                          device uchar* n [[buffer(2)]], constant uint4& p [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
  // p: triangles, bytes from one to the next, whether they are RtHitPrims
  // (else a bare normal is written).
  if (t >= p.x) return;
  const float3 a = v[idx[t * 3]], b = v[idx[t * 3 + 1]], c = v[idx[t * 3 + 2]];
  const float3 nn = cross(b - a, c - a);
  device uchar* at = n + t * p.y;
  if (p.z == 0u) {
    *(device packed_float3*)at = packed_float3(nn);
    return;
  }
  // The UVs written on the CPU; the normal and the LOD constant here.
  device RtHitPrim& h = *(device RtHitPrim*)at;
  h.normal = rt_oct_encode(nn);
  const float2 e1 = float2(as_type<half2>(h.duv[0])), e2 = float2(as_type<half2>(h.duv[1]));
  const float ua = abs(e1.x * e2.y - e1.y * e2.x), wa = length(nn);
  h.lod = ua > 0.0 && wa > 0.0 ? 0.5 * log2(ua / wa) : 0.0;
}
)";

// Whether world * vp is wvp, to the precision float matrices made apart
// can agree to (each element against the size of its own terms).
bool SameProduct(const float (*w)[4], const double* vp, const float (*wvp)[4]) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) {
      double v = 0, mag = 0;
      for (int i = 0; i < 4; i++) {
        const double t = double(w[r][i]) * vp[i * 4 + k];
        v += t;
        mag += std::fabs(t);
      }
      if (!(std::fabs(v - double(wvp[r][k])) <= 2e-4 * mag + 1e-6)) return false;
    }
  return true;
}

double Det3(double a, double b, double c, double d, double e, double f, double g, double h, double i) {
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// Whether `wvp` (object to clip, rows) shows triangles mirrored: those
// clockwise in object space (D3D's front faces when nothing mirrors)
// counterclockwise on screen. A triangle's screen orientation is the sign
// of its plane at the camera's position times that of the minor of rows
// 0-2 in columns x, y and w (by Cauchy-Binet: the camera's position is the
// point those columns send to 0). Depth (column z) has no part in it: a
// projection storing depth reversed winds as one that does not. An
// orthographic matrix (column w 0, 0, 0, 1) has no camera position: the
// minor of rows 0-2 in columns x, y and z decides, its sign turned where
// the draw keeps the greatest depth (`reversed`: the camera looks toward
// depth 0).
// Assumes the shader makes its position's x and y with the same rows as z
// and w: hlsl::Program::position_matrix proves only z and w.
bool Mirrors(const float (*m)[4], bool reversed) {
  double big = 0, w = 0;
  for (int r = 0; r < 3; r++) {
    w = std::max(w, std::fabs(double(m[r][3])));
    for (int k = 0; k < 3; k++) big = std::max(big, std::fabs(double(m[r][k])));
  }
  if (w > 1e-7 * big)
    return Det3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]) < 0;
  const double d = Det3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
  return reversed ? d > 0 : d < 0;
}

class MetalRtScene final : public RtScene {
 public:
  MetalRtScene(id<MTLDevice> dev, const RtSceneOptions& o) : dev_(dev), opt_(o) {}
  ~MetalRtScene() override {
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *))
      if (rs_ && rs_queue_) [rs_queue_ removeResidencySet:(id<MTLResidencySet>)rs_];
  }

  bool Init() {
    NSError* err = nil;
    id<MTLLibrary> cap = [dev_ newLibraryWithSource:@(kRtCaptureMsl) options:nil error:&err];
    if (cap) {
      convert_pipe_ = [dev_ newComputePipelineStateWithFunction:[cap newFunctionWithName:@"rt_convert_cs"] error:&err];
      if (convert_pipe_ && (opt_.debug == RtDebug::Normals || opt_.hit_data))
        normals_pipe_ = [dev_ newComputePipelineStateWithFunction:[cap newFunctionWithName:@"rt_normals_cs"]
                                                            error:&err];
    }
    if (!convert_pipe_ || ((opt_.debug == RtDebug::Normals || opt_.hit_data) && !normals_pipe_)) {
      std::fprintf(stderr, "gfx: ray tracing captured draws: %s\n", err.localizedDescription.UTF8String);
      return false;
    }
    if (opt_.debug == RtDebug::Off) return true;
    id<MTLLibrary> lib = [dev_ newLibraryWithSource:@((std::string(kRtMsl) + kRtDebugMsl).c_str())
                                            options:nil
                                              error:&err];
    if (!lib) {
      std::fprintf(stderr, "gfx: ray tracing debug view: %s\n", err.localizedDescription.UTF8String);
      return false;
    }
    debug_pipe_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rt_debug_cs"] error:&err];
    if (!debug_pipe_) {
      std::fprintf(stderr, "gfx: ray tracing debug view: %s\n", err.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }

  bool UseResidency(id<MTLCommandQueue> queue) override {
    if (rs_ || !queue) return rs_ != nil;
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
      MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
      d.label = @"ray-traced scene's BLASes";
      NSError* err = nil;
      id<MTLResidencySet> set = [dev_ newResidencySetWithDescriptor:d error:&err];
      if (!set) {
        std::fprintf(stderr, "gfx: ray tracing's residency set: %s\n", err.localizedDescription.UTF8String);
        return false;
      }
      rs_ = set;
      // What the scene already holds.
      for (const auto& [k, b] : cache_) {
        Resident(b->as);
        for (const RtBlas::Piece& c : b->pieces) Resident(c.as);
      }
      for (const auto& b : retired_) {
        Resident(b->as);
        for (const RtBlas::Piece& c : b->pieces) Resident(c.as);
      }
      for (const auto& b : cap_live_) Resident(b->as);
      for (const PooledAs& a : as_pool_) Resident(a.as);
      for (const auto& [k, v] : warm_)
        for (const Warm& w : v) Resident(w.as);
      CommitResidency();
      [queue addResidencySet:set];
      rs_queue_ = queue;
      return true;
    }
    return false;
  }

  void SetAllocator(RtAlloc alloc, void* ctx) override {
    alloc_ = alloc;
    alloc_ctx_ = ctx;
  }

  void BeginFrame(uint64_t serial, uint64_t completed) override {
    // (The caller's frame slots promise this slot's last frame is done;
    // `completed` may lag behind that, so it is not checked against it.)
    // Builds recorded but never encoded (no EndFrame since): those BLASes
    // were never built.
    if (cur_batch_) {
      DropUnbuilt(*cur_batch_);
      cur_batch_ = nullptr;
    }
    // Likewise captures never converted: their instances stay out.
    // (Not kept for refits: a build that never ran left nothing to refit.)
    if (cur_cap_) {
      for (const RtBuildBatch::Job& j : cur_cap_->jobs) j.blas->ready = j.blas->keyed = false;
      cur_cap_ = nullptr;
    }
    actor_ = nullptr;  // (an actor open then: never made, its instance out)
    actor_batch_ = nullptr;
    serial_ = serial;
    ended_ = false;
    FrameSlot& s = slots_[serial % kSlots];
    for (size_t i = 0; i < s.batches_used; i++) s.batches[i]->jobs.clear();
    s.batches_used = 0;
    s.frames_used = 0;
    // Scratch: the last (largest) buffer kept, unless far above what the
    // slot's last frame needed.
    if (s.scratch.size() > 1) s.scratch.erase(s.scratch.begin(), s.scratch.end() - 1);
    if (!s.scratch.empty() && s.scratch.back().length > 4 * std::max<size_t>(s.scratch_used, kMinScratch))
      s.scratch.clear();
    s.scratch_used = 0;
    // The pool's buffers (LaneScratch): each lane's last (largest) kept if
    // it held what the slot's last frame took from the lane and is not far
    // above it.
    for (FrameSlot::ScratchLane& l : s.lanes) {
      if (l.bufs.size() > 1) l.bufs.erase(l.bufs.begin(), l.bufs.end() - 1);
      if (!l.bufs.empty() &&
          (l.bufs.back().length < l.need || l.bufs.back().length > 4 * std::max<size_t>(l.need, kMinLaneScratch)))
        l.bufs.clear();
      l.prev_need = l.need;
      l.used = l.need = 0;
    }
    s.lane_next = 0;
    for (size_t i = 0; i < s.cap_batches_used; i++) {
      RtCaptureBatch& b = *s.cap_batches[i];
      b.ranges.clear();
      b.jobs.clear();
      b.normals.clear();
    }
    s.cap_batches_used = 0;
    // Capture buffers: the last (largest) kept if it held what the slot's
    // last frame needed and is not far above it.
    if (s.cap.size() > 1) s.cap.erase(s.cap.begin(), s.cap.end() - 1);
    if (!s.cap.empty() && (s.cap.back().capacity < s.cap_need ||
                           s.cap.back().capacity > 4 * std::max(s.cap_need, kMinCaptureVertices)))
      s.cap.clear();
    s.cap_prev_need = s.cap_need;
    s.cap_used = s.cap_need = 0;
    // Captured BLASes, under the same rule as retired ones: pooled for
    // later captures; those with a key kept for that key's next draws
    // (refit).
    std::erase_if(cap_live_, [&](std::unique_ptr<RtBlas>& b) {
      if (b->last_serial + kSlots - 1 > completed || b->last_frame + 2 > frame_) return false;
      // (An actor whose BLAS was never made has none.)
      if (b->as && b->keyed)
        warm_[b->key].push_back({b->as, b->bytes, frame_, b->build_scratch, b->refit_scratch, b->refits,
                                 b->triangles, b->stride, b->nth, std::move(b->part_shape)});
      else if (b->as)
        as_pool_.push_back({b->as, b->as.size, b->bytes, frame_});
      *b = RtBlas{};
      cap_spare_.push_back(std::move(b));
      return true;
    });
    PurgeRetained();
    // Retired BLASes: once the GPU is past every frame that may trace a
    // TLAS naming them (an RtFrame is valid until its slot is reused:
    // kSlots - 1 frames after its own), and no instance that may still go
    // into a TLAS (this or the last presented frame's) names them.
    std::erase_if(retired_, [&](const std::unique_ptr<RtBlas>& b) {
      if (b->last_serial + kSlots - 1 > completed || b->last_frame + 2 > frame_) return false;
      NotResident(b->as);
      for (const RtBlas::Piece& c : b->pieces) NotResident(c.as);
      retired_bytes_ -= b->bytes;
      return true;
    });
    // Textures material tables name: the same rule (a TLAS naming one may
    // be traced until its slot horizon; carried instances name it a
    // presented frame longer).
    std::erase_if(held_, [&](const auto& kv) {
      const HeldTex& h = *kv.second;
      if (h.retained || h.last_serial + kSlots - 1 > completed || h.last_frame + 2 > frame_) return false;
      held_bytes_ -= h.bytes;
      return true;
    });
  }

  bool Add(const RtMeshKey& key, const RtGeometry& g, const float (*world)[4], const float (*wvp)[4],
           uint32_t cull, uint32_t zfunc, const RtCut* cut, RtView view, const RtAlbedo* albedo) override {
    vote_ = 0;
    if (ended_ || view == RtView::None) return false;
    const bool main = view == RtView::Main;
    const uint32_t draw = draws_++;
    // Affine (column 3 is 0, 0, 0, 1).
    if (!(std::fabs(world[0][3]) < 1e-5f && std::fabs(world[1][3]) < 1e-5f && std::fabs(world[2][3]) < 1e-5f &&
          std::fabs(world[3][3] - 1.0f) < 1e-5f)) {
      Skip(RtSkip::NotAffine);
      return false;
    }
    // Another view's mirrored world (a reflection drawn by flipping the
    // world): not where the mesh is.
    if (!main && Det3(world[0][0], world[0][1], world[0][2], world[1][0], world[1][1], world[1][2], world[2][0],
                      world[2][1], world[2][2]) < 0) {
      Skip(RtSkip::OtherMirrored);
      return false;
    }
    // The view-projection a main view's draw was drawn with: one of the
    // frame's, or a new one (world^-1 * wvp), counted once the draw is
    // taken. The end of the frame keeps the most used. Other views' draws
    // have no part in it (kAnyCamera).
    const bool reversed = zfunc == 5 || zfunc == 7;  // D3DCMP_GREATER, GREATEREQUAL
    uint32_t cam = 0;
    while (main && cam < cams_.size() && !SameProduct(world, cams_[cam].vp, wvp)) cam++;
    const bool new_cam = main && cam == cams_.size();
    Camera fresh{};
    if (new_cam) {
      if (cams_.size() >= kCameras) {
        Skip(RtSkip::OtherCamera);
        return false;
      }
      double w[16], iw[16], m[16];
      for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) {
          w[r * 4 + k] = world[r][k];
          m[r * 4 + k] = wvp[r][k];
        }
      if (!Invert4(w, iw)) {
        Skip(RtSkip::NotAffine);
        return false;
      }
      Mul4(iw, m, fresh.vp);
      fresh.reversed = reversed;
      fresh.id = ++camera_seq_;
    }
    // The same geometry at the same place this presented frame drawn by
    // another view too (into the main view and into each of a shadow map's
    // cascades, say; perhaps with another shader, so not what alpha-cuts
    // it): one instance, the main view's (its material), listed with any
    // camera. (The main view's own draws are taken as they come, as before
    // other views were: a mesh it draws twice is two instances.)
    Placed placed;
    placed.key = Geometry(key);
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 3; k++) placed.world[r * 3 + k] = world[r][k];
    // The key's hash and its geometry's, each made once (one where they are
    // the same key), for every lookup of this draw and of its instance
    // later in the frame (Pending::placed). The cache is looked up first
    // where its BLAS gives the geometry's hash (it has no other effect);
    // another view's opaque draw with hit data looks for the main view's
    // BLAS by its geometry first, and the cache only without one.
    auto whole = [&] { return placed.key == key; };  // (nothing in the key alpha-cuts it)
    const bool shares = opt_.hit_data && !main && !cut;
    size_t kh = 0, gh = 0;
    bool have_kh = false, looked = false;
    RtBlas* cached = nullptr;
    auto look_up = [&] {
      if (!have_kh) kh = KeyHash(key);
      have_kh = looked = true;
      if (auto it = cache_.find(KeyRef{&key, kh}); it != cache_.end()) cached = it->second.get();
    };
    if (!shares) {
      look_up();
      gh = cached ? cached->geo_hash : whole() ? kh : KeyHash(placed.key);
    } else {
      gh = KeyHash(placed.key);
      if (whole()) kh = gh, have_kh = true;
    }
    placed.hash = PlacedHashOf(gh, placed.world);
    // (An entry of an earlier frame is not this frame's place.)
    auto seen = placed_.find(placed);
    const bool here = seen != placed_.end() && seen->second.epoch == placed_epoch_;
    if (here && !main) {
      if (!seen->second.other) pending_[seen->second.index].camera = kAnyCamera;  // whichever camera wins
      stats_.deduped++;
      return false;
    }
    const bool replace = here && seen->second.other;  // a Main draw of an Other's instance
    // An opaque draw of another view of a mesh the main view drew with UVs
    // (hit data): the main view's BLAS (the key differs by the UVs).
    // (Looked up first: a BLAS the other view built before the main view
    // drew the mesh then goes unused and ages out.)
    RtBlas* b = nullptr;
    if (shares)
      if (auto al = by_geometry_.find(KeyRef{&placed.key, gh}); al != by_geometry_.end()) {
        b = al->second;
        stats_.shared_blas++;
      }
    if (!b) {
      if (!looked) look_up();
      b = cached;
    }
    if (!b && !(b = Build(key, g, kh, gh))) return false;
    if (opt_.hit_data && main && b->hit) {
      // (Written only where it names another BLAS: the last main view's
      // draw of the geometry this frame wins.)
      if (auto al = by_geometry_.find(KeyRef{&placed.key, gh}); al == by_geometry_.end())
        by_geometry_.emplace(placed.key, b);
      else if (al->second != b)
        al->second = b;
    }
    if (new_cam) cams_.push_back(fresh);
    b->last_frame = frame_;
    // Alpha-cut, unless no hit can fail its test: the product of the
    // texel (no less than texel_min) and the vertex alpha the primitive
    // data holds (no less than the mesh's least, where the material reads
    // it) passes the test from its least value to 1 (most of a city's cut
    // draws: an opaque DXT texture times vertex alphas all 255).
    bool is_cut = cut && b->cut;
    if (is_cut) {
      const float va = (cut->material.flags & kRtMatVertexAlpha) ? float(b->alpha_lo) / 255.0f : 1.0f;
      const float lo = std::clamp(cut->texel_min, 0.0f, 1.0f) * va;
      if (lo > 0.0f && RtCutAlwaysPasses(cut->material, lo)) {
        is_cut = false;
        stats_.cut_opaque++;
      }
    }
    const uint32_t material = AddMaterial(is_cut ? cut : nullptr, b->hit && b->uvs ? albedo : nullptr);
    if (replace) {
      seen->second.other = false;
      stats_.deduped++;
    } else if (!here) {
      // (An earlier frame's entry of this place taken over: no new node.)
      const PlacedAt at{placed_epoch_, uint32_t(pending_.size()), !main};
      placed_now_++;
      if (seen != placed_.end())
        seen->second = at;
      else
        placed_.emplace(placed, at);
    }
    Pending& p = replace ? pending_[seen->second.index] : pending_.emplace_back();
    auto& d = p.desc;
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 3; k++) d.transformationMatrix.columns[r].elements[k] = world[r][k];
    // Front faces are D3D's: clockwise on screen. Rays decide in object
    // space, so where the draw's matrices mirror (Mirrors) the other
    // winding; and the other again where the game culls clockwise faces
    // (it draws the counterclockwise ones). No culling for D3DCULL_NONE.
    // Alpha-cut: non-opaque (rays test each candidate), else opaque (no
    // candidates).
    uint32_t opts =
        is_cut ? MTLAccelerationStructureInstanceOptionNonOpaque : MTLAccelerationStructureInstanceOptionOpaque;
    if (Mirrors(wvp, reversed) != (cull == 2)) opts |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
    if (cull == 1) opts |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
    d.options = MTLAccelerationStructureInstanceOptions(opts);
    d.mask = is_cut ? kRtMaskCut : kRtMaskOpaque;
    d.intersectionFunctionTableOffset = 0;
    d.accelerationStructureID = b->id;
    p.blas = b;
    p.info = {draw, b->mesh, b->triangles, opts | (main ? 0u : kRtInfoOtherView)};
    // (Taking an Other's place: listed with any camera, as it was.)
    p.camera = main && !replace ? cam : kAnyCamera;
    p.material = material;
    p.placed = placed.hash;
    // Retained from an earlier frame: this draw is its instance now.
    if (!retained_.empty())
      if (auto r = retained_.find(placed); r != retained_.end()) {
        r->second.drawn = frame_;
        r->second.index = replace ? seen->second.index : uint32_t(pending_.size() - 1);
      }
    if (main) {
      cams_[cam].count++;
      vote_ = cams_[cam].id;
    }
    return true;
  }

  void RetainedSlots(const RtFrame* f, std::vector<std::pair<uint32_t, uint64_t>>* out) override {
    out->clear();
    if (f) *out = static_cast<const FrameData*>(f)->retained_slots;
  }

  void DropRetained(const uint64_t* ids, size_t n) override {
    if (!n || retained_.empty()) return;
    std::vector<uint64_t> want(ids, ids + n);
    std::sort(want.begin(), want.end());
    std::erase_if(retained_, [&](auto& kv) {
      Retained& r = kv.second;
      if (r.drawn == frame_ || !std::binary_search(want.begin(), want.end(), r.id)) return false;
      Drop(kv.first, r, RtRetainDrop::Replaced);
      return true;
    });
  }

  void DescribeNextTlas(std::vector<RtInstanceSummary>* out) override {
    describe_ = out;
    if (out) out->clear();
  }

  uint64_t TakeVote() override {
    const uint64_t v = vote_;
    vote_ = 0;
    return v;
  }

  uint64_t CameraId() const override {
    const int best = BestCamera();
    return best >= 0 ? cams_[size_t(best)].id : 0;
  }

  uint64_t CameraOf(const float (*world)[4], const float (*wvp)[4]) const override {
    if (!world || !wvp) return 0;
    for (const Camera& c : cams_)
      if (SameProduct(world, c.vp, wvp)) return c.id;
    return 0;
  }

  void Skip(RtSkip why) override { stats_.skipped[size_t(why)]++; }

  RtCaptureSlot AddCaptured(const RtGeometry& g, uint32_t first, uint32_t count, uint32_t cull,
                            const RtCut* cut, const RtAlbedo* albedo, const RtMeshKey* key, uint64_t actor) override {
    if (ended_) return {};
    const uint32_t draw = draws_++;
    const uint32_t tris = g.primitives;
    if (!tris || !count || !IndexCount(g)) {
      SkipCapture(RtSkip::BadIndices);
      return {};
    }
    if (tris > opt_.max_mesh_triangles) {
      SkipCapture(RtSkip::TooLarge);
      return {};
    }
    if (frame_caps_ >= kCapturesPerFrame || uint64_t(frame_cap_tris_) + tris > kCaptureTrianglesPerFrame ||
        uint64_t(frame_cap_verts_) + count > kCaptureVerticesPerFrame) {
      SkipCapture(RtSkip::CaptureBudget);
      return {};
    }
    if (!alloc_) {
      SkipCapture(RtSkip::NoMemory);
      return {};
    }
    // The triangles as a list over the captured vertices, then their
    // primitive data: an alpha-cut draw's RtCutPrim (from the vertex
    // streams, here) and the debug view's normals (made on the GPU); with
    // hit data RtHitPrim, its UVs here, its normal and LOD constant on the
    // GPU.
    const bool hit = opt_.hit_data && g.uv.data, debug = opt_.debug == RtDebug::Normals, normals = debug || hit;
    const uint32_t stride =
        hit ? uint32_t(sizeof(RtHitPrim)) : (cut ? uint32_t(sizeof(RtCutPrim)) : 0) + (debug ? 12 : 0);
    const size_t tri_bytes = size_t(tris) * 12;
    const size_t prim_at = (tri_bytes + 255) & ~size_t(255);
    const RtUpload u = alloc_(alloc_ctx_, stride ? prim_at + size_t(tris) * stride : tri_bytes, 256);
    if (!u.cpu) {
      SkipCapture(RtSkip::NoMemory);
      return {};
    }
    // The triangles and primitive data made on the CPU, kept by `key` for
    // the next frames' draws of the same (their positions alone change):
    // gathering the UVs took 10-25 ns a triangle, every frame.
    const size_t region = stride ? prim_at + size_t(tris) * stride : tri_bytes;
    const bool keep = (cut || hit) && key && key->attr_version && (!g.indices || key->index_version);
    CapPrims* kept = nullptr;
    if (keep)
      if (auto it = cap_prims_.find(*key); it != cap_prims_.end() && it->second.stride == stride &&
                                           it->second.bytes.size() == region)
        kept = &it->second;
    auto* tri = reinterpret_cast<uint32_t*>(u.cpu);
    // (UVs that cannot be read, of a draw not alpha-cut: zeros, and no
    // albedo.)
    bool uvs = true;
    if (kept) {
      std::memcpy(u.cpu, kept->bytes.data(), region);
      uvs = kept->uvs;
      kept->last_frame = frame_;
      stats_.capture_prims_reused++;
    } else {
      RtTriangles(g, first, tri);
    }
    if ((cut || hit) && !kept)
      for (uint32_t t = 0; t < tris; t++) {
        RtCutPrim cp;
        if (!RtCutTriangle(g, first, tri + size_t(t) * 3, &cp)) {
          if (cut) {
            SkipCapture(RtSkip::BadIndices);
            return {};
          }
          cp = RtCutPrim{};
          cp.alpha = 0xFFFFFFu;
          uvs = false;
        }
        if (hit) {
          // (The normal and LOD constant made on the GPU with the positions.)
          const float zero[3] = {};
          RtHitPrim hp;
          RtHitTriangle(cp, zero, zero, zero, &hp);
          std::memcpy(u.cpu + prim_at + size_t(t) * stride, &hp, sizeof hp);
        } else {
          std::memcpy(u.cpu + prim_at + size_t(t) * stride, &cp, sizeof cp);
        }
      }
    if (keep && !kept && cap_prim_bytes_ + region <= kCapturePrimBytes) {
      CapPrims& c = cap_prims_[*key];
      cap_prim_bytes_ -= c.bytes.size();
      c.bytes.assign(u.cpu, u.cpu + region);
      c.stride = stride;
      c.uvs = uvs;
      c.last_frame = frame_;
      cap_prim_bytes_ += region;
    }
    FrameSlot& s = slots_[serial_ % kSlots];
    const auto [cb, v0] = CaptureSpace(s, count);
    if (!cb.buf) {
      SkipCapture(RtSkip::NoMemory);
      return {};
    }
    const size_t vert_off = size_t(cb.capacity) * 16 + size_t(v0) * 12;
    MTLAccelerationStructureTriangleGeometryDescriptor* geo = [MTLAccelerationStructureTriangleGeometryDescriptor new];
    geo.vertexBuffer = cb.buf;
    geo.vertexBufferOffset = vert_off;  // a multiple of 12 and of 256 (CaptureSpace)
    geo.vertexStride = 12;
    geo.vertexFormat = MTLAttributeFormatFloat3;
    geo.indexBuffer = u.buf;
    geo.indexBufferOffset = u.off;
    geo.indexType = MTLIndexTypeUInt32;
    geo.triangleCount = tris;
    // (A single draw's alpha-cut instance is made non-opaque; an actor's
    // geometries say it each, its instance not.)
    geo.opaque = actor && key ? !cut : YES;
    if (stride) {
      geo.primitiveDataBuffer = u.buf;
      geo.primitiveDataBufferOffset = u.off + prim_at;
      geo.primitiveDataStride = stride;
      geo.primitiveDataElementSize = stride;
    }
    if (actor && key) {
      // A part of an actor: the BLAS made once its parts are all in
      // (FinishActor).
      EnsureBatch();
      if (!actor_ || actor_->actor != actor || actor_->cull != cull || actor_->parts >= kActorParts ||
          actor_batch_ != cur_cap_)
        StartActor(actor, cull, draw);
      RtBlas* b = actor_;
      b->parts++;
      b->triangles += tris;
      b->part_keys.push_back(*key);
      b->part_shape.push_back(uint64_t(tris) << 32 | stride);
      [b->part_geos addObject:geo];
      RtMaterial m{};
      HeldTex *tex = nullptr, *alb = nullptr;  // (held: Hold)
      MakeMaterial(cut, hit && uvs ? albedo : nullptr, &m, &tex, &alb);
      if (hit) m.flags |= kRtMatHitPrim;
      b->prim_data = b->prim_data || stride != 0;
      b->part_mat.push_back(m);
      b->part_tex.push_back(tex);
      b->part_albedo.push_back(alb);
      b->cut_parts += cut ? 1 : 0;
      b->mask |= cut ? kRtMaskCapturedCut : kRtMaskCapturedOpaque;
      CaptureWork(cb, v0, count, vert_off, u, prim_at, tris, stride, hit, cut != nullptr, normals);
      frame_caps_++;
      frame_cap_tris_ += tris;
      frame_cap_verts_ += count;
      return {cb.buf, size_t(v0) * 16, vert_off};
    }
    FinishActor();
    RtBuildBatch::Job job;
    @autoreleasepool {
      MTLPrimitiveAccelerationStructureDescriptor* desc = [MTLPrimitiveAccelerationStructureDescriptor new];
      desc.geometryDescriptors = @[ geo ];
      // Made every frame: a fast build, refittable where the key names the
      // triangles (a later frame's draw of the same refits it).
      desc.usage = MTLAccelerationStructureUsagePreferFastBuild | (key ? MTLAccelerationStructureUsageRefit : 0);
      const uint32_t nth = key ? cap_nth_[*key]++ : 0;
      CapturedAs a;
      if (!AcquireCaptured(desc, key, nth, tris, stride, nullptr, s, &a)) {
        SkipCapture(RtSkip::NoMemory);
        return {};
      }
      job.desc = desc;
      job.as = a.as;
      job.scratch = a.scratch;
      job.scratch_off = a.scratch_off;
      job.refit = a.refit;
      if (cap_spare_.empty()) cap_spare_.push_back(std::make_unique<RtBlas>());
      cap_live_.push_back(std::move(cap_spare_.back()));
      cap_spare_.pop_back();
      RtBlas* b = cap_live_.back().get();
      b->as = a.as;
      b->id = a.as.gpuResourceID;
      b->bytes = a.bytes;
      b->triangles = tris;
      b->captured = true;
      b->ready = false;
      b->cull = cull;
      b->cut = cut != nullptr;
      b->hit = hit;
      b->uvs = uvs;
      b->last_serial = serial_;
      b->last_frame = frame_;
      b->keyed = key != nullptr;
      if (key) b->key = *key;
      b->refits = a.refits;
      b->nth = nth;
      b->stride = stride;
      b->prim_data = stride != 0;
      b->build_scratch = a.build_scratch;
      b->refit_scratch = a.refit_scratch;
      job.blas = b;
    }
    CaptureWork(cb, v0, count, vert_off, u, prim_at, tris, stride, hit, cut != nullptr, normals);
    RtBlas* b = job.blas;
    cur_cap_->jobs.push_back(std::move(job));
    Pending& p = pending_.emplace_back();
    auto& d = p.desc;
    // In world space already: the identity.
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 3; k++) d.transformationMatrix.columns[r].elements[k] = r == k ? 1.0f : 0.0f;
    d.options = MTLAccelerationStructureInstanceOptionOpaque;  // (TakeCaptures gives them)
    d.mask = cut ? kRtMaskCapturedCut : kRtMaskCapturedOpaque;
    d.intersectionFunctionTableOffset = 0;
    d.accelerationStructureID = b->id;
    p.blas = b;
    p.info = {draw, 0, tris, kRtInfoCaptured};
    p.camera = kCapturedCamera;
    p.material = AddMaterial(cut, hit && uvs ? albedo : nullptr);
    pending_captured_++;
    stats_.capture_draws++;
    frame_caps_++;
    frame_cap_tris_ += tris;
    frame_cap_verts_ += count;
    return {cb.buf, size_t(v0) * 16, vert_off};
  }

  void SkipCapture(RtSkip why) override { stats_.capture_skipped[size_t(why)]++; }

  void NoteUncut() override { stats_.uncut++; }

  // (With the scratch pool the builds stay for the next Snapshot's or
  // EndFrame's work, Work: one encoder with the captured draws'.)
  bool PendingBuilds() const override { return cur_batch_ != nullptr && !scratch_lanes_; }

  const RtBuildBatch* TakeBuilds() override {
    const RtBuildBatch* b = cur_batch_;
    cur_batch_ = nullptr;
    return b;
  }

  const RtFrame* Snapshot() override {
    if (ended_) return nullptr;
    PurgeRetained();
    DropReplaced(false);
    const int best = BestCamera();
    const double* vp = best >= 0 ? cams_[best].vp : have_prev_cam_ ? prev_cam_ : nullptr;
    const bool reversed = best >= 0 ? cams_[best].reversed : prev_reversed_;
    FrameData* f = Work(vp, reversed);
    size_t slots = 0;
    if (const size_t n = vp ? TlasCount(best, true, &slots) : 0) {
      if (!f) f = &NextFrame();
      MakeTlas(*f, best, true, vp, reversed, n, slots);
      snap_counts_ = {frame_, f->instances, tlas_captured_, tlas_cut_, tlas_albedo_, tlas_other_, tlas_retained_};
    }
    snap_at_ = pending_.size();
    snapped_ = true;
    stats_.snapshots++;
    return f;
  }

  const RtFrame* EndFrame(bool presented) override {
    if (ended_) return nullptr;
    ended_ = true;
    PurgeRetained();
    const int best = BestCamera();
    const double* vp = best >= 0 ? cams_[best].vp : have_prev_cam_ ? prev_cam_ : nullptr;
    const bool reversed = best >= 0 ? cams_[best].reversed : prev_reversed_;
    FrameData* f = Work(vp, reversed);
    if (!presented) return f;  // the instances go on into the next frame
    // (Main draws of the other cameras. Captured ones not converted are
    // counted where they were left out.)
    for (const Pending& p : pending_)
      if (p.camera != kCapturedCamera && !Listed(p, best)) stats_.skipped[size_t(RtSkip::OtherCamera)]++;
    DropReplaced(true);
    Retain(best);
    size_t slots = 0;
    if (const size_t n = opt_.end_tlas && vp ? TlasCount(best, false, &slots) : 0) {
      FrameData& tl = f ? *f : NextFrame();
      f = &tl;
      MakeTlas(tl, best, false, vp, reversed, n, slots);
      stats_.instances += tl.instances;
      stats_.captured += tlas_captured_;
      stats_.cut += tlas_cut_;
      stats_.albedo += tlas_albedo_;
      stats_.other += tlas_other_;
      stats_.retained += tlas_retained_;
    } else if (!opt_.end_tlas && snap_counts_.frame == frame_) {
      // (Its last snapshot's: what the frame's effects traced.)
      const SnapCounts& c = snap_counts_;
      stats_.instances += c.instances;
      stats_.captured += c.captured;
      stats_.cut += c.cut;
      stats_.albedo += c.albedo;
      stats_.other += c.other;
      stats_.retained += c.retained;
    }
    // What the next presented frame's snapshots take from this one: the
    // instances after the last snapshot (with their materials).
    carry_.clear();
    carry_mat_.clear();
    if (snapped_)
      for (size_t i = snap_at_; i < pending_.size(); i++)
        // (Not those of other views: drawn again each frame.)
        if (pending_[i].camera != kAnyCamera && Listed(pending_[i], best)) {
          // (Its retained instance stays out of the next frame's snapshots:
          // this is it.)
          if (pending_[i].camera != kCapturedCamera)
            if (auto r = retained_.find(PlacedOf(pending_[i])); r != retained_.end()) r->second.carried = frame_ + 1;
          Pending& c = carry_.emplace_back(pending_[i]);
          if (c.material != kNoMaterial) {
            carry_mat_.push_back(pending_mat_[c.material]);
            c.material = uint32_t(carry_mat_.size() - 1);
          }
        }
    pending_mat_.clear();
    if (best >= 0) {
      std::memcpy(prev_cam_, cams_[best].vp, sizeof prev_cam_);
      prev_reversed_ = cams_[best].reversed;
      have_prev_cam_ = true;
    }
    // (The frame's first TLAS's, where sampled: MakeTlas.)
    if (static_sample_.frame == frame_) {
      stats_.static_samples++;
      stats_.static_listed += static_sample_.listed;
      stats_.static_distinct += static_sample_.distinct;
    }
    frame_builds_ = frame_tris_ = 0;  // the budgets are per presented frame
    frame_build_ns_ = 0;
    frame_caps_ = frame_cap_tris_ = frame_cap_verts_ = 0;
    cap_nth_.clear();
    pending_captured_ = 0;
    pending_.clear();
    replaced_from_ = 0;
    NextPlacedEpoch();
    cams_.clear();
    snap_at_ = 0;
    snapped_ = false;
    draws_ = 0;
    frame_++;
    stats_.frames++;
    Sweep();
    Trim();
    // Captured draws' primitive data no draw took for a while goes.
    std::erase_if(cap_prims_, [&](const auto& kv) {
      if (frame_ - kv.second.last_frame <= kPoolIdleFrames) return false;
      cap_prim_bytes_ -= kv.second.bytes.size();
      return true;
    });
    // Pooled BLAS memory no capture has taken for a while goes.
    std::erase_if(as_pool_, [&](const PooledAs& a) {
      if (frame_ - a.since <= kPoolIdleFrames) return false;
      cap_blas_bytes_ -= a.bytes;
      NotResident(a.as);
      return true;
    });
    std::erase_if(warm_, [&](auto& kv) {
      std::erase_if(kv.second, [&](const Warm& w) {
        if (frame_ - w.since <= kPoolIdleFrames) return false;
        cap_blas_bytes_ -= w.bytes;
        NotResident(w.as);
        return true;
      });
      return kv.second.empty();
    });
    return f && (f->builds || f->captures || f->tlas) ? f : nullptr;
  }

  void EncodeBuilds(id<MTLCommandBuffer> cmd, const RtBuildBatch* b) const override {
    EncodeJobs(cmd, b ? &b->jobs : nullptr, nullptr);
  }

  void EncodeFrame(id<MTLCommandBuffer> cmd, const RtFrame* frame) const override {
    if (!frame) return;
    const FrameData* f = static_cast<const FrameData*>(frame);
    // Captured positions to world space, then every BLAS build, then the
    // TLAS: each encoder after the one before (the buffers and structures
    // are tracked).
    EncodeConversion(cmd, f->captures);
    EncodeJobs(cmd, f->captures ? &f->captures->jobs : nullptr, f->builds ? &f->builds->jobs : nullptr);
    if (!f->tlas) return;
    id<MTLAccelerationStructureCommandEncoder> e = TimedAccelerationStructure(cmd, "rt TLAS build");
    // The instances name their BLASes by resource ID: resident, and ordered
    // after their builds.
    [e useResources:f->blas count:f->blas_count usage:MTLResourceUsageRead];
    [e buildAccelerationStructure:f->tlas descriptor:f->desc scratchBuffer:f->scratch scratchBufferOffset:f->scratch_off];
    [e endEncoding];
  }

  void EncodeDebug(id<MTLCommandBuffer> cmd, const RtFrame* f, id<MTLTexture> target) override {
    if (!debug_pipe_ || !f || !f->tlas || !target) return;
    const NSUInteger w = target.width, h = target.height;
    if (!debug_tex_ || debug_tex_.width != w || debug_tex_.height != h) {
      // (The command buffers that used the one before hold it.)
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:target.pixelFormat
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageShaderWrite;
      debug_tex_ = [dev_ newTextureWithDescriptor:td];
      if (!debug_tex_) return;
    }
    struct {
      float inv_vp[16];
      uint32_t size[2];
      uint32_t mode, reversed;
    } p = {};
    static_assert(sizeof p == 80);
    std::memcpy(p.inv_vp, f->inv_view_proj, sizeof p.inv_vp);
    p.size[0] = uint32_t(w);
    p.size[1] = uint32_t(h);
    p.mode = opt_.debug == RtDebug::Normals ? 1 : 2;
    p.reversed = f->depth_reversed ? 1 : 0;
    id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt debug view");
    [e setComputePipelineState:debug_pipe_];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    [e setBytes:&p length:sizeof p atIndex:1];
    [e setBuffer:f->info offset:f->info_offset atIndex:2];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:3];
    RtUseResources(e, f);
    [e setTexture:target atIndex:0];
    [e setTexture:debug_tex_ atIndex:1];
    [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [e endEncoding];
    id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder];
    [bl copyFromTexture:debug_tex_ toTexture:target];
    [bl endEncoding];
  }

  RtStats Stats() const override {
    RtStats s = stats_;
    s.key_hashes = key_hashes_;
    s.cached = cache_.size();
    s.blas_bytes = blas_bytes_;
    s.retired = retired_.size();
    s.retired_bytes = retired_bytes_;
    s.total_builds = total_builds_;
    s.capture_blas_bytes = cap_blas_bytes_;
    s.capture_buffer_bytes = CaptureBufferBytes();
    s.cut_textures = held_.size();
    s.capture_prim_bytes = cap_prim_bytes_;
    s.cut_texture_bytes = held_bytes_;
    s.retained_kept = retained_.size();
    s.places = placed_.size();
    return s;
  }

  std::string TakeStats() override {
    const RtStats s = Stats();
    stats_ = {};
    key_hashes_ = 0;
    if (!s.frames) return {};
    const double f = double(s.frames);
    char b[400];
    std::snprintf(b, sizeof b,
                  "ray tracing per frame %.0f instances, %.2f snapshots, %.2f BLAS builds (%.0f triangles; %llu "
                  "meshes in %llu pieces); %llu BLAS cached (%.1f MB), %llu let go waiting for the GPU (%.1f MB); "
                  "skipped per frame:",
                  double(s.instances) / f, double(s.snapshots) / f, double(s.builds) / f,
                  double(s.build_triangles) / f, (unsigned long long)s.pieced, (unsigned long long)s.pieces,
                  (unsigned long long)s.cached, double(s.blas_bytes) / 1048576.0,
                  (unsigned long long)s.retired, double(s.retired_bytes) / 1048576.0);
    std::string out = b;
    bool any = false;
    for (size_t i = 0; i < size_t(RtSkip::Count); i++) {
      if (!s.skipped[i]) continue;
      std::snprintf(b, sizeof b, "%s %s %.1f", any ? "," : "", RtSkipName(RtSkip(i)), double(s.skipped[i]) / f);
      out += b;
      any = true;
    }
    if (!any) out += " none";
    std::snprintf(b, sizeof b,
                  "; captured per frame %.1f instances, %.1f BLAS builds of %.1f draws (%.1f of them refits; %.0f "
                  "triangles), %.1f MB (BLASes and buffers); captures skipped per frame:",
                  double(s.captured) / f, double(s.capture_builds) / f, double(s.capture_draws) / f,
                  double(s.capture_refits) / f,
                  double(s.capture_triangles) / f,
                  double(s.capture_blas_bytes + s.capture_buffer_bytes) / 1048576.0);
    out += b;
    any = false;
    for (size_t i = 0; i < size_t(RtSkip::Count); i++) {
      if (!s.capture_skipped[i]) continue;
      std::snprintf(b, sizeof b, "%s %s %.1f", any ? "," : "", RtSkipName(RtSkip(i)), double(s.capture_skipped[i]) / f);
      out += b;
      any = true;
    }
    if (!any) out += " none";
    std::snprintf(b, sizeof b,
                  "; alpha-cut per frame %.1f instances, %.1f discarding draws taken as opaque, %.1f cut draws "
                  "opaque by their least alpha, %llu textures held "
                  "(%.1f MB); other passes per frame %.1f instances, %.1f draws already placed",
                  double(s.cut) / f, double(s.uncut) / f, double(s.cut_opaque) / f, (unsigned long long)s.cut_textures,
                  double(s.cut_texture_bytes) / 1048576.0, double(s.other) / f, double(s.deduped) / f);
    out += b;
    const uint64_t* rd = s.retained_dropped;
    std::snprintf(b, sizeof b,
                  "; retained per frame %.1f instances (%llu kept), let go per frame %.1f in view, %.1f replaced, "
                  "%.1f BLAS gone, %.1f idle, %.1f over the cap (%.1f of them within 100 m), %.1f the camera came closer to",
                  double(s.retained) / f, (unsigned long long)s.retained_kept,
                  double(rd[size_t(RtRetainDrop::InView)]) / f, double(rd[size_t(RtRetainDrop::Replaced)]) / f,
                  double(rd[size_t(RtRetainDrop::Gone)]) / f,
                  double(rd[size_t(RtRetainDrop::Idle)]) / f, double(rd[size_t(RtRetainDrop::Cap)]) / f,
                  double(s.retained_cap_near) / f,
                  double(rd[size_t(RtRetainDrop::Closer)]) / f);
    out += b;
    if (opt_.hit_data) {
      std::snprintf(b, sizeof b,
                    "; with an albedo per frame %.1f instances; %.1f other views' draws on the main view's BLAS; "
                    "%.1f captured draws' data kept (%.1f MB)",
                    double(s.albedo) / f, double(s.shared_blas) / f, double(s.capture_prims_reused) / f,
                    double(s.capture_prim_bytes) / 1048576.0);
      out += b;
    }
    if (s.static_samples) {
      // (Means over the frames sampled, one in kStaticSampleEvery.)
      const double n = double(s.static_samples);
      std::snprintf(b, sizeof b,
                    "; static instances in a TLAS %.0f, at %.0f distinct world matrices and instance options (%llu "
                    "frames sampled)",
                    double(s.static_listed) / n, double(s.static_distinct) / n, (unsigned long long)s.static_samples);
      out += b;
    }
    return out;
  }

  uint64_t MemoryBytes() const override {
    uint64_t n = blas_bytes_ + retired_bytes_ + cap_blas_bytes_ + CaptureBufferBytes() + cap_prim_bytes_ + held_bytes_ +
                 retained_.size() * (sizeof(Placed) + sizeof(Retained) + 4 * sizeof(void*));
    // The places kept (a node each: the next one, the hash, the entry; the
    // buckets), Retain's sets (their nodes the pool's).
    n += placed_.size() * (2 * sizeof(void*) + sizeof(Placed) + sizeof(PlacedAt)) +
         (placed_.bucket_count() + static_last_.bucket_count() + static_now_.bucket_count()) * sizeof(void*) +
         static_pool_.Bytes();
    for (const FrameSlot& s : slots_) {
      for (const auto& f : s.frames)
        if (f->store) n += f->store.allocatedSize;
      for (id<MTLBuffer> b : s.scratch) n += b.allocatedSize;
      for (const FrameSlot::ScratchLane& l : s.lanes)
        for (id<MTLBuffer> b : l.bufs) n += b.allocatedSize;
    }
    return n;
  }

 private:
  static constexpr size_t kCameras = 4;               // view-projections told apart in a frame
  static constexpr size_t kMinScratch = size_t(8) << 20;  // a frame slot's first scratch buffer
  // The scratch pool. A frame slot's BLAS builds and refits take their
  // scratch from kScratchLanes buffers in turn (LaneScratch), each still a
  // range of its own: Metal runs the builds of one encoder side by side,
  // but not two whose scratch is in the same buffer (WWDC22 10105,
  // "Maximize your Metal ray tracing performance": "builds which use the
  // same scratch buffer can't run in parallel", so "looping through a small
  // pool of scratch buffers"). One buffer for the whole slot (as before)
  // may have left a frame's 30-60 refits one after the other: 'rt BLAS
  // builds' was 0.55 ms + 15 us a job. Whether Metal tells ranges of one
  // buffer apart is not documented: the pass's time with
  // RECOMP_RT_BLAS_POOL=1 against the default says. With the pool the
  // static builds also wait for the frame's encoder (PendingBuilds).
  // RECOMP_RT_BLAS_POOL=n: n buffers (at most kMaxScratchLanes); 0: no
  // pool, all as before (one buffer a slot, static builds encoded between
  // render passes). A lane's first buffer is kMinLaneScratch (8 of them: a
  // slot's 8 MB as before); a slot's TLASes have a lane of their own (their
  // builds are in another encoder, and their scratch, 1-2 MB at 8000
  // instances, would make every lane it lands on grow to hold it).
  static constexpr uint32_t kScratchLanes = 8, kMaxScratchLanes = 16;
  static constexpr size_t kMinLaneScratch = size_t(1) << 20;
  static uint32_t ScratchLanesWanted() {
    const char* e = std::getenv("RECOMP_RT_BLAS_POOL");
    if (!e || !*e) return kScratchLanes;
    return uint32_t(std::min<unsigned long>(std::strtoul(e, nullptr, 10), kMaxScratchLanes));
  }
  // A static mesh's BLAS: default usage. RECOMP_RT_BLAS_FAST_TRACE=1 (off
  // by default, to measure): MTLAccelerationStructureUsagePreferFastIntersection
  // (macOS 26, iOS 26: "prioritize intersection performance over
  // acceleration structure build time"). They are built once (0.5-5 a
  // frame, a few hundred triangles each) and every ray's triangle tests
  // walk them. The frame's build budget counts builds and triangles, not
  // GPU time: a slower build shows in 'rt BLAS builds', not as skips.
  // Only BLASes built with it set have it: compare fresh launches.
  static MTLAccelerationStructureUsage StaticUsageWanted() {
    const char* e = std::getenv("RECOMP_RT_BLAS_FAST_TRACE");
    if (!e || *e != '1') return MTLAccelerationStructureUsageNone;
#if defined(__MAC_26_0)  // (an SDK that has the constant)
    if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
      std::fprintf(stderr, "gfx: ray tracing: static BLASes built for fast intersection (RECOMP_RT_BLAS_FAST_TRACE)\n");
      return MTLAccelerationStructureUsagePreferFastIntersection;
    }
#endif
    std::fprintf(stderr, "gfx: ray tracing: RECOMP_RT_BLAS_FAST_TRACE needs macOS 26 or iOS 26: static BLASes as "
                         "before\n");
    return MTLAccelerationStructureUsageNone;
  }
  // The TLAS: a fast build, as it is built anew for every snapshot and
  // frame end. RECOMP_RT_TLAS_QUALITY=1 (off by default, to measure):
  // default usage, a slower build ('rt TLAS build') for a tree that may
  // trace faster.
  static MTLAccelerationStructureUsage TlasUsageWanted() {
    const char* e = std::getenv("RECOMP_RT_TLAS_QUALITY");
    if (!e || *e != '1') return MTLAccelerationStructureUsagePreferFastBuild;
    std::fprintf(stderr, "gfx: ray tracing: TLAS built with default usage, not a fast build "
                         "(RECOMP_RT_TLAS_QUALITY)\n");
    return MTLAccelerationStructureUsageNone;
  }
  static constexpr size_t kKeptGather = size_t(4) << 20;  // Build's working memory kept between builds (elements)
  // Pending::camera of a captured draw's instance (in world space: no
  // camera of its own).
  static constexpr uint32_t kCapturedCamera = UINT32_MAX;
  // Pending::camera of another view's draw (RtView::Other): listed with any.
  static constexpr uint32_t kAnyCamera = UINT32_MAX - 1;
  // A frame slot's first capture buffer, in vertices (28 bytes each), and
  // the multiple capacities are of: 192 = 3 * 64, so that world positions
  // (12 bytes a vertex from capacity * 16) start at multiples of 256, as
  // captures do (each at a multiple of 64 vertices).
  static constexpr uint32_t kMinCaptureVertices = 192u << 6;
  static constexpr uint32_t kCaptureAlign = 64, kCaptureCapacityStep = 192;
  // Pooled BLAS memory of captured draws (as_pool_) unused for this many
  // presented frames is let go.
  static constexpr uint64_t kPoolIdleFrames = 60;
  // A captured BLAS kept for its key's refits (warm_) that long unused may
  // go to another key's draw (PooledBlas).
  static constexpr uint64_t kWarmIdleFrames = 3;
  // InView's margin: a retained instance is let go only well inside the
  // view (5% of its half-width in from each side), not where a jittered or
  // slightly different projection could have culled it.
  static constexpr double kViewMargin = 0.05;

  struct Pending {
    MTLIndirectAccelerationStructureInstanceDescriptor desc;
    RtBlas* blas;
    RtInstanceInfo info;
    uint32_t camera;
    uint32_t material;  // its material in pending_mat_ (carry_mat_ for carry_), or kNoMaterial
    size_t placed;      // a static draw's: its Placed's hash (PlacedOf)
  };
  static_assert(sizeof(Pending) == 112, "written for every draw taken: kept small");
  static constexpr uint32_t kNoMaterial = UINT32_MAX;
  // A texture material tables name, held until no TLAS naming it may be
  // traced (the rule retired BLASes follow).
  struct HeldTex {
    id<MTLTexture> tex;
    uint64_t bytes = 0;
    uint64_t last_serial = 0;  // the last frame whose TLAS named it
    uint64_t last_frame = 0;   // the last presented frame that added an instance naming it
    uint64_t mark = 0;         // the last TLAS that listed it (RtFrame::textures)
    uint32_t retained = 0;     // retained instances naming it (held while any does)
  };
  // An instance's material (alpha-cut, an albedo; its textures' IDs set
  // at MakeTlas).
  struct PendingMat {
    RtMaterial m;
    HeldTex* tex;     // the alpha's; null: none
    HeldTex* albedo;  // null: none
  };
  struct Camera {
    double vp[16];  // world to clip, rows
    uint32_t count;
    bool reversed;  // its first draw kept the greatest depth
    uint64_t id;    // RtScene::CameraId's number (never reused)
  };
  // A TLAS and what encoding and tracing it use; reused with its frame slot
  // (`store`, the TLAS itself, kept and grown).
  struct FrameData : RtFrame {
    MTLInstanceAccelerationStructureDescriptor* desc = nil;
    __unsafe_unretained id<MTLBuffer> scratch = nil;
    size_t scratch_off = 0;
    const RtBuildBatch* builds = nullptr;
    const RtCaptureBatch* captures = nullptr;
    std::vector<__unsafe_unretained id<MTLResource>> used;      // RtFrame::blas
    std::vector<__unsafe_unretained id<MTLResource>> held;      // RtFrame::textures
    id<MTLAccelerationStructure> store = nil;
    void Reset() {
      static_cast<RtFrame&>(*this) = RtFrame{};
      desc = nil;
      scratch = nil;
      scratch_off = 0;
      builds = nullptr;
      captures = nullptr;
      used.clear();
      held.clear();
      retained_slots.clear();
    }
    std::vector<std::pair<uint32_t, uint64_t>> retained_slots;  // RetainedSlots
  };
  // Per frame slot (serial % kSlots): what its frame's encoded work uses,
  // kept until the slot is reused.
  struct FrameSlot {
    std::vector<std::unique_ptr<RtBuildBatch>> batches;  // reused
    size_t batches_used = 0;
    std::vector<std::unique_ptr<FrameData>> frames;  // reused
    size_t frames_used = 0;
    std::vector<id<MTLBuffer>> scratch;  // the last one being filled
    size_t scratch_used = 0;
    // With the scratch pool (kScratchLanes) instead: its lanes, a BLAS
    // job's scratch from the next in turn (lane_next), and after them the
    // TLASes' lane. `used`: of bufs.back(); `need`: what the frame took
    // from the lane in all, `prev_need` the slot's last frame's (the size
    // of its next buffer).
    struct ScratchLane {
      std::vector<id<MTLBuffer>> bufs;  // the last one being filled
      size_t used = 0, need = 0, prev_need = 0;
    };
    ScratchLane lanes[kMaxScratchLanes + 1];
    uint32_t lane_next = 0;
    // Captured draws: their batches (reused), and the private buffers their
    // vertex shaders write (the last one being filled; vertices).
    std::vector<std::unique_ptr<RtCaptureBatch>> cap_batches;
    size_t cap_batches_used = 0;
    struct CaptureBuf {
      id<MTLBuffer> buf;
      uint32_t capacity;  // vertices: captures at 16 bytes each from 0, world positions at 12 from capacity * 16
    };
    std::vector<CaptureBuf> cap;
    uint32_t cap_used = 0, cap_need = 0, cap_prev_need = 0;  // vertices: of cap.back(), this frame, its last one
  };
  // A captured draw's BLAS with a key (RtBlas::keyed) no frame uses now,
  // kept for that key's next draws (warm_): its memory, the presented frame
  // it was let go, scratch sizes, refits since its build, what it holds.
  struct Warm {
    id<MTLAccelerationStructure> as = nil;
    uint64_t bytes = 0, since = 0, build_scratch = 0, refit_scratch = 0;
    uint32_t refits = 0, triangles = 0, stride = 0;
    uint32_t nth = 0;  // its frame's draw of the key it was: the first 0
    std::vector<uint64_t> shape;  // an actor's parts (RtBlas::part_shape); empty: a single draw's
  };
  // Memory of a captured draw's BLAS not in use: `size` usable.
  struct PooledAs {
    id<MTLAccelerationStructure> as;
    uint64_t size, bytes;
    uint64_t since;  // the presented frame it was pooled
  };
  // Where a mesh's data is (RtMeshKey::Place): its BLAS now, and how often
  // new contents there were built.
  struct Place {
    RtBlas* current = nullptr;
    uint64_t vertex_version = 0, index_version = 0, attr_version = 0;  // the last contents seen there
    uint64_t since = 0;         // presented frame they were first seen
    uint64_t window = 0;        // presented frame the rebuild count started
    uint64_t last_frame = 0;    // presented frame last drawn
    uint32_t rebuilds = 0;
    bool rewritten = false;     // left out (RtSkip::Rebuilt)
  };

  // A mesh key and its hash (RtMeshKeyHash), made once: what the maps of
  // mesh keys below are looked up by, without hashing the key again.
  struct KeyRef {
    const RtMeshKey* key;
    size_t hash;
  };
  // RtMeshKeyHash, counted (RtStats::key_hashes); a KeyRef's hash as it is.
  struct KeyHasher {
    using is_transparent = void;
    uint64_t* count = nullptr;
    size_t operator()(const RtMeshKey& k) const {
      if (count) ++*count;
      return RtMeshKeyHash()(k);
    }
    size_t operator()(const KeyRef& r) const { return r.hash; }
  };
  struct KeyEq {
    using is_transparent = void;
    bool operator()(const RtMeshKey& a, const RtMeshKey& b) const { return a == b; }
    bool operator()(const RtMeshKey& a, const KeyRef& b) const { return a == *b.key; }
    bool operator()(const KeyRef& a, const RtMeshKey& b) const { return *a.key == b; }
  };
  template <typename V>
  using KeyMap = std::unordered_map<RtMeshKey, V, KeyHasher, KeyEq>;
  // A key hashed for a static draw's bookkeeping (counted).
  size_t KeyHash(const RtMeshKey& k) { return KeyHasher{&key_hashes_}(k); }

  // A mesh's geometry (Geometry: its key without what alpha-cuts it) at a
  // world matrix (rows 0-3, columns 0-2: affine), as Add dedupes instances
  // of two views; the instance's place in pending_ and whether it is still
  // the other view's, as of placed_epoch_ `epoch` (this frame's if that is
  // placed_epoch_ now).
  struct PlacedAt {
    uint64_t epoch;
    uint32_t index;
    bool other;
  };
  static RtMeshKey Geometry(const RtMeshKey& k) {
    RtMeshKey g = k;
    g.attr_version = g.attr_buffer = 0;
    g.attr_stream_offset = g.attr_stride = g.uv_offset = g.alpha_offset = 0;
    g.uv_type = g.alpha_type = g.uv_lanes = g.alpha_lane = 0;
    g.cut = 0;
    return g;
  }
  // With its hash, PlacedHashOf(RtMeshKeyHash(key), world), made once with
  // it: the maps of Placed take it as it is.
  struct Placed {
    RtMeshKey key;
    float world[12];
    size_t hash;
    bool operator==(const Placed& o) const {
      return hash == o.hash && key == o.key && std::memcmp(world, o.world, sizeof world) == 0;
    }
  };
  static_assert(sizeof(Placed) == sizeof(RtMeshKey) + 48 + sizeof(size_t), "no padding");
  static size_t PlacedHashOf(size_t key_hash, const float world[12]) {
    uint64_t h = key_hash;
    uint32_t w[12];
    std::memcpy(w, world, sizeof w);
    for (uint32_t x : w) {
      h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
      h *= 0xBF58476D1CE4E5B9ull;
    }
    return size_t(h ^ (h >> 31));
  }
  struct PlacedHash {
    size_t operator()(const Placed& p) const { return p.hash; }
  };
  // Memory for node-based containers emptied and refilled every frame
  // (Retain's sets): a block freed waits for the next of its size instead
  // of going back to the heap. Single nodes only (bucket arrays pass).
  class BlockPool {
   public:
    BlockPool() = default;
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() {
      for (List& l : lists_)
        while (l.head) {
          Block* b = l.head;
          l.head = b->next;
          ::operator delete(b);
        }
    }
    void* Get(size_t bytes) {
      for (List& l : lists_)
        if (l.bytes == bytes && l.head) {
          Block* b = l.head;
          l.head = b->next;
          return b;
        }
      bytes_ += bytes;
      return ::operator new(bytes);
    }
    // The blocks taken from the heap, in use and waiting (all kept until
    // the pool goes).
    uint64_t Bytes() const { return bytes_; }
    void Put(void* p, size_t bytes) {
      List* l = nullptr;
      for (List& x : lists_)
        if (x.bytes == bytes) l = &x;
      if (!l) l = &lists_.emplace_back(List{bytes, nullptr});
      Block* b = static_cast<Block*>(p);
      b->next = l->head;
      l->head = b;
    }

   private:
    struct Block {
      Block* next;
    };
    struct List {
      size_t bytes;
      Block* head;
    };
    std::vector<List> lists_;
    uint64_t bytes_ = 0;
  };
  template <typename T>
  struct PoolAlloc {
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    BlockPool* pool;
    explicit PoolAlloc(BlockPool* p) : pool(p) {}
    template <typename U>
    PoolAlloc(const PoolAlloc<U>& o) : pool(o.pool) {}
    T* allocate(size_t n) {
      static_assert(alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__ && sizeof(T) >= sizeof(void*));
      return static_cast<T*>(n == 1 ? pool->Get(sizeof(T)) : ::operator new(n * sizeof(T)));
    }
    void deallocate(T* p, size_t n) {
      if (n == 1)
        pool->Put(p, sizeof(T));
      else
        ::operator delete(p);
    }
    template <typename U>
    bool operator==(const PoolAlloc<U>& o) const {
      return pool == o.pool;
    }
  };

  // Retained instances (rt_scene.h), by geometry and world matrix: the
  // instance as its last draw listed it (camera kAnyCamera, kRtInfoRetained;
  // its material in `mat`), its world-space bounds, the last presented frame
  // that drew it; `drawn` a presented frame whose draw pending_[index] is it,
  // `carried` the one whose snapshots carry_ lists it in.
  struct Retained {
    Pending p;
    PendingMat mat;
    bool has_mat = false;
    float lo[3] = {}, hi[3] = {};
    uint64_t last_frame = 0, drawn = 0, carried = 0;
    uint32_t index = 0;
    uint32_t in_view = 0;  // presented frames in a row the camera saw all of it undrawn
    // The least squared distance from the main camera to its bounds' centre
    // at a draw of it (negative: no camera known at one).
    double seen2 = -1.0;
    uint64_t id = 0;  // its number (RetainedSlots, DropRetained)
  };
  using RetainedNode = std::pair<const Placed, Retained>;

  // A texture a material names, held (null for nil).
  HeldTex* Hold(__unsafe_unretained id<MTLTexture> t) {
    if (!t) return nullptr;
    std::unique_ptr<HeldTex>& slot = held_[(__bridge const void*)t];
    if (!slot) {
      slot = std::make_unique<HeldTex>();
      slot->tex = t;
      slot->bytes = t.allocatedSize;
      held_bytes_ += slot->bytes;
    }
    slot->last_frame = frame_;
    return slot.get();
  }

  // An instance's material, its textures held: alpha-cut with `cut`, its
  // albedo `albedo` (with hit data); its number in pending_mat_, or
  // kNoMaterial with neither.
  uint32_t AddMaterial(const RtCut* cut, const RtAlbedo* albedo) {
    PendingMat p{RtMaterial{}, nullptr, nullptr};
    if (!MakeMaterial(cut, albedo, &p.m, &p.tex, &p.albedo)) return kNoMaterial;
    pending_mat_.push_back(p);
    return uint32_t(pending_mat_.size() - 1);
  }

  // The material AddMaterial makes, in `m` (zero with neither), its
  // textures held (`tex` the alpha's, `alb` the albedo's; null: none).
  // False with neither.
  bool MakeMaterial(const RtCut* cut, const RtAlbedo* albedo, RtMaterial* m, HeldTex** tex, HeldTex** alb) {
    *m = RtMaterial{};
    *tex = *alb = nullptr;
    const bool with_albedo = opt_.hit_data && albedo && albedo->texture;
    if (!cut && !with_albedo) return false;
    PendingMat p{RtMaterial{}, nullptr, nullptr};
    if (cut) {
      p.m = cut->material;
      p.tex = Hold(cut->texture);
      p.m.flags = (p.m.flags | kRtMatCut) & ~uint32_t(kRtMatTexture);
      if (p.tex) p.m.flags |= kRtMatTexture;
    }
    p.m.alpha_texture = MTLResourceID{};
    p.m.flags &= ~uint32_t(kRtMatAlbedo | kRtMatHitPrim);
    p.m.albedo_texture = MTLResourceID{};
    p.m.albedo_max_level = 0;
    p.m.albedo_flags = 0;
    p.m.albedo_border = 0;
    if (with_albedo) {
      p.albedo = Hold(albedo->texture);
      p.m.flags |= kRtMatAlbedo;
      p.m.albedo_max_level = albedo->max_level;
      p.m.albedo_flags = (albedo->address_u & 7u) | (albedo->address_v & 7u) << 3;
      p.m.albedo_border = albedo->border;
    }
    *m = p.m;
    *tex = p.tex;
    *alb = p.albedo;
    return true;
  }

  FrameData& NextFrame() {
    FrameSlot& s = slots_[serial_ % kSlots];
    if (s.frames_used == s.frames.size()) s.frames.push_back(std::make_unique<FrameData>());
    FrameData& f = *s.frames[s.frames_used++];
    f.Reset();
    return f;
  }

  int BestCamera() const {
    int best = -1;
    for (size_t c = 0; c < cams_.size(); c++)
      if (best < 0 || cams_[c].count > cams_[size_t(best)].count) best = int(c);
    return best;
  }

  // Instance `p`'s summary (RtInstanceSummary) onto describe_.
  void Describe(const Pending& p, bool carried, bool retained) {
    RtInstanceSummary s;
    s.view = p.camera == kCapturedCamera ? 2 : p.camera == kAnyCamera ? 1 : 0;
    s.carried = carried;
    s.retained = retained;
    s.draw = p.info.draw;
    s.mesh = p.info.mesh;
    s.triangles = p.info.triangles;
    if (!p.blas->captured) {
      WorldBounds(p, s.lo, s.hi);
      s.box = true;
    }
    describe_->push_back(s);
  }

  // A static instance's world-space bounds: its BLAS's corners through the
  // world matrix (columns of the instance's transform: its rows 0-3).
  static void WorldBounds(const Pending& p, float lo[3], float hi[3]) {
    const RtBlas* b = p.blas;
    const auto& m = p.desc.transformationMatrix;
    for (int k = 0; k < 3; k++) lo[k] = INFINITY, hi[k] = -INFINITY;
    for (int c = 0; c < 8; c++) {
      const float x = (c & 1) ? b->hi[0] : b->lo[0], y = (c & 2) ? b->hi[1] : b->lo[1], z = (c & 4) ? b->hi[2] : b->lo[2];
      for (int k = 0; k < 3; k++) {
        const float w = x * m.columns[0].elements[k] + y * m.columns[1].elements[k] + z * m.columns[2].elements[k] +
                        m.columns[3].elements[k];
        lo[k] = std::min(lo[k], w);
        hi[k] = std::max(hi[k], w);
      }
    }
  }

  // A TLAS in `f` of the pending instances drawn with camera `cam` (none if
  // negative) and, with `carry`, the last presented frame's carried ones;
  // `vp` the camera's world to clip, `reversed` its depth's direction.
  // `n` and `slots`: TlasCount's, called just before (its
  // listed_retained_).
  void MakeTlas(FrameData& f, int cam, bool carry, const double* vp, bool reversed, size_t n, size_t slots) {
    using Desc = MTLIndirectAccelerationStructureInstanceDescriptor;
    CommitResidency();  // (its BLASes resident before the work that traces it runs)
    tlas_captured_ = tlas_cut_ = tlas_other_ = tlas_albedo_ = tlas_retained_ = 0;
    if (!n) return;
    double inv[16];
    if (!Invert4(vp, inv)) {
      stats_.skipped[size_t(RtSkip::BadCamera)] += n;
      return;
    }
    const RtUpload ib = alloc_ ? alloc_(alloc_ctx_, n * sizeof(Desc), 256) : RtUpload{nil, 0, nullptr};
    const RtUpload info = alloc_ ? alloc_(alloc_ctx_, slots * sizeof(RtInstanceInfo), 256) : RtUpload{nil, 0, nullptr};
    const RtUpload mat = alloc_ ? alloc_(alloc_ctx_, slots * sizeof(RtMaterial), 256) : RtUpload{nil, 0, nullptr};
    if (!ib.cpu || !info.cpu || !mat.cpu) {
      stats_.skipped[size_t(RtSkip::NoMemory)] += n;
      return;
    }
    auto* out = reinterpret_cast<Desc*>(ib.cpu);
    auto* out_info = reinterpret_cast<RtInstanceInfo*>(info.cpu);
    auto* out_mat = reinterpret_cast<RtMaterial*>(mat.cpu);
    const uint64_t seq = ++tlas_seq_;
    uint32_t i = 0, slot = 0;  // instances, their first slots
    // One presented frame in kStaticSampleEvery, its first TLAS: the
    // static instances' world matrices and options, to count the distinct
    // ones (RtStats::static_distinct).
    const bool sample = frame_ % kStaticSampleEvery == 0 && static_sample_.frame != frame_;
    if (sample) static_hashes_.clear();
    auto hold = [&](HeldTex* h) {
      h->last_serial = serial_;
      if (h->mark != seq) {
        h->mark = seq;
        f.held.push_back(h->tex);
      }
      return h->tex.gpuResourceID;
    };
    // (kind: 0 this frame's, 1 carried, 2 retained.)
    auto put = [&](const Pending& p, const PendingMat* pm, int kind) {
      if (describe_) Describe(p, kind == 1, kind == 2);
      Desc d = p.desc;
      d.userID = slot;
      RtInstanceInfo info = p.info;
      RtBlas* b = p.blas;
      if (b->captured) {
        d.options = MTLAccelerationStructureInstanceOptions(b->opts);
        info.flags = b->opts | kRtInfoCaptured;
        tlas_captured_++;
      }
      if (b->parts) {
        // An actor: its BLAS, mask and triangles known once it was made;
        // a slot a part.
        d.accelerationStructureID = b->id;
        d.mask = b->mask;
        info.triangles = b->triangles;
        for (uint32_t k = 0; k < b->parts; k++) {
          RtMaterial m = b->part_mat[k];
          if (auto* t = static_cast<HeldTex*>(b->part_tex[k])) m.alpha_texture = hold(t);
          if (auto* t = static_cast<HeldTex*>(b->part_albedo[k])) {
            m.albedo_texture = hold(t);
            tlas_albedo_++;
          }
          if (m.flags & kRtMatCut) tlas_cut_++;
          out_info[slot + k] = info;
          std::memcpy(&out_mat[slot + k], &m, sizeof m);
        }
        out[i] = d;
        i++;
        slot += b->parts;
        b->last_serial = serial_;
        if (b->mark != seq) {
          b->mark = seq;
          f.used.push_back(b->as);
        }
        return;
      }
      RtMaterial m{};
      if (pm) {
        const PendingMat& c = *pm;
        m = c.m;
        if (c.tex) m.alpha_texture = hold(c.tex);
        if (c.albedo) {
          m.albedo_texture = hold(c.albedo);
          tlas_albedo_++;
        }
        if (m.flags & kRtMatCut) tlas_cut_++;
      }
      if (b->hit) m.flags |= kRtMatHitPrim;
      if (kind == 2)
        tlas_retained_++;
      else if (p.camera == kAnyCamera)
        tlas_other_++;
      if (sample && !b->captured) {
        // Its world matrix and options as one number (64 bits: two that
        // differ count as one once in some 10^12 samples of 8000).
        float w[12];
        static_assert(sizeof d.transformationMatrix == sizeof w);
        std::memcpy(w, &d.transformationMatrix, sizeof w);
        static_hashes_.push_back(PlacedHashOf(size_t(d.options), w));
      }
      out[i] = d;
      out_info[slot] = info;
      std::memcpy(&out_mat[slot], &m, sizeof m);
      i++;
      slot++;
      // (Its other pieces: the same instance but for the BLAS, a slot each.)
      for (const RtBlas::Piece& c : b->pieces) {
        if (describe_) Describe(p, kind == 1, kind == 2);
        Desc dc = d;
        dc.userID = slot;
        dc.accelerationStructureID = c.id;
        out[i] = dc;
        out_info[slot] = info;
        std::memcpy(&out_mat[slot], &m, sizeof m);
        i++;
        slot++;
      }
      b->last_serial = serial_;
      if (b->mark != seq) {
        b->mark = seq;
        f.used.push_back(b->as);
        for (const RtBlas::Piece& c : b->pieces) f.used.push_back(c.as);
      }
    };
    auto mat_of = [](const Pending& p, const std::vector<PendingMat>& mats) {
      return p.material != kNoMaterial ? &mats[p.material] : nullptr;
    };
    for (const Pending& p : pending_)
      if (Listed(p, cam)) put(p, mat_of(p, pending_mat_), 0);
    if (carry)
      for (const Pending& p : carry_) put(p, mat_of(p, carry_mat_), 1);
    f.retained_slots.clear();
    for (const Retained* r : listed_retained_) {
      const uint32_t first = slot;
      put(r->p, r->has_mat ? &r->mat : nullptr, 2);
      for (uint32_t k = first; k < slot; k++) f.retained_slots.emplace_back(k, r->id);
    }
    if (sample) {
      // (EndFrame counts it once the frame is presented.)
      const uint64_t listed = static_hashes_.size();
      std::sort(static_hashes_.begin(), static_hashes_.end());
      const uint64_t distinct =
          uint64_t(std::unique(static_hashes_.begin(), static_hashes_.end()) - static_hashes_.begin());
      static_sample_ = {frame_, listed, distinct};
    }
    describe_ = nullptr;
    for (int k = 0; k < 16; k++) f.inv_view_proj[k] = float(inv[k]);
    std::memcpy(f.inv_view_proj_exact, inv, sizeof inv);
    f.camera_id = cam >= 0 ? cams_[size_t(cam)].id : 0;
    f.depth_reversed = reversed;
    @autoreleasepool {
      MTLInstanceAccelerationStructureDescriptor* desc = [MTLInstanceAccelerationStructureDescriptor new];
      desc.instanceDescriptorBuffer = ib.buf;
      desc.instanceDescriptorBufferOffset = ib.off;
      desc.instanceDescriptorStride = sizeof(Desc);
      desc.instanceCount = n;
      desc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeIndirect;
      desc.usage = tlas_usage_;  // a fast build: rebuilt every frame (TlasUsageWanted)
      const MTLAccelerationStructureSizes sz = [dev_ accelerationStructureSizesWithDescriptor:desc];
      const NSUInteger need = sz.accelerationStructureSize;
      // Grown by half again, made anew when far above need.
      if (!f.store || f.store.size < need || f.store.size > 4 * need + 65536)
        f.store = [dev_ newAccelerationStructureWithSize:need + need / 2];
      const auto [sb, so] = Scratch(slots_[serial_ % kSlots], sz.buildScratchBufferSize, true);
      if (!f.store || !sb) {
        stats_.skipped[size_t(RtSkip::NoMemory)] += n;
        f.used.clear();
        f.held.clear();
        tlas_captured_ = tlas_cut_ = tlas_other_ = tlas_albedo_ = tlas_retained_ = 0;
        return;
      }
      f.desc = desc;
      f.scratch = sb;
      f.scratch_off = so;
      f.tlas = f.store;
      f.instances = uint32_t(n);
      f.blas = f.used.data();
      f.blas_count = uint32_t(f.used.size());
      f.blas_resident = rs_ != nil;
      f.info = info.buf;
      f.info_offset = info.off;
      f.materials = mat.buf;
      f.materials_offset = mat.off;
      f.textures = f.held.data();
      f.texture_count = uint32_t(f.held.size());
    }
  }

  // Whether pending instance `p` goes into a TLAS of camera `cam` (none if
  // negative): drawn with it, another view's (any camera), or a captured
  // draw's converted (in world space, any camera).
  bool Listed(const Pending& p, int cam) const {
    if (p.camera == kAnyCamera) return true;
    return p.camera == kCapturedCamera ? p.blas->ready : cam >= 0 && p.camera == uint32_t(cam);
  }

  // The instances a TLAS of camera `cam` (MakeTlas) lists, and their slots
  // in the materials and instance info (`slots`): one an instance, one a
  // part for an actor's (AddCaptured). The retained ones listed, in
  // retained_'s order, into listed_retained_ (for MakeTlas next).
  size_t TlasCount(int cam, bool carry, size_t* slots) {
    size_t n = 0, s = 0;
    // (A mesh in pieces: an instance and a slot a piece.)
    for (const Pending& p : pending_)
      if (Listed(p, cam)) {
        n += 1 + p.blas->pieces.size();
        s += p.blas->parts ? p.blas->parts : 1 + p.blas->pieces.size();
      }
    if (carry)
      for (const Pending& p : carry_) {
        n += 1 + p.blas->pieces.size();
        s += p.blas->parts ? p.blas->parts : 1 + p.blas->pieces.size();
      }
    listed_retained_.clear();
    for (const auto& [k, r] : retained_)
      if (RetainedListed(r, cam, carry)) {
        listed_retained_.push_back(&r);
        n += 1 + r.p.blas->pieces.size();
        s += 1 + r.p.blas->pieces.size();
      }
    *slots = s;
    return n;
  }

  // This presented frame's places (placed_) forgotten: a new epoch. Past
  // kPlacedSlack times as many entries as the epoch that ended took (at
  // least kPlacedFloor), only its own stay (for the next frame's draws of
  // their places): the map holds at most that many and this epoch's, and
  // this walks at most that many and the epoch's and lets go at most
  // kPlacedSlack times as many as the epoch before took (or kPlacedFloor),
  // in no frame more. (Nothing at all where the frames draw the same
  // places: only the frame's draws look at it.)
  void NextPlacedEpoch() {
    const uint64_t took = std::max<uint64_t>(placed_now_, kPlacedFloor);
    placed_now_ = 0;
    placed_epoch_++;
    if (placed_.size() <= kPlacedSlack * took) return;
    stats_.places_dropped +=
        std::erase_if(placed_, [&](const auto& kv) { return kv.second.epoch + 1 < placed_epoch_; });
  }

  // ---- retained instances (rt_scene.h)

  // A static instance's mesh geometry and world matrix (Placed), as Add
  // keyed its draw (with the hash it made).
  static Placed PlacedOf(const Pending& p) {
    Placed k;
    k.key = Geometry(p.blas->key);
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 3; c++) k.world[r * 3 + c] = p.desc.transformationMatrix.columns[r].elements[c];
    k.hash = p.placed;
    return k;
  }

  // Whether retained instance `r` goes into a TLAS of camera `cam` (with
  // `carry`: a snapshot listing carry_): not when this frame's draw of it is
  // listed, nor when carry_ holds it.
  bool RetainedListed(const Retained& r, int cam, bool carry) const {
    if (r.p.blas->retired || (carry && r.carried == frame_)) return false;
    return !(r.drawn == frame_ && r.index < pending_.size() && Listed(pending_[r.index], cam));
  }

  // What a retained instance holds: its BLAS, its material's textures.
  void Ref(Retained& r, int by) {
    r.p.blas->retained += uint32_t(by);
    if (!r.has_mat) return;
    if (r.mat.tex) r.mat.tex->retained += uint32_t(by);
    if (r.mat.albedo) r.mat.albedo->retained += uint32_t(by);
  }

  // (Before the entry is erased.)
  void Drop(const Placed& k, Retained& r, RtRetainDrop why) {
    Ref(r, -1);
    stats_.retained_dropped[size_t(why)]++;
    Unindex(k);
  }

  // The world matrix alone of `k` (its key zero): retained_world_'s key.
  static Placed WorldOf(const Placed& k) {
    static const size_t zero = RtMeshKeyHash()(RtMeshKey{});  // (the same key every time)
    Placed w;
    w.key = RtMeshKey{};
    std::memcpy(w.world, k.world, sizeof w.world);
    w.hash = PlacedHashOf(zero, w.world);
    return w;
  }

  // A retained instance in retained_geo_ and retained_world_ (`n` stays
  // where it is until erased).
  void Index(RetainedNode* n) {
    retained_geo_[n->first.key].push_back(n);
    retained_world_[WorldOf(n->first)].push_back(n);
  }
  void Unindex(const Placed& k) {
    auto out = [&](auto& map, const auto& key) {
      auto it = map.find(key);
      if (it == map.end()) return;
      std::erase_if(it->second, [&](const RetainedNode* n) { return n->first == k; });
      if (it->second.empty()) map.erase(it);
    };
    out(retained_geo_, k.key);
    out(retained_world_, WorldOf(k));
  }

  // Whether world-space boxes a and b are much the same: on every axis
  // they overlap by at least half the larger extent.
  static bool Similar(const float alo[3], const float ahi[3], const float blo[3], const float bhi[3]) {
    for (int k = 0; k < 3; k++) {
      const double ov = double(std::min(ahi[k], bhi[k])) - std::max(alo[k], blo[k]);
      const double e = std::max(double(ahi[k]) - alo[k], double(bhi[k]) - blo[k]);
      if (!(ov >= 0.5 * e - 1e-3)) return false;
    }
    return true;
  }

  // Retained instances this presented frame's static draws replace, not
  // drawn this frame themselves: of the same geometry at another world
  // matrix with much the same bounds (a vehicle or prop that moved off),
  // checked once per draw (from replaced_from_); with `lod` (the end of
  // the frame, when what the frame drew is known), also of other geometry
  // at the same world matrix with much the same bounds (another level of
  // detail drawn in its place), against every draw.
  void DropReplaced(bool lod) {
    if (retained_.empty()) {
      replaced_from_ = pending_.size();
      return;
    }
    victims_.clear();
    for (size_t i = lod ? 0 : replaced_from_; i < pending_.size(); i++) {
      const Pending& p = pending_[i];
      if (p.camera == kCapturedCamera) continue;
      const Placed at = PlacedOf(p);
      const auto g = i >= replaced_from_ ? retained_geo_.find(KeyRef{&at.key, p.blas->geo_hash}) : retained_geo_.end();
      const auto w = lod ? retained_world_.find(WorldOf(at)) : retained_world_.end();
      if (g == retained_geo_.end() && w == retained_world_.end()) continue;
      // (The draw's bounds made only for a candidate not drawn this frame:
      // its own retained instance, drawn, is the usual one.)
      float lo[3], hi[3];
      bool bounds = false;
      auto consider = [&](RetainedNode* n) {
        if (n->second.drawn == frame_ || n->first == at) return;
        if (!bounds) WorldBounds(p, lo, hi), bounds = true;
        if (!Similar(lo, hi, n->second.lo, n->second.hi)) return;
        if (std::find(victims_.begin(), victims_.end(), n) == victims_.end()) victims_.push_back(n);
      };
      if (g != retained_geo_.end())
        for (RetainedNode* n : g->second) consider(n);
      if (w != retained_world_.end())
        for (RetainedNode* n : w->second) consider(n);
    }
    replaced_from_ = pending_.size();
    for (RetainedNode* n : victims_) {
      const Placed k = n->first;
      Drop(k, n->second, RtRetainDrop::Replaced);
      retained_.erase(k);
    }
  }

  // Retained instances whose BLAS left the cache (Retire, DropUnbuilt) go.
  void PurgeRetained() {
    if (!retained_gone_) return;
    retained_gone_ = false;
    std::erase_if(retained_, [&](auto& kv) {
      if (!kv.second.p.blas->retired) return false;
      Drop(kv.first, kv.second, RtRetainDrop::Gone);
      return true;
    });
  }

  // The end of a presented frame, camera `best` its TLAS's: its static
  // instances that TLAS lists retained (or renewed: material, last draw)
  // once static, drawn at the same world matrix the presented frame before
  // too (what moves leaves no copies behind); those not drawn let go where
  // the camera has seen all of them for kInViewFrames frames in a row (not
  // at once: a game's occlusion culling leaves out what is hidden behind
  // something nearer), idle for retain_frames, and past max_retained the
  // farthest from the camera.
  void Retain(int best) {
    if (!opt_.retain_frames) return;
    const double* vp = best >= 0 ? cams_[size_t(best)].vp : nullptr;
    double eye[3];
    const bool have_eye = vp && CameraOf(vp, eye);
    static_now_.clear();
    for (uint32_t i = 0; i < pending_.size(); i++) {
      const Pending& p = pending_[i];
      // (Not one whose BLAS left the cache since its draw: new contents.)
      if (p.camera == kCapturedCamera || p.blas->retired || !Listed(p, best)) continue;
      const Placed k = PlacedOf(p);
      static_now_.insert(k);
      auto it = retained_.find(k);
      const bool fresh = it == retained_.end();
      if (fresh) {
        if (!static_last_.count(k)) continue;
        it = retained_.try_emplace(k).first;
        it->second.id = ++retained_ids_;
        Index(&*it);
      }
      Retained& r = it->second;
      if (!fresh) Ref(r, -1);
      r.p = p;
      r.p.camera = kAnyCamera;
      r.p.material = kNoMaterial;
      r.p.info.flags |= kRtInfoRetained;
      r.has_mat = p.material != kNoMaterial;
      if (r.has_mat) r.mat = pending_mat_[p.material];
      if (fresh) WorldBounds(p, r.lo, r.hi);
      if (have_eye) {
        const double d2 = CentreDistance2(r.lo, r.hi, eye);
        r.seen2 = r.seen2 < 0.0 ? d2 : std::min(r.seen2, d2);
      }
      r.last_frame = r.drawn = frame_;
      r.index = i;
      r.in_view = 0;
      Ref(r, 1);
    }
    std::swap(static_last_, static_now_);
    // (To measure, RECOMP_RT_RETAIN_DUMP=1: every 120 frames the retained
    // instances not drawn this frame within 30 m of the camera.)
    static const bool dump = [] {
      const char* e = std::getenv("RECOMP_RT_RETAIN_DUMP");
      return e && *e == '1';
    }();
    if (dump && have_eye && frame_ % 120 == 0) {
      std::fprintf(stderr, "rt retain dump: frame %llu camera (%.2f %.2f %.2f), %zu retained\n",
                   (unsigned long long)frame_, eye[0], eye[1], eye[2], retained_.size());
      for (const auto& [k, r] : retained_) {
        if (r.last_frame == frame_ || !Near(r.lo, r.hi, eye, 30.0)) continue;
        std::fprintf(stderr,
                     "rt retain dump:   mesh %u (%u triangles) draw %u box (%.2f %.2f %.2f)-(%.2f %.2f %.2f) undrawn "
                     "%llu frames, nearest drawn from %.1f m, centre now %.1f m\n",
                     r.p.info.mesh, r.p.info.triangles, r.p.info.draw, r.lo[0], r.lo[1], r.lo[2], r.hi[0], r.hi[1],
                     r.hi[2], (unsigned long long)(frame_ - r.last_frame), r.seen2 >= 0 ? std::sqrt(r.seen2) : -1.0,
                     std::sqrt(CentreDistance2(r.lo, r.hi, eye)));
      }
    }
    // (Read each frame: one getenv; the tests set it.)
    const char* ive = std::getenv("RECOMP_RT_IN_VIEW_FRAMES");
    const uint32_t in_view_frames = ive && *ive ? uint32_t(std::strtoul(ive, nullptr, 10)) : 0u;
    std::erase_if(retained_, [&](auto& kv) {
      Retained& r = kv.second;
      if (r.last_frame == frame_) return false;
      RtRetainDrop why;
      // (One the camera stands within the bounds of is drawn as long as the
      // game uses it: no view or occlusion test leaves out what holds the
      // camera. Not drawn, it was replaced: a whole station's far model by
      // the near ones inside it, whose centre stays far off however close
      // the camera is. Kept, its coarse floors and walls lie a centimetre or
      // two off the near ones' and shadow them. What lies on what the camera
      // sees from outside it the device finds by rays: DropRetained.)
      if (have_eye && Near(r.lo, r.hi, eye, 0.0)) {
        why = RtRetainDrop::Closer;
      } else if (have_eye && r.seen2 >= 0.0 &&
                 CentreDistance2(r.lo, r.hi, eye) < kRetainCloser * kRetainCloser * r.seen2) {
        why = RtRetainDrop::Closer;
      } else if (frame_ - r.last_frame > opt_.retain_frames && !(have_eye && Near(r.lo, r.hi, eye, kRetainNear))) {
        why = RtRetainDrop::Idle;
      } else if (in_view_frames && vp && InView(r.lo, r.hi, vp)) {
        if (++r.in_view <= in_view_frames) return false;
        why = RtRetainDrop::InView;
      } else {
        r.in_view = 0;
        return false;
      }
      Drop(kv.first, r, why);
      return true;
    });
    if (retained_.size() <= opt_.max_retained) return;
    using It = decltype(retained_)::iterator;
    std::vector<It> all;
    all.reserve(retained_.size());
    for (It it = retained_.begin(); it != retained_.end(); ++it) all.push_back(it);
    const size_t over = all.size() - opt_.max_retained;
    // The farthest from the camera go first: driving fast, the cap fills
    // with what is far behind, and the longest undrawn (the rule before)
    // were often buildings beside the road the game had stopped drawing,
    // whose shadows then left the road ahead. Without a camera, the
    // longest undrawn.
    // (RECOMP_RT_RETAIN_CAP_OLDEST=1: the rule before, to compare.)
    static const bool oldest = [] {
      const char* e = std::getenv("RECOMP_RT_RETAIN_CAP_OLDEST");
      return e && *e == '1';
    }();
    if (have_eye && !oldest)
      std::nth_element(all.begin(), all.begin() + ptrdiff_t(over), all.end(), [&](const It& a, const It& b) {
        return CentreDistance2(a->second.lo, a->second.hi, eye) > CentreDistance2(b->second.lo, b->second.hi, eye);
      });
    else
      std::nth_element(all.begin(), all.begin() + ptrdiff_t(over), all.end(),
                       [](const It& a, const It& b) { return a->second.last_frame < b->second.last_frame; });
    for (size_t i = 0; i < over; i++) {
      const Placed k = all[i]->first;
      if (have_eye && Near(all[i]->second.lo, all[i]->second.hi, eye, 100.0)) stats_.retained_cap_near++;
      Drop(k, all[i]->second, RtRetainDrop::Cap);
      retained_.erase(k);
    }
  }

  // The camera's place for `vp` (world to clip, rows: a point's clip
  // coordinates are x vp[0..3] + y vp[4..7] + z vp[8..11] + vp[12..15]):
  // where clip x, y and w are all 0. False where that cannot be solved (an
  // orthographic view).
  static bool CameraOf(const double* vp, double out[3]) {
    const int cols[3] = {0, 1, 3};
    double m[3][4];
    for (int r = 0; r < 3; r++) {
      const int k = cols[r];
      m[r][0] = vp[k];
      m[r][1] = vp[4 + k];
      m[r][2] = vp[8 + k];
      m[r][3] = -vp[12 + k];
    }
    const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                       m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (!(std::fabs(det) > 1e-12)) return false;
    for (int c = 0; c < 3; c++) {
      double a[3][3];
      for (int r = 0; r < 3; r++)
        for (int q = 0; q < 3; q++) a[r][q] = q == c ? m[r][3] : m[r][q];
      out[c] = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])) /
               det;
    }
    return std::isfinite(out[0]) && std::isfinite(out[1]) && std::isfinite(out[2]);
  }

  // Whether the world-space box lo-hi comes within `d` of point `p`.
  static double CentreDistance2(const float lo[3], const float hi[3], const double p[3]) {
    double s = 0;
    for (int k = 0; k < 3; k++) {
      const double v = 0.5 * (double(lo[k]) + double(hi[k])) - p[k];
      s += v * v;
    }
    return s;
  }

  static bool Near(const float lo[3], const float hi[3], const double p[3], double d) {
    double s = 0;
    for (int k = 0; k < 3; k++) {
      const double v = p[k] < lo[k] ? lo[k] - p[k] : (p[k] > hi[k] ? p[k] - hi[k] : 0.0);
      s += v * v;
    }
    return s <= d * d;
  }

  // Whether camera `vp` (world to clip, rows) sees all of the world-space
  // box lo-hi: every corner in front of it and inside its view, kViewMargin
  // of the view's half-width in from each side (a conservative test: one
  // partly outside, or behind the camera, is not).
  static bool InView(const float lo[3], const float hi[3], const double* vp) {
    for (int c = 0; c < 8; c++) {
      const double x = (c & 1) ? hi[0] : lo[0], y = (c & 2) ? hi[1] : lo[1], z = (c & 4) ? hi[2] : lo[2];
      double clip[4];
      for (int k = 0; k < 4; k++) clip[k] = x * vp[k] + y * vp[4 + k] + z * vp[8 + k] + vp[12 + k];
      const double w = clip[3], m = (1.0 - kViewMargin) * w;
      if (!(w > 0) || !(std::fabs(clip[0]) <= m) || !(std::fabs(clip[1]) <= m) || !(clip[2] >= 0) || !(clip[2] <= w))
        return false;
    }
    return true;
  }

  // The builds and captures not taken yet, captures converted with camera
  // `vp` (world to clip; null: none known), in a frame of this slot; null
  // if there are none.
  FrameData* Work(const double* vp, bool reversed) {
    const RtBuildBatch* b = TakeBuilds();
    const RtCaptureBatch* c = TakeCaptures(vp, reversed);
    if (!b && !c) return nullptr;
    FrameData& f = NextFrame();
    f.builds = b;
    f.captures = c;
    return &f;
  }

  // The captures added since the last call, converted to world space with
  // camera `vp` (world to clip, rows), their instances' winding from it;
  // null if there are none or no camera (those left out).
  const RtCaptureBatch* TakeCaptures(const double* vp, bool reversed) {
    FinishActor();
    RtCaptureBatch* b = cur_cap_;
    cur_cap_ = nullptr;
    if (!b) return nullptr;
    if (!vp || !ClipToWorld(vp, b->convert)) {
      for (const RtBuildBatch::Job& j : b->jobs) j.blas->ready = j.blas->keyed = false;
      stats_.capture_skipped[size_t(RtSkip::BadCamera)] += b->jobs.size();
      b->jobs.clear();
      b->ranges.clear();
      b->normals.clear();
      return nullptr;
    }
    // The vertices are in world space and the instances' matrices the
    // identity, so the camera alone mirrors the screen (Mirrors of the
    // view-projection; the draw's world matrix has its part in the
    // positions). Then as Add: the other winding where it mirrors, and again
    // where the game culls clockwise faces; no culling for D3DCULL_NONE.
    float m[4][4];
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 4; k++) m[r][k] = float(vp[r * 4 + k]);
    const bool mirrored = Mirrors(m, reversed);
    for (const RtBuildBatch::Job& j : b->jobs) {
      RtBlas* x = j.blas;
      // (An actor of opaque and alpha-cut parts: its geometries say which.)
      const bool mixed = x->parts && x->cut_parts && x->cut_parts < x->parts;
      uint32_t opts = mixed    ? MTLAccelerationStructureInstanceOptionNone
                      : x->cut ? MTLAccelerationStructureInstanceOptionNonOpaque
                               : MTLAccelerationStructureInstanceOptionOpaque;
      if (mirrored != (x->cull == 2)) opts |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
      if (x->cull == 1) opts |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
      x->opts = opts;
      x->ready = true;
      stats_.capture_builds++;
      if (j.refit) stats_.capture_refits++;
      stats_.capture_triangles += x->triangles;
    }
    return b;
  }

  // RtCaptureBatch::convert for camera `vp` (world to clip, rows): with
  // row vectors (p, 1) * vp = clip, the clip position's x, y and w are
  // p * M + t, M rows 0-2 and t row 3 of those columns, so p = (clip - t) *
  // M^-1. Not z: the depth a projection makes from w (a w - b) loses most
  // of its bits, kilometres out a metre's worth. Orthographic cameras (w
  // the same everywhere) use x, y and z, as precise there. M^-1 in double,
  // stored as float; subtracting t first keeps the products near the
  // camera's size. False if M is singular.
  static bool ClipToWorld(const double* vp, float* out) {
    double big = 0, w = 0;
    for (int r = 0; r < 3; r++) {
      w = std::max(w, std::fabs(vp[r * 4 + 3]));
      for (int k = 0; k < 3; k++) big = std::max(big, std::fabs(vp[r * 4 + k]));
    }
    const int cols[3] = {0, 1, w > 1e-7 * big ? 3 : 2};
    double m[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) m[i][j] = vp[i * 4 + cols[j]];
    const double det = Det3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    double scale = 0;
    for (const auto& row : m)
      for (double v : row) scale = std::max(scale, std::fabs(v));
    if (!(std::fabs(det) > 1e-12 * scale * scale * scale)) return false;
    double inv[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        // The cofactor of m[j][i], over the determinant.
        const int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
        inv[i][j] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det;
      }
    std::memset(out, 0, 16 * sizeof(float));
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) out[k * 4 + cols[j]] = float(inv[j][k]);
      out[12 + cols[j]] = float(vp[12 + cols[j]]);
    }
    return true;
  }

  // The capture's positions converted and (the debug view's, hit data)
  // its normals made in the batch open (made if none is).
  void CaptureWork(const FrameSlot::CaptureBuf& cb, uint32_t v0, uint32_t count, size_t vert_off, const RtUpload& u,
                   size_t prim_at, uint32_t tris, uint32_t stride, bool hit, bool cut, bool normals) {
    EnsureBatch();
    // Captures of a batch in one buffer follow each other: one range.
    std::vector<RtCaptureBatch::Range>& ranges = cur_cap_->ranges;
    if (!ranges.empty() && ranges.back().buf == cb.buf)
      ranges.back().count = v0 + count - ranges.back().first;
    else
      ranges.push_back({cb.buf, v0, count, cb.capacity});
    if (normals)
      cur_cap_->normals.push_back({cb.buf, vert_off, u.buf, u.off,
                                   u.off + prim_at + (!hit && cut ? sizeof(RtCutPrim) : 0), tris, stride,
                                   hit ? 1u : 0u});
  }

  // The batch captures go into (made if none is open).
  void EnsureBatch() {
    if (cur_cap_) return;
    FrameSlot& s = slots_[serial_ % kSlots];
    if (s.cap_batches_used == s.cap_batches.size()) s.cap_batches.push_back(std::make_unique<RtCaptureBatch>());
    cur_cap_ = s.cap_batches[s.cap_batches_used++].get();
  }

  // What AcquireCaptured gives: the BLAS's memory, whether this frame
  // refits it, its refits since its build, the scratch.
  struct CapturedAs {
    id<MTLAccelerationStructure> as = nil;
    uint64_t bytes = 0, build_scratch = 0, refit_scratch = 0;
    bool refit = false;
    uint32_t refits = 0;
    __unsafe_unretained id<MTLBuffer> scratch = nil;
    size_t scratch_off = 0;
  };

  // A captured BLAS for `desc` (`triangles` in all; a single draw's
  // primitive data `stride`; an actor's parts `shape`, null for a single
  // draw): the one an earlier frame made for the same key (refit while it
  // has been fewer than kCaptureRefits times), else pooled memory; the
  // scratch for its build or refit in frame slot `s`. False if there is
  // no memory (nothing taken).
  bool AcquireCaptured(MTLPrimitiveAccelerationStructureDescriptor* desc, const RtMeshKey* key, uint32_t nth,
                       uint32_t triangles, uint32_t stride, const std::vector<uint64_t>* shape, FrameSlot& s,
                       CapturedAs* out) {
    *out = CapturedAs{};
    // The BLAS an earlier frame made for the same key: of those let go
    // first, one that was its frame's draw of the key in the same place in
    // draw order as this one (the game draws its people in much the same
    // order every frame: likely this draw's own a few frames ago, in much
    // the same pose and place), else any.
    Warm warm;
    bool have_warm = false;
    if (key)
      if (auto it = warm_.find(*key); it != warm_.end()) {
        std::vector<Warm>& v = it->second;
        size_t pick = v.size();
        for (size_t i = 0; i < v.size(); i++)
          if (v[i].triangles == triangles && v[i].stride == stride &&
              (shape ? v[i].shape == *shape : v[i].shape.empty())) {
            if (pick == v.size()) pick = i;
            if (v[i].nth == nth) {
              pick = i;
              break;
            }
          }
        if (pick < v.size()) {
          warm = std::move(v[pick]);
          v.erase(v.begin() + ptrdiff_t(pick));
          have_warm = true;
        }
        if (v.empty()) warm_.erase(it);
      }
    if (have_warm) {
      out->as = warm.as;
      out->bytes = warm.bytes;
      out->build_scratch = warm.build_scratch;
      out->refit_scratch = warm.refit_scratch;
      out->refit = warm.refits < kCaptureRefits;
    } else {
      const MTLAccelerationStructureSizes sz = [dev_ accelerationStructureSizesWithDescriptor:desc];
      out->build_scratch = sz.buildScratchBufferSize;
      out->refit_scratch = sz.refitScratchBufferSize;
      out->as = PooledBlas(sz.accelerationStructureSize, &out->bytes);
    }
    std::pair<__unsafe_unretained id<MTLBuffer>, size_t> sc{nil, 0};
    if (out->as) sc = Scratch(s, out->refit ? out->refit_scratch : out->build_scratch);
    if (!sc.first) {
      if (have_warm)
        warm_[*key].push_back(std::move(warm));
      else if (out->as)
        as_pool_.push_back({out->as, out->as.size, out->bytes, frame_});
      *out = CapturedAs{};
      return false;
    }
    out->refits = out->refit ? warm.refits + 1 : 0;
    out->scratch = sc.first;
    out->scratch_off = sc.second;
    return true;
  }

  // A new actor (AddCaptured's `actor`) open for its parts, the one open
  // before finished: its BLAS record (no memory yet) and its instance.
  void StartActor(uint64_t actor, uint32_t cull, uint32_t draw) {
    FinishActor();
    if (cap_spare_.empty()) cap_spare_.push_back(std::make_unique<RtBlas>());
    cap_live_.push_back(std::move(cap_spare_.back()));
    cap_spare_.pop_back();
    RtBlas* b = cap_live_.back().get();
    b->captured = true;
    b->ready = false;
    b->cull = cull;
    b->actor = actor;
    b->hit = opt_.hit_data;
    b->last_serial = serial_;
    b->last_frame = frame_;
    b->part_geos = [NSMutableArray new];
    actor_ = b;
    actor_batch_ = cur_cap_;
    Pending& p = pending_.emplace_back();
    auto& d = p.desc;
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 3; k++) d.transformationMatrix.columns[r].elements[k] = r == k ? 1.0f : 0.0f;
    d.options = MTLAccelerationStructureInstanceOptionNone;  // (TakeCaptures gives them)
    d.mask = 0;  // (MakeTlas: the parts')
    d.intersectionFunctionTableOffset = 0;
    p.blas = b;
    p.info = {draw, 0, 0, kRtInfoCaptured};
    p.camera = kCapturedCamera;
    p.material = kNoMaterial;  // (the parts': RtBlas::part_mat)
    pending_captured_++;
  }

  // The actor open (StartActor) made: one BLAS of its parts, refit from an
  // earlier frame's of the same parts where there is one, its build or
  // refit in its batch's work. Without memory its instance stays out (its
  // parts counted).
  void FinishActor() {
    RtBlas* b = actor_;
    actor_ = nullptr;
    if (!b) return;
    RtCaptureBatch* batch = actor_batch_ ? actor_batch_ : cur_cap_;
    actor_batch_ = nullptr;
    // Its key: the parts' (their keys and shapes); a refit wants them all
    // the same and in the same order.
    RtMeshKeyHash hk;
    uint64_t h1 = 0xcbf29ce484222325ull, h2 = 0x84222325cbf29ce4ull;
    for (uint32_t i = 0; i < b->parts; i++) {
      const uint64_t k = hk(b->part_keys[i]);
      h1 = (h1 ^ k) * 0x100000001b3ull;
      h2 = (h2 ^ (k + b->part_shape[i] * 0x9E3779B97F4A7C15ull)) * 0x100000001b3ull;
    }
    RtMeshKey key;
    key.cut = 0xFFFFFFFFu;  // (no single draw's key has this)
    key.vertex_version = h1;
    key.index_version = h2;
    key.primitives = b->triangles;
    key.min_index = b->parts;
    @autoreleasepool {
      MTLPrimitiveAccelerationStructureDescriptor* desc = [MTLPrimitiveAccelerationStructureDescriptor new];
      desc.geometryDescriptors = b->part_geos;
      desc.usage = MTLAccelerationStructureUsagePreferFastBuild | MTLAccelerationStructureUsageRefit;
      const uint32_t nth = cap_nth_[key]++;
      CapturedAs a;
      FrameSlot& s = slots_[serial_ % kSlots];
      if (!batch || !AcquireCaptured(desc, &key, nth, b->triangles, 0, &b->part_shape, s, &a)) {
        stats_.capture_skipped[size_t(RtSkip::NoMemory)] += b->parts;
        b->part_geos = nil;
        return;
      }
      b->as = a.as;
      b->id = a.as.gpuResourceID;
      b->bytes = a.bytes;
      b->keyed = true;
      b->key = key;
      b->refits = a.refits;
      b->nth = nth;
      b->stride = 0;
      b->build_scratch = a.build_scratch;
      b->refit_scratch = a.refit_scratch;
      b->cut = b->cut_parts == b->parts;
      RtBuildBatch::Job job;
      job.desc = desc;
      job.as = a.as;
      job.scratch = a.scratch;
      job.scratch_off = a.scratch_off;
      job.refit = a.refit;
      job.blas = b;
      batch->jobs.push_back(std::move(job));
    }
    b->part_geos = nil;
    stats_.capture_draws += b->parts;
  }

  // Room for `count` captured vertices in frame slot `s`'s capture buffer:
  // the buffer and the first vertex's number in it (a multiple of
  // kCaptureAlign); a null buffer if none can be made.
  std::pair<FrameSlot::CaptureBuf, uint32_t> CaptureSpace(FrameSlot& s, uint32_t count) {
    const uint32_t n = (count + kCaptureAlign - 1) / kCaptureAlign * kCaptureAlign;
    s.cap_need += n;
    if (!s.cap.empty() && uint64_t(s.cap_used) + n <= s.cap.back().capacity) {
      const uint32_t v0 = s.cap_used;
      s.cap_used += n;
      return {s.cap.back(), v0};
    }
    // Twice the last one, or what the slot's last frame needed; at most the
    // frame's budget (with the alignment's gaps).
    constexpr uint32_t kMost = kCaptureVerticesPerFrame + kCapturesPerFrame * kCaptureAlign;
    uint32_t cap = std::max({kMinCaptureVertices, s.cap_prev_need, s.cap.empty() ? 0u : s.cap.back().capacity * 2});
    cap = std::max(std::min(cap, kMost), n);
    cap = (cap + kCaptureCapacityStep - 1) / kCaptureCapacityStep * kCaptureCapacityStep;
    id<MTLBuffer> buf = [dev_ newBufferWithLength:size_t(cap) * 28 options:MTLResourceStorageModePrivate];
    if (!buf) return {{nil, 0}, 0};
    s.cap.push_back({buf, cap});
    s.cap_used = n;
    return {s.cap.back(), 0};
  }

  uint64_t CaptureBufferBytes() const {
    uint64_t n = 0;
    for (const FrameSlot& s : slots_)
      for (const FrameSlot::CaptureBuf& c : s.cap) n += c.buf.allocatedSize;
    return n;
  }

  // A BLAS made (in the residency set, UseResidency) and one let go (out
  // of it): the set's changes take effect at its next commit
  // (CommitResidency: before a snapshot's or a frame end's TLAS).
  void Resident(id<MTLAccelerationStructure> as) {
    if (!rs_ || !as) return;
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
      [(id<MTLResidencySet>)rs_ addAllocation:as];
      rs_dirty_ = true;
    }
  }
  void NotResident(id<MTLAccelerationStructure> as) {
    if (!rs_ || !as) return;
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
      [(id<MTLResidencySet>)rs_ removeAllocation:as];
      rs_dirty_ = true;
    }
  }
  void CommitResidency() {
    if (!rs_ || !rs_dirty_) return;
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) [(id<MTLResidencySet>)rs_ commit];
    rs_dirty_ = false;
  }

  // A captured draw's BLAS of at least `size` bytes: pooled memory that
  // fits (not more than twice the need), else new, within half the BLAS
  // memory cap (the pool's oldest let go to stay in it); nil if not.
  // `bytes`: what it takes.
  id<MTLAccelerationStructure> PooledBlas(uint64_t size, uint64_t* bytes) {
    size_t best = as_pool_.size();
    for (size_t i = 0; i < as_pool_.size(); i++) {
      const PooledAs& a = as_pool_[i];
      if (a.size >= size && a.size <= 2 * size + 65536 && (best == as_pool_.size() || a.size < as_pool_[best].size))
        best = i;
    }
    if (best < as_pool_.size()) {
      id<MTLAccelerationStructure> as = as_pool_[best].as;
      *bytes = as_pool_[best].bytes;
      as_pool_[best] = std::move(as_pool_.back());
      as_pool_.pop_back();
      return as;
    }
    // Else one kept for refits that its key's draws have left alone for a
    // few frames (their draws take theirs at once), built anew here.
    for (auto it = warm_.begin(); it != warm_.end(); ++it)
      for (size_t i = 0; i < it->second.size(); i++) {
        const Warm& w = it->second[i];
        if (frame_ - w.since < kWarmIdleFrames || w.as.size < size || w.as.size > 2 * size + 65536) continue;
        id<MTLAccelerationStructure> as = w.as;
        *bytes = w.bytes;
        it->second.erase(it->second.begin() + ptrdiff_t(i));
        if (it->second.empty()) warm_.erase(it);
        return as;
      }
    const uint64_t want = size + size / 4;  // room for the same mesh a little larger
    const uint64_t cap = opt_.max_blas_bytes / 2;
    while (cap_blas_bytes_ + want > cap && !as_pool_.empty()) {
      auto oldest = std::min_element(as_pool_.begin(), as_pool_.end(),
                                     [](const PooledAs& a, const PooledAs& b) { return a.since < b.since; });
      cap_blas_bytes_ -= oldest->bytes;
      NotResident(oldest->as);
      *oldest = std::move(as_pool_.back());
      as_pool_.pop_back();
    }
    // Then those kept for refits, the longest unused first.
    while (cap_blas_bytes_ + want > cap && !warm_.empty()) {
      auto key = warm_.end();
      size_t at = 0;
      for (auto it = warm_.begin(); it != warm_.end(); ++it)
        for (size_t i = 0; i < it->second.size(); i++)
          if (key == warm_.end() || it->second[i].since < key->second[at].since) {
            key = it;
            at = i;
          }
      cap_blas_bytes_ -= key->second[at].bytes;
      NotResident(key->second[at].as);
      key->second.erase(key->second.begin() + ptrdiff_t(at));
      if (key->second.empty()) warm_.erase(key);
    }
    if (cap_blas_bytes_ + want > cap) return nil;
    id<MTLAccelerationStructure> as = [dev_ newAccelerationStructureWithSize:want];
    if (!as) return nil;
    Resident(as);
    *bytes = as.allocatedSize;
    cap_blas_bytes_ += *bytes;
    return as;
  }

  // BLAS builds and refits, in one encoder: Metal may run those of one
  // encoder at once, each with a scratch range of its own, the ranges
  // dealt over the slot's pool of buffers (kScratchLanes: two in one
  // buffer do not run at once).
  void EncodeJobs(id<MTLCommandBuffer> cmd, const std::vector<RtBuildBatch::Job>* a,
                  const std::vector<RtBuildBatch::Job>* b) const {
    if ((!a || a->empty()) && (!b || b->empty())) return;
    id<MTLAccelerationStructureCommandEncoder> e = TimedAccelerationStructure(cmd, "rt BLAS builds");
    for (const auto* jobs : {a, b})
      if (jobs)
        for (const RtBuildBatch::Job& j : *jobs) {
          if (j.refit) {
            // In place: the new positions, and the primitive data (the
            // normals made with them) where there is any.
            const bool prims = j.blas && j.blas->prim_data;
            if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *))
              [e refitAccelerationStructure:j.as
                                 descriptor:j.desc
                                destination:j.as
                              scratchBuffer:j.scratch
                        scratchBufferOffset:j.scratch_off
                                    options:MTLAccelerationStructureRefitOptionVertexData |
                                            (prims ? MTLAccelerationStructureRefitOptionPerPrimitiveData : 0)];
            else
              [e refitAccelerationStructure:j.as
                                 descriptor:j.desc
                                destination:j.as
                              scratchBuffer:j.scratch
                        scratchBufferOffset:j.scratch_off];
            continue;
          }
          [e buildAccelerationStructure:j.as descriptor:j.desc scratchBuffer:j.scratch scratchBufferOffset:j.scratch_off];
        }
    [e endEncoding];
  }

  // Captured positions to world space (RtCaptureBatch::convert), and the
  // debug view's normals of their triangles.
  void EncodeConversion(id<MTLCommandBuffer> cmd, const RtCaptureBatch* c) const {
    if (!c || c->jobs.empty()) return;
    struct {
      float convert[16];
      uint32_t count, pad[3];
    } p = {};
    static_assert(sizeof p == 80);
    std::memcpy(p.convert, c->convert, sizeof p.convert);
    // Concurrent dispatches: each capture's are small (a few thousand
    // threads), and one after the other each waited for the last to drain.
    // One barrier between the positions and the normals that read them.
    id<MTLComputeCommandEncoder> e = TimedCompute(cmd, "rt capture convert", MTLDispatchTypeConcurrent);
    [e setComputePipelineState:convert_pipe_];
    for (const RtCaptureBatch::Range& r : c->ranges) {
      p.count = r.count;
      [e setBuffer:r.buf offset:size_t(r.first) * 16 atIndex:0];
      [e setBuffer:r.buf offset:size_t(r.capacity) * 16 + size_t(r.first) * 12 atIndex:1];
      [e setBytes:&p length:sizeof p atIndex:2];
      [e dispatchThreads:MTLSizeMake(r.count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    if (normals_pipe_ && !c->normals.empty()) {
      [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [e setComputePipelineState:normals_pipe_];
      for (const RtCaptureBatch::Normals& n : c->normals) {
        const uint32_t np[4] = {n.triangles, n.stride, n.hit, 0};
        [e setBuffer:n.vbuf offset:n.voff atIndex:0];
        [e setBuffer:n.ibuf offset:n.ioff atIndex:1];
        [e setBuffer:n.ibuf offset:n.noff atIndex:2];
        [e setBytes:np length:sizeof np atIndex:3];
        [e dispatchThreads:MTLSizeMake(n.triangles, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
    }
    [e endEncoding];
  }

  // GPU scratch for one build in frame slot `s`, `bytes` of its own
  // (`tlas`: a TLAS's, else a BLAS's build or refit). With the scratch
  // pool, from the next of its lanes (the TLASes' own).
  std::pair<__unsafe_unretained id<MTLBuffer>, size_t> Scratch(FrameSlot& s, size_t bytes, bool tlas = false) {
    bytes = (std::max<size_t>(bytes, 1) + 255) & ~size_t(255);
    if (scratch_lanes_)
      return LaneScratch(tlas ? s.lanes[kMaxScratchLanes] : s.lanes[s.lane_next++ % scratch_lanes_], bytes);
    if (!s.scratch.empty() && s.scratch_used + bytes <= s.scratch.back().length) {
      const size_t off = s.scratch_used;
      s.scratch_used += bytes;
      return {s.scratch.back(), off};
    }
    const size_t size = std::max({bytes, kMinScratch, s.scratch.empty() ? 0 : size_t(s.scratch.back().length) * 2});
    id<MTLBuffer> b = [dev_ newBufferWithLength:size options:MTLResourceStorageModePrivate];
    if (!b) return {nil, 0};
    s.scratch.push_back(b);
    s.scratch_used = bytes;
    return {b, 0};
  }

  // `bytes` (a multiple of 256) of scratch in pool lane `l`: a range of
  // its own in the lane's buffer; a buffer that is full is kept for the
  // frame's jobs that have ranges in it, and the next is twice its size,
  // or what the slot's last frame took from the lane and a quarter more
  // (the jobs land on other lanes as their number changes).
  std::pair<__unsafe_unretained id<MTLBuffer>, size_t> LaneScratch(FrameSlot::ScratchLane& l, size_t bytes) {
    l.need += bytes;
    if (!l.bufs.empty() && l.used + bytes <= l.bufs.back().length) {
      const size_t off = l.used;
      l.used += bytes;
      return {l.bufs.back(), off};
    }
    const size_t size = std::max({bytes, kMinLaneScratch, l.prev_need + l.prev_need / 4,
                                  l.bufs.empty() ? 0 : size_t(l.bufs.back().length) * 2});
    id<MTLBuffer> b = [dev_ newBufferWithLength:size options:MTLResourceStorageModePrivate];
    if (!b) return {nil, 0};
    l.bufs.push_back(b);
    l.used = bytes;
    return {b, 0};
  }

  // Out of the cache; kept (retired_) until nothing may use it.
  // The other views' alias of `b` (by_geometry_) forgotten.
  void Unalias(RtBlas* b) {
    const RtMeshKey geo = Geometry(b->key);
    auto al = by_geometry_.find(KeyRef{&geo, b->geo_hash});
    if (al != by_geometry_.end() && al->second == b) by_geometry_.erase(al);
  }

  void Retire(RtBlas* b) {
    auto it = cache_.find(KeyRef{&b->key, b->key_hash});
    if (it == cache_.end() || it->second.get() != b) return;
    Unalias(b);
    b->retired = true;
    if (b->retained) retained_gone_ = true;  // (PurgeRetained, before it may be freed)
    auto pl = places_.find(b->key.Place());
    if (pl != places_.end() && pl->second.current == b) pl->second.current = nullptr;
    blas_bytes_ -= b->bytes;
    retired_bytes_ += b->bytes;
    retired_.push_back(std::move(it->second));
    cache_.erase(it);
  }

  // BLASes whose builds were never encoded: gone, with their instances.
  void DropUnbuilt(const RtBuildBatch& batch) {
    for (const RtBuildBatch::Job& j : batch.jobs) j.blas->mark = UINT64_MAX;
    std::erase_if(pending_, [](const Pending& p) { return p.blas->mark == UINT64_MAX; });
    for (Camera& c : cams_) c.count = 0;
    for (const Pending& p : pending_)
      if (p.camera != kAnyCamera && p.camera != kCapturedCamera) cams_[p.camera].count++;
    // (Their places forgotten: a later draw of one adds it again.)
    NextPlacedEpoch();
    snap_at_ = std::min(snap_at_, pending_.size());
    // Their retained instances (none should name a BLAS this new) go before
    // the BLASes do; the others' draws this frame found again.
    for (const RtBuildBatch::Job& j : batch.jobs) {
      j.blas->retired = true;
      if (j.blas->retained) retained_gone_ = true;
    }
    PurgeRetained();
    replaced_from_ = 0;
    for (auto& [k, r] : retained_)
      if (r.drawn == frame_) r.drawn = 0;
    for (uint32_t i = 0; i < pending_.size() && !retained_.empty(); i++)
      if (pending_[i].camera != kCapturedCamera)
        if (auto r = retained_.find(PlacedOf(pending_[i])); r != retained_.end()) {
          r->second.drawn = frame_;
          r->second.index = i;
        }
    for (const RtBuildBatch::Job& j : batch.jobs) {
      auto pl = places_.find(j.blas->key.Place());
      if (pl != places_.end() && pl->second.current == j.blas) pl->second.current = nullptr;
      auto it = cache_.find(KeyRef{&j.blas->key, j.blas->key_hash});
      if (it != cache_.end() && it->second.get() == j.blas) {
        Unalias(j.blas);
        blas_bytes_ -= j.blas->bytes;
        NotResident(j.blas->as);
        for (const RtBlas::Piece& c : j.blas->pieces) NotResident(c.as);
        cache_.erase(it);
      }
    }
  }

  // Every 32 presented frames: BLASes unused for kEvictAfter of them go.
  void Sweep() {
    if (frame_ - last_sweep_ < 32) return;
    last_sweep_ = frame_;
    lru_.clear();
    for (const auto& [k, b] : cache_)
      if (frame_ - b->last_frame > kEvictAfter && !b->retained) lru_.push_back(b.get());
    for (RtBlas* b : lru_) Retire(b);
    std::erase_if(places_, [&](const auto& kv) {
      return !kv.second.current && frame_ - kv.second.last_frame > kEvictAfter;
    });
  }

  // At max_blas_bytes (where builds stop): the least recently used go,
  // down to 7/8 of it (those of the frame that just ended and the one
  // before stay). A BLAS a retained instance names is in use (as for
  // Sweep), though no draw has added it lately: those go only after every
  // other, the least recently drawn first.
  void Trim() {
    if (blas_bytes_ < opt_.max_blas_bytes) return;
    lru_.clear();
    for (const auto& [k, b] : cache_)
      if (b->last_frame + 2 < frame_) lru_.push_back(b.get());
    std::sort(lru_.begin(), lru_.end(), [](const RtBlas* a, const RtBlas* b) {
      if ((a->retained != 0) != (b->retained != 0)) return b->retained != 0;
      return a->last_frame < b->last_frame;
    });
    for (RtBlas* b : lru_) {
      if (blas_bytes_ <= opt_.max_blas_bytes / 8 * 7) break;
      Retire(b);
    }
  }

  // A mesh not cached: its data gathered and its BLAS queued for building
  // in this frame's work; null if it cannot be (counted). `key_hash`,
  // `geo_hash`: RtMeshKeyHash of the key and of its geometry.
  // A static mesh larger than kPieceMin metres along some axis, of at least
  // kPieceMinTriangles triangles: its triangles (`tri`, three indices each)
  // reordered by the cell of a grid of kPieceCell-metre cells (larger where
  // that makes more than kPieceMax) their centres lie in, and the runs of
  // one cell in `runs` (first triangle, count); `runs` empty where it stays
  // whole (one cell).
  static constexpr float kPieceMin = 24.0f, kPieceCell = 12.0f;
  static constexpr uint32_t kPieceMinTriangles = 128, kPieceMax = 48;
  static void Pieces(const std::vector<float>& pos, std::vector<uint32_t>* tri, uint32_t tris, const float lo[3],
                     const float hi[3], std::vector<std::pair<uint32_t, uint32_t>>* runs) {
    runs->clear();
    static const bool off = [] {
      const char* e = std::getenv("RECOMP_RT_PIECES");
      return e && *e == '0';
    }();
    const float ext = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    if (off || !(ext > kPieceMin) || tris < kPieceMinTriangles || tri->size() < size_t(tris) * 3) return;
    std::vector<uint64_t> keyed(tris);
    for (float cell = kPieceCell;; cell *= 1.5f) {
      for (uint32_t t = 0; t < tris; t++) {
        uint64_t key = 0;
        for (int k = 0; k < 3; k++) {
          const float c = (pos[size_t((*tri)[t * 3]) * 4 + k] + pos[size_t((*tri)[t * 3 + 1]) * 4 + k] +
                           pos[size_t((*tri)[t * 3 + 2]) * 4 + k]) /
                          3.0f;
          const uint64_t at = uint64_t(std::clamp((c - lo[k]) / cell, 0.0f, 1023.0f));
          key = key * 1024 + at;
        }
        keyed[t] = key << 32 | t;
      }
      std::sort(keyed.begin(), keyed.end());
      uint32_t cells = 0;
      for (uint32_t t = 0; t < tris; t++) cells += t == 0 || (keyed[t] >> 32) != (keyed[t - 1] >> 32);
      if (cells <= kPieceMax) break;
    }
    std::vector<uint32_t> sorted(size_t(tris) * 3);
    for (uint32_t t = 0; t < tris; t++) {
      const uint32_t from = uint32_t(keyed[t]);
      for (int k = 0; k < 3; k++) sorted[size_t(t) * 3 + k] = (*tri)[size_t(from) * 3 + k];
      if (t == 0 || (keyed[t] >> 32) != (keyed[t - 1] >> 32))
        runs->emplace_back(t, 1u);
      else
        runs->back().second++;
    }
    if (runs->size() <= 1) {
      runs->clear();
      return;
    }
    std::copy(sorted.begin(), sorted.end(), tri->begin());
  }

  RtBlas* Build(const RtMeshKey& key, const RtGeometry& g, size_t key_hash, size_t geo_hash) {
    const uint32_t tris = g.primitives;
    if (tris > opt_.max_mesh_triangles) {
      Skip(RtSkip::TooLarge);
      return nullptr;
    }
    if (frame_builds_ >= kBuildsPerFrame ||
        (frame_builds_ && uint64_t(frame_tris_) + tris > opt_.build_triangles_per_frame) ||
        (frame_builds_ && opt_.build_ms_per_frame > 0.0f &&
         double(frame_build_ns_) >= double(opt_.build_ms_per_frame) * 1e6)) {
      Skip(RtSkip::BuildBudget);
      return nullptr;
    }
    // (The time from here to the return, whatever the way out: the frame's
    // budget of it, RtSceneOptions::build_ms_per_frame.)
    struct BuildTime {
      uint64_t* sum;
      std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
      ~BuildTime() {
        *sum += uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
      }
    } build_time{&frame_build_ns_};
    if (!alloc_ || blas_bytes_ >= opt_.max_blas_bytes) {
      Skip(RtSkip::NoMemory);
      return nullptr;
    }
    // New contents where a mesh's data is: the BLAS of the old ones goes at
    // once; built too often, the place is left out until it stays the same.
    Place& pl = places_[key.Place()];
    pl.last_frame = frame_;
    if (pl.vertex_version != key.vertex_version || pl.index_version != key.index_version ||
        pl.attr_version != key.attr_version) {
      pl.vertex_version = key.vertex_version;
      pl.index_version = key.index_version;
      pl.attr_version = key.attr_version;
      pl.since = frame_;
    }
    if (pl.rewritten) {
      if (frame_ - pl.since < opt_.calm_frames) {
        Skip(RtSkip::Rebuilt);
        return nullptr;
      }
      pl.rewritten = false;
      pl.rebuilds = 0;
    }
    if (pl.current) {
      Retire(pl.current);
      if (frame_ - pl.window > opt_.rebuild_window) {
        pl.window = frame_;
        pl.rebuilds = 0;
      }
      if (++pl.rebuilds >= opt_.rebuild_limit) {
        pl.rewritten = true;
        Skip(RtSkip::Rebuilt);
        return nullptr;
      }
    }
    if (!RtGather(g, &gathered_)) {
      Skip(RtSkip::BadIndices);
      return nullptr;
    }
    // Primitive data: an alpha-cut mesh's RtCutPrim, then the debug view's
    // normal; with hit data and UVs RtHitPrim (both, and the LOD constant).
    const bool hit = opt_.hit_data && g.uv.data, normals = opt_.debug == RtDebug::Normals && !hit,
               cut = key.cut != 0;
    const uint32_t stride =
        hit ? uint32_t(sizeof(RtHitPrim)) : (cut ? uint32_t(sizeof(RtCutPrim)) : 0) + (normals ? 12 : 0);
    const std::vector<float>& pos = gathered_.positions;
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t v = 0; v + 3 < pos.size(); v += 4)
      for (int k = 0; k < 3; k++) {
        lo[k] = std::min(lo[k], pos[v + k]);
        hi[k] = std::max(hi[k], pos[v + k]);
      }
    // A mesh this large in pieces (RtBlas::pieces): its triangles sorted by
    // the cell of a grid their centres lie in, a run a piece.
    std::vector<std::pair<uint32_t, uint32_t>> runs;  // first triangle, count
    Pieces(pos, &gathered_.indices, tris, lo, hi, &runs);
    const std::vector<uint32_t>& tri = gathered_.indices;
    const size_t pos_bytes = pos.size() * 4, tri_bytes = tri.size() * 4;
    const size_t tri_at = (pos_bytes + 255) & ~size_t(255);
    const size_t prim_at = (tri_at + tri_bytes + 255) & ~size_t(255);
    const size_t total = stride ? prim_at + size_t(tris) * stride : tri_at + tri_bytes;
    const RtUpload u = alloc_(alloc_ctx_, total, 256);
    if (!u.cpu) {
      Skip(RtSkip::NoMemory);
      return nullptr;
    }
    std::memcpy(u.cpu, pos.data(), pos_bytes);
    std::memcpy(u.cpu + tri_at, tri.data(), tri_bytes);
    // (UVs that cannot be read, of a mesh not alpha-cut: zeros, and no
    // albedo.)
    bool uvs = true;
    uint32_t alpha_lo = 255;  // the least vertex alpha the primitive data holds
    if (cut || hit)
      for (uint32_t t = 0; t < tris; t++) {
        const uint32_t* c = &tri[size_t(t) * 3];
        const uint32_t v[3] = {gathered_.used[c[0]], gathered_.used[c[1]], gathered_.used[c[2]]};
        RtCutPrim cp;
        if (!RtCutTriangle(g, gathered_.first, v, &cp)) {
          if (cut) {
            Skip(RtSkip::BadIndices);
            return nullptr;
          }
          cp = RtCutPrim{};
          cp.alpha = 0xFFFFFFu;
          uvs = false;
        }
        for (int k = 0; k < 3; k++) alpha_lo = std::min(alpha_lo, (cp.alpha >> (8 * k)) & 0xFFu);
        if (hit) {
          RtHitPrim hp;
          RtHitTriangle(cp, &pos[size_t(c[0]) * 4], &pos[size_t(c[1]) * 4], &pos[size_t(c[2]) * 4], &hp);
          std::memcpy(u.cpu + prim_at + size_t(t) * stride, &hp, sizeof hp);
        } else {
          std::memcpy(u.cpu + prim_at + size_t(t) * stride, &cp, sizeof cp);
        }
      }
    if (normals) {
      // Each triangle's (unnormalised) geometric normal, object space.
      uint8_t* nrm = u.cpu + prim_at + (cut ? sizeof(RtCutPrim) : 0);
      for (uint32_t t = 0; t < tris; t++) {
        const float* a = &pos[size_t(tri[t * 3]) * 4];
        const float* b = &pos[size_t(tri[t * 3 + 1]) * 4];
        const float* c = &pos[size_t(tri[t * 3 + 2]) * 4];
        const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        std::memcpy(nrm + size_t(t) * stride, n, 12);
      }
    }
    if (pos.capacity() > kKeptGather || tri.capacity() > kKeptGather || gathered_.remap.capacity() > kKeptGather ||
        gathered_.used.capacity() > kKeptGather)
      gathered_ = RtGathered{};  // after an outlier
    FrameSlot& s = slots_[serial_ % kSlots];
    if (runs.empty()) runs.emplace_back(0u, tris);
    std::vector<RtBuildBatch::Job> jobs;
    jobs.reserve(runs.size());
    for (const auto& [first, count] : runs) {
      RtBuildBatch::Job job;
      @autoreleasepool {
        MTLAccelerationStructureTriangleGeometryDescriptor* geo =
            [MTLAccelerationStructureTriangleGeometryDescriptor new];
        geo.vertexBuffer = u.buf;
        geo.vertexBufferOffset = u.off;
        geo.vertexStride = 16;
        geo.vertexFormat = MTLAttributeFormatFloat3;
        geo.indexBuffer = u.buf;
        geo.indexBufferOffset = u.off + tri_at + size_t(first) * 12;
        geo.indexType = MTLIndexTypeUInt32;
        geo.triangleCount = count;
        geo.opaque = YES;  // (alpha-cut instances are made non-opaque)
        if (stride) {
          geo.primitiveDataBuffer = u.buf;
          geo.primitiveDataBufferOffset = u.off + prim_at + size_t(first) * stride;
          geo.primitiveDataStride = stride;
          geo.primitiveDataElementSize = stride;
        }
        MTLPrimitiveAccelerationStructureDescriptor* desc = [MTLPrimitiveAccelerationStructureDescriptor new];
        desc.geometryDescriptors = @[ geo ];
        // Default usage, no fast build: these are built once and traced for
        // as long as the mesh is drawn (StaticUsageWanted: or built for fast
        // intersection, to measure).
        desc.usage = static_usage_;
        const MTLAccelerationStructureSizes sz = [dev_ accelerationStructureSizesWithDescriptor:desc];
        id<MTLAccelerationStructure> as = [dev_ newAccelerationStructureWithSize:sz.accelerationStructureSize];
        std::pair<__unsafe_unretained id<MTLBuffer>, size_t> sc{nil, 0};
        if (as) sc = Scratch(s, sz.buildScratchBufferSize);
        if (!as || !sc.first) {
          Skip(RtSkip::NoMemory);
          return nullptr;  // (pieces made so far go with this frame's memory: never resident)
        }
        job.desc = desc;
        job.as = as;
        job.scratch = sc.first;
        job.scratch_off = sc.second;
      }
      jobs.push_back(std::move(job));
    }
    for (const RtBuildBatch::Job& j : jobs) Resident(j.as);
    if (!cur_batch_) {
      if (s.batches_used == s.batches.size()) s.batches.push_back(std::make_unique<RtBuildBatch>());
      cur_batch_ = s.batches[s.batches_used++].get();
    }
    std::unique_ptr<RtBlas>& slot = cache_[key];
    slot = std::make_unique<RtBlas>();
    RtBlas* b = slot.get();
    b->as = jobs[0].as;
    b->id = jobs[0].as.gpuResourceID;
    b->bytes = 0;
    for (size_t k = 0; k < jobs.size(); k++) {
      b->bytes += jobs[k].as.allocatedSize;
      if (k) {
        RtBlasPiece c;
        c.as = jobs[k].as;
        c.id = jobs[k].as.gpuResourceID;
        b->pieces.push_back(c);
      }
    }
    b->key = key;
    b->key_hash = key_hash;
    b->geo_hash = geo_hash;
    // (Where Add made them, perhaps late: Retire and the maps find the BLAS
    // by them.)
    assert(b->key_hash == RtMeshKeyHash()(b->key) && b->geo_hash == RtMeshKeyHash()(Geometry(b->key)));
    b->mesh = next_mesh_++;
    b->triangles = tris;
    b->cut = cut;
    b->alpha_lo = uint8_t(alpha_lo);
    b->hit = hit;
    b->uvs = uvs;
    std::memcpy(b->lo, lo, sizeof lo);
    std::memcpy(b->hi, hi, sizeof hi);
    b->last_serial = serial_;
    b->last_frame = frame_;
    pl.current = b;
    blas_bytes_ += b->bytes;
    if (!b->pieces.empty()) stats_.pieced++, stats_.pieces += 1 + b->pieces.size();
    for (RtBuildBatch::Job& j : jobs) {
      j.blas = b;
      cur_batch_->jobs.push_back(std::move(j));
    }
    frame_builds_++;
    frame_tris_ += tris;
    total_builds_++;
    stats_.builds++;
    stats_.build_triangles += tris;
    return b;
  }

  id<MTLDevice> dev_;
  RtSceneOptions opt_;
  // Read once, when the scene is made: the scratch pool's lanes (0: none,
  // the path before it), a static BLAS's usage and the TLAS's.
  const uint32_t scratch_lanes_ = ScratchLanesWanted();
  const MTLAccelerationStructureUsage static_usage_ = StaticUsageWanted();
  const MTLAccelerationStructureUsage tlas_usage_ = TlasUsageWanted();
  RtAlloc alloc_ = nullptr;
  void* alloc_ctx_ = nullptr;
  uint64_t serial_ = 0;
  uint64_t frame_ = 1;  // presented frames, counted from 1
  uint64_t retained_ids_ = 0;  // Retained::id's last
  uint64_t last_sweep_ = 0, tlas_seq_ = 0;
  bool ended_ = false;  // EndFrame since BeginFrame
  // (Their hashes counted: RtStats::key_hashes, since the last TakeStats.)
  uint64_t key_hashes_ = 0;
  KeyMap<std::unique_ptr<RtBlas>> cache_{0, KeyHasher{&key_hashes_}};
  KeyMap<Place> places_{0, KeyHasher{&key_hashes_}};
  // With hit data: the BLAS a main-view draw with UVs used, by its geometry
  // (Geometry), for other views' opaque draws of it (Add).
  KeyMap<RtBlas*> by_geometry_{0, KeyHasher{&key_hashes_}};
  // Captured draws' triangles and primitive data made on the CPU, by
  // AddCaptured's key, at most kCapturePrimBytes in all.
  struct CapPrims {
    std::vector<uint8_t> bytes;
    uint32_t stride = 0;
    bool uvs = true;
    uint64_t last_frame = 0;
  };
  static constexpr uint64_t kCapturePrimBytes = 64ull << 20;
  std::unordered_map<RtMeshKey, CapPrims, RtMeshKeyHash> cap_prims_;
  uint64_t cap_prim_bytes_ = 0;
  std::vector<std::unique_ptr<RtBlas>> retired_;
  std::vector<RtBlas*> lru_;  // Sweep and Trim's working list
  uint64_t blas_bytes_ = 0, retired_bytes_ = 0, total_builds_ = 0;
  uint32_t next_mesh_ = 1;
  FrameSlot slots_[kSlots];
  RtBuildBatch* cur_batch_ = nullptr;  // builds not taken yet
  RtCaptureBatch* cur_cap_ = nullptr;  // captures not taken yet
  // Captured draws this presented frame (their budgets), their instances
  // in pending_; their BLASes while anything may use them, records to
  // reuse, and memory (theirs and the pool's); the last TLAS's.
  uint32_t frame_caps_ = 0, frame_cap_tris_ = 0, frame_cap_verts_ = 0;
  size_t pending_captured_ = 0;
  std::vector<std::unique_ptr<RtBlas>> cap_live_, cap_spare_;
  std::vector<PooledAs> as_pool_;
  std::unordered_map<RtMeshKey, std::vector<Warm>, RtMeshKeyHash> warm_;  // by AddCaptured's key
  std::unordered_map<RtMeshKey, uint32_t, RtMeshKeyHash> cap_nth_;  // this presented frame's draws of each key
  // The actor whose parts are being added (StartActor), and the batch
  // they are in.
  RtBlas* actor_ = nullptr;
  RtCaptureBatch* actor_batch_ = nullptr;
  // The residency set every BLAS is in (UseResidency; nil: none, each
  // frame's declared where traced), with changes not committed yet.
  id rs_ = nil;
  id<MTLCommandQueue> rs_queue_ = nil;  // the queue that has it
  bool rs_dirty_ = false;
  uint64_t cap_blas_bytes_ = 0;
  uint32_t tlas_captured_ = 0;
  uint32_t frame_builds_ = 0, frame_tris_ = 0, draws_ = 0;  // builds, their triangles, draws: this presented frame
  uint64_t frame_build_ns_ = 0;  // Build's time on the drawing thread this presented frame
  // The presented frame's instances so far, in draw order, and its
  // cameras; the last snapshot's place in them.
  std::vector<Pending> pending_;
  std::vector<Camera> cams_;
  size_t snap_at_ = 0;
  bool snapped_ = false;
  // The last presented frame's instances after its last snapshot, and its
  // camera.
  std::vector<Pending> carry_;
  // Instances' materials: pending_'s (this presented frame), carry_'s; the
  // textures they name, by the texture.
  std::vector<PendingMat> pending_mat_, carry_mat_;
  std::unordered_map<const void*, std::unique_ptr<HeldTex>> held_;
  uint64_t held_bytes_ = 0;
  uint32_t tlas_cut_ = 0;    // alpha-cut instances in the last TLAS
  uint32_t tlas_albedo_ = 0; // instances with an albedo in the last TLAS
  uint32_t tlas_other_ = 0;  // other views' instances in the last TLAS
  // The geometries and world matrices of this presented frame's instances
  // (Add): the entries of epoch placed_epoch_ (a new one each presented
  // frame, and when DropUnbuilt forgets them), placed_now_ of them. Those
  // of earlier ones stay for the next draw of their place, as many as
  // NextPlacedEpoch keeps.
  std::unordered_map<Placed, PlacedAt, PlacedHash> placed_;
  uint64_t placed_epoch_ = 1, placed_now_ = 0;
  std::unordered_map<Placed, Retained, PlacedHash> retained_;
  // By geometry (Placed::key), by world matrix (WorldOf), for DropReplaced.
  KeyMap<std::vector<RetainedNode*>> retained_geo_{0, KeyHasher{&key_hashes_}};
  std::unordered_map<Placed, std::vector<RetainedNode*>, PlacedHash> retained_world_;
  std::vector<RetainedNode*> victims_;  // DropReplaced's working list
  std::vector<const Retained*> listed_retained_;  // TlasCount's for MakeTlas
  size_t replaced_from_ = 0;            // pending_ checked by DropReplaced so far
  // The static instances (Placed) the last presented frame's TLAS listed,
  // and this one's (Retain); their nodes reused from frame to frame.
  BlockPool static_pool_;
  using StaticSet = std::unordered_set<Placed, PlacedHash, std::equal_to<Placed>, PoolAlloc<Placed>>;
  StaticSet static_last_ = StaticSet(PoolAlloc<Placed>(&static_pool_));
  StaticSet static_now_ = StaticSet(PoolAlloc<Placed>(&static_pool_));
  bool retained_gone_ = false;  // a BLAS a retained instance names left the cache (PurgeRetained)
  uint32_t tlas_retained_ = 0;  // retained instances in the last TLAS
  // The static instances of presented frame `frame`'s first TLAS, where
  // sampled (MakeTlas; RtStats::static_listed, static_distinct), and its
  // working list.
  struct StaticSample {
    uint64_t frame = 0, listed = 0, distinct = 0;
  } static_sample_;
  std::vector<uint64_t> static_hashes_;
  // The instance counts of the presented frame `frame`'s last snapshot
  // (the frame's, without RtSceneOptions::end_tlas).
  struct SnapCounts {
    uint64_t frame = 0;
    uint32_t instances = 0, captured = 0, cut = 0, albedo = 0, other = 0, retained = 0;
  } snap_counts_;
  uint64_t camera_seq_ = 0, vote_ = 0;
  std::vector<RtInstanceSummary>* describe_ = nullptr;  // DescribeNextTlas  // cameras numbered (Camera::id); the last Add's vote (TakeVote)
  double prev_cam_[16] = {};
  bool prev_reversed_ = false, have_prev_cam_ = false;
  RtGathered gathered_;
  RtStats stats_;
  // Encoding side.
  id<MTLComputePipelineState> convert_pipe_ = nil, normals_pipe_ = nil;
  id<MTLComputePipelineState> debug_pipe_ = nil;
  id<MTLTexture> debug_tex_ = nil;
};

}  // namespace

uint64_t RtDefaultBlasBytes(id<MTLDevice> dev) {
#if TARGET_OS_OSX
  // A Mac: an eighth of what Metal recommends the process keeps resident
  // (several GB on Apple silicon), at most 1 GiB; a city's static
  // geometry in view took a few hundred MB.
  const uint64_t ws = dev ? uint64_t(dev.recommendedMaxWorkingSetSize) : 0;
  return std::clamp<uint64_t>(ws / 8, 128ull << 20, 1ull << 30);
#else
  // iPad, Apple TV: the system ends a process for its footprint, and a
  // game's own budget there takes most of what it allows; 256 MB keeps
  // the scene near the camera.
  (void)dev;
  return 256ull << 20;
#endif
}

std::unique_ptr<RtScene> MakeRtScene(id<MTLDevice> dev, const RtSceneOptions& opt) {
  if (!RtSupported(dev)) return nullptr;
  RtSceneOptions o = opt;
  if (!o.max_blas_bytes) o.max_blas_bytes = RtDefaultBlasBytes(dev);
  auto s = std::make_unique<MetalRtScene>(dev, o);
  if (!s->Init()) return nullptr;
  return s;
}

}  // namespace recomp::gfx
