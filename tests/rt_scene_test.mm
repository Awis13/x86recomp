// The ray-traced scene (gfx/rt_scene.h). On the CPU: half floats, mesh
// keys, vertex ranges, index ranges kept by where the indices are,
// gathering (only the vertices used, interleaved FLOAT3, FLOAT4 and
// FLOAT16_4 positions), strips and fans as lists. On a
// GPU with hardware ray tracing, through RtScene alone: BLASes (inputs
// overwritten before tracing), TLASes of instances at D3D world matrices
// (rotated and sheared, kilometres out), the frame's camera by vote, D3D's
// winding (mirrored worlds and view-projections, reversed depth, culling),
// the cache (versions, places rebuilt, budgets, memory cap, eviction
// waiting for the GPU, builds never encoded), frames a readback splits,
// snapshots with the last frame's carried instances, frame slots reused
// with earlier frames in flight, and the debug view. Alpha-cut geometry
// (RtCut): vertex inputs read on the CPU, primitive data, the tests on the
// CPU, keys; on the GPU holes where the texel's alpha fails, wrap and
// clamp, vertex alpha, NEVER, texkill thresholds, opaque instances
// unaffected, any hit and masks, the material table's layout, texture
// levels, bilinear filtering, X8R8G8B8's swizzle through a resource ID,
// carried instances, a texture released after Add held to its horizon,
// and a captured draw. Retained instances: kept outside the view, let go in
// it undrawn after kInViewFrames, when replaced (moved off, another level of
// detail), with their BLAS, when idle and past the cap; in snapshots; what
// moves and captured draws not retained; another view's listed on two
// frames of 16 kept in every TLAS; their BLASes the last to go at the BLAS
// memory cap. The main view's size at a render scale. The bookkeeping of
// static draws frame after frame: mesh keys hashed once a draw, one
// geometry at one place from two views (either first), other views' BLAS
// of a geometry with two alpha-cut variants, the instances' order, places
// forgotten when a build never encoded drops its draw, the places kept
// bounded where every draw moves each frame, another view's BLAS built with
// a key hash made late let go by the sweep; and its CPU time a draw
// (printed).
//
//   rt_scene_test        (skipped, exit 77, without a GPU that traces rays)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/rt_alpha.h"
#include "runtime/gfx/rt_scene.h"

namespace {

using namespace recomp::gfx;
namespace hlsl = recomp::hlsl;

int Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  return ok ? 0 : 1;
}

uint16_t ToHalf(float f) {  // exact values only (the test's)
  uint32_t b;
  std::memcpy(&b, &f, 4);
  const uint32_t sign = (b >> 16) & 0x8000u;
  if ((b & 0x7FFFFFFFu) == 0) return uint16_t(sign);
  const int exp = int((b >> 23) & 0xFF) - 127 + 15;
  return uint16_t(sign | uint32_t(exp) << 10 | ((b >> 13) & 0x3FFu));
}

using Pos = std::vector<std::array<float, 3>>;

// An interleaved vertex buffer: `stride` bytes a vertex, the position at
// `offset` (FLOAT3, FLOAT4 or FLOAT16_4), other bytes 0xAB.
std::vector<uint8_t> Vertices(const Pos& pos, uint32_t stride, uint32_t offset, uint32_t type) {
  std::vector<uint8_t> v(pos.size() * stride, 0xAB);
  for (size_t i = 0; i < pos.size(); i++) {
    uint8_t* p = v.data() + i * stride + offset;
    if (type == 16) {
      const uint16_t h[4] = {ToHalf(pos[i][0]), ToHalf(pos[i][1]), ToHalf(pos[i][2]), ToHalf(1.0f)};
      std::memcpy(p, h, 8);
    } else {
      std::memcpy(p, pos[i].data(), 12);
      if (type == 3) {
        const float w = 1.0f;
        std::memcpy(p + 12, &w, 4);
      }
    }
  }
  return v;
}

struct Mesh {
  std::vector<uint8_t> vb;
  std::vector<uint16_t> ib;
  RtGeometry g;
  RtMeshKey key;
};

// A mesh in buffer number `buffer` (its place); its contents version 1.
Mesh MakeMesh(const Pos& pos, std::vector<uint16_t> ib, uint32_t stride, uint32_t offset, uint32_t type, uint32_t prim,
              uint32_t primitives, int32_t base_vertex, uint64_t buffer) {
  Mesh m;
  m.vb = Vertices(pos, stride, offset, type);
  m.ib = std::move(ib);
  m.g.positions = m.vb.data() + offset;
  m.g.size = m.vb.size() - offset;
  m.g.stride = stride;
  m.g.type = type;
  m.g.prim = prim;
  m.g.primitives = primitives;
  m.g.base_vertex = base_vertex;
  m.g.indices = m.ib.empty() ? nullptr : m.ib.data();
  m.g.index_size = 2;
  m.key.vertex_version = buffer * 16 + 1;
  m.key.index_version = m.ib.empty() ? 0 : buffer * 16 + 2;
  m.key.vertex_buffer = buffer;
  m.key.index_buffer = m.ib.empty() ? 0 : buffer + (1u << 20);
  m.key.base_vertex = base_vertex;
  m.key.prim = prim;
  m.key.primitives = primitives;
  m.key.offset = offset;
  m.key.stride = stride;
  m.key.type = uint16_t(type);
  m.key.index_size = m.ib.empty() ? 0 : 2;
  return m;
}

// ---- CPU

int CpuChecks() {
  int failures = 0;
  failures += Check(HalfToFloat(0x3C00) == 1.0f && HalfToFloat(0xC000) == -2.0f && HalfToFloat(0x7BFF) == 65504.0f &&
                        HalfToFloat(0x0001) == std::ldexp(1.0f, -24) && HalfToFloat(0x8000) == 0.0f &&
                        std::signbit(HalfToFloat(0x8000)) && std::isinf(HalfToFloat(0x7C00)) &&
                        std::isnan(HalfToFloat(0x7E00)) && HalfToFloat(0x3555) == 0.333251953125f,
                    "half floats");

  // Keys: equal ones hash alike; each field tells keys apart; the place
  // leaves the versions out.
  {
    RtMeshKey a;
    a.vertex_version = 7;
    a.index_version = 8;
    a.vertex_buffer = 3;
    a.index_buffer = 4;
    a.prim = 4;
    a.primitives = 10;
    a.stride = 32;
    a.type = 2;
    a.index_size = 2;
    RtMeshKey b = a;
    bool ok = a == b && RtMeshKeyHash{}(a) == RtMeshKeyHash{}(b);
    RtMeshKey v[16];
    for (RtMeshKey& k : v) k = a;
    v[0].vertex_version++, v[1].index_version++, v[2].vertex_buffer++, v[3].index_buffer++, v[4].vertex_offset += 32;
    v[5].index_offset += 2, v[6].base_vertex++, v[7].min_index++, v[8].vertex_count++, v[9].prim = 5;
    v[10].primitives++, v[11].offset += 4, v[12].stride += 4, v[13].type = 16, v[14].index_size = 4;
    v[15].vertex_offset = 1u << 31;
    for (const RtMeshKey& k : v) ok = ok && !(k == a) && RtMeshKeyHash{}(k) != RtMeshKeyHash{}(a);
    RtMeshKey p = a.Place();
    ok = ok && p.vertex_version == 0 && p.index_version == 0 && p.vertex_buffer == 3 && v[0].Place() == p &&
         v[1].Place() == p && !(v[2].Place() == p);
    failures += Check(ok, "mesh keys and places");
  }

  // Strip of an odd count: indices 10-14 with base vertex 2 use vertices
  // 12-16; triangle 1 swapped to the strip's winding.
  {
    Pos pos(20);
    for (size_t i = 0; i < pos.size(); i++) pos[i] = {float(i), 0.5f * float(i), -float(i)};
    Mesh m = MakeMesh(pos, {10, 11, 12, 13, 14}, 28, 8, 2, 5, 3, 2, 1);
    uint32_t first = 0, count = 0;
    bool ok = RtVertexRange(m.g, &first, &count) && first == 12 && count == 5;
    uint32_t t[9] = {};
    RtTriangles(m.g, first, t);
    const uint32_t want[9] = {0, 1, 2, 2, 1, 3, 2, 3, 4};
    ok = ok && std::memcmp(t, want, sizeof t) == 0;
    RtGathered gt;
    ok = ok && RtGather(m.g, &gt) && gt.positions.size() == 20 && gt.indices.size() == 9 &&
         std::memcmp(gt.indices.data(), want, sizeof want) == 0;
    for (uint32_t i = 0; i < 5 && ok; i++)
      ok = gt.positions[i * 4] == float(12 + i) && gt.positions[i * 4 + 1] == 0.5f * float(12 + i) &&
           gt.positions[i * 4 + 2] == -float(12 + i) && gt.positions[i * 4 + 3] == 0.0f;
    failures += Check(ok, "strip (3 triangles) as a list, FLOAT3 positions at offset 8 of 28-byte vertices");
  }
  // Fans: not indexed from base vertex 3; indexed.
  {
    Pos pos(10, {0, 0, 0});
    Mesh m = MakeMesh(pos, {}, 12, 0, 2, 6, 4, 3, 1);
    uint32_t first = 0, count = 0;
    bool ok = RtVertexRange(m.g, &first, &count) && first == 3 && count == 6;
    uint32_t t[12] = {};
    RtTriangles(m.g, first, t);
    const uint32_t want[12] = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5};
    ok = ok && std::memcmp(t, want, sizeof t) == 0;
    Mesh f = MakeMesh(pos, {9, 4, 6, 2}, 12, 0, 2, 6, 2, 0, 2);
    ok = ok && RtVertexRange(f.g, &first, &count) && first == 2 && count == 8;
    RtTriangles(f.g, first, t);
    const uint32_t want_i[6] = {7, 2, 4, 7, 4, 0};
    failures += Check(ok && std::memcmp(t, want_i, sizeof want_i) == 0, "fans as lists, indexed and not");
  }
  // FLOAT16_4 and FLOAT4 positions; 32-bit indices; only the vertices
  // used gathered (a table over a narrow range, sorted over a wide one).
  {
    Pos pos = {{0.5f, -2.0f, 1024.0f}, {-0.25f, 3.0f, 0.125f}, {7, 8, 9}};
    bool ok = true;
    for (uint32_t type : {16u, 3u}) {
      Mesh m = MakeMesh(pos, {}, 24, 4, type, 4, 1, 0, 1);
      RtGathered gt;
      ok = ok && RtGather(m.g, &gt) && gt.positions.size() == 12;
      const std::vector<float>& p = gt.positions;
      ok = ok && p[0] == 0.5f && p[1] == -2.0f && p[2] == 1024.0f && p[4] == -0.25f && p[5] == 3.0f && p[6] == 0.125f &&
           p[8] == 7 && p[10] == 9;
    }
    for (uint32_t span : {1000u, 70000u}) {
      Pos wide(span + 1);
      for (uint32_t i = 0; i <= span; i++) wide[i] = {float(i), 0, 0};
      Mesh m = MakeMesh(wide, {}, 12, 0, 2, 4, 2, 0, 3);
      const uint32_t idx[6] = {span, 1, span / 2, span / 2, 1, 5};
      m.g.indices = idx;
      m.g.index_size = 4;
      RtGathered gt;
      ok = ok && RtGather(m.g, &gt) && gt.positions.size() == 16 && gt.indices.size() == 6;
      // Each corner's position is the vertex the draw's index named.
      for (int k = 0; k < 6 && ok; k++) ok = gt.positions[size_t(gt.indices[k]) * 4] == float(idx[k]);
    }
    failures += Check(ok, "FLOAT16_4 and FLOAT4 positions; 32-bit indices; only the vertices used gathered");
  }
  // Ranges refused: an index past the stream, a vertex before 0, lines, an
  // unsupported type, primitive counts whose indices overflow 32 bits.
  {
    Pos pos(4, {0, 0, 0});
    Mesh m = MakeMesh(pos, {0, 1, 4}, 12, 0, 2, 4, 1, 0, 1);
    uint32_t first, count;
    bool ok = !RtVertexRange(m.g, &first, &count);
    m.ib = {0, 1, 3};
    m.g.indices = m.ib.data();
    ok = ok && RtVertexRange(m.g, &first, &count) && count == 4;
    m.g.base_vertex = 1;
    ok = ok && !RtVertexRange(m.g, &first, &count);
    m.g.base_vertex = -1;
    ok = ok && !RtVertexRange(m.g, &first, &count);
    m.g.base_vertex = 0;
    m.g.prim = 2;
    ok = ok && !RtVertexRange(m.g, &first, &count);
    m.g.prim = 4;
    m.g.type = 15;  // FLOAT16_2
    ok = ok && !RtVertexRange(m.g, &first, &count) && !RtPositionType(15) && RtPositionType(2) &&
         RtPositionType(3) && RtPositionType(16);
    Mesh big = MakeMesh(pos, {}, 12, 0, 2, 4, (UINT32_MAX - 2) / 3 + 1, 0, 1);
    big.g.size = SIZE_MAX;
    ok = ok && !RtVertexRange(big.g, &first, &count);
    big.g.prim = 5;
    big.g.primitives = UINT32_MAX;
    ok = ok && !RtVertexRange(big.g, &first, &count);
    failures += Check(ok, "ranges outside the stream, lines, other types and overflowing counts refused");
  }
  // The index range kept by where the indices are (RtIndexRangeMemo): the
  // scan's result, then a hit; any part of the key changed a miss that
  // scans again; version 0 and draws not indexed scanned every time and
  // counted neither way; two keys in one slot each other's misses.
  {
    Pos pos(30, {0, 0, 0});
    // A list of 2 triangles at index 0; 3 more indices after them.
    Mesh m = MakeMesh(pos, {5, 9, 7, 3, 8, 6, 12, 4, 10}, 12, 0, 2, 4, 2, 2, 1);
    const uint64_t kBuf = 77, kVer = 3;
    struct Result {
      bool ok = false;
      uint32_t first = 0, count = 0;
      bool operator==(const Result& o) const { return ok == o.ok && (!ok || (first == o.first && count == o.count)); }
    };
    auto scan = [](const RtGeometry& g) {
      Result r;
      r.ok = RtIndexRange(g, &r.first, &r.count);
      return r;
    };
    auto kept = [](RtIndexRangeMemo& memo, const RtGeometry& g, uint64_t buffer, uint64_t version, uint32_t offset) {
      Result r;
      r.ok = memo.Range(g, buffer, version, offset, &r.first, &r.count);
      return r;
    };
    RtIndexRangeMemo memo;
    const Result want = scan(m.g);
    bool ok = want.ok && want.first == 5 && want.count == 7 && kept(memo, m.g, kBuf, kVer, 0) == want &&
              memo.misses() == 1 && memo.hits() == 0;
    failures += Check(ok, "index range memo: the scan's range on an indexed mesh, a miss");
    ok = kept(memo, m.g, kBuf, kVer, 0) == want && memo.misses() == 1 && memo.hits() == 1;
    failures += Check(ok, "index range memo: the same draw again a hit with the same range");

    // Each part of the key changed: a miss, and the range of what is there now.
    struct Variant {
      const char* what;
      RtGeometry g;
      uint64_t version;
      uint32_t offset;
      Result want;
    };
    std::vector<Variant> variants;
    auto add = [&](const char* what, RtGeometry g, uint64_t version, uint32_t offset, uint32_t first, uint32_t count) {
      Result w;
      w.ok = true, w.first = first, w.count = count;
      variants.push_back({what, g, version, offset, w});
    };
    add("version", m.g, kVer + 1, 0, 5, 7);
    RtGeometry g = m.g;
    g.indices = m.ib.data() + 3;  // indices 3 8 6 12 4 10
    add("index offset", g, kVer, 6, 5, 10);
    g = m.g;
    g.base_vertex = 4;
    add("base vertex", g, kVer, 0, 7, 7);
    g = m.g;
    g.primitives = 1;  // 5 9 7
    add("primitive count", g, kVer, 0, 7, 5);
    g = m.g;
    g.prim = 5;  // a strip of 2: 5 9 7 3
    add("primitive type", g, kVer, 0, 5, 7);
    // (The same bytes read as 32-bit indices: other indices there.)
    const std::vector<uint32_t> wide = {1, 2, 3, 20, 21, 22};
    g = m.g;
    g.indices = wide.data();
    g.index_size = 4;
    add("index size", g, kVer, 0, 3, 22);
    for (const Variant& v : variants) {
      RtIndexRangeMemo fresh;
      kept(fresh, m.g, kBuf, kVer, 0);
      const Result r = kept(fresh, v.g, kBuf, v.version, v.offset);
      const Result again = kept(fresh, v.g, kBuf, v.version, v.offset);
      const std::string what =
          std::string("index range memo: a changed ") + v.what + " a miss with the new range (then a hit)";
      failures += Check(r == v.want && r == scan(v.g) && again == v.want && fresh.misses() == 2 && fresh.hits() == 1,
                        what.c_str());
    }

    // A refused range is kept as refused.
    {
      RtIndexRangeMemo refused;
      g = m.g;
      g.base_vertex = -10;
      Result r;
      r.first = 123, r.count = 456;
      r.ok = refused.Range(g, kBuf, kVer, 0, &r.first, &r.count);
      Result h;
      h.first = 123, h.count = 456;
      h.ok = refused.Range(g, kBuf, kVer, 0, &h.first, &h.count);
      failures +=
          Check(!r.ok && !h.ok && r.first == 123 && h.count == 456 && refused.misses() == 1 && refused.hits() == 1,
                "index range memo: a refused range refused again (a hit), the outputs untouched");
    }

    // Version 0 and a draw not indexed: scanned, counted neither way.
    {
      RtIndexRangeMemo by;
      bool same = true;
      for (int i = 0; i < 2; i++) same = same && kept(by, m.g, kBuf, 0, 0) == want;
      Mesh flat = MakeMesh(pos, {}, 12, 0, 2, 4, 3, 2, 2);
      const Result fw = scan(flat.g);
      for (int i = 0; i < 2; i++) same = same && kept(by, flat.g, kBuf, kVer, 0) == fw;
      same = same && fw.ok && fw.first == 2 && fw.count == 9 && by.hits() == 0 && by.misses() == 0;
      failures += Check(same, "index range memo: version 0 and a draw not indexed bypass it");
    }

    // Two keys in one slot (a table of one): each evicts the other, both right.
    {
      RtIndexRangeMemo one(1);
      g = m.g;
      g.indices = m.ib.data() + 3;
      const Result wa = want, wb = scan(g);
      bool both = wb.ok && !(wa == wb);
      for (int i = 0; i < 3; i++)
        both = both && kept(one, m.g, kBuf, kVer, 0) == wa && kept(one, g, kBuf + 1, kVer, 6) == wb;
      both = both && one.misses() == 6 && one.hits() == 0;
      failures += Check(both, "index range memo: two keys in one slot evict each other, both ranges right");
    }
  }
  bool named = true;
  for (size_t i = 0; i < size_t(RtSkip::Count); i++) named = named && std::strcmp(RtSkipName(RtSkip(i)), "?") != 0;
  failures += Check(named, "skip reasons named");
  // The main view's size: the game's size of a target (a scene target an
  // upscaler makes at 0.67 of it keeps that), not its texture's.
  {
    RtDrawPass p;
    p.depth = p.test = p.write = p.color = true;
    p.main_size = RtMainSize(1920, 1080, 1920, 1080);
    bool ok = RtViewOf(p, false, true) == RtView::Main;
    ok = ok && !RtMainSize(int(std::lround(1920 * 0.67)), int(std::lround(1080 * 0.67)), 1920, 1080);
    ok = ok && !RtMainSize(960, 540, 1920, 1080) && !RtMainSize(1920, 1080, 0, 0);
    p.main_size = RtMainSize(960, 540, 1920, 1080);  // a half-size scene buffer
    ok = ok && RtViewOf(p, false, true) == RtView::None;
    failures += Check(ok, "a scene target at render scale 0.67 is the main view (by the game's size); a half-size "
                          "one is not");
  }
  return failures;
}

// ---- GPU

// (After kRtMsl.) Rays as effects trace them: alpha-cut candidates
// tested (RtTraceClosest, or RtTraceAny: t 0 for a hit), instances by mask.
constexpr const char* kTraceMsl = R"(
struct Query { float4 o; float4 d; };  // o.w: the ray's length; d.w: 1 to cull back faces
struct Hit { float t; uint id; uint front; uint prim; float4 n; };
struct TraceParams { uint normals, any, mask, pad; };
kernel void trace_cs(instance_acceleration_structure scene [[buffer(0)]],
                     const device Query* q [[buffer(1)]], device Hit* out [[buffer(2)]],
                     constant TraceParams& p [[buffer(3)]], const device RtMaterial* materials [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
  ray r(q[i].o.xyz, q[i].d.xyz, 0.0, q[i].o.w);
  intersection_params ip;
  if (q[i].d.w > 0.5) ip.set_triangle_cull_mode(triangle_cull_mode::back);
  Hit o = {};
  if (p.any != 0) {
    o.t = RtTraceAny(scene, r, p.mask, materials, ip) ? 0.0 : -1.0;
    o.id = 0xFFFFFFFFu;
    out[i] = o;
    return;
  }
  const RtHit h = RtTraceClosest(scene, r, p.mask, materials, ip);
  o.t = h.hit ? h.distance : -1.0;
  o.id = h.instance;
  o.front = h.front ? 1 : 0;
  o.prim = h.primitive;
  if (h.hit && p.normals != 0) {
    const float4x3 m = h.world_to_object;
    const uint at = (materials[h.instance].flags & RT_MAT_CUT) != 0u ? 28u : 0u;
    const float3 no = float3(*(const device packed_float3*)((const device uchar*)h.prim + at));
    o.n = float4(normalize(no * float3x3(m[0], m[1], m[2])), 0.0);
  }
  out[i] = o;
}
// The material table as the GPU reads it (layout check).
kernel void material_echo_cs(const device RtMaterial* m [[buffer(0)]], device float4* out [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  out[0] = float4(m[0].alpha_scale, m[0].kill_offset, m[0].max_level, as_type<float>(m[0].flags));
  out[1] = float4(float(sizeof(RtMaterial)), float(sizeof(RtCutPrim)), m[1].alpha_offset, as_type<float>(m[1].reserved[2]));
}
)";

struct Query {
  float o[4];  // origin, length
  float d[4];  // direction, cull back faces (1)
};
struct Hit {
  float t;
  uint32_t id, front, prim;
  float n[4];
};

struct Arena {
  id<MTLBuffer> buf = nil;
  size_t used = 0;
  bool full = false;
};

RtUpload ArenaAlloc(void* ctx, size_t bytes, size_t align) {
  auto* a = static_cast<Arena*>(ctx);
  const size_t off = (a->used + align - 1) & ~(align - 1);
  if (off + bytes > a->buf.length) {
    a->full = true;
    return {nil, 0, nullptr};
  }
  a->used = off + bytes;
  return {a->buf, off, static_cast<uint8_t*>(a->buf.contents) + off};
}

RtUpload NoAlloc(void*, size_t, size_t) { return {nil, 0, nullptr}; }

struct Gpu {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  id<MTLComputePipelineState> trace = nil, echo = nil;
  Arena arena;
  Arena slot_arena[RtScene::kSlots];  // for frames in flight

  bool Init() {
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(kRtMsl) + kTraceMsl).c_str()) options:nil error:&e];
    if (!lib) {
      std::printf("FAIL trace kernel: %s\n", e.localizedDescription.UTF8String);
      return false;
    }
    trace = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"trace_cs"] error:&e];
    echo = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"material_echo_cs"] error:&e];
    arena.buf = [dev newBufferWithLength:64u << 20 options:MTLResourceStorageModeShared];
    for (Arena& a : slot_arena) a.buf = [dev newBufferWithLength:1u << 20 options:MTLResourceStorageModeShared];
    return trace && echo && arena.buf;
  }

  // (Without retained instances unless `retain`: the checks before
  // RetainChecks count each frame's own.)
  std::unique_ptr<RtScene> Scene(RtSceneOptions o = {}, bool retain = false) {
    if (!retain) o.retain_frames = 0;
    std::unique_ptr<RtScene> s = MakeRtScene(dev, o);
    if (s) s->SetAllocator(ArenaAlloc, &arena);
    return s;
  }
  std::unique_ptr<RtScene> Scene(RtDebug debug) {
    RtSceneOptions o;
    o.debug = debug;
    return Scene(o);
  }

  bool Commit(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }

  // One presented frame: BeginFrame (every earlier frame finished), the
  // adds, the frame's work encoded and run.
  template <typename F>
  const RtFrame* Frame(RtScene& s, uint64_t serial, F adds) {
    arena.used = 0;
    s.BeginFrame(serial, serial - 1);
    adds();
    const RtFrame* f = s.EndFrame(true);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    s.EncodeFrame(cb, f);
    if (!Commit(cb)) return nullptr;
    return f;
  }

  // Encodes the queries' rays against f into cb; their hits land in the
  // returned buffer once cb is done.
  id<MTLBuffer> EncodeTrace(id<MTLCommandBuffer> cb, const RtFrame* f, const std::vector<Query>& q, bool normals,
                            bool any = false, uint32_t mask = 0xFF) {
    id<MTLBuffer> qb = [dev newBufferWithBytes:q.data() length:q.size() * sizeof(Query) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hb = [dev newBufferWithLength:q.size() * sizeof(Hit) options:MTLResourceStorageModeShared];
    const uint32_t params[4] = {normals ? 1u : 0u, any ? 1u : 0u, mask, 0};
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:trace];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    RtUseResources(e, f);
    [e setBuffer:qb offset:0 atIndex:1];
    [e setBuffer:hb offset:0 atIndex:2];
    [e setBytes:params length:sizeof params atIndex:3];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:4];
    [e dispatchThreads:MTLSizeMake(q.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    return hb;
  }

  std::vector<Hit> Trace(const RtFrame* f, const std::vector<Query>& q, bool normals = false, bool any = false,
                         uint32_t mask = 0xFF) {
    std::vector<Hit> out(q.size());
    for (Hit& h : out) h.t = -2;  // not traced
    if (!f || !f->tlas || q.empty()) return out;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBuffer> hb = EncodeTrace(cb, f, q, normals, any, mask);
    if (Commit(cb)) std::memcpy(out.data(), hb.contents, out.size() * sizeof(Hit));
    return out;
  }
};

using Mat = float[4][4];

void Translate(Mat m, float x, float y, float z, float scale = 1.0f) {
  std::memset(m, 0, sizeof(Mat));
  m[0][0] = m[1][1] = m[2][2] = scale;
  m[3][0] = x, m[3][1] = y, m[3][2] = z, m[3][3] = 1;
}

void MulRows(const Mat a, const Mat b, Mat o) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) {
      float v = 0;
      for (int i = 0; i < 4; i++) v += a[r][i] * b[i][k];
      o[r][k] = v;
    }
}

// D3D's left-handed perspective (row vectors), depth 0..1 at near..far, or
// reversed (1 at near, 0 at far).
void Perspective(Mat m, float fov_y, float aspect, float n, float f, bool reversed = false) {
  std::memset(m, 0, sizeof(Mat));
  const float ys = 1.0f / std::tan(fov_y * 0.5f);
  m[0][0] = ys / aspect;
  m[1][1] = ys;
  m[2][2] = reversed ? n / (n - f) : f / (f - n);
  m[2][3] = 1;
  m[3][2] = reversed ? -n * f / (n - f) : -n * f / (f - n);
}

// p (w = 1) times m.
std::array<float, 3> Apply(const Mat m, float x, float y, float z) {
  std::array<float, 3> o{};
  for (int k = 0; k < 3; k++) o[k] = x * m[0][k] + y * m[1][k] + z * m[2][k] + m[3][k];
  return o;
}

Query Ray(float ox, float oy, float oz, float dx, float dy, float dz, bool cull_back = false, float len = 1e5f) {
  return {{ox, oy, oz, len}, {dx, dy, dz, cull_back ? 1.0f : 0.0f}};
}

bool Near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol; }

// The unit square in z = 0 whose triangles are clockwise seen from -z (D3D's
// front faces toward a viewer there): v0 (0,0) v1 (0,1) v2 (1,0) v3 (1,1),
// as a strip.
const Pos kSquare = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}};

Mesh Square(uint64_t buffer) { return MakeMesh(kSquare, {0, 1, 2, 3}, 16, 0, 2, 5, 2, 0, buffer); }

// BLASes from interleaved data: a FLOAT3 strip of 3 triangles (16-bit
// indices, a negative base vertex) and a FLOAT16_4 list; inputs overwritten
// before the rays are traced; normals read back.
int BlasChecks(Gpu& g) {
  auto s = g.Scene(RtDebug::Normals);
  if (!s) return Check(false, "scene made");
  Pos pos(10, {1000, 1000, 1000});
  const Pos zig = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {2, 0, 0}};
  for (int i = 0; i < 5; i++) pos[3 + i] = zig[i];
  Mesh strip = MakeMesh(pos, {5, 6, 7, 8, 9}, 28, 8, 2, 5, 3, -2, 1);
  Mesh half = MakeMesh({{10, 0, 0}, {10, 2, 0}, {12, 0, 0}, {12, 2, 0}}, {0, 1, 2, 2, 1, 3}, 20, 4, 16, 4, 2, 0, 2);
  Mat w;
  Translate(w, 0, 0, 5);
  const RtFrame* f = g.Frame(*s, 1, [&] {
    s->Add(strip.key, strip.g, w, w, 3, 4);
    s->Add(half.key, half.g, w, w, 3, 4);
  });
  // The build inputs gone (the material table, which rays read, kept).
  std::vector<uint8_t> table;
  if (f && f->materials) {
    const uint8_t* m = static_cast<const uint8_t*>(f->materials.contents) + f->materials_offset;
    table.assign(m, m + size_t(f->instances) * sizeof(RtMaterial));
  }
  std::memset(g.arena.buf.contents, 0xFF, g.arena.buf.length);
  if (!table.empty())
    std::memcpy(static_cast<uint8_t*>(f->materials.contents) + f->materials_offset, table.data(), table.size());
  const std::vector<Hit> h = g.Trace(f, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(0.75f, 0.75f, 0, 0, 0, 1),
                                         Ray(1.7f, 0.2f, 0, 0, 0, 1), Ray(10.5f, 1.5f, 1, 0, 0, 1),
                                         Ray(11.5f, 0.5f, 1, 0, 0, 1), Ray(2.5f, 0.5f, 0, 0, 0, 1)},
                                     true);
  int failures = 0;
  failures += Check(f && f->instances == 2 && s->Stats().total_builds == 2, "two BLASes built, two instances");
  failures += Check(Near(h[0].t, 5) && Near(h[1].t, 5) && Near(h[2].t, 5) && h[0].id == 0 && h[2].id == 0 &&
                        h[0].prim != h[1].prim && h[1].prim != h[2].prim,
                    "FLOAT3 strip (3 triangles, stride 28, offset 8, 16-bit indices): every triangle at 5");
  failures += Check(h[0].front == 1 && h[1].front == 1 && h[2].front == 1,
                    "the strip's triangles all front-facing (winding restored)");
  failures += Check(Near(h[3].t, 4) && Near(h[4].t, 4) && h[3].id == 1 && h[4].id == 1,
                    "FLOAT16_4 list (stride 20, offset 4): both triangles at 4");
  failures += Check(h[5].t == -1, "nothing outside the meshes");
  failures += Check(Near(h[0].n[2], -1) && Near(h[4].n[2], -1) && Near(h[0].n[0], 0) && Near(h[4].n[1], 0),
                    "triangle normals kept in the BLAS");
  // An indexed fan around the square's centre, clockwise from -z.
  Mesh fan = MakeMesh({{0.5f, 0.5f, 0}, {0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}, {0, 1, 2, 3, 4, 1}, 12, 0, 2, 6, 4,
                      0, 3);
  Translate(w, 20, 0, 5);
  f = g.Frame(*s, 2, [&] { s->Add(fan.key, fan.g, w, w, 3, 4); });
  const std::vector<Hit> hf = g.Trace(f, {Ray(20.5f, 0.1f, 0, 0, 0, 1), Ray(20.9f, 0.5f, 0, 0, 0, 1),
                                          Ray(20.5f, 0.9f, 0, 0, 0, 1), Ray(20.1f, 0.5f, 0, 0, 0, 1)});
  bool fan_ok = true;
  for (const Hit& x : hf) fan_ok = fan_ok && Near(x.t, 5) && x.front == 1;
  failures += Check(fan_ok && hf[0].prim != hf[1].prim, "an indexed fan: four triangles, all front-facing");
  return failures;
}

// Instances of one BLAS at different world matrices, drawn with one
// camera; one drawn with another left out; rotated and sheared; kilometres
// from the origin.
int TlasChecks(Gpu& g) {
  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  Mesh m = Square(1);
  Mat vp, vp2, wa, wb, wc, a, b, c;
  Perspective(vp, 1.0f, 1.5f, 0.5f, 5000.0f);
  Perspective(vp2, 0.7f, 1.0f, 0.1f, 100.0f);
  Translate(wa, 0, 0, 5);
  Translate(wb, 10, 0, 7, 2.0f);  // 2 x 2 at x 10..12
  Translate(wc, 20, 0, 3);
  MulRows(wa, vp, a);
  MulRows(wb, vp, b);
  MulRows(wc, vp2, c);
  const RtFrame* f = g.Frame(*s, 1, [&] {
    s->Add(m.key, m.g, wa, a, 3, 4);
    s->Add(m.key, m.g, wc, c, 3, 4);  // another camera
    s->Add(m.key, m.g, wb, b, 3, 4);
  });
  const RtStats st = s->Stats();
  int failures = 0;
  failures += Check(f && f->instances == 2 && f->blas_count == 1 && st.total_builds == 1 &&
                        st.skipped[size_t(RtSkip::OtherCamera)] == 1,
                    "one BLAS, two instances; the other camera's draw left out");
  std::vector<Hit> h = g.Trace(f, {Ray(0.25f, 0.75f, 0, 0, 0, 1), Ray(11.5f, 1.5f, 0, 0, 0, 1),
                                   Ray(12.5f, 1.5f, 0, 0, 0, 1), Ray(20.5f, 0.5f, 0, 0, 0, 1)});
  failures += Check(Near(h[0].t, 5) && h[0].id == 0, "instance 0 hit at 5 (translation in row 3)");
  failures += Check(Near(h[1].t, 7) && h[1].id == 1, "instance 1 (scaled 2) hit at 7");
  failures += Check(h[2].t == -1 && h[3].t == -1, "nothing past the scaled square, nor the left-out instance");
  const auto* info = f ? reinterpret_cast<const RtInstanceInfo*>(static_cast<const uint8_t*>(f->info.contents) +
                                                                 f->info_offset)
                       : nullptr;
  failures += Check(info && info[0].draw == 0 && info[1].draw == 2 && info[0].triangles == 2 &&
                        info[0].mesh == info[1].mesh,
                    "instance info: draw numbers, one mesh");
  bool cam = f != nullptr;
  for (int r = 0; r < 4 && cam; r++)
    for (int k = 0; k < 4; k++) {
      float v = 0;
      for (int i = 0; i < 4; i++) v += vp[r][i] * f->inv_view_proj[i * 4 + k];
      cam = cam && Near(v, r == k ? 1.0f : 0.0f, 1e-4f);
    }
  failures += Check(cam, "the frame's camera: world^-1 * wvp, inverted");

  // Rotated about x and z, sheared and scaled: a transposed 3x3 would put
  // the square elsewhere.
  Mat rs = {{0.8f, 0.5f, 0.2f, 0}, {-0.6f, 1.1f, 0.3f, 0}, {0.1f, -0.2f, 0.9f, 0}, {3, -2, 8, 1}};
  Mat rsv;
  MulRows(rs, vp, rsv);
  f = g.Frame(*s, 2, [&] { s->Add(m.key, m.g, rs, rsv, 3, 4); });
  std::vector<Query> q;
  const float in[4][2] = {{0.9f, 0.1f}, {0.1f, 0.9f}, {0.85f, 0.85f}, {0.1f, 0.1f}};
  const float out[3][2] = {{1.2f, 0.5f}, {0.5f, -0.2f}, {-0.1f, 0.95f}};
  for (const auto& p : in) {
    const auto w = Apply(rs, p[0], p[1], 0);
    q.push_back(Ray(w[0], w[1], w[2] - 10, 0, 0, 1));
  }
  for (const auto& p : out) {
    const auto w = Apply(rs, p[0], p[1], 0);
    q.push_back(Ray(w[0], w[1], w[2] - 10, 0, 0, 1));
  }
  h = g.Trace(f, q);
  bool sheared = f && f->instances == 1;
  for (int i = 0; i < 4; i++) sheared = sheared && Near(h[i].t, 10, 1e-3f);
  for (int i = 4; i < 7; i++) sheared = sheared && h[i].t == -1;
  failures += Check(sheared, "a rotated and sheared world matrix: the square where its rows put it");

  // Kilometres out: two draws 4 km from the origin, the camera near them,
  // one camera.
  Mat view, kvp, w1, w2, k1, k2;
  Translate(view, -3990, -10, -3990);
  Perspective(vp2, 1.2f, 1.78f, 0.3f, 3000.0f);
  MulRows(view, vp2, kvp);
  Mat rot = {{0.9553365f, 0.2955202f, 0, 0}, {-0.2955202f, 0.9553365f, 0, 0}, {0, 0, 1, 0}, {4000, 12, 4000, 1}};
  std::memcpy(w1, rot, sizeof(Mat));
  std::memcpy(w2, rot, sizeof(Mat));
  w2[3][0] = 4003.5f, w2[3][2] = 4001;
  MulRows(w1, kvp, k1);
  MulRows(w2, kvp, k2);
  f = g.Frame(*s, 3, [&] {
    s->Add(m.key, m.g, w1, k1, 3, 4);
    s->Add(m.key, m.g, w2, k2, 3, 4);
  });
  const auto p1 = Apply(w1, 0.5f, 0.5f, 0), p2 = Apply(w2, 0.5f, 0.5f, 0);
  h = g.Trace(f, {Ray(p1[0], p1[1], p1[2] - 2, 0, 0, 1), Ray(p2[0], p2[1], p2[2] - 2, 0, 0, 1)});
  failures += Check(f && f->instances == 2 && s->Stats().skipped[size_t(RtSkip::OtherCamera)] == 1 &&
                        Near(h[0].t, 2, 3e-3f) && h[0].id == 0 && Near(h[1].t, 2, 3e-3f) && h[1].id == 1,
                    "4 km from the origin: one camera, both instances in place");
  return failures;
}

// The frame's camera by vote: the most used, not the first; a fifth camera
// past the four told apart left out at once.
int CameraChecks(Gpu& g) {
  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  Mesh m = Square(1);
  Mat vp[5], w, wvp;
  for (int i = 0; i < 5; i++) Perspective(vp[i], 0.6f + 0.2f * float(i), 1.5f, 0.5f, 1000.0f);
  const int order[7] = {0, 1, 2, 2, 3, 4, 2};
  const RtFrame* f = g.Frame(*s, 1, [&] {
    for (int i = 0; i < 7; i++) {
      Translate(w, float(i) * 3, 0, 5);
      MulRows(w, vp[order[i]], wvp);
      s->Add(m.key, m.g, w, wvp, 3, 4);
    }
  });
  bool cam = f != nullptr;
  for (int r = 0; r < 4 && cam; r++)
    for (int k = 0; k < 4; k++) {
      float v = 0;
      for (int i = 0; i < 4; i++) v += vp[2][r][i] * f->inv_view_proj[i * 4 + k];
      cam = cam && Near(v, r == k ? 1.0f : 0.0f, 1e-4f);
    }
  const std::vector<Hit> h = g.Trace(f, {Ray(6.5f, 0.5f, 0, 0, 0, 1), Ray(18.5f, 0.5f, 0, 0, 0, 1),
                                         Ray(0.5f, 0.5f, 0, 0, 0, 1), Ray(15.5f, 0.5f, 0, 0, 0, 1)});
  return Check(cam && f->instances == 3 && s->Stats().skipped[size_t(RtSkip::OtherCamera)] == 4 && h[0].id == 0 &&
                   h[1].id == 2 && h[2].t == -1 && h[3].t == -1,
               "the camera most draws share wins (not the first); a fifth camera left out");
}

// D3D's front faces (clockwise on screen) front-facing for rays: also
// mirrored by the world matrix or the view-projection, not by reversed
// depth; D3DCULL_NONE culls nothing; D3DCULL_CW draws the other side.
int WindingChecks(Gpu& g) {
  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  // One triangle, clockwise seen from -z: its D3D front faces -z.
  Mesh m = MakeMesh({{0, 0, 0}, {0, 1, 0}, {1, 0, 0}}, {}, 12, 0, 2, 4, 1, 0, 1);
  Mat plain, mirrored, none, cw, mirrored_cw;
  Translate(plain, 0, 0, 5);
  Translate(mirrored, 20, 0, 5);
  mirrored[0][0] = -1;  // x mirrored: (20, 0) (20, 1) (19, 0), D3D's front now faces +z
  Translate(none, 40, 0, 5);
  Translate(cw, 60, 0, 5);
  Translate(mirrored_cw, 80, 0, 5);
  mirrored_cw[0][0] = -1;
  // (An orthographic view-projection, the identity: the world matrix alone
  // decides.)
  const RtFrame* f = g.Frame(*s, 1, [&] {
    s->Add(m.key, m.g, plain, plain, 3, 4);
    s->Add(m.key, m.g, mirrored, mirrored, 3, 4);
    s->Add(m.key, m.g, none, none, 1, 4);
    s->Add(m.key, m.g, cw, cw, 2, 4);
    s->Add(m.key, m.g, mirrored_cw, mirrored_cw, 2, 4);
  });
  std::vector<Hit> h = g.Trace(f, {
      Ray(0.2f, 0.2f, 0, 0, 0, 1),            // 0: plain, from its front
      Ray(0.2f, 0.2f, 10, 0, 0, -1),          // 1: plain, from behind
      Ray(0.2f, 0.2f, 10, 0, 0, -1, true),    // 2: plain, from behind, back faces culled
      Ray(19.8f, 0.2f, 10, 0, 0, -1, true),   // 3: mirrored, from its D3D front (+z)
      Ray(19.8f, 0.2f, 0, 0, 0, 1),           // 4: mirrored, from behind
      Ray(40.2f, 0.2f, 10, 0, 0, -1, true),   // 5: no culling, from behind, back faces culled
      Ray(60.2f, 0.2f, 10, 0, 0, -1, true),   // 6: CW culled (drawn from behind), from behind
      Ray(60.2f, 0.2f, 0, 0, 0, 1, true),     // 7: CW culled, from the side it does not draw
      Ray(79.8f, 0.2f, 0, 0, 0, 1, true),     // 8: mirrored and CW culled: drawn from -z again
      Ray(79.8f, 0.2f, 10, 0, 0, -1, true),   // 9: ... not from +z
  });
  int failures = 0;
  failures += Check(f && f->instances == 5, "five instances");
  failures += Check(Near(h[0].t, 5) && h[0].id == 0 && h[0].front == 1, "a clockwise D3D triangle is front-facing");
  failures += Check(Near(h[1].t, 5) && h[1].front == 0 && h[2].t == -1, "seen from behind: back-facing, culled");
  failures += Check(Near(h[3].t, 5) && h[3].id == 1 && h[3].front == 1 && h[4].front == 0,
                    "a mirrored world matrix (det < 0): front-facing from its D3D front");
  failures += Check(Near(h[5].t, 5) && h[5].id == 2, "D3DCULL_NONE: not culled from behind");
  failures += Check(Near(h[6].t, 5) && h[6].id == 3 && h[6].front == 1 && h[7].t == -1,
                    "D3DCULL_CW: the side drawn is front-facing");
  failures += Check(Near(h[8].t, 5) && h[8].id == 4 && h[8].front == 1 && h[9].t == -1,
                    "a mirrored world matrix with D3DCULL_CW: the side drawn is front-facing");

  // View-projections, each the frame's camera: perspective (plain, mirrored
  // by x negated, storing depth reversed, with a mirrored world matrix) and
  // orthographic storing depth reversed (depth 1 at z 0, 0 at z 10: drawn
  // keeping the greatest depth).
  Mat vp, w, wvp;
  bool ok = true;
  for (int kind = 0; kind < 5; kind++) {
    Translate(w, 0, 0, 5);
    if (kind == 3) w[0][0] = -1;  // (0, 0) (0, 1) (-1, 0)
    if (kind < 4) {
      Perspective(vp, 1.0f, 1.0f, 0.5f, 100.0f, kind == 2);
      if (kind == 1) vp[0][0] = -vp[0][0];
    } else {
      Translate(vp, 0, 0, 1);
      vp[2][2] = -0.1f;
    }
    const uint32_t zfunc = kind == 2 || kind == 4 ? 5 : 4;  // GREATER, LESSEQUAL
    MulRows(w, vp, wvp);
    const float x = kind == 3 ? -0.2f : 0.2f;
    f = g.Frame(*s, 2 + uint64_t(kind), [&] { s->Add(m.key, m.g, w, wvp, 3, zfunc); });
    h = g.Trace(f, {Ray(x, 0.2f, 0, 0, 0, 1), Ray(x, 0.2f, 10, 0, 0, -1)});
    // Mirrored on screen: D3D draws (as front) the side facing +z.
    const uint32_t front_from_minus_z = kind == 1 || kind == 3 ? 0 : 1;
    const bool kind_ok = f && Near(h[0].t, 5) && h[0].front == front_from_minus_z && h[1].front == 1 - front_from_minus_z;
    if (!kind_ok) std::printf("  (view-projection %d)\n", kind);
    ok = ok && kind_ok;
  }
  failures += Check(ok, "mirrored view-projections and world matrices flip the front side, reversed depth does not "
                        "(perspective and orthographic)");
  return failures;
}

// The cache: the same key not built again, a new version is and the
// old one goes at once (freed once the GPU is past it); a place rebuilt too
// often is left out until it stays the same; budgets; the memory cap;
// eviction after kEvictAfter presented frames; builds never encoded.
int CacheChecks(Gpu& g) {
  int failures = 0;
  Mat w;
  Translate(w, 0, 0, 5);
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    Mesh m = Square(1);
    const RtFrame* f = g.Frame(*s, 1, [&] { s->Add(m.key, m.g, w, w, 3, 4); });
    failures += Check(f && s->Stats().total_builds == 1 && s->Stats().cached == 1, "frame 1: built");
    f = g.Frame(*s, 2, [&] { s->Add(m.key, m.g, w, w, 3, 4); });
    std::vector<Hit> h = g.Trace(f, {Ray(0.5f, 0.5f, 0, 0, 0, 1)});
    failures += Check(f && s->Stats().total_builds == 1 && Near(h[0].t, 5),
                      "frame 2: the same key not built again, its BLAS traced");
    // New contents in the same place: the square moved to z 2 in the data;
    // the GPU (as BeginFrame is told) not done with frame 2 yet.
    Mesh m2 = MakeMesh({{0, 0, -3}, {0, 1, -3}, {1, 0, -3}, {1, 1, -3}}, {0, 1, 2, 3}, 16, 0, 2, 5, 2, 0, 1);
    m2.key.vertex_version++;
    g.arena.used = 0;
    s->BeginFrame(3, 1);
    s->Add(m2.key, m2.g, w, w, 3, 4);
    f = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, f);
    g.Commit(cb);
    h = g.Trace(f, {Ray(0.5f, 0.5f, 0, 0, 0, 1)});
    RtStats st = s->Stats();
    failures += Check(f && st.total_builds == 2 && st.cached == 1 && st.retired == 1 && Near(h[0].t, 2),
                      "a new version built, the old one out of the cache at once");
    s->BeginFrame(4, 1);
    s->EndFrame(true);
    failures += Check(s->Stats().retired == 1, "the old one kept while the GPU may use it");
    // Frame 2's TLAS (it named the old BLAS) may be traced until its slot
    // is reused: the BLAS stays until the GPU is past frame 2 + kSlots - 1.
    s->BeginFrame(5, 4);
    s->EndFrame(true);
    failures += Check(s->Stats().retired == 1, "kept while a TLAS naming it may still be traced (its slot not reused)");
    s->BeginFrame(6, 5);
    s->EndFrame(true);
    failures += Check(s->Stats().retired == 0 && s->Stats().retired_bytes == 0,
                      "freed once the GPU is past the last frame that may trace it");

    // Eviction: kEvictAfter presented frames unused (frames with no draw).
    const uint64_t mem = s->MemoryBytes();
    uint64_t serial = 7;
    for (uint64_t i = 0; i < RtScene::kEvictAfter + 40; i++, serial++) {
      s->BeginFrame(serial, serial - 1);
      s->EndFrame(true);
    }
    st = s->Stats();
    failures += Check(st.cached == 0 && st.retired == 0 && s->MemoryBytes() < mem,
                      "unused for kEvictAfter presented frames: let go");

    // A build never encoded (no EndFrame before the next BeginFrame): out of
    // the cache, built again.
    const uint64_t builds = s->Stats().total_builds;
    s->BeginFrame(serial, serial - 1);
    s->Add(m2.key, m2.g, w, w, 3, 4);
    const bool queued = s->Stats().cached == 1;
    serial++;
    f = g.Frame(*s, serial++, [&] { s->Add(m2.key, m2.g, w, w, 3, 4); });
    h = g.Trace(f, {Ray(0.5f, 0.5f, 0, 0, 0, 1)});
    failures += Check(queued && s->Stats().cached == 1 && s->Stats().total_builds == builds + 2 && f &&
                          f->instances == 1 && Near(h[0].t, 2),
                      "a build never encoded: dropped, built again");
  }
  {
    // A place rebuilt with new contents three times in the window: left
    // out, until the same contents have stayed calm_frames.
    RtSceneOptions o;
    o.calm_frames = 5;
    auto s = g.Scene(o);
    Mesh m = Square(1);
    uint64_t serial = 1;
    for (int i = 0; i < 4; i++) {
      g.Frame(*s, serial++, [&] { s->Add(m.key, m.g, w, w, 3, 4); });
      m.key.vertex_version++;
    }
    m.key.vertex_version--;
    RtStats st = s->Stats();
    const bool left_out = st.skipped[size_t(RtSkip::Rebuilt)] == 1 && st.total_builds == 3 && st.cached == 0;
    int frames_out = 0;
    for (int i = 0; i < 10; i++) {
      const RtFrame* f = g.Frame(*s, serial++, [&] { s->Add(m.key, m.g, w, w, 3, 4); });
      if (!f) frames_out++;
      else break;
    }
    st = s->Stats();
    failures += Check(left_out && frames_out >= 3 && frames_out <= 5 && st.cached == 1 && st.total_builds == 4,
                      "a place rebuilt too often left out, then built once its contents stay");
  }
  {
    // Budgets: kBuildsPerFrame builds a frame, and triangles.
    RtSceneOptions o;
    o.build_triangles_per_frame = 3;
    o.max_mesh_triangles = 2;
    auto s = g.Scene();
    auto t = g.Scene(o);
    std::vector<Mesh> ms;
    for (uint32_t i = 0; i < RtScene::kBuildsPerFrame + 6; i++) ms.push_back(Square(100 + i));
    uint32_t added = 0;
    const RtFrame* f = g.Frame(*s, 1, [&] {
      for (const Mesh& m : ms) added += s->Add(m.key, m.g, w, w, 3, 4) ? 1 : 0;
    });
    RtStats st = s->Stats();
    failures += Check(f && added == RtScene::kBuildsPerFrame && st.total_builds == RtScene::kBuildsPerFrame &&
                          st.skipped[size_t(RtSkip::BuildBudget)] == 6 && f->instances == added,
                      "a frame builds kBuildsPerFrame, the rest wait");
    added = 0;
    f = g.Frame(*s, 2, [&] {
      for (const Mesh& m : ms) added += s->Add(m.key, m.g, w, w, 3, 4) ? 1 : 0;
    });
    failures += Check(f && added == ms.size() && s->Stats().total_builds == ms.size(), "the next frame builds the rest");
    Mesh three = MakeMesh(Pos(5, {0, 0, 0}), {}, 12, 0, 2, 4, 1, 0, 7);
    three.g.primitives = three.key.primitives = 3;
    three.vb.resize(9 * 12);
    three.g.positions = three.vb.data();
    three.g.size = three.vb.size();
    g.Frame(*t, 1, [&] {
      t->Add(ms[0].key, ms[0].g, w, w, 3, 4);  // 2 triangles
      t->Add(ms[1].key, ms[1].g, w, w, 3, 4);  // 2 more: past 3
      t->Add(three.key, three.g, w, w, 3, 4);  // past max_mesh_triangles
    });
    st = t->Stats();
    failures += Check(st.total_builds == 1 && st.skipped[size_t(RtSkip::BuildBudget)] == 1 &&
                          st.skipped[size_t(RtSkip::TooLarge)] == 1,
                      "the triangle budget and the mesh size limit");
    // The time budget: with next to none, a frame's first build goes and
    // the others wait, one more each frame after.
    RtSceneOptions ot;
    ot.build_ms_per_frame = 1e-6f;
    auto u = g.Scene(ot);
    added = 0;
    g.Frame(*u, 1, [&] {
      for (uint32_t i = 0; i < 4; i++) added += u->Add(ms[i].key, ms[i].g, w, w, 3, 4) ? 1 : 0;
    });
    st = u->Stats();
    failures += Check(added == 1 && st.total_builds == 1 && st.skipped[size_t(RtSkip::BuildBudget)] == 3,
                      "past the time budget a frame's other builds wait");
    added = 0;
    g.Frame(*u, 2, [&] {
      for (uint32_t i = 0; i < 4; i++) added += u->Add(ms[i].key, ms[i].g, w, w, 3, 4) ? 1 : 0;
    });
    failures += Check(added == 2 && u->Stats().total_builds == 2, "the next frame builds one more of them");
  }
  {
    // The memory cap: past it the least recently used go at the end of a
    // frame (not those of the last two), and nothing is built.
    uint64_t one = 0;
    {
      auto p = g.Scene();
      Mesh m = Square(1);
      g.Frame(*p, 1, [&] { p->Add(m.key, m.g, w, w, 3, 4); });
      one = p->Stats().blas_bytes;
    }
    RtSceneOptions o;
    o.max_blas_bytes = one * 3;
    auto s = g.Scene(o);
    std::vector<Mesh> ms;
    for (int i = 0; i < 12; i++) ms.push_back(Square(200 + uint64_t(i)));
    bool ok = one > 0;
    for (uint64_t i = 0; i < 8; i++) {
      g.Frame(*s, i + 1, [&] { s->Add(ms[i].key, ms[i].g, w, w, 3, 4); });
      ok = ok && s->Stats().blas_bytes <= o.max_blas_bytes;
    }
    const RtStats st = s->Stats();
    ok = ok && st.cached >= 2 && st.cached <= 3 && st.total_builds == 8;
    g.Frame(*s, 9, [&] {
      for (int i = 8; i < 12; i++) s->Add(ms[i].key, ms[i].g, w, w, 3, 4);
    });
    ok = ok && s->Stats().skipped[size_t(RtSkip::NoMemory)] >= 1 && s->MemoryBytes() >= s->Stats().blas_bytes;
    failures += Check(ok, "BLAS memory capped: least recently used let go, builds past it refused");
  }
  {
    // No memory for a build: left out, counted.
    auto s = g.Scene();
    s->SetAllocator(NoAlloc, nullptr);
    Mesh m = Square(1);
    s->BeginFrame(1, 0);
    const bool added = s->Add(m.key, m.g, w, w, 3, 4);
    const RtFrame* f = s->EndFrame(true);
    failures += Check(!added && !f && s->Stats().skipped[size_t(RtSkip::NoMemory)] == 1,
                      "no memory for a build: left out");
    s->BeginFrame(2, 1);
    s->EndFrame(true);
    failures += Check(!s->Add(m.key, m.g, w, w, 3, 4), "no draw taken after EndFrame");
  }
  return failures;
}

// A readback ends a frame before Present: only its builds are encoded;
// its instances go into the presented frame's TLAS. Snapshots: the
// instances so far and those the last frame added after its snapshot.
int FrameChecks(Gpu& g) {
  int failures = 0;
  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  Mesh m = Square(1), n = Square(2);
  Mat wa, wb;
  Translate(wa, 0, 0, 5);
  Translate(wb, 10, 0, 5);
  g.arena.used = 0;
  s->BeginFrame(1, 0);
  s->Add(m.key, m.g, wa, wa, 3, 4);
  const RtFrame* split = s->EndFrame(false);
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  s->EncodeFrame(cb, split);
  g.Commit(cb);
  const bool builds_only = split && !split->tlas;
  const RtFrame* f = g.Frame(*s, 2, [&] { s->Add(n.key, n.g, wb, wb, 3, 4); });
  std::vector<Hit> h = g.Trace(f, {Ray(0.5f, 0.5f, 0, 0, 0, 1), Ray(10.5f, 0.5f, 0, 0, 0, 1)});
  failures += Check(builds_only && f && f->instances == 2 && f->blas_count == 2 && s->Stats().frames == 1 &&
                        Near(h[0].t, 5) && Near(h[1].t, 5),
                    "a frame split by a readback: one TLAS of both parts");

  // Snapshots: frame 3 draws A, snapshots, draws B; frame 4 draws A moved
  // to x 20, snapshots (A at 20, B where frame 3 had it), draws B moved to
  // x 30 (the end-of-frame TLAS: A at 20, B at 30).
  Mat wa2, wb2;
  Translate(wa2, 20, 0, 5);
  Translate(wb2, 30, 0, 5);
  const RtFrame* snap[2] = {};
  const RtFrame* end[2] = {};
  for (int i = 0; i < 2; i++) {
    g.arena.used = 0;
    s->BeginFrame(3 + uint64_t(i), 2 + uint64_t(i));
    s->Add(m.key, m.g, i ? wa2 : wa, i ? wa2 : wa, 3, 4);
    snap[i] = s->Snapshot();
    s->Add(n.key, n.g, i ? wb2 : wb, i ? wb2 : wb, 3, 4);
    end[i] = s->EndFrame(true);
    cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, snap[i]);
    s->EncodeFrame(cb, end[i]);
    g.Commit(cb);
    if (i == 0) {
      h = g.Trace(snap[0], {Ray(0.5f, 0.5f, 0, 0, 0, 1), Ray(10.5f, 0.5f, 0, 0, 0, 1)});
      failures += Check(snap[0] && snap[0]->instances == 1 && Near(h[0].t, 5) && h[1].t == -1,
                        "a first snapshot: the instances so far");
    }
  }
  h = g.Trace(snap[1], {Ray(20.5f, 0.5f, 0, 0, 0, 1), Ray(10.5f, 0.5f, 0, 0, 0, 1), Ray(0.5f, 0.5f, 0, 0, 0, 1),
                        Ray(30.5f, 0.5f, 0, 0, 0, 1)});
  failures += Check(snap[1] && snap[1]->instances == 2 && Near(h[0].t, 5) && Near(h[1].t, 5) && h[2].t == -1 &&
                        h[3].t == -1,
                    "a snapshot: this frame's instances so far, the last frame's after its snapshot (one frame old)");
  h = g.Trace(end[1], {Ray(20.5f, 0.5f, 0, 0, 0, 1), Ray(30.5f, 0.5f, 0, 0, 0, 1), Ray(10.5f, 0.5f, 0, 0, 0, 1)});
  failures += Check(end[1] && end[1]->instances == 2 && Near(h[0].t, 5) && Near(h[1].t, 5) && h[2].t == -1,
                    "the end of the frame: every instance at its place now");

  // New contents in a place whose old BLAS the last frame's carried
  // instances name: the snapshot still traces the old one, freed only once
  // the GPU is past the last frame that may trace that snapshot.
  Mesh n2 = MakeMesh({{0, 0, -3}, {0, 1, -3}, {1, 0, -3}, {1, 1, -3}}, {0, 1, 2, 3}, 16, 0, 2, 5, 2, 0, 2);
  n2.key.vertex_version++;
  Mat wb3;
  Translate(wb3, 40, 0, 5);
  g.arena.used = 0;
  s->BeginFrame(5, 4);
  s->Add(n2.key, n2.g, wb3, wb3, 3, 4);
  const bool retired_now = s->Stats().retired == 1;
  const RtFrame* sn = s->Snapshot();
  const RtFrame* en = s->EndFrame(true);
  cb = [g.queue commandBuffer];
  s->EncodeFrame(cb, sn);
  s->EncodeFrame(cb, en);
  g.Commit(cb);
  h = g.Trace(sn, {Ray(30.5f, 0.5f, 0, 0, 0, 1), Ray(40.5f, 0.5f, 0, 0, 0, 1)});
  bool ok = retired_now && sn && sn->instances == 2 && Near(h[0].t, 5) && Near(h[1].t, 2);
  for (uint64_t serial = 6; serial <= 8; serial++) {
    s->BeginFrame(serial, serial - 1);
    s->EndFrame(true);
    ok = ok && s->Stats().retired == 1;
  }
  s->BeginFrame(9, 8);
  s->EndFrame(true);
  failures += Check(ok && s->Stats().retired == 0,
                    "a retired BLAS the carried instances name: in the snapshot, kept until its slot horizon");
  return failures;
}

// Frame slots reused while earlier frames are in flight: each frame's TLAS
// traced in its own command buffer, all committed before any is waited for.
int InFlightChecks(Gpu& g) {
  auto s = g.Scene();
  if (!s) return Check(false, "scene made");
  Mesh m = Square(1);
  constexpr int kFrames = 10;
  id<MTLCommandBuffer> cbs[kFrames + 1] = {};
  id<MTLBuffer> hits[kFrames + 1] = {};
  uint32_t instances[kFrames + 1] = {};
  for (int i = 1; i <= kFrames; i++) {
    // The frame kSlots back done (its slot reused): what BeginFrame is told.
    if (i > RtScene::kSlots) [cbs[i - RtScene::kSlots] waitUntilCompleted];
    Arena& a = g.slot_arena[i % RtScene::kSlots];
    a.used = 0;
    s->SetAllocator(ArenaAlloc, &a);
    s->BeginFrame(uint64_t(i), uint64_t(std::max(0, i - RtScene::kSlots)));
    Mat w;
    Translate(w, float(i) * 3, 0, 5);
    s->Add(m.key, m.g, w, w, 3, 4);
    const RtFrame* f = s->EndFrame(true);
    instances[i] = f ? f->instances : 0;
    cbs[i] = [g.queue commandBuffer];
    s->EncodeFrame(cbs[i], f);
    if (f) hits[i] = g.EncodeTrace(cbs[i], f, {Ray(float(i) * 3 + 0.5f, 0.5f, 0, 0, 0, 1),
                                               Ray(float(i - 1) * 3 + 0.5f, 0.5f, 0, 0, 0, 1)}, false);
    [cbs[i] commit];
  }
  bool ok = true;
  for (int i = 1; i <= kFrames; i++) {
    [cbs[i] waitUntilCompleted];
    const Hit* h = hits[i] ? static_cast<const Hit*>(hits[i].contents) : nullptr;
    ok = ok && cbs[i].status == MTLCommandBufferStatusCompleted && h && instances[i] == 1 && Near(h[0].t, 5) &&
         h[1].t == -1;
  }
  s->SetAllocator(ArenaAlloc, &g.arena);
  return Check(ok, "frame slots reused with earlier frames in flight: each frame's TLAS its own");
}

// The debug view: an orthographic camera (the identity) and a perspective
// one, a square over the left half of the view; the right half darkened.
int DebugChecks(Gpu& g) {
  int failures = 0;
  for (int kind = 0; kind < 3; kind++) {
    const RtDebug mode = kind == 1 ? RtDebug::Normals : RtDebug::Instances;
    auto s = g.Scene(mode);
    if (!s) return failures + Check(false, "scene made");
    Mesh m = Square(1);
    Mat w, vp, wvp;
    if (kind < 2) {
      Translate(w, -1.01f, -1, 0.5f);
      w[1][1] = 2;  // x -1.01..-0.01, y -1..1 (no pixel centre on its diagonal)
      std::memcpy(wvp, w, sizeof(Mat));
    } else {
      // 90 degrees across a 2:1 view, the square at z 5, far wider than the
      // view to the left of x -0.01.
      Translate(w, -100.01f, -50, 5);
      w[0][0] = w[1][1] = 100;
      Perspective(vp, 1.5707964f, 2.0f, 0.5f, 100.0f, true);  // reversed depth: the camera from w = 0
      MulRows(w, vp, wvp);
    }
    const RtFrame* f = g.Frame(*s, 1, [&] { s->Add(m.key, m.g, w, wvp, 3, 4); });
    constexpr int kW = 16, kH = 8;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:kW
                                                                                 height:kH
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> target = [g.dev newTextureWithDescriptor:td];
    std::vector<uint32_t> px(kW * kH, 0xFFFFFFFFu);
    [target replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:px.data() bytesPerRow:kW * 4];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeDebug(cb, f, target);
    bool ok = f && g.Commit(cb);
    [target getBytes:px.data() bytesPerRow:kW * 4 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
    for (int y = 0; y < kH && ok; y++)
      for (int x = 0; x < kW; x++) {
        const uint32_t p = px[y * kW + x];
        const uint32_t b = p & 255, gr = (p >> 8) & 255, r = (p >> 16) & 255, a = p >> 24;
        const bool dark = a == 255 && r >= 75 && r <= 78 && gr == r && b == r;
        if (x >= kW / 2) {
          ok = ok && dark;
        } else if (mode == RtDebug::Normals) {
          // Normal (0, 0, -1) toward the camera: (0.5, 0.5, 0) as a colour.
          ok = ok && a == 255 && r >= 126 && r <= 129 && gr >= 126 && gr <= 129 && b <= 1;
        } else {
          ok = ok && a == 255 && !dark && p != 0xFFFFFFFFu;
        }
      }
    const char* what[3] = {"debug view: instance colours where rays hit, darkened elsewhere",
                           "debug view: normals where rays hit, darkened elsewhere",
                           "debug view, a perspective camera storing depth reversed: rays from its position"};
    failures += Check(ok, what[kind]);
  }
  // Orthographic, depth reversed (1 at z 0, 0 at z 10, drawn keeping the
  // greatest): rays from depth 1 (z 0) meet the square over the left half
  // at z 5 before the one over the whole view at z 8; from depth 0 they
  // would meet the far one first everywhere.
  {
    auto s = g.Scene(RtDebug::Instances);
    Mesh m = Square(1);
    Mat vp, w1, w2, v1, v2;
    Translate(vp, 0, 0, 1);
    vp[2][2] = -0.1f;
    Translate(w1, -1.04f, -1, 5);
    w1[1][1] = 2;
    Translate(w2, -1.5f, -1.5f, 8);
    w2[0][0] = w2[1][1] = 3;
    MulRows(w1, vp, v1);
    MulRows(w2, vp, v2);
    const RtFrame* f = g.Frame(*s, 1, [&] {
      s->Add(m.key, m.g, w1, v1, 1, 5);
      s->Add(m.key, m.g, w2, v2, 1, 5);
    });
    constexpr int kW = 16, kH = 8;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:kW
                                                                                 height:kH
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> target = [g.dev newTextureWithDescriptor:td];
    std::vector<uint32_t> px(kW * kH, 0xFFFFFFFFu);
    [target replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:px.data() bytesPerRow:kW * 4];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeDebug(cb, f, target);
    bool ok = s && f && f->depth_reversed && g.Commit(cb);
    [target getBytes:px.data() bytesPerRow:kW * 4 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
    // Pixels away from the triangles' edges (the view darkens those): two
    // of each square, one in each of its triangles; no pixel without a hit.
    const uint32_t left = px[3 * kW + 2], right = px[3 * kW + 10];
    ok = ok && px[5 * kW + 5] == left && px[6 * kW + 13] == right && left != right && (left >> 24) == 255;
    for (uint32_t p : px) {
      const uint32_t r = (p >> 16) & 255;
      ok = ok && (p >> 24) == 255 && !(r >= 75 && r <= 78 && ((p >> 8) & 255) == r && (p & 255) == r);
    }
    failures += Check(ok,
                      "debug view, an orthographic camera storing depth reversed: rays from depth 1");
  }
  return failures;
}


// ---- alpha-cut geometry

// The unit square (kSquare's strip) in buffer `buffer`, 24-byte vertices:
// the position (FLOAT3) at 0, UV = xy * uv_scale (FLOAT2) at 12, a
// D3DCOLOR at 20 whose alpha is alpha[vertex]; key and geometry alpha-cut.
struct CutMesh {
  std::vector<uint8_t> vb;
  std::vector<uint16_t> ib = {0, 1, 2, 3};
  RtGeometry g;
  RtMeshKey key;
};
CutMesh CutSquare(uint64_t buffer, float uv_scale, const uint8_t alpha[4]) {
  CutMesh m;
  m.vb.resize(4 * 24);
  for (int i = 0; i < 4; i++) {
    uint8_t* v = m.vb.data() + i * 24;
    std::memcpy(v, kSquare[size_t(i)].data(), 12);
    const float uv[2] = {kSquare[size_t(i)][0] * uv_scale, kSquare[size_t(i)][1] * uv_scale};
    std::memcpy(v + 12, uv, 8);
    const uint32_t c = uint32_t(alpha[i]) << 24 | 0x00FFFFFFu;
    std::memcpy(v + 20, &c, 4);
  }
  RtGeometry& g = m.g;
  g.positions = m.vb.data();
  g.size = m.vb.size();
  g.stride = 24;
  g.type = 2;
  g.prim = 5;
  g.primitives = 2;
  g.indices = m.ib.data();
  g.index_size = 2;
  g.uv = {m.vb.data() + 12, m.vb.size() - 12, 24, 1, {0, 1}};
  g.alpha = {m.vb.data() + 20, m.vb.size() - 20, 24, 4, {3, 0}};
  RtMeshKey& k = m.key;
  k.vertex_version = buffer * 16 + 1;
  k.index_version = buffer * 16 + 2;
  k.vertex_buffer = buffer;
  k.index_buffer = buffer + (1u << 20);
  k.prim = 5;
  k.primitives = 2;
  k.stride = 24;
  k.type = 2;
  k.index_size = 2;
  k.cut = 1;
  k.attr_version = k.vertex_version;
  k.attr_buffer = buffer;
  k.attr_stride = 24;
  k.uv_offset = 12;
  k.uv_type = 1;
  k.uv_lanes = 0 | 1 << 2;
  k.alpha_offset = 20;
  k.alpha_type = 4;
  k.alpha_lane = 3;
  return m;
}

// An RGBA8 texture, w x h, texels as uint32 (R in the low byte), levels
// given one after the other; `x8`: alpha read as 1 (a swizzle, as the
// backend makes X8R8G8B8 textures).
id<MTLTexture> RgbaTexture(Gpu& g, int w, int h, const std::vector<uint32_t>& texels, int levels = 1, bool x8 = false) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:NSUInteger(w)
                                                                               height:NSUInteger(h)
                                                                            mipmapped:levels > 1];
  td.mipmapLevelCount = NSUInteger(levels);
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  if (x8)
    td.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue,
                                               MTLTextureSwizzleOne);
  id<MTLTexture> t = [g.dev newTextureWithDescriptor:td];
  size_t at = 0;
  for (int l = 0; l < levels; l++) {
    const int lw = std::max(w >> l, 1), lh = std::max(h >> l, 1);
    [t replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(lw), NSUInteger(lh))
         mipmapLevel:NSUInteger(l)
           withBytes:texels.data() + at
         bytesPerRow:NSUInteger(lw) * 4];
    at += size_t(lw) * size_t(lh);
  }
  return t;
}

// A material: the texture's alpha (nil: none) against `ref` by D3DCMP
// `func`, D3DTADDRESS `address` both ways, nearest; more flag bits in
// `bits`.
RtCut Cut(id<MTLTexture> t, uint32_t func, uint32_t ref, uint32_t address = 1, uint32_t bits = 0) {
  RtCut c;
  c.texture = t;
  c.material.alpha_scale = 1.0f;
  c.material.flags = RtMaterialFlags(bits, func, ref, 3, address, address);
  return c;
}

constexpr uint32_t kSolid = 0xFF808080u, kHole = 0x00808080u;
const uint8_t kOpaqueVerts[4] = {255, 255, 255, 255};

int AlphaCpuChecks() {
  int failures = 0;
  // Inputs as D3D9 fills the register.
  {
    const uint8_t color[4] = {0x10, 0x20, 0x30, 0x40};  // B, G, R, A
    const uint16_t half[4] = {0x3C00, 0xC000, 0x3800, 0x0000};
    const int16_t shorts[4] = {32767, -32768, 16384, 0};
    float o[4];
    bool ok = true;
    RtFetch(color, 4, o);
    ok = ok && Near(o[0], 0x30 / 255.0f) && Near(o[2], 0x10 / 255.0f) && Near(o[3], 0x40 / 255.0f);
    RtFetch(reinterpret_cast<const uint8_t*>(half), 15, o);
    ok = ok && o[0] == 1.0f && o[1] == -2.0f && o[2] == 0.0f && o[3] == 1.0f;
    RtFetch(reinterpret_cast<const uint8_t*>(shorts), 10, o);
    ok = ok && o[0] == 1.0f && o[1] == -1.0f && Near(o[2], 16384.0f / 32767.0f) && o[3] == 0.0f;
    const float f2[2] = {0.25f, 0.75f};
    RtFetch(reinterpret_cast<const uint8_t*>(f2), 1, o);
    ok = ok && o[0] == 0.25f && o[1] == 0.75f && o[2] == 0.0f && o[3] == 1.0f;
    RtAttribute a{reinterpret_cast<const uint8_t*>(f2), 8, 8, 1, {1, 0}};
    float x = 0;
    ok = ok && RtReadLane(a, 0, 1, &x) && x == 0.75f && !RtReadLane(a, 1, 0, &x) && RtAttributeBytes(16) == 8 &&
         RtAttributeBytes(17) == 0;
    a.stride = 0;  // one value for every vertex
    ok = ok && RtReadLane(a, 1000, 0, &x) && x == 0.25f;
    failures += Check(ok, "vertex inputs read as D3D9 fills the register (D3DCOLOR, FLOAT16_2, SHORT4N, FLOAT2)");
  }
  // A triangle's primitive data: UVs and alpha bytes per corner.
  {
    const uint8_t va[4] = {0, 128, 255, 255};
    CutMesh m = CutSquare(1, 2.0f, va);
    const uint32_t tri[3] = {0, 1, 2};
    RtCutPrim p;
    bool ok = RtCutTriangle(m.g, 0, tri, &p) && p.uv[1][1] == 2.0f && p.uv[2][0] == 2.0f && p.uv[0][0] == 0.0f &&
              p.alpha == (0u | 128u << 8 | 255u << 16);
    const uint32_t past[3] = {0, 1, 4};
    ok = ok && !RtCutTriangle(m.g, 0, past, &p);
    RtGeometry none = m.g;
    none.uv = {};
    none.alpha = {};
    ok = ok && RtCutTriangle(none, 0, tri, &p) && p.alpha == 0xFFFFFFu && p.uv[2][0] == 0.0f;
    failures += Check(ok, "alpha-cut primitive data: corner UVs and alpha bytes; outside the stream refused");
  }
  // The tests on the CPU (constant alphas).
  {
    RtMaterial m{};
    m.alpha_scale = 1;
    m.flags = RtMaterialFlags(0, 7, 128, 3, 1, 1);  // GREATEREQUAL 128
    bool ok = RtAlphaPasses(m, 128.0f / 255.0f) && !RtAlphaPasses(m, 127.0f / 255.0f);
    m.flags = RtMaterialFlags(0, 1, 0, 3, 1, 1);  // NEVER
    ok = ok && !RtAlphaPasses(m, 1.0f);
    m.flags = RtMaterialFlags(kRtMatKill, 0, 0, 3, 1, 1);
    m.kill_scale = 1;
    m.kill_offset = -0.5f;
    ok = ok && RtAlphaPasses(m, 0.5f) && !RtAlphaPasses(m, 0.49f);
    failures += Check(ok, "alpha tests on the CPU: GREATEREQUAL at 8 bits, NEVER, a texkill threshold");
  }
  // Alpha-cut keys: their UV and alpha inputs tell them apart.
  {
    CutMesh m = CutSquare(1, 1.0f, kOpaqueVerts);
    RtMeshKey a = m.key, v[6];
    for (RtMeshKey& k : v) k = a;
    v[0].attr_version++, v[1].attr_buffer++, v[2].uv_offset += 4, v[3].uv_type = 15, v[4].alpha_lane = 2, v[5].cut = 0;
    bool ok = a.Place().attr_version == 0 && v[0].Place() == a.Place();
    for (const RtMeshKey& k : v) ok = ok && !(k == a) && RtMeshKeyHash{}(k) != RtMeshKeyHash{}(a);
    failures += Check(ok, "alpha-cut keys: the UVs' and alpha's stream and inputs in the key, the version not in the place");
  }
  return failures;
}


// Where a draw's alpha comes from (RtResolveAlpha): a draw of 24-byte
// vertices, POSITION (slot 0), TEXCOORD0 (slot 1, FLOAT2 at 12) and
// COLOR0 (slot 2, D3DCOLOR at 20, or in stream 1).
int ResolveChecks() {
  int failures = 0;
  std::vector<uint8_t> vb(4 * 24, 0), vb1(4 * 4, 0);
  VertexInput in[3] = {{0, 0, 2}, {0, 12, 1}, {0, 20, 4}};
  uint32_t ss[16][14] = {};
  float c[224][4] = {};
  ShaderDraw d;
  d.inputs = in;
  d.input_count = 3;
  d.streams[0] = {vb.data(), uint32_t(vb.size()), 24};
  d.streams[1] = {vb1.data(), uint32_t(vb1.size()), 4};
  d.ps_float = c;
  RtAlphaTexture tex[16];
  for (RtAlphaTexture& t : tex) t = {true, true, false};
  const std::vector<hlsl::MovedLane> moved = {{{hlsl::kTexcoord, 0}, 0, 1, 0},
                                              {{hlsl::kTexcoord, 0}, 1, 1, 1},
                                              {{hlsl::kColor, 0}, 3, 2, 3}};
  auto shaders = [&](const hlsl::PixelAlpha* pa, bool discards = false) {
    RtAlphaShaders sh;
    sh.pixel = pa;
    sh.discards = discards;
    sh.sampler_type.fill(hlsl::ParamType::Sampler2D);
    sh.moved = &moved;
    sh.tex_slot[0] = 1;
    sh.color0_slot = 2;
    return sh;
  };
  // The bytecode's value: s1's alpha at TEXCOORD0 times COLOR0.a times
  // c5.y (0.5), less 0.25.
  hlsl::PixelAlpha known;
  hlsl::AlphaValue& v = known.color0;
  v.known = true;
  v.sampler = 1;
  v.texcoord = 0;
  v.color = 0;
  v.constant = 5;
  v.constant_lane = 1;
  v.scale = 1.0f;
  v.offset = -0.25f;
  c[5][1] = 0.5f;
  const hlsl::PixelAlpha unknown;
  {
    const RtAlphaShaders sh = shaders(&known);
    RtAlphaResult r = RtResolveAlpha(d, sh, tex, false, 8, 0);
    bool ok = r.kind == RtAlphaKind::Opaque;
    r = RtResolveAlpha(d, sh, tex, true, 1, 0);
    ok = ok && r.kind == RtAlphaKind::Skip;
    failures += Check(ok, "resolve: nothing discarded opaque; ALPHAFUNC NEVER left out");
    r = RtResolveAlpha(d, sh, tex, true, 7, 0x180);  // GREATEREQUAL 128 (ALPHAREF's low byte)
    const RtMaterial& m = r.material;
    ok = r.kind == RtAlphaKind::Cut && !r.generic && r.sampler == 1 && r.uv.data == vb.data() + 12 && r.uv.type == 1 &&
         r.uv.lane[0] == 0 && r.uv.lane[1] == 1 && r.alpha.data == vb.data() + 20 && r.alpha.lane[0] == 3 &&
         r.stream == 0 && m.alpha_scale == 0.5f && m.alpha_offset == -0.25f &&
         m.flags == RtMaterialFlags(kRtMatVertexAlpha, 7, 0x80, 3, 1, 1);
    failures += Check(ok, "resolve: the bytecode's texel, UVs and colour back to inputs, constants from the registers, "
                          "D3D9's default sampler state");
    RtAlphaShaders hook = sh;
    hook.hook = true;
    hook.hook_sampler = 3;
    hook.hook_texcoord = 0;
    r = RtResolveAlpha(d, hook, tex, true, 7, 128);
    failures += Check(r.kind == RtAlphaKind::Cut && r.sampler == 3 && r.material.alpha_scale == 1.0f && !r.generic,
                      "resolve: the game's hook before the bytecode");
    r = RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 128);
    failures += Check(r.kind == RtAlphaKind::Cut && r.generic && r.sampler == 0 && r.uv.data == vb.data() + 12 &&
                          r.alpha.data == vb.data() + 20,
                      "resolve: nothing known: sampler 0 at TEXCOORD0 times COLOR0's alpha");
  }
  // Known but not followed back (UVs the vertex shader computes): left out,
  // not guessed.
  {
    hlsl::PixelAlpha tiled = known;
    tiled.color0.texcoord = 5;
    failures += Check(RtResolveAlpha(d, shaders(&tiled), tex, true, 7, 128).kind == RtAlphaKind::Skip,
                      "resolve: a known alpha whose UVs are not moved from an input: left out");
  }
  // UVs and colour in different streams; a texture missing or a cube.
  {
    in[2].stream = 1;
    in[2].offset = 0;
    bool ok = RtResolveAlpha(d, shaders(&known), tex, true, 7, 128).kind == RtAlphaKind::Skip;
    in[2] = {0, 20, 4};
    tex[1].present = false;
    ok = ok && RtResolveAlpha(d, shaders(&known), tex, true, 7, 128).kind == RtAlphaKind::Skip;
    tex[1] = {true, false, false};
    ok = ok && RtResolveAlpha(d, shaders(&known), tex, true, 7, 128).kind == RtAlphaKind::Skip;
    tex[1] = {true, true, false};
    RtAlphaShaders cube = shaders(&known);
    cube.sampler_type[1] = hlsl::ParamType::SamplerCube;
    ok = ok && RtResolveAlpha(d, cube, tex, true, 7, 128).kind == RtAlphaKind::Skip;
    failures += Check(ok, "resolve: UVs and colour in two streams, no texture, a cube: left out");
  }
  // An opaque (X8R8G8B8) texture: no texture read, the vertex alpha alone;
  // without a colour input, the alpha 1 everywhere: opaque.
  {
    tex[0].opaque = true;
    RtAlphaResult r = RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 128);
    bool ok = r.kind == RtAlphaKind::Cut && r.sampler < 0 && !r.uv.data && r.alpha.data && r.stream == 0;
    RtAlphaShaders nocolor = shaders(&unknown);
    nocolor.color0_slot = -1;
    r = RtResolveAlpha(d, nocolor, tex, true, 5, 128);
    ok = ok && r.kind == RtAlphaKind::Opaque;
    tex[0].opaque = false;
    failures += Check(ok, "resolve: an X8R8G8B8 texture's alpha read as 1 (no texture); alpha 1 everywhere: opaque");
  }
  // A texture whose least alpha passes the test (alpha_min): its alpha
  // alone opaque; with the vertex alpha, or below the reference, still cut.
  {
    RtAlphaShaders nocolor = shaders(&unknown);
    nocolor.color0_slot = -1;
    tex[0].alpha_min = 200.0f / 255.0f;
    bool ok = RtResolveAlpha(d, nocolor, tex, true, 7, 128).kind == RtAlphaKind::Opaque;  // GREATEREQUAL 128
    ok = ok && RtResolveAlpha(d, nocolor, tex, true, 5, 200).kind == RtAlphaKind::Cut;    // GREATER 200
    ok = ok && RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 128).kind == RtAlphaKind::Cut;  // times COLOR0.a
    tex[0].alpha_min = 0.0f;
    ok = ok && RtResolveAlpha(d, nocolor, tex, true, 7, 128).kind == RtAlphaKind::Cut;
    failures += Check(ok, "resolve: a texture's least alpha passing the test: opaque; with vertex alpha or below: cut");
  }
  // MinAlpha: DXT1 four-colour and three-colour blocks (index 3 transparent
  // only where used, and only inside the image), DXT3, DXT5, A8R8G8B8.
  {
    const uint8_t four[8] = {0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF};      // c0 > c1, index 3: opaque
    const uint8_t three[8] = {0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xC0};    // c0 <= c1, texel 15 index 3
    bool ok = MinAlpha(four, 8, Format::DXT1, 4, 4) == 255 && MinAlpha(three, 8, Format::DXT1, 4, 4) == 0 &&
              MinAlpha(three, 8, Format::DXT1, 2, 2) == 255;  // texel 15 outside a 2x2 level
    uint8_t dxt3[16] = {};
    for (int i = 0; i < 8; i++) dxt3[i] = 0xFF;
    dxt3[3] = 0x8F;  // texel 7: 8 * 17 = 136
    ok = ok && MinAlpha(dxt3, 16, Format::DXT3, 4, 4) == 136;
    uint8_t dxt5[16] = {250, 240};  // all indices 0: 250
    ok = ok && MinAlpha(dxt5, 16, Format::DXT5, 4, 4) == 250;
    dxt5[2] = 0x01;  // texel 0 index 1: 240
    ok = ok && MinAlpha(dxt5, 16, Format::DXT5, 4, 4) == 240;
    const uint8_t argb[8] = {0, 0, 0, 255, 0, 0, 0, 7};
    ok = ok && MinAlpha(argb, 8, Format::A8R8G8B8, 2, 1) == 7 && MinAlpha(argb, 8, Format::X8R8G8B8, 2, 1) == 255 &&
         MinAlpha(argb, 8, Format::R32F, 2, 1) == 0;
    failures += Check(ok, "MinAlpha: DXT1 transparent texels only where used and inside, DXT3, DXT5, A8R8G8B8");
  }
  // The sampler state: ADDRESSU MIRROR, ADDRESSV BORDER, an opaque border,
  // MAGFILTER linear, MAXMIPLEVEL 3.
  {
    ss[0][1] = 2, ss[0][2] = 4, ss[0][4] = 0xFF000000u, ss[0][5] = 2, ss[0][9] = 3;
    d.sampler_states = ss;
    const RtAlphaResult r = RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 128);
    d.sampler_states = nullptr;
    failures += Check(r.kind == RtAlphaKind::Cut && r.material.max_level == 3.0f &&
                          r.material.flags ==
                              RtMaterialFlags(kRtMatVertexAlpha | kRtMatLinear | kRtMatBorderAlpha, 7, 128, 3, 2, 4),
                      "resolve: address modes, border alpha, filter and MAXMIPLEVEL from the sampler state");
  }
  // An alpha the same everywhere (a register's): passing, opaque; never,
  // left out. Tests every alpha passes: opaque.
  {
    hlsl::PixelAlpha k;
    k.color0.known = true;
    k.color0.constant = 3;
    k.color0.scale = 1.0f;
    c[3][0] = 1.0f;
    bool ok = RtResolveAlpha(d, shaders(&k), tex, true, 5, 128).kind == RtAlphaKind::Opaque;
    c[3][0] = 0.2f;
    ok = ok && RtResolveAlpha(d, shaders(&k), tex, true, 5, 128).kind == RtAlphaKind::Skip;
    ok = ok && RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 0).kind == RtAlphaKind::Opaque &&
         RtResolveAlpha(d, shaders(&unknown), tex, true, 4, 255).kind == RtAlphaKind::Opaque &&
         RtResolveAlpha(d, shaders(&unknown), tex, true, 3, 255).kind == RtAlphaKind::Cut;
    // The edges: NOTEQUAL, LESS 255, GREATER 0 do not pass everything; a
    // float texture's alpha may lie past 1.
    ok = ok && RtResolveAlpha(d, shaders(&unknown), tex, true, 6, 0).kind == RtAlphaKind::Cut &&
         RtResolveAlpha(d, shaders(&unknown), tex, true, 2, 255).kind == RtAlphaKind::Cut &&
         RtResolveAlpha(d, shaders(&unknown), tex, true, 5, 0).kind == RtAlphaKind::Cut;
    tex[0].unit = false;
    ok = ok && RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 0).kind == RtAlphaKind::Cut;
    tex[0].unit = true;
    in[2].type = 3;  // a FLOAT4 colour: unclamped in ps_3_0
    ok = ok && RtResolveAlpha(d, shaders(&unknown), tex, true, 7, 0).kind == RtAlphaKind::Cut;
    in[2].type = 4;
    failures += Check(ok, "resolve: a constant alpha decided on the CPU; GREATEREQUAL 0 and LESSEQUAL 255 opaque; "
                          "EQUAL, NOTEQUAL, LESS 255, GREATER 0, a float texture and a float vertex colour not");
  }
  // Discards without the alpha test: a known texkill cut; one that may
  // read a texel the usual guess (GREATER 0); one that reads none (a clip
  // plane), or HLSL source: opaque, counted apart.
  {
    hlsl::PixelAlpha kill;
    kill.kills = 1;
    kill.kill = known.color0;
    kill.kill.offset = -0.5f;
    kill.kill.scale = 1.0f;
    RtAlphaResult r = RtResolveAlpha(d, shaders(&kill, true), tex, false, 8, 0);
    bool ok = r.kind == RtAlphaKind::Cut && (r.material.flags & kRtMatKill) && r.material.kill_scale == 0.5f &&
              r.material.kill_offset == -0.5f && ((r.material.flags >> 8) & 15) == 0;
    hlsl::PixelAlpha reads;
    reads.kills = 1;
    reads.kill_reads_texel = true;
    r = RtResolveAlpha(d, shaders(&reads, true), tex, false, 8, 0);
    ok = ok && r.kind == RtAlphaKind::Cut && r.generic && ((r.material.flags >> 8) & 15) == 5;
    hlsl::PixelAlpha plane;
    plane.kills = 1;
    ok = ok && RtResolveAlpha(d, shaders(&plane, true), tex, false, 8, 0).kind == RtAlphaKind::Uncut &&
         RtResolveAlpha(d, shaders(nullptr, true), tex, false, 8, 0).kind == RtAlphaKind::Uncut &&
         RtResolveAlpha(d, shaders(nullptr, true), tex, true, 7, 128).generic;
    // A texkill of 0.5 - P cuts (P above 0.5); one of 1 - P never does.
    hlsl::PixelAlpha neg = kill;
    neg.kill.constant = -1;
    neg.kill.color = -1;
    neg.kill.scale = -1.0f;
    neg.kill.offset = 0.5f;
    ok = ok && RtResolveAlpha(d, shaders(&neg, true), tex, false, 8, 0).kind == RtAlphaKind::Cut;
    neg.kill.offset = 1.0f;
    ok = ok && RtResolveAlpha(d, shaders(&neg, true), tex, false, 8, 0).kind == RtAlphaKind::Opaque;
    failures += Check(ok, "resolve: a known texkill cut; one reading a texel GREATER 0 on the usual guess; a clip "
                          "plane or HLSL source opaque (uncut); a texkill with a negative scale cut, or opaque where "
                          "it never discards");
  }
  return failures;
}

// Alpha-cut instances traced: a 2 x 2 texture whose texel (1, 0) has alpha
// 0, UV = xy, alpha >= 128 kept; the debug view's normals kept as well
// (primitive data 40 bytes a triangle).
int AlphaChecks(Gpu& g) {
  int failures = 0;
  auto s = g.Scene(RtDebug::Normals);
  if (!s) return Check(false, "scene made");
  id<MTLTexture> tex = RgbaTexture(g, 2, 2, {kSolid, kHole, kSolid, kSolid});
  CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts);
  CutMesh wide = CutSquare(2, 2.0f, kOpaqueVerts);  // UV = 2 xy
  const uint8_t va[4] = {0, 255, 255, 255};        // vertex (0, 0)'s alpha 0
  CutMesh faded = CutSquare(3, 1.0f, va);
  Mesh back = Square(4);
  Mat w0, wback, wwrap, wclamp, wfade, wnever, wopq, wkill;
  Translate(w0, 0, 0, 5);
  Translate(wback, -1, -1, 10, 3.0f);  // opaque, behind the first: x, y -1..2 at z 10
  Translate(wwrap, 10, 0, 5);
  Translate(wclamp, 20, 0, 5);
  Translate(wfade, 30, 0, 5);
  Translate(wnever, 40, 0, 5);
  Translate(wopq, 50, 0, 5);
  Translate(wkill, 60, 0, 5);
  const RtCut test = Cut(tex, 7, 128);  // GREATEREQUAL 128, wrap
  const RtCut clamp = Cut(tex, 7, 128, 3);
  const RtCut fade = Cut(tex, 7, 128, 1, kRtMatVertexAlpha);
  const RtCut never = Cut(tex, 1, 0);
  RtCut kill = Cut(tex, 0, 0, 1, kRtMatKill);
  kill.material.kill_scale = 1.0f;
  kill.material.kill_offset = -0.5f;
  bool added = true;
  const RtFrame* f = g.Frame(*s, 1, [&] {
    added = added && s->Add(cm.key, cm.g, w0, w0, 3, 4, &test);       // 0
    added = added && s->Add(back.key, back.g, wback, wback, 3, 4);    // 1
    added = added && s->Add(wide.key, wide.g, wwrap, wwrap, 3, 4, &test);  // 2
    added = added && s->Add(wide.key, wide.g, wclamp, wclamp, 3, 4, &clamp);  // 3: the same BLAS, clamped
    added = added && s->Add(faded.key, faded.g, wfade, wfade, 3, 4, &fade);   // 4
    added = added && s->Add(cm.key, cm.g, wnever, wnever, 3, 4, &never);      // 5
    added = added && s->Add(back.key, back.g, wopq, wopq, 3, 4);              // 6: opaque
    added = added && s->Add(cm.key, cm.g, wkill, wkill, 3, 4, &kill);         // 7: texkill
  });
  const RtStats st = s->Stats();
  failures += Check(added && f && f->instances == 8 && f->texture_count == 1 && st.cut == 6 && st.cut_textures == 1 &&
                        st.total_builds == 4,
                    "alpha-cut instances added: one texture held, BLASes shared across materials");
  const auto* info = f ? reinterpret_cast<const RtInstanceInfo*>(static_cast<const uint8_t*>(f->info.contents) +
                                                                 f->info_offset)
                       : nullptr;
  failures += Check(info && (info[0].flags & MTLAccelerationStructureInstanceOptionNonOpaque) &&
                        (info[1].flags & MTLAccelerationStructureInstanceOptionOpaque),
                    "alpha-cut instances non-opaque, the others opaque");
  std::vector<Hit> h = g.Trace(f, {Ray(0.25f, 0.25f, 0, 0, 0, 1),   // 0: solid
                                   Ray(0.75f, 0.25f, 0, 0, 0, 1),   // 1: the hole: the square behind
                                   Ray(0.75f, 0.75f, 0, 0, 0, 1),   // 2: solid (the other triangle)
                                   Ray(10.25f, 0.25f, 0, 0, 0, 1),  // 3: UV (0.5, 0.5): solid
                                   Ray(10.375f, 0.125f, 0, 0, 0, 1),  // 4: UV (0.75, 0.25): hole
                                   Ray(10.875f, 0.625f, 0, 0, 0, 1),  // 5: UV (1.75, 1.25), wrapped: hole
                                   Ray(20.875f, 0.625f, 0, 0, 0, 1),  // 6: clamped: texel (1, 1), solid
                                   Ray(20.875f, 0.125f, 0, 0, 0, 1),  // 7: UV (1.75, 0.25) clamped: hole
                                   Ray(30.4f, 0.4f, 0, 0, 0, 1),    // 8: vertex alpha 0.8 x texel 1: kept
                                   Ray(30.1f, 0.1f, 0, 0, 0, 1),    // 9: vertex alpha 0.2 x texel 1: cut
                                   Ray(30.75f, 0.2f, 0, 0, 0, 1),   // 10: vertex alpha high, texel 0: cut
                                   Ray(40.25f, 0.25f, 0, 0, 0, 1),  // 11: NEVER: nothing
                                   Ray(50.75f, 0.25f, 0, 0, 0, 1),  // 12: opaque: hit
                                   Ray(60.25f, 0.25f, 0, 0, 0, 1),  // 13: texkill, alpha 1: kept
                                   Ray(60.75f, 0.25f, 0, 0, 0, 1)},  // 14: texkill, alpha 0: cut
                               true);
  failures += Check(Near(h[0].t, 5) && h[0].id == 0 && h[0].front == 1 && Near(h[1].t, 10) && h[1].id == 1 &&
                        Near(h[2].t, 5) && h[2].id == 0,
                    "through the texel of alpha 0 the ray goes on to the square behind; elsewhere it stops");
  failures += Check(Near(h[3].t, 5) && h[3].id == 2 && h[4].t == -1 && h[5].t == -1,
                    "UVs past 1 wrapped (D3DTADDRESS_WRAP)");
  failures += Check(Near(h[6].t, 5) && h[6].id == 3 && h[7].t == -1,
                    "the same mesh with another material: clamped (D3DTADDRESS_CLAMP)");
  failures += Check(Near(h[8].t, 5) && h[9].t == -1 && h[10].t == -1, "the vertex alpha times the texel's alpha");
  failures += Check(h[11].t == -1 && Near(h[12].t, 5) && h[12].id == 6, "ALPHAFUNC NEVER: never hit; opaque instances "
                                                                         "hit as before");
  failures += Check(Near(h[13].t, 5) && h[14].t == -1, "a texkill threshold (alpha - 0.5 < 0 cut)");
  failures += Check(Near(h[0].n[2], -1) && Near(h[2].n[2], -1) && Near(h[3].n[2], -1) && Near(h[12].n[2], -1),
                    "debug normals after the alpha-cut primitive data (and without it)");
  // Any hit and masks.
  h = g.Trace(f, {Ray(0.75f, 0.25f, 0, 0, 0, 1), Ray(10.375f, 0.125f, 0, 0, 0, 1), Ray(10.25f, 0.25f, 0, 0, 0, 1),
                  Ray(40.25f, 0.25f, 0, 0, 0, 1)},
              false, true);
  failures += Check(h[0].t == 0 && h[1].t == -1 && h[2].t == 0 && h[3].t == -1,
                    "any hit: through the hole onto the square behind, not through a hole alone, the solid part, "
                    "not NEVER");
  h = g.Trace(f, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(50.5f, 0.5f, 0, 0, 0, 1)}, false, false, kRtMaskOpaque);
  failures += Check(Near(h[0].t, 10) && h[0].id == 1 && Near(h[1].t, 5),
                    "the opaque mask: alpha-cut instances not seen");
  // The table as the GPU reads it.
  {
    RtMaterial m[2] = {};
    m[0].alpha_scale = 0.5f;
    m[0].kill_offset = -0.25f;
    m[0].max_level = 3;
    m[0].flags = 0xA5C3F00Fu;
    m[1].alpha_offset = 7;
    m[1].reserved[2] = 0x3F800000u;
    id<MTLBuffer> mb = [g.dev newBufferWithBytes:m length:sizeof m options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [g.dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g.echo];
    [e setBuffer:mb offset:0 atIndex:0];
    [e setBuffer:ob offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
    const float* o = static_cast<const float*>(ob.contents);
    uint32_t flags = 0;
    bool ok = g.Commit(cb);
    std::memcpy(&flags, &o[3], 4);
    ok = ok && o[0] == 0.5f && o[1] == -0.25f && o[2] == 3.0f && flags == 0xA5C3F00Fu && o[4] == 64.0f &&
         o[5] == 28.0f && o[6] == 7.0f && o[7] == 1.0f;
    failures += Check(ok, "RtMaterial and RtCutPrim: the same layout in C++ and MSL");
  }
  // Traced opaque by their least alpha (RtCut::texel_min): a texture whose
  // least texel passes, alone or times vertex alphas all 255, opaque; times
  // a vertex alpha of 0, or with the least texel not known, still cut.
  {
    RtCut sure = Cut(tex, 7, 128);
    sure.texel_min = 200.0f / 255.0f;
    RtCut sure_va = Cut(tex, 7, 128, 1, kRtMatVertexAlpha);
    sure_va.texel_min = 1.0f;
    RtCut equal = Cut(tex, 3, 255);  // EQUAL: never taken as opaque
    equal.texel_min = 1.0f;
    const uint64_t cut_before = s->Stats().cut_opaque;
    const RtFrame* f2 = g.Frame(*s, 2, [&] {
      s->Add(cm.key, cm.g, w0, w0, 3, 4, &sure);             // 0: opaque
      s->Add(cm.key, cm.g, wwrap, wwrap, 3, 4, &sure_va);    // 1: opaque (vertex alphas 255)
      s->Add(faded.key, faded.g, wfade, wfade, 3, 4, &sure_va);  // 2: cut (a vertex alpha 0)
      s->Add(cm.key, cm.g, wclamp, wclamp, 3, 4, &test);      // 3: cut (texel_min not known)
      s->Add(cm.key, cm.g, wkill, wkill, 3, 4, &equal);       // 4: cut
    });
    const auto* in = f2 ? reinterpret_cast<const RtInstanceInfo*>(static_cast<const uint8_t*>(f2->info.contents) +
                                                                  f2->info_offset)
                        : nullptr;
    auto cut = [&](int i) { return (in[i].flags & MTLAccelerationStructureInstanceOptionNonOpaque) != 0; };
    const bool ok = in && f2->instances == 5 && !cut(0) && !cut(1) && cut(2) && cut(3) && cut(4) &&
                    s->Stats().cut_opaque - cut_before == 2 && RtCutAlwaysPasses(sure.material, 0.5f) &&
                    !RtCutAlwaysPasses(sure.material, 0.4f) && !RtCutAlwaysPasses(equal.material, 1.0f);
    failures += Check(ok, "alpha-cut draws whose least texel times least vertex alpha passes traced opaque; a "
                          "vertex alpha that fails, an unknown least texel or an EQUAL test still cut");
  }
  return failures;
}

// Texture levels, bilinear filtering, X8R8G8B8 alpha (a swizzle read
// through a resource ID), a carried alpha-cut instance, a texture released
// after Add held to its horizon, and an alpha-cut draw captured.
int AlphaMoreChecks(Gpu& g) {
  int failures = 0;
  Mat w0, w1, w2, w3;
  Translate(w0, 0, 0, 5);
  Translate(w1, 10, 0, 5);
  Translate(w2, 20, 0, 5);
  Translate(w3, 30, 0, 5);
  {
    auto s = g.Scene();
    if (!s) return Check(false, "scene made");
    // 2 x 2 solid at level 0, alpha 0 at level 1.
    id<MTLTexture> mips = RgbaTexture(g, 2, 2, {kSolid, kSolid, kSolid, kSolid, kHole}, 2);
    // 2 x 1: alpha 0 and 1, bilinear, clamped: alpha 0.5 at u 0.5.
    id<MTLTexture> ramp = RgbaTexture(g, 2, 1, {0x00000000u, 0xFF000000u});
    // Alpha 0 stored, read as 1 (X8R8G8B8).
    id<MTLTexture> x8 = RgbaTexture(g, 2, 2, {kHole, kHole, kHole, kHole}, 1, true);
    CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts);
    RtCut level0 = Cut(mips, 7, 128), level1 = Cut(mips, 7, 128);
    level1.material.max_level = 1;
    const RtCut lin = Cut(ramp, 7, 128, 3, kRtMatLinear), x8c = Cut(x8, 7, 128);
    const RtFrame* f = g.Frame(*s, 1, [&] {
      s->Add(cm.key, cm.g, w0, w0, 3, 4, &level0);
      s->Add(cm.key, cm.g, w1, w1, 3, 4, &level1);
      s->Add(cm.key, cm.g, w2, w2, 3, 4, &lin);
      s->Add(cm.key, cm.g, w3, w3, 3, 4, &x8c);
    });
    const std::vector<Hit> h = g.Trace(f, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(10.25f, 0.25f, 0, 0, 0, 1),
                                           Ray(20.52f, 0.25f, 0, 0, 0, 1), Ray(20.45f, 0.25f, 0, 0, 0, 1),
                                           Ray(20.9f, 0.6f, 0, 0, 0, 1), Ray(30.75f, 0.25f, 0, 0, 0, 1)});
    failures += Check(f && f->texture_count == 3 && Near(h[0].t, 5) && h[1].t == -1,
                      "MAXMIPLEVEL: the level read (level 0 solid, level 1 cut)");
    failures += Check(Near(h[2].t, 5) && h[3].t == -1 && Near(h[4].t, 5),
                      "bilinear, clamped: alpha 0.54 just right of the texels' midpoint kept, 0.4 left of it cut, "
                      "1 past the last texel's centre");
    failures += Check(Near(h[5].t, 5), "an X8R8G8B8 texture (alpha swizzled to 1) read through its resource ID: "
                                       "alpha 1");
  }
  // Addressing by hand: MIRROR, MIRRORONCE (UVs below 0 and past 1) and
  // BORDER (nearest, the border's alpha 0 and 1; bilinear taps past the
  // edge reading the border); the texkill's threshold at exactly 0 kept;
  // two passing alpha-cut instances one behind the other: the nearer.
  {
    auto s = g.Scene();
    id<MTLTexture> tex = RgbaTexture(g, 2, 2, {kSolid, kHole, kSolid, kSolid});
    id<MTLTexture> solid = RgbaTexture(g, 2, 2, {kSolid, kSolid, kSolid, kSolid});
    // Alpha 128 and 127 (0x80, 0x7F).
    id<MTLTexture> edge = RgbaTexture(g, 2, 1, {0x80808080u, 0x7F808080u});
    CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts), wide = CutSquare(2, 2.0f, kOpaqueVerts),
            neg = CutSquare(3, -1.0f, kOpaqueVerts);
    const RtCut mirror = Cut(tex, 7, 128, 2), once = Cut(tex, 7, 128, 5), border0 = Cut(tex, 7, 128, 4),
                border1 = Cut(tex, 7, 128, 4, kRtMatBorderAlpha), lin0 = Cut(solid, 7, 160, 4, kRtMatLinear),
                lin1 = Cut(solid, 7, 160, 4, kRtMatLinear | kRtMatBorderAlpha), test = Cut(tex, 7, 128);
    RtCut kill = Cut(edge, 0, 0, 3, kRtMatKill);
    kill.material.kill_scale = 1.0f;
    kill.material.kill_offset = -(128.0f / 255.0f);
    Mat w[10];
    for (int i = 0; i < 10; i++) Translate(w[i], float(i) * 10, 0, 5);
    Mat behind;
    Translate(behind, 90, 0, 7);
    const RtFrame* f = g.Frame(*s, 1, [&] {
      s->Add(wide.key, wide.g, w[0], w[0], 3, 4, &mirror);  // 0
      s->Add(neg.key, neg.g, w[1], w[1], 3, 4, &once);      // 1: UV -xy
      s->Add(wide.key, wide.g, w[2], w[2], 3, 4, &once);    // 2: UV past 1
      s->Add(wide.key, wide.g, w[3], w[3], 3, 4, &border0);  // 3
      s->Add(wide.key, wide.g, w[4], w[4], 3, 4, &border1);  // 4
      s->Add(cm.key, cm.g, w[5], w[5], 3, 4, &lin0);         // 5
      s->Add(cm.key, cm.g, w[6], w[6], 3, 4, &lin1);         // 6
      s->Add(cm.key, cm.g, w[7], w[7], 3, 4, &kill);         // 7
      s->Add(cm.key, cm.g, behind, behind, 3, 4, &test);     // 8: behind 9
      s->Add(cm.key, cm.g, w[9], w[9], 3, 4, &test);         // 9
    });
    const std::vector<Hit> h = g.Trace(f, {
        Ray(0.625f, 0.125f, 0, 0, 0, 1),   // 0: MIRROR, UV (1.25, 0.25): texel 1 mirrored, the hole
        Ray(0.875f, 0.125f, 0, 0, 0, 1),   // 1: MIRROR, UV (1.75, 0.25): texel 0, solid
        Ray(10.75f, 0.25f, 0, 0, 0, 1),    // 2: MIRRORONCE, UV (-0.75, -0.25): texel (1, 0), the hole
        Ray(10.25f, 0.25f, 0, 0, 0, 1),    // 3: MIRRORONCE, UV (-0.25, -0.25): texel (0, 0), solid
        Ray(20.875f, 0.125f, 0, 0, 0, 1),  // 4: MIRRORONCE, UV (1.75, 0.25): clamped to texel 1, the hole
        Ray(30.875f, 0.125f, 0, 0, 0, 1),  // 5: BORDER, outside, border alpha 0: cut
        Ray(30.25f, 0.25f, 0, 0, 0, 1),    // 6: BORDER, inside, solid
        Ray(40.875f, 0.125f, 0, 0, 0, 1),  // 7: BORDER, outside, border alpha 1: kept
        Ray(50.95f, 0.25f, 0, 0, 0, 1),    // 8: bilinear, u 0.95: 0.6 of the texel, 0.4 border 0: 153 < 160
        Ray(50.8f, 0.25f, 0, 0, 0, 1),     // 9: bilinear, u 0.8: 0.9: kept
        Ray(60.95f, 0.25f, 0, 0, 0, 1),    // 10: bilinear, border alpha 1: kept
        Ray(70.25f, 0.25f, 0, 0, 0, 1),    // 11: texkill: alpha 128/255 - 128/255 = 0: kept
        Ray(70.75f, 0.25f, 0, 0, 0, 1),    // 12: texkill: alpha 127/255: cut
        Ray(90.25f, 0.25f, 0, 0, 0, 1),    // 13: two passing instances: the nearer (9, t 5)
    });
    failures += Check(h[0].t == -1 && Near(h[1].t, 5), "D3DTADDRESS_MIRROR past 1");
    failures += Check(h[2].t == -1 && Near(h[3].t, 5) && h[4].t == -1, "D3DTADDRESS_MIRRORONCE below 0 and past 1");
    failures += Check(h[5].t == -1 && Near(h[6].t, 5) && Near(h[7].t, 5), "D3DTADDRESS_BORDER: the border's alpha");
    failures += Check(h[8].t == -1 && Near(h[9].t, 5) && Near(h[10].t, 5),
                      "bilinear taps past the edge read the border (alpha 0 and 1)");
    failures += Check(Near(h[11].t, 5) && h[12].t == -1, "a texkill keeps exactly 0 (discards below it)");
    failures += Check(Near(h[13].t, 5) && h[13].id == 9, "two passing alpha-cut instances in line: the nearer");
  }
  // A carried alpha-cut instance: added after frame 2's snapshot, in frame
  // 3's snapshot with its material.
  {
    auto s = g.Scene();
    id<MTLTexture> tex = RgbaTexture(g, 2, 2, {kSolid, kHole, kSolid, kSolid});
    CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts);
    Mesh m = Square(2);
    const RtCut test = Cut(tex, 7, 128);
    const RtFrame* snap = nullptr;
    for (uint64_t serial = 2; serial <= 3; serial++) {
      g.arena.used = 0;
      s->BeginFrame(serial, serial - 1);
      s->Add(m.key, m.g, w1, w1, 3, 4);
      snap = s->Snapshot();
      if (serial == 2) s->Add(cm.key, cm.g, w0, w0, 3, 4, &test);
      const RtFrame* end = s->EndFrame(true);
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      s->EncodeFrame(cb, snap);
      s->EncodeFrame(cb, end);
      g.Commit(cb);
    }
    const std::vector<Hit> h = g.Trace(snap, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(0.75f, 0.25f, 0, 0, 0, 1)});
    failures += Check(snap && snap->instances == 2 && snap->texture_count == 1 && Near(h[0].t, 5) && h[0].id == 1 &&
                          h[1].t == -1,
                      "an alpha-cut instance carried into the next frame's snapshot, with its material");
  }
  // A texture released right after Add: held (and traced right) until no
  // frame that may trace its table is in flight.
  {
    auto s = g.Scene();
    CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts);
    Mesh m = Square(2);
    __weak id<MTLTexture> weak = nil;
    const RtFrame* f1 = nullptr;
    bool held = true;
    for (uint64_t serial = 1; serial <= 5; serial++) {
      @autoreleasepool {
        Arena& a = g.slot_arena[serial % RtScene::kSlots];
        a.used = 0;
        s->SetAllocator(ArenaAlloc, &a);
        s->BeginFrame(serial, serial - 1);
        if (serial <= 4) held = held && s->Stats().cut_textures == (serial == 1 ? 0u : 1u);
        if (serial == 1) {
          id<MTLTexture> tex = RgbaTexture(g, 2, 2, {kSolid, kHole, kSolid, kSolid});
          weak = tex;
          const RtCut c = Cut(tex, 7, 128);
          s->Add(cm.key, cm.g, w0, w0, 3, 4, &c);
          tex = nil;  // the caller's reference gone
        }
        s->Add(m.key, m.g, w1, w1, 3, 4);
        const RtFrame* f = s->EndFrame(true);
        if (serial == 1) f1 = f;
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        s->EncodeFrame(cb, f);
        g.Commit(cb);
        if (serial == 3) {
          // Frame 1's TLAS (its slot not reused yet), after two more frames.
          const std::vector<Hit> h = g.Trace(f1, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(0.75f, 0.25f, 0, 0, 0, 1)});
          held = held && weak != nil && Near(h[0].t, 5) && h[1].t == -1;
        }
      }
    }
    s->SetAllocator(ArenaAlloc, &g.arena);
    failures += Check(held && s->Stats().cut_textures == 0 && weak == nil,
                      "a texture released after Add: held and traced right until its slot horizon, then let go");
  }
  // A captured alpha-cut draw: its UVs from the vertex stream (the clip
  // positions written where the capture would, the identity camera).
  {
    auto s = g.Scene();
    id<MTLTexture> tex = RgbaTexture(g, 2, 2, {kSolid, kHole, kSolid, kSolid});
    CutMesh cm = CutSquare(1, 1.0f, kOpaqueVerts);
    Mesh m = Square(2);
    const RtCut test = Cut(tex, 7, 128);
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    s->Add(m.key, m.g, w1, w1, 3, 4);  // the camera: the identity
    RtGeometry cg = cm.g;
    cg.positions = nullptr;
    const RtCaptureSlot slot = s->AddCaptured(cg, 0, 4, 3, &test);
    const RtFrame* f = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    if (slot.buf) {
      // What the capture variant writes: clip positions (identity camera),
      // the square at z 5.
      float clip[16];
      for (int i = 0; i < 4; i++) {
        clip[i * 4] = kSquare[size_t(i)][0];
        clip[i * 4 + 1] = kSquare[size_t(i)][1];
        clip[i * 4 + 2] = 5;
        clip[i * 4 + 3] = 1;
      }
      id<MTLBuffer> src = [g.dev newBufferWithBytes:clip length:sizeof clip options:MTLResourceStorageModeShared];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      [bl copyFromBuffer:src sourceOffset:0 toBuffer:slot.buf destinationOffset:slot.capture_off size:sizeof clip];
      [bl endEncoding];
    }
    s->EncodeFrame(cb, f);
    const bool ran = g.Commit(cb);
    const std::vector<Hit> h = g.Trace(f, {Ray(0.25f, 0.25f, 0, 0, 0, 1), Ray(0.75f, 0.25f, 0, 0, 0, 1),
                                           Ray(0.75f, 0.75f, 0, 0, 0, 1)});
    const RtStats st = s->Stats();
    failures += Check(ran && slot.buf && f && f->instances == 2 && st.captured == 1 && st.cut == 1 &&
                          Near(h[0].t, 5) && h[0].id == 1 && h[1].t == -1 && Near(h[2].t, 5),
                      "a captured alpha-cut draw: UVs from the vertex stream, cut where the texel's alpha is 0");
  }
  return failures;
}

// Retained instances: a static instance (drawn at the same place two frames
// in a row) stays in the scene after the frames that drew it while the
// camera does not see all of it; let go when the camera has seen it undrawn
// for kInViewFrames, when it is replaced (moved off, another level of
// detail), when its BLAS goes, when idle, past the cap; what moves and
// captured draws never retained; one instance with this frame's draw;
// another view's drawn on two frames of 16 kept between them; its BLAS
// kept at the memory cap while another can go.
int RetainChecks(Gpu& g) {
  int failures = 0;
  Mat vp, wf, wb, wq, a, b, q;
  Perspective(vp, 1.0f, 1.0f, 0.5f, 1000.0f);  // looking down +z from the origin
  Translate(wf, 0, 0, 5);                      // in front, all in view
  Translate(wb, 0, 0, -5);                     // behind the camera
  Translate(wq, -50, -50, 8, 100.0f);          // in front, mostly outside the view
  MulRows(wf, vp, a);
  MulRows(wb, vp, b);
  MulRows(wq, vp, q);
  Mesh m = Square(1), n = Square(2), big = Square(3);
  const Query fore = Ray(0.5f, 0.5f, 0, 0, 0, 1), aft = Ray(0.5f, 0.5f, 0, 0, 0, -1);
  auto at = [&](float x, float z, Mat w, Mat wvp) {
    Translate(w, x, 0, z);
    MulRows(w, vp, wvp);
  };
  {
    auto s = g.Scene({}, true);
    if (!s) return Check(false, "scene made");
    std::vector<RtInstanceSummary> sum;
    uint64_t serial = 1;
    auto all3 = [&] {
      s->Add(m.key, m.g, wf, a, 3, 4);
      s->Add(n.key, n.g, wb, b, 3, 4);
      s->Add(big.key, big.g, wq, q, 3, 4);
    };
    const RtFrame* f = g.Frame(*s, serial++, all3);
    bool ok = f && f->instances == 3 && s->Stats().retained_kept == 0;  // (drawn once: not yet static)
    f = g.Frame(*s, serial++, all3);
    ok = ok && f && f->instances == 3 && s->Stats().retained == 0 && s->Stats().retained_kept == 3;
    s->DescribeNextTlas(&sum);
    f = g.Frame(*s, serial++, [&] { s->Add(m.key, m.g, wf, a, 3, 4); });
    std::vector<Hit> h = g.Trace(f, {fore, aft});
    int flagged = 0;
    for (const RtInstanceSummary& x : sum) flagged += x.retained ? 1 : 0;
    ok = ok && f && f->instances == 3 && s->Stats().retained == 2 && flagged == 2 && sum.size() == 3 &&
         Near(h[0].t, 5) && Near(h[1].t, 5);
    failures += Check(ok, "an instance drawn (twice) and then not, outside the view: still in the next TLAS (a ray "
                          "hits it), once with this frame's draw of the other");
    // In the view and not drawn, by default: kept (the game hides what lies
    // behind something nearer, and it still casts shadows).
    ok = true;
    unsetenv("RECOMP_RT_IN_VIEW_FRAMES");
    for (uint32_t k = 0; k < RtScene::kInViewFrames + 5; k++) {
      f = g.Frame(*s, serial++, [&] { s->Add(n.key, n.g, wb, b, 3, 4); });
      ok = ok && f && f->instances == 3;
    }
    h = g.Trace(f, {fore});
    ok = ok && Near(h[0].t, 5) && s->Stats().retained_dropped[size_t(RtRetainDrop::InView)] == 0;
    failures += Check(ok, "in the view and not drawn, by default: kept (a ray still hits it)");
    // With RECOMP_RT_IN_VIEW_FRAMES=n: kept for n frames, then let go; the
    // one mostly outside kept.
    setenv("RECOMP_RT_IN_VIEW_FRAMES", std::to_string(RtScene::kInViewFrames).c_str(), 1);
    ok = true;
    for (uint32_t k = 0; k < RtScene::kInViewFrames; k++) {
      f = g.Frame(*s, serial++, [&] { s->Add(n.key, n.g, wb, b, 3, 4); });
      ok = ok && f && f->instances == 3;
    }
    h = g.Trace(f, {fore});
    ok = ok && Near(h[0].t, 5) && s->Stats().retained_dropped[size_t(RtRetainDrop::InView)] == 0;
    f = g.Frame(*s, serial++, [&] { s->Add(n.key, n.g, wb, b, 3, 4); });
    h = g.Trace(f, {fore, aft});
    RtStats st = s->Stats();
    ok = ok && f && f->instances == 2 && st.retained_dropped[size_t(RtRetainDrop::InView)] == 1 &&
         st.retained_kept == 2 && Near(h[0].t, 8) && Near(h[1].t, 5);
    failures += Check(ok, "in the view and not drawn: kept for kInViewFrames frames, then let go (the ray goes on to "
                          "the one mostly outside the view, kept)");
    // A snapshot before any draw: the retained ones (the last camera's);
    // the frame's draw of one is that instance.
    g.arena.used = 0;
    s->BeginFrame(serial, serial - 1);
    serial++;
    const RtFrame* sn = s->Snapshot();
    s->Add(n.key, n.g, wb, b, 3, 4);
    const RtFrame* en = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, sn);
    s->EncodeFrame(cb, en);
    g.Commit(cb);
    h = g.Trace(sn, {fore, aft});
    ok = sn && sn->instances == 2 && Near(h[0].t, 8) && Near(h[1].t, 5) && en && en->instances == 2;
    failures += Check(ok, "a snapshot before the frame's draws: the retained instances; then one with their draws");
    // New contents where the retained one's data is: let go with its BLAS.
    Mesh n2 = n;
    n2.key.vertex_version++;
    f = g.Frame(*s, serial++, [&] { s->Add(n2.key, n2.g, wb, b, 3, 4); });
    st = s->Stats();
    // (The square mostly outside the view still there; the old one gone, the
    // new one not static yet.)
    ok = f && f->instances == 2 && st.retained_dropped[size_t(RtRetainDrop::Gone)] == 1 && st.retained_kept == 1;
    failures += Check(ok, "a retained instance whose buffer has new contents: let go with its BLAS");
    unsetenv("RECOMP_RT_IN_VIEW_FRAMES");
  }
  // What moves: never retained; a static one that moves off a little: its
  // copy let go, in the frame's snapshot already; another level of detail
  // drawn at its place: let go.
  {
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    uint64_t serial = 1;
    Mat w, wvp;
    bool ok = true;
    for (int k = 0; k < 5; k++) {
      at(0.1f * float(k), -5, w, wvp);  // (moving behind the camera)
      const RtFrame* f = g.Frame(*s, serial++, [&] { s->Add(n.key, n.g, w, wvp, 3, 4); });
      ok = ok && f && f->instances == 1 && s->Stats().retained_kept == 0;
    }
    failures += Check(ok, "an instance moving every frame leaves no retained copies");
    at(0, -5, w, wvp);
    for (int k = 0; k < 2; k++) g.Frame(*s, serial++, [&] { s->Add(n.key, n.g, w, wvp, 3, 4); });
    ok = s->Stats().retained_kept == 1;
    Mat w2, wvp2;
    at(0.2f, -5, w2, wvp2);
    g.arena.used = 0;
    s->BeginFrame(serial, serial - 1);
    serial++;
    s->Add(n.key, n.g, w2, wvp2, 3, 4);
    const RtFrame* sn = s->Snapshot();
    const RtFrame* en = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, sn);
    s->EncodeFrame(cb, en);
    g.Commit(cb);
    ok = ok && sn && sn->instances == 1 && en && en->instances == 1 && s->Stats().retained_kept == 0 &&
         s->Stats().retained_dropped[size_t(RtRetainDrop::Replaced)] == 1;
    failures += Check(ok, "a static instance moved off a little: its retained copy let go (in the snapshot already)");
    // Levels of detail: hi drawn twice, then lo (other geometry, much the
    // same bounds) at the same place.
    Mesh hi = Square(4), lo = Square(5);
    at(5, -5, w, wvp);
    for (int k = 0; k < 2; k++) g.Frame(*s, serial++, [&] { s->Add(hi.key, hi.g, w, wvp, 3, 4); });
    ok = s->Stats().retained_kept == 1;
    const RtFrame* f = g.Frame(*s, serial++, [&] { s->Add(lo.key, lo.g, w, wvp, 3, 4); });
    ok = ok && f && f->instances == 1 && s->Stats().retained_kept == 0 &&
         s->Stats().retained_dropped[size_t(RtRetainDrop::Replaced)] == 2;
    failures += Check(ok, "another level of detail drawn at a retained instance's place: the retained one let go");
  }
  // Idle: let go after retain_frames presented frames without a draw.
  {
    RtSceneOptions o;
    o.retain_frames = 3;
    auto s = g.Scene(o, true);
    if (!s) return failures + Check(false, "scene made");
    for (uint64_t k = 1; k <= 2; k++) g.Frame(*s, k, [&] { s->Add(n.key, n.g, wb, b, 3, 4); });
    bool ok = true;
    const RtFrame* f = nullptr;
    for (uint64_t k = 3; k <= 5; k++) {
      f = g.Frame(*s, k, [] {});
      ok = ok && f && f->instances == 1 && s->Stats().retained_kept == 1;
    }
    std::vector<Hit> h = g.Trace(f, {aft});
    f = g.Frame(*s, 6, [] {});
    ok = ok && Near(h[0].t, 5) && !f && s->Stats().retained_kept == 0 &&
         s->Stats().retained_dropped[size_t(RtRetainDrop::Idle)] == 1;
    failures += Check(ok, "a retained instance not drawn for retain_frames frames: let go");
  }
  // A far level of detail: drawn from afar only, let go once the camera is
  // at less than kRetainCloser of that distance and nothing draws it; kept
  // until then.
  {
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    Mesh own = Square(7);
    Mat wfar, far_wvp;
    at(0, 100, wfar, far_wvp);
    for (uint64_t k = 1; k <= 2; k++) g.Frame(*s, k, [&] { s->Add(n.key, n.g, wfar, far_wvp, 3, 4); });
    // The camera `z` along the view from where it was, with a draw of its own just ahead.
    auto from = [&](uint64_t serial, float z) {
      Mat view, vp2, w, wvp;
      Translate(view, 0, 0, -z);
      MulRows(view, vp, vp2);
      Translate(w, 0, 0, z + 5);
      MulRows(w, vp2, wvp);
      return g.Frame(*s, serial, [&] { s->Add(own.key, own.g, w, wvp, 3, 4); });
    };
    static_assert(RtScene::kRetainCloser == 0.5, "the distances of this test");
    const RtFrame* f = from(3, 40);
    bool ok = f && f->instances == 2 && s->Stats().retained_dropped[size_t(RtRetainDrop::Closer)] == 0;
    f = from(4, 90);
    const RtStats st = s->Stats();
    ok = ok && f && f->instances == 1 && st.retained_dropped[size_t(RtRetainDrop::Closer)] == 1;
    failures += Check(ok, "a retained instance drawn only from afar, the camera then at under half that distance "
                          "without a draw of it: let go (kept at over half)");
  }
  // The cap: the farthest from the camera let go first (here the last
  // drawn: the longest undrawn stays).
  {
    RtSceneOptions o;
    o.max_retained = 2;
    auto s = g.Scene(o, true);
    if (!s) return failures + Check(false, "scene made");
    Mat w[3], wvp[3];
    for (int k = 0; k < 3; k++) at(10.0f * float(k), -5, w[k], wvp[k]);
    for (int k = 0; k < 6; k++) g.Frame(*s, uint64_t(k) + 1, [&] { s->Add(n.key, n.g, w[k / 2], wvp[k / 2], 3, 4); });
    const RtFrame* f = g.Frame(*s, 7, [] {});
    std::vector<Hit> h = g.Trace(f, {Ray(0.5f, 0.5f, 0, 0, 0, -1), Ray(10.5f, 0.5f, 0, 0, 0, -1),
                                     Ray(20.5f, 0.5f, 0, 0, 0, -1)});
    const RtStats st = s->Stats();
    failures += Check(f && f->instances == 2 && st.retained_kept == 2 &&
                          st.retained_dropped[size_t(RtRetainDrop::Cap)] == 1 && Near(h[0].t, 5) && Near(h[1].t, 5) &&
                          h[2].t == -1,
                      "past max_retained: the farthest from the camera let go");
  }
  // Another view's static instance listed only on the first two of every
  // 16 presented frames (a shadow map drawn that sparsely), with the main
  // view's own draw each frame: in every TLAS, between the pairs as well,
  // behind the camera and in its view undrawn alike (14 frames in a row
  // undrawn, fewer than kInViewFrames).
  {
    static_assert(16 - 2 < RtScene::kInViewFrames, "the gaps of this schedule");
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    Mesh own = Square(6);
    Mat ws, side;
    Translate(ws, 3, 0, 5);  // the main view's draw, in view off to the side
    MulRows(ws, vp, side);
    bool ok = true;
    for (uint64_t f = 0; f < 80; f++) {
      const RtFrame* fr = g.Frame(*s, f + 1, [&] {
        s->Add(own.key, own.g, ws, side, 3, 4);
        if (f % 16 < 2) {
          s->Add(m.key, m.g, wf, a, 3, 4, nullptr, RtView::Other);  // in view
          s->Add(n.key, n.g, wb, b, 3, 4, nullptr, RtView::Other);  // behind the camera
        }
      });
      const std::vector<Hit> h = g.Trace(fr, {fore, aft});
      const bool here = fr && fr->instances == 3 && Near(h[0].t, 5) && Near(h[1].t, 5);
      if (!here) std::printf("  frame %llu: not all three in its TLAS\n", (unsigned long long)f);
      ok = ok && here;
    }
    const RtStats st = s->Stats();
    ok = ok && st.retained_kept == 3 && st.retained_dropped[size_t(RtRetainDrop::InView)] == 0 &&
         st.retained_dropped[size_t(RtRetainDrop::Idle)] == 0;
    failures += Check(ok, "another view's static instances listed on two frames of 16: in every TLAS, in view "
                          "undrawn and behind the camera");
  }
  // BLAS memory at its cap: the BLAS a retained instance names (not drawn
  // lately, the oldest) goes after one no instance names; when no other is
  // old enough, it goes too, and its instance with it.
  {
    uint64_t one = 0;
    {
      auto p = g.Scene();
      if (!p) return failures + Check(false, "scene made");
      Mesh sq = Square(7);
      g.Frame(*p, 1, [&] { p->Add(sq.key, sq.g, wf, a, 3, 4); });
      one = p->Stats().blas_bytes;
    }
    Mesh r = Square(8), d = Square(9), e = Square(10), c = Square(11);
    RtSceneOptions o;
    o.max_blas_bytes = one * 4;
    auto s = g.Scene(o, true);
    if (!s) return failures + Check(false, "scene made");
    uint64_t serial = 1;
    for (int k = 0; k < 2; k++) g.Frame(*s, serial++, [&] { s->Add(r.key, r.g, wb, b, 3, 4); });  // retained
    for (Mesh* x : {&d, &e, &c}) g.Frame(*s, serial++, [&] { s->Add(x->key, x->g, wf, a, 3, 4); });
    // (Four BLASes, at the cap, after the last of those frames: the
    // retained one and d old enough to go. Then a frame drawing nothing.)
    const RtFrame* f = g.Frame(*s, serial++, [] {});
    const std::vector<Hit> h = g.Trace(f, {aft});
    RtStats st = s->Stats();
    bool ok = one > 0 && f && f->instances == 1 && Near(h[0].t, 5) && st.retained_kept == 1 &&
              st.retained_dropped[size_t(RtRetainDrop::Gone)] == 0 && st.blas_bytes == one * 3;
    failures +=
        Check(ok, "BLAS memory capped: one no instance names goes before an older one a retained instance names");

    o.max_blas_bytes = one * 3;
    auto t = g.Scene(o, true);
    if (!t) return failures + Check(false, "scene made");
    serial = 1;
    for (int k = 0; k < 2; k++) g.Frame(*t, serial++, [&] { t->Add(r.key, r.g, wb, b, 3, 4); });
    for (Mesh* x : {&d, &e}) g.Frame(*t, serial++, [&] { t->Add(x->key, x->g, wf, a, 3, 4); });
    // (Three BLASes, at the cap: only the retained one old enough to go.
    // Then a frame drawing nothing: no instance left.)
    f = g.Frame(*t, serial++, [] {});
    st = t->Stats();
    ok =
        !f && st.retained_kept == 0 && st.retained_dropped[size_t(RtRetainDrop::Gone)] == 1 && st.blas_bytes == one * 2;
    failures += Check(ok, "BLAS memory capped, nothing else old enough: the retained instance's BLAS goes, and it");
  }
  // Captured draws are not retained.
  {
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    bool two = true;
    for (uint64_t k = 1; k <= 2; k++) {
      g.arena.used = 0;
      s->BeginFrame(k, k - 1);
      s->Add(n.key, n.g, wb, b, 3, 4);
      RtGeometry cg = m.g;
      cg.positions = nullptr;
      const RtCaptureSlot slot = s->AddCaptured(cg, 0, 4, 3);
      const RtFrame* f = s->EndFrame(true);
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      s->EncodeFrame(cb, f);
      g.Commit(cb);
      two = two && slot.buf && f && f->instances == 2;
    }
    const RtFrame* f = g.Frame(*s, 3, [] {});
    failures += Check(two && f && f->instances == 1 && s->Stats().retained_kept == 1,
                      "a captured draw not retained (the static one is)");
  }
  return failures;
}

// What the scene's bookkeeping of static draws gives, frame after frame:
// the mesh keys a repeated draw hashes (once, every later lookup of it and
// its instance in the frame taking that), the instances of one geometry at
// one place drawn by two views (either first), the other views' BLAS of a
// geometry the main view drew in two alpha-cut variants (the last drawn),
// the instances' order, places forgotten when a build never encoded drops
// its draws in the middle of a frame, the places kept (and let go a frame)
// bounded where every draw moves each frame, and another view's BLAS built
// with its key's hash made late found by it when it goes.
int BookkeepingChecks(Gpu& g) {
  int failures = 0;
  Mat vp;
  Perspective(vp, 1.0f, 1.0f, 0.5f, 1000.0f);
  auto place = [&](float x, Mat w, Mat wvp) {
    Translate(w, x, 0, 5);
    MulRows(w, vp, wvp);
  };
  // A presented frame: the draws before a snapshot and after it, the
  // frame's work committed; the end of frame's TLAS summarised into `sum`.
  auto run = [&](RtScene& s, uint64_t serial, auto before, auto after, std::vector<RtInstanceSummary>* sum = nullptr) {
    g.arena.used = 0;
    s.BeginFrame(serial, serial - 1);
    before();
    const RtFrame* sn = s.Snapshot();
    after();
    s.DescribeNextTlas(sum);
    const RtFrame* en = s.EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s.EncodeFrame(cb, sn);
    s.EncodeFrame(cb, en);
    g.Commit(cb);
    return en;
  };
  RtCut cut;
  cut.material.alpha_scale = 1.0f;
  cut.material.flags = RtMaterialFlags(0, 7, 128, 3, 3, 3);  // GREATEREQUAL 128 (no texture: never always passes)
  {
    // Each kind of draw repeated, with a main view's draw elsewhere (the
    // frame's camera), both retained after their second frame: one key
    // hash a draw a frame from the third frame on.
    enum Kind { kMain, kOther, kReplaced, kDeduped, kMainCut, kKinds };
    const char* names[kKinds] = {"the main view's", "another view's", "another view's then the main view's",
                                 "the main view's then another view's", "the main view's alpha-cut"};
    const uint32_t draws[kKinds] = {2, 2, 3, 3, 2};  // (the camera's draw included)
    Mesh sq = Square(40), other_sq = Square(39);
    const uint8_t opaque_va[4] = {255, 255, 255, 255};
    CutMesh cm = CutSquare(41, 1.0f, opaque_va);
    Mat w, wvp, w2, wvp2;
    place(0, w, wvp);
    place(3, w2, wvp2);
    for (int hit = 0; hit < 2; hit++)
      for (int k = 0; k < kKinds; k++) {
        RtSceneOptions o;
        o.hit_data = hit != 0;
        auto s = g.Scene(o, true);
        if (!s) return failures + Check(false, "scene made");
        auto main_draw = [&] {
          if (k == kMainCut)
            s->Add(cm.key, cm.g, w, wvp, 3, 4, &cut);
          else
            s->Add(sq.key, sq.g, w, wvp, 3, 4);
        };
        auto other_draw = [&] { s->Add(sq.key, sq.g, w, wvp, 3, 4, nullptr, RtView::Other); };
        auto adds = [&] {
          s->Add(other_sq.key, other_sq.g, w2, wvp2, 3, 4);
          if (k == kOther || k == kReplaced) other_draw();
          if (k != kOther) main_draw();
          if (k == kDeduped) other_draw();
        };
        uint64_t worst = 0;
        bool listed = true;
        for (uint64_t f = 1; f <= 6; f++) {
          const uint64_t before = s->Stats().key_hashes;
          const RtFrame* fr = run(*s, f, adds, [] {});
          const uint64_t used = s->Stats().key_hashes - before;
          if (f >= 3) worst = std::max(worst, used);
          listed = listed && fr && fr->instances == 2;
        }
        const RtStats st = s->Stats();
        char what[200];
        std::snprintf(what, sizeof what,
                      "%s draw repeated%s, and the main view's of another mesh: %llu key "
                      "hashes a frame (at most %u), an instance each, retained",
                      names[k], hit ? " (hit data)" : "", (unsigned long long)worst, draws[k]);
        failures += Check(worst <= draws[k] && listed && st.retained_kept == 2 && st.total_builds == 2, what);
      }
  }
  {
    // One geometry at one place drawn by both kinds of view, either first,
    // frame after frame: one instance, listed with any camera; drawn by
    // the other view alone in a later frame: not taken for the earlier
    // frames' instance; the main view's twice: two instances, the other
    // view's draw then deduped against the first.
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    Mesh sq = Square(42);
    Mat w, wvp;
    place(0, w, wvp);
    auto main_draw = [&] { s->Add(sq.key, sq.g, w, wvp, 3, 4); };
    auto other_draw = [&] { s->Add(sq.key, sq.g, w, wvp, 3, 4, nullptr, RtView::Other); };
    std::vector<RtInstanceSummary> sum;
    struct Want {
      uint32_t instances;
      uint64_t deduped;
      std::vector<uint8_t> views;
    };
    const Want want[5] = {{1, 1, {1}}, {1, 2, {1}}, {1, 2, {1}}, {1, 2, {0}}, {2, 3, {1, 0}}};
    bool ok = true;
    for (uint64_t f = 1; f <= 5; f++) {
      const RtFrame* fr = run(
          *s, f, [] {},
          [&] {
            if (f == 1) other_draw(), main_draw();
            if (f == 2) main_draw(), other_draw();
            if (f == 3) other_draw();
            if (f == 4) main_draw();
            if (f == 5) main_draw(), main_draw(), other_draw();
          },
          &sum);
      const Want& x = want[f - 1];
      bool here = fr && fr->instances == x.instances && s->Stats().deduped == x.deduped && sum.size() == x.views.size();
      for (size_t i = 0; here && i < sum.size(); i++) here = sum[i].view == x.views[i] && !sum[i].retained;
      if (!here) std::printf("  frame %llu: not as expected\n", (unsigned long long)f);
      ok = ok && here;
    }
    failures += Check(ok, "one geometry at one place from both views, either first, frame after frame: one instance "
                          "listed with any camera; another view's alone later not taken for an earlier frame's; the "
                          "main view's twice two instances, another view's after them the first's");
  }
  {
    // Hit data: two alpha-cut variants of one geometry drawn by the main
    // view; another view's opaque draw of the geometry (elsewhere) takes the
    // BLAS of the one drawn last, whichever that is.
    RtSceneOptions o;
    o.hit_data = true;
    auto s = g.Scene(o);
    if (!s) return failures + Check(false, "scene made");
    const uint8_t va[4] = {255, 255, 255, 255};
    CutMesh x = CutSquare(43, 1.0f, va), y = x;
    y.key.alpha_lane = 2;  // (another variant: its alpha read from blue)
    y.g.alpha.lane[0] = 2;
    Mesh bare = Square(43);  // (the same vertex buffer and indices: the geometry's key)
    bare.key = x.key;
    bare.key.cut = 0;
    bare.key.attr_version = bare.key.attr_buffer = 0;
    bare.key.attr_stride = bare.key.uv_offset = bare.key.alpha_offset = 0;
    bare.key.uv_type = bare.key.alpha_type = bare.key.uv_lanes = bare.key.alpha_lane = 0;
    bare.g = x.g;
    bare.g.uv = bare.g.alpha = RtAttribute{};
    Mat w, wvp, w2, wvp2;
    place(0, w, wvp);
    place(3, w2, wvp2);
    std::vector<RtInstanceSummary> sum;
    bool ok = true;
    uint32_t mesh_x = 0, mesh_y = 0;
    for (uint64_t f = 1; f <= 4; f++) {
      const bool x_last = f % 2 == 0;
      const RtFrame* fr = run(
          *s, f, [] {},
          [&] {
            CutMesh& first = x_last ? y : x;
            CutMesh& last = x_last ? x : y;
            s->Add(first.key, first.g, w, wvp, 3, 4, &cut);
            s->Add(last.key, last.g, w, wvp, 3, 4, &cut);
            s->Add(bare.key, bare.g, w2, wvp2, 3, 4, nullptr, RtView::Other);
          },
          &sum);
      if (!fr || sum.size() != 3) {
        ok = false;
        continue;
      }
      if (f == 1) mesh_x = sum[0].mesh, mesh_y = sum[1].mesh;
      const uint32_t want = x_last ? mesh_x : mesh_y;
      ok = ok && fr->instances == 3 && sum[2].view == 1 && sum[2].mesh == want;
    }
    const RtStats st = s->Stats();
    failures +=
        Check(ok && mesh_x != 0 && mesh_y != 0 && mesh_x != mesh_y && st.shared_blas == 4 && st.total_builds == 2,
              "hit data: two alpha-cut variants of a geometry; another view's draw of it on the BLAS of the "
              "one the main view drew last, frame after frame");
  }
  {
    // The order of a frame's instances (the TLAS's user IDs) with another
    // view's draws taken over and deduped, the same in every frame.
    auto s = g.Scene({}, true);
    if (!s) return failures + Check(false, "scene made");
    Mesh a = Square(44), b = Square(45), c = Square(46), d = Square(47);
    Mat w[4], wvp[4];
    for (int k = 0; k < 4; k++) place(2.0f * float(k), w[k], wvp[k]);
    std::vector<RtInstanceSummary> sum;
    bool ok = true;
    for (uint64_t f = 1; f <= 4; f++) {
      const RtFrame* fr = run(
          *s, f,
          [&] {
            s->Add(a.key, a.g, w[0], wvp[0], 3, 4, nullptr, RtView::Other);  // draw 0
            s->Add(b.key, b.g, w[1], wvp[1], 3, 4, nullptr, RtView::Other);  // 1
            s->Add(c.key, c.g, w[2], wvp[2], 3, 4);                          // 2
          },
          [&] {
            s->Add(a.key, a.g, w[0], wvp[0], 3, 4);                          // 3: takes draw 0's place
            s->Add(c.key, c.g, w[2], wvp[2], 3, 4, nullptr, RtView::Other);  // 4: deduped
            s->Add(d.key, d.g, w[3], wvp[3], 3, 4);                          // 5
          },
          &sum);
      const uint32_t draws[4] = {3, 1, 2, 5};
      const uint8_t views[4] = {1, 1, 1, 0};
      bool here = fr && fr->instances == 4 && sum.size() == 4;
      for (size_t i = 0; here && i < 4; i++) here = sum[i].draw == draws[i] && sum[i].view == views[i];
      ok = ok && here;
    }
    failures += Check(ok && s->Stats().retained_kept == 4, "the order of a frame's instances, other views' draws taken "
                                                           "over and deduped among them, the same frame after frame");
  }
  {
    // A build never encoded drops its draw in the middle of a presented
    // frame: the frame's places forgotten (another view's draw of a place
    // the frame has is a new instance), then found again by later draws.
    auto s = g.Scene();
    if (!s) return failures + Check(false, "scene made");
    Mesh m = Square(48), n = Square(49);
    Mat w, wvp, w2, wvp2;
    place(0, w, wvp);
    place(3, w2, wvp2);
    run(*s, 1, [] {}, [&] { s->Add(m.key, m.g, w, wvp, 3, 4); });
    g.arena.used = 0;
    s->BeginFrame(2, 1);
    s->Add(m.key, m.g, w, wvp, 3, 4);    // draw 0
    s->Add(n.key, n.g, w2, wvp2, 3, 4);  // 1: built, never encoded
    std::vector<RtInstanceSummary> sum;
    const RtFrame* fr = run(
        *s, 3, [] {},
        [&] {
          s->Add(m.key, m.g, w, wvp, 3, 4, nullptr, RtView::Other);  // 2: a new instance
          s->Add(n.key, n.g, w2, wvp2, 3, 4);                        // 3: built again
          s->Add(m.key, m.g, w, wvp, 3, 4, nullptr, RtView::Other);  // 4: deduped
        },
        &sum);
    const RtStats st = s->Stats();
    const bool ok = fr && fr->instances == 3 && st.deduped == 1 && st.total_builds == 3 && sum.size() == 3 &&
                    sum[0].draw == 0 && sum[0].view == 0 && sum[1].draw == 2 && sum[1].view == 1 && sum[2].draw == 3 &&
                    sum[2].view == 0;
    failures += Check(ok, "a build never encoded dropped in the middle of a frame: its places forgotten, found again "
                          "by the frame's later draws");
  }
  {
    // Every draw at a new world matrix every frame (what moves, a world
    // drawn relative to the camera), from both kinds of view: the places
    // kept stay within kPlacedSlack times a frame's (kPlacedFloor at
    // least), and this frame's while it is drawn; no frame lets go of more
    // than that many at once; every place either kept or let go; one key
    // hash a draw, an instance each.
    auto s = g.Scene();
    if (!s) return failures + Check(false, "scene made");
    Mesh m = Square(50);
    constexpr uint32_t kDraws = 1000;
    constexpr uint64_t kFrames = 40;
    const uint64_t cap = RtScene::kPlacedSlack * std::max<uint64_t>(kDraws, RtScene::kPlacedFloor);
    uint64_t most = 0, most_drawing = 0, most_dropped = 0, worst_hashes = 0;
    bool listed = true;
    for (uint64_t f = 1; f <= kFrames; f++) {
      const RtStats before = s->Stats();
      const RtFrame* fr = run(
          *s, f, [] {},
          [&] {
            for (uint32_t i = 0; i < kDraws; i++) {
              Mat w, wvp;
              Translate(w, float(i % 40) - 20.0f + 0.01f * float(f), float(i / 40) - 12.0f, 50.0f);
              MulRows(w, vp, wvp);
              s->Add(m.key, m.g, w, wvp, 3, 4, nullptr, i % 2 ? RtView::Other : RtView::Main);
            }
            most_drawing = std::max(most_drawing, s->Stats().places);
          });
      const RtStats st = s->Stats();
      most = std::max(most, st.places);
      most_dropped = std::max(most_dropped, st.places_dropped - before.places_dropped);
      if (f >= 2) worst_hashes = std::max(worst_hashes, st.key_hashes - before.key_hashes);
      listed = listed && fr && fr->instances == kDraws;
    }
    const RtStats st = s->Stats();
    char what[300];
    std::snprintf(what, sizeof what,
                  "every draw at a new world matrix every frame: at most %llu places kept (bound %llu; %llu while "
                  "drawn, bound %llu), at most %llu let go in a frame (bound %llu), %llu let go in all, one key "
                  "hash a draw, an instance each",
                  (unsigned long long)most, (unsigned long long)cap, (unsigned long long)most_drawing,
                  (unsigned long long)(cap + kDraws), (unsigned long long)most_dropped, (unsigned long long)cap,
                  (unsigned long long)st.places_dropped);
    failures += Check(most <= cap && most_drawing <= cap + kDraws && most_dropped <= cap &&
                          st.places_dropped + st.places == kDraws * kFrames && worst_hashes <= kDraws && listed &&
                          st.total_builds == 1,
                      what);
  }
  {
    // Hit data: another view's opaque draw of a mesh with UVs that no main
    // view's draw gave a BLAS to share: its own BLAS, built with the key's
    // hash made after its geometry's (the one such case); unused, it goes
    // at the scene's sweep, which finds it by that hash. (With a main view's
    // draw of another mesh elsewhere: the frame's camera.)
    RtSceneOptions o;
    o.hit_data = true;
    auto s = g.Scene(o);
    if (!s) return failures + Check(false, "scene made");
    const uint8_t va[4] = {255, 255, 255, 255};
    CutMesh x = CutSquare(51, 1.0f, va);
    x.key.cut = 0;  // (opaque: its UVs alone, for the albedo)
    x.key.alpha_offset = x.key.alpha_type = x.key.alpha_lane = 0;
    x.g.alpha = RtAttribute{};
    Mesh other = Square(52);
    Mat w, wvp, w2, wvp2;
    place(0, w, wvp);
    place(3, w2, wvp2);
    const RtFrame* fr = run(
        *s, 1, [] {},
        [&] {
          s->Add(other.key, other.g, w2, wvp2, 3, 4);
          s->Add(x.key, x.g, w, wvp, 3, 4, nullptr, RtView::Other);
        });
    const RtStats st = s->Stats();
    const bool built = fr && fr->instances == 2 && st.total_builds == 2 && st.cached == 2 && st.shared_blas == 0;
    uint64_t serial = 2;
    for (uint64_t i = 0; i < RtScene::kEvictAfter + 40; i++, serial++) {
      s->BeginFrame(serial, serial - 1);
      s->EndFrame(true);
    }
    const RtStats aged = s->Stats();
    failures += Check(built && aged.cached == 0 && aged.retired == 0 && aged.blas_bytes == 0,
                      "hit data: another view's opaque draw of a mesh with UVs and no main view's BLAS: built with "
                      "the key's hash made late, let go by the sweep once unused");
  }
  return failures;
}

// The scene's CPU time a static draw repeated frame after frame costs
// (Add, and the snapshot's and end of frame's bookkeeping of its instance):
// kBenchMeshes meshes at their own world matrices, each drawn by another
// view (a shadow map, first) and the main view, retained instances on, a
// snapshot half way; the frames' GPU work never committed. Printed, not
// checked (a timing).
int BenchChecks(Gpu& g) {
  constexpr uint32_t kBenchMeshes = 3000, kWarm = 64, kTimed = 40;
  RtSceneOptions o;
  o.hit_data = true;
  auto s = g.Scene(o, true);
  if (!s) return Check(false, "scene made");
  std::vector<Mesh> ms;
  ms.reserve(kBenchMeshes);
  std::vector<std::array<Mat, 3>> mats(kBenchMeshes);  // world, the main view's wvp, the other view's
  Mat vp, light;
  Perspective(vp, 1.0f, 1.0f, 0.5f, 5000.0f);
  Translate(light, 0, 0, 0, 0.01f);
  for (uint32_t i = 0; i < kBenchMeshes; i++) {
    ms.push_back(Square(1000 + i));
    Translate(mats[i][0], float(i % 60) * 2.0f - 60.0f, float(i / 60) * 2.0f - 50.0f, 100.0f);
    MulRows(mats[i][0], vp, mats[i][1]);
    MulRows(mats[i][0], light, mats[i][2]);
  }
  uint64_t serial = 1;
  auto frame = [&] {
    g.arena.used = 0;
    s->BeginFrame(serial, serial - 1);
    serial++;
    for (uint32_t i = 0; i < kBenchMeshes; i++)
      s->Add(ms[i].key, ms[i].g, mats[i][0], mats[i][2], 3, 4, nullptr, RtView::Other);
    for (uint32_t i = 0; i < kBenchMeshes / 2; i++) s->Add(ms[i].key, ms[i].g, mats[i][0], mats[i][1], 3, 4);
    s->Snapshot();
    for (uint32_t i = kBenchMeshes / 2; i < kBenchMeshes; i++) s->Add(ms[i].key, ms[i].g, mats[i][0], mats[i][1], 3, 4);
    return s->EndFrame(true);
  };
  for (uint32_t k = 0; k < kWarm; k++) frame();  // (the builds, kBuildsPerFrame a frame)
  const RtStats warm = s->Stats();
  const auto t0 = std::chrono::steady_clock::now();
  const RtFrame* f = nullptr;
  for (uint32_t k = 0; k < kTimed; k++) f = frame();
  const auto t1 = std::chrono::steady_clock::now();
  const double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  const RtStats st = s->Stats();
  std::printf("  scene bookkeeping: %.0f ns a draw (%u meshes, each drawn by two views, %u frames)\n",
              ns / (double(kTimed) * kBenchMeshes * 2), kBenchMeshes, kTimed);
  return Check(warm.cached == kBenchMeshes && st.total_builds == warm.total_builds && f &&
                   f->instances == kBenchMeshes && st.retained_kept == kBenchMeshes,
               "bookkeeping benchmark: every mesh built in the warm-up, one instance each, all retained");
}
}  // namespace

int main() {
  @autoreleasepool {
    int failures = CpuChecks() + AlphaCpuChecks() + ResolveChecks();
    Gpu g;
    if (!g.dev || !RtSupported(g.dev)) {
      std::printf("skipped: %s\n", g.dev ? "the GPU does not trace rays in hardware (Apple9)" : "no Metal device");
      return failures ? 1 : 77;
    }
    if (!g.Init()) return 1;
    failures += BlasChecks(g);
    failures += TlasChecks(g);
    failures += CameraChecks(g);
    failures += WindingChecks(g);
    failures += CacheChecks(g);
    failures += FrameChecks(g);
    failures += InFlightChecks(g);
    failures += DebugChecks(g);
    failures += AlphaChecks(g);
    failures += AlphaMoreChecks(g);
    failures += RetainChecks(g);
    failures += BookkeepingChecks(g);
    failures += BenchChecks(g);
    if (g.arena.full) failures += Check(false, "test arena large enough");
    return failures ? 1 : 0;
  }
}
