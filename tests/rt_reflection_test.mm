// Ray-traced reflections (gfx/rt_reflection.h) and the scene's hit data
// (gfx/rt_scene.h: RtHitPrim, RtMaterial version 2, RtAlbedo).
// On the CPU: a triangle's normal and LOD constant (RtHitTriangle); where a
// draw's albedo comes from (RtResolveAlbedo: sampler 0 at TEXCOORD0, the
// game's hook, an alpha-cut draw's texture and UVs, refused for a cube
// sampler, a missing texture or input, UVs other than the cut's); a draw's
// constants and the kernel's parameters (the light's sign, the colour times
// its w, the blur's register or the exponent's scale, wetness only with a
// slope); one snapshot per depth contents within a command buffer
// (RtSnapshotReuse).
// On a GPU with hardware ray tracing: the BLAS memory hit data adds (printed:
// the same mesh with and without); the material table and RtHitPrim as the
// GPU reads them (version 2, the albedo's level and addressing, the
// normal, the UVs and the LOD constant against the CPU's), the albedo's
// level from ray cones against the analytic one at three distances
// (rising with distance) and on a scaled instance; and a mirror (the ground,
// y = 0) in front of a textured box (five quads, both halves of the texture
// told apart) seen from a camera above, with a synthetic G-buffer (depth
// reversed in a depth texture or not in a float one, normals, specular
// intensity and gloss): every judged pixel's reflection against an analytic
// trace (a 1 and the texel times the light (wrapped diffuse, a shadow ray)
// plus the ambient, times the scale, where the ray hits the box; the hit's
// distance; a 0 and the sky's distance where it misses; nothing where the
// surface is not shiny), an occluder over the box (its faces then get the
// ambient only, a hit on it, without an albedo, a 0), an alpha-cut quad in
// front whose hole lets rays through to the box, gloss blurrier than the
// cutoff (no ray at all), wet pixels (spec 0, the stencil 0) traced and dry
// ones not, the counters; the shadow mask traced against the same snapshot
// (one TLAS for both) against the same analytic shadows.
//
//   rt_reflection_test        (skipped, exit 77, without a GPU that traces rays)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/rt_alpha.h"
#include "runtime/gfx/rt_reflection.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"
#include "runtime/hlsl/hlsl.h"

namespace {

using namespace recomp::gfx;
namespace hlsl = recomp::hlsl;

int Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  return ok ? 0 : 1;
}

using Mat = float[4][4];

struct V3 {
  double x, y, z;
};
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 Cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 Unit(V3 a) { return a * (1.0 / std::sqrt(Dot(a, a))); }
double Len(V3 a) { return std::sqrt(Dot(a, a)); }
double Sat(double v) { return std::clamp(v, 0.0, 1.0); }

// D3D's left-handed camera at `eye` looking at `at` (y up), its
// perspective (depth 0..1 at near..far), as world to clip rows.
void Camera(V3 eye, V3 at, double fov_y, double aspect, double n, double f, double vp[16]) {
  const V3 fw = Unit(at - eye), rt = Unit(Cross({0, 1, 0}, fw)), up = Cross(fw, rt);
  const double view[16] = {rt.x, up.x, fw.x, 0, rt.y, up.y, fw.y, 0, rt.z, up.z, fw.z, 0,
                           -Dot(rt, eye), -Dot(up, eye), -Dot(fw, eye), 1};
  const double ys = 1.0 / std::tan(fov_y * 0.5);
  double proj[16] = {};
  proj[0] = ys / aspect;
  proj[5] = ys;
  proj[10] = f / (f - n);
  proj[11] = 1;
  proj[14] = -n * f / (f - n);
  Mul4(view, proj, vp);
}

void Identity(Mat m, double scale = 1.0) {
  std::memset(m, 0, sizeof(Mat));
  m[0][0] = m[1][1] = m[2][2] = float(scale);
  m[3][3] = 1;
}

// world * vp (double) as float rows.
void Wvp(const Mat w, const double* vp, Mat out) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) {
      double v = 0;
      for (int i = 0; i < 4; i++) v += double(w[r][i]) * vp[i * 4 + k];
      out[r][k] = float(v);
    }
}

// A quad as a strip of 2 triangles: 24-byte vertices (FLOAT3 position,
// FLOAT2 uv, D3DCOLOR), in buffer number `buffer`; p[0] at uv (0, 0), p[1]
// (0, 1), p[2] (1, 0), p[3] (1, 1). Its UVs in the key and the geometry
// (an albedo's, or with `cut` an alpha-cut draw's).
struct Quad {
  std::vector<uint8_t> vb;
  std::vector<uint16_t> ib = {0, 1, 2, 3};
  RtGeometry g;
  RtMeshKey key;
};
Quad MakeQuad(const V3 p[4], uint64_t buffer, bool cut) {
  Quad q;
  q.vb.resize(4 * 24);
  const float uv[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (int i = 0; i < 4; i++) {
    uint8_t* v = q.vb.data() + i * 24;
    const float pos[3] = {float(p[i].x), float(p[i].y), float(p[i].z)};
    std::memcpy(v, pos, 12);
    std::memcpy(v + 12, uv[i], 8);
    const uint32_t c = 0xFFFFFFFFu;
    std::memcpy(v + 20, &c, 4);
  }
  RtGeometry& g = q.g;
  g.positions = q.vb.data();
  g.size = q.vb.size();
  g.stride = 24;
  g.type = 2;
  g.prim = 5;
  g.primitives = 2;
  g.indices = q.ib.data();
  g.index_size = 2;
  g.uv = {q.vb.data() + 12, q.vb.size() - 12, 24, 1, {0, 1}};
  RtMeshKey& k = q.key;
  k.vertex_version = buffer * 16 + 1;
  k.index_version = buffer * 16 + 2;
  k.vertex_buffer = buffer;
  k.index_buffer = buffer + (1u << 20);
  k.prim = 5;
  k.primitives = 2;
  k.stride = 24;
  k.type = 2;
  k.index_size = 2;
  k.cut = cut ? 1 : 0;
  k.attr_version = k.vertex_version;
  k.attr_buffer = buffer;
  k.attr_stride = 24;
  k.uv_type = 1;
  k.uv_offset = 12;
  k.uv_lanes = 0 | 1 << 2;
  k.alpha_type = 17;
  return q;
}

// ---- CPU

int CpuChecks() {
  int failures = 0;
  {
    // A right triangle of legs 2 and 4 (area 4) over UVs of legs 0.5 and 1
    // (area 0.25): 0.5 log2(0.25 / 4) = -2.
    RtCutPrim cp{};
    cp.uv[1][0] = 0.5f;
    cp.uv[2][1] = 1.0f;
    const float a[3] = {1, 1, 1}, b[3] = {3, 1, 1}, c[3] = {1, 5, 1};
    cp.uv[0][0] = 100.25f, cp.uv[1][0] += 100.25f, cp.uv[2][0] = 100.25f;  // tiled far out
    cp.alpha = 0x00FF8040u;
    RtHitPrim hp;
    RtHitTriangle(cp, a, b, c, &hp);
    RtCutPrim none{};
    RtHitPrim hn;
    RtHitTriangle(none, a, b, c, &hn);
    float n[3];
    RtOctDecode(hp.normal, n);
    const bool ok = n[0] == 0 && n[1] == 0 && n[2] == 1 && std::fabs(hp.lod + 2.0f) < 1e-6f && hp.uv0[0] == 100.25f &&
                    HalfToFloat(uint16_t(hp.duv[0])) == 0.5f && HalfToFloat(uint16_t(hp.duv[0] >> 16)) == 0 &&
                    HalfToFloat(uint16_t(hp.duv[1])) == 0 && HalfToFloat(uint16_t(hp.duv[1] >> 16)) == 1.0f &&
                    hp.alpha == 0x00FF8040u && hn.lod == 0;
    failures += Check(ok, "RtHitTriangle: corner 0's UV, the others' less it as halves, the alphas, the normal "
                          "octahedral, 0.5 log2(uv area / area), 0 without UVs");
  }
  {
    // Halves: exact ones, rounding to even, subnormals, overflow, NaN.
    const struct {
      float f;
      uint16_t h;
    } kHalves[] = {{0.5f, 0x3800},
                   {-2.25f, 0xC080},
                   {65504.0f, 0x7BFF},
                   {65520.0f, 0x7C00},
                   {1.0f + 1.0f / 4096, 0x3C00},
                   {1.0f + 3.0f / 4096, 0x3C01},
                   {1.0f + 2.0f / 4096, 0x3C00},
                   {1.0f + 6.0f / 4096, 0x3C02},
                   {5.9604645e-08f, 0x0001},
                   {2.9802322e-08f, 0x0000},
                   {6.1035156e-05f, 0x0400},
                   {-0.0f, 0x8000},
                   {1e10f, 0x7C00}};
    bool ok = true;
    for (const auto& k : kHalves) {
      const uint16_t got = FloatToHalf(k.f);
      if (got != k.h) std::printf("  half of %g: %04x, want %04x\n", double(k.f), got, k.h);
      ok = ok && got == k.h;
    }
    const uint16_t nan = FloatToHalf(std::nanf(""));
    ok = ok && (nan & 0x7C00) == 0x7C00 && (nan & 0x3FF) != 0;
    // Octahedral normals: directions around the sphere back within 1e-4;
    // a zero one +z.
    double worst = 0;
    for (int i = 0; i < 64; i++) {
      const double th = std::acos(1.0 - 2.0 * (i + 0.5) / 64), ph = i * 2.399963;
      const float d[3] = {float(std::sin(th) * std::cos(ph) * 3), float(std::sin(th) * std::sin(ph) * 3),
                          float(std::cos(th) * 3)};
      float o[3];
      RtOctDecode(RtOctNormal(d), o);
      for (int k = 0; k < 3; k++) worst = std::max(worst, std::fabs(o[k] - d[k] / 3.0));
    }
    const float zero[3] = {};
    float z[3];
    RtOctDecode(RtOctNormal(zero), z);
    std::printf("  octahedral normals: worst component error %.2g\n", worst);
    failures += Check(ok && worst < 1e-4 && z[2] == 1, "FloatToHalf rounds as IEEE; octahedral normals within 1e-4");
  }
  {
    // Albedo sources: 24-byte vertices, POSITION (slot 0), TEXCOORD0 (slot
    // 1, FLOAT2 at 12), TEXCOORD1 (slot 2, FLOAT2 at 0 of stream 1).
    std::vector<uint8_t> vb(4 * 24, 0), vb1(4 * 8, 0);
    VertexInput in[3] = {{0, 0, 2}, {0, 12, 1}, {1, 0, 1}};
    uint32_t ss[16][14] = {};
    ss[0][1] = 3, ss[0][2] = 2, ss[0][9] = 2;  // CLAMP, MIRROR, MAXMIPLEVEL 2
    ss[0][4] = 0xFF102030u;                    // BORDERCOLOR
    ShaderDraw d;
    d.inputs = in;
    d.input_count = 3;
    d.streams[0] = {vb.data(), uint32_t(vb.size()), 24};
    d.streams[1] = {vb1.data(), uint32_t(vb1.size()), 8};
    d.sampler_states = ss;
    RtAlphaTexture tex[16];
    for (RtAlphaTexture& t : tex) t = {true, true, false};
    RtAlphaShaders sh;
    sh.sampler_type.fill(hlsl::ParamType::Sampler2D);
    sh.tex_slot[0] = 1;
    sh.tex_slot[1] = 2;
    RtAlbedoResult r = RtResolveAlbedo(d, sh, tex, nullptr);
    bool ok = r.on && r.sampler == 0 && r.uv.data == vb.data() + 12 && r.uv.stride == 24 && r.uv.type == 1 &&
              r.stream == 0 && r.albedo.max_level == 2 && r.albedo.address_u == 3 && r.albedo.address_v == 2 &&
              r.albedo.border == 0xFF102030u;
    failures += Check(ok, "albedo: D3D9's usual sampler 0 at TEXCOORD0, its MAXMIPLEVEL, addressing and border");
    RtAlphaShaders hooked = sh;
    hooked.albedo_hook = true;
    hooked.albedo_sampler = 3;
    hooked.albedo_texcoord = 1;
    r = RtResolveAlbedo(d, hooked, tex, nullptr);
    ok = r.on && r.sampler == 3 && r.uv.data == vb1.data() && r.stream == 1 && r.albedo.max_level == 0 &&
         r.albedo.address_u == 0;
    failures += Check(ok, "albedo: the game's hook (s3 at TEXCOORD1, its sampler's states)");
    // An alpha-cut draw: its texture and UVs.
    RtAlphaResult cut;
    cut.kind = RtAlphaKind::Cut;
    cut.sampler = 5;
    cut.uv = {vb1.data(), vb1.size(), 8, 1, {1, 0}};
    cut.uv_in = in[2];
    cut.stream = 1;
    r = RtResolveAlbedo(d, sh, tex, &cut);
    ok = r.on && r.sampler == 5 && r.uv.data == vb1.data() && r.uv.lane[0] == 1 && r.uv.lane[1] == 0 && r.stream == 1;
    // The hook naming other UVs than the cut's: none (one set a triangle).
    RtAlphaShaders other = hooked;
    other.albedo_texcoord = 0;
    const RtAlbedoResult x = RtResolveAlbedo(d, other, tex, &cut);
    failures += Check(ok && !x.on, "albedo: an alpha-cut draw's texture and UVs; another set of UVs refused");
    RtAlphaShaders cube = sh;
    cube.sampler_type[0] = hlsl::ParamType::SamplerCube;
    RtAlphaTexture missing[16];
    for (RtAlphaTexture& t : missing) t = {true, true, false};
    missing[0].present = false;
    RtAlphaShaders no_uv = sh;
    no_uv.tex_slot[0] = -1;
    failures += Check(!RtResolveAlbedo(d, cube, tex, nullptr).on && !RtResolveAlbedo(d, sh, missing, nullptr).on &&
                          !RtResolveAlbedo(d, no_uv, tex, nullptr).on,
                      "albedo: none for a cube sampler, a missing texture, a missing TEXCOORD0");
  }
  {
    float c[224][4] = {};
    c[17][0] = 0.6f, c[17][1] = -0.8f;            // away from the light
    c[18][0] = 1, c[18][1] = 0.5f, c[18][3] = 2;  // colour, times w
    c[37][2] = 0.25f;
    c[38][0] = 0.125f;
    c[72][0] = 0.01f, c[72][1] = 0.75f;
    RayReflection r;
    r.light_register = 17;
    r.colour_register = 18;
    r.ambient_registers[0] = 37;
    r.ambient_registers[1] = 38;
    r.blur_register = 72;
    r.wet_register = 72;
    r.wet_component = 1;
    RtReflectionInputs in = RtReflectionInputsFor(r, c);
    bool ok = in.light[0] == -0.6f && in.light[1] == 0.8f && in.colour[0] == 2 && in.colour[1] == 1 &&
              in.ambient[0][2] == 0.25f && in.ambient[1][0] == 0.125f && in.blur_k == 0.01f && in.wet == 0;
    r.wet_slope = 4;
    r.blur_register = 224;
    r.exponent_scale = 256;
    in = RtReflectionInputsFor(r, c);
    ok = ok && in.wet == 0.75f && in.blur_k == 1.0f / 256;
    failures += Check(ok, "inputs: the light toward it, the colour times its w, the blur's register (or 1 / the "
                          "exponent's scale), wetness only with a slope");
  }
  {
    RtSnapshotMemo m;
    RtShadowWant w;
    int dummy = 0;
    w.depth = &dummy;
    w.epoch = 5;
    w.depth_camera = w.scene_camera = 2;
    const bool first = RtSnapshotReuse(&m, w, 10), again = RtSnapshotReuse(&m, w, 10);
    w.light[0] = 1;  // (the light is the mask's business, not the snapshot's)
    const bool light = RtSnapshotReuse(&m, w, 10);
    const bool serial = RtSnapshotReuse(&m, w, 11);
    w.epoch = 6;
    const bool epoch = RtSnapshotReuse(&m, w, 11);
    m = {};
    const bool reset = RtSnapshotReuse(&m, w, 11);
    failures += Check(!first && again && light && !serial && !epoch && !reset,
                      "one snapshot per depth contents and cameras within a command buffer, again after a write, "
                      "another command buffer, a presented frame");
  }
  {
    // The passes' cache: a key (shader config, constants, textures) keeps
    // its entry and its choice; another config, constants or texture
    // another entry; a third key takes over the least recently used (traced
    // again), the other kept; NaN constants are the same bit for bit.
    RtReflectionCache c;
    RtReflectionKey a, b;
    int t1 = 1, t2 = 2;
    a.textures[0] = &t1;
    a.inputs.wet = std::nanf("");
    b = a;
    b.config.output_scale = 0.5f;  // another shader's
    RtShadowWant w;
    w.depth = &t1;
    w.depth_camera = w.scene_camera = 1;
    w.main_size = true;
    w.light[2] = 1;
    const int ea = RtReflectionEntry(&c, a);
    const bool trace_a = RtShadowNext(&c.entry[ea].memo, w) == RtShadowStep::Trace;
    const int eb = RtReflectionEntry(&c, b);
    const bool trace_b = RtShadowNext(&c.entry[eb].memo, w) == RtShadowStep::Trace;
    const int ea2 = RtReflectionEntry(&c, a);
    const bool reuse_a = RtShadowNext(&c.entry[ea2].memo, w) == RtShadowStep::Reuse;
    RtReflectionKey x = a;
    x.textures[0] = &t2;
    RtReflectionKey y = a;
    y.inputs.light[0] = 1;
    RtReflectionKey z = a;
    z.config.wet_slope = 4;
    const bool differ = !RtReflectionSameKey(a, x) && !RtReflectionSameKey(a, y) && !RtReflectionSameKey(a, z) &&
                        !RtReflectionSameKey(a, b) && RtReflectionSameKey(a, a);
    const int ex = RtReflectionEntry(&c, x);  // takes b's (used before a's last use)
    const bool x_trace = ex == eb && RtShadowNext(&c.entry[ex].memo, w) == RtShadowStep::Trace;
    const int ea3 = RtReflectionEntry(&c, a);  // a still there
    const bool a_kept = ea3 == ea && RtShadowNext(&c.entry[ea3].memo, w) == RtShadowStep::Reuse;
    const int eb2 = RtReflectionEntry(&c, b);  // b again: x's (now the least recently used), traced again
    const bool b_again = eb2 == ex && RtShadowNext(&c.entry[eb2].memo, w) == RtShadowStep::Trace;
    failures += Check(ea != eb && trace_a && trace_b && ea2 == ea && reuse_a && differ && x_trace && a_kept && b_again,
                      "reflection passes: one per shader config, constants and textures (two kept, NaNs bit for "
                      "bit), reused while the depth stays, the least recently used taken over");
  }
  return failures;
}

// ---- GPU

struct Arena {
  id<MTLBuffer> buf = nil;
  size_t used = 0;
};

RtUpload ArenaAlloc(void* ctx, size_t bytes, size_t align) {
  auto* a = static_cast<Arena*>(ctx);
  const size_t off = (a->used + align - 1) & ~(align - 1);
  if (off + bytes > a->buf.length) return {nil, 0, nullptr};
  a->used = off + bytes;
  return {a->buf, off, static_cast<uint8_t*>(a->buf.contents) + off};
}

struct Gpu {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  Arena arena;
  std::unique_ptr<RtReflector> reflector;
  std::unique_ptr<RtShadowMasker> masker;
  id<MTLComputePipelineState> probe = nil;
  // The outputs the last case traced into (Case::reuse).
  id<MTLTexture> last_out = nil, last_dist = nil;

  bool Init();
  bool Commit(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }
  id<MTLTexture> Texture(MTLPixelFormat pf, uint32_t w, uint32_t h, const void* data, size_t row) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [dev newTextureWithDescriptor:td];
    [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data bytesPerRow:row];
    return t;
  }
  // The texels of `t` (bpp bytes each).
  std::vector<uint8_t> Read(id<MTLTexture> t, uint32_t bpp) {
    const uint32_t w = uint32_t(t.width), h = uint32_t(t.height);
    id<MTLBuffer> out = [dev newBufferWithLength:size_t(w) * h * bpp options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(w, h, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:w * bpp
        destinationBytesPerImage:size_t(w) * h * bpp];
    [bl endEncoding];
    if (!Commit(cb)) return {};
    const auto* p = static_cast<const uint8_t*>(out.contents);
    return std::vector<uint8_t>(p, p + size_t(w) * h * bpp);
  }
};

// The material table, RtHitPrim and the albedo's level as the GPU reads
// them: per ray (rays[2i]: origin, the cone's width per unit of distance;
// rays[2i + 1]: direction, the texture's side) four float4s.
constexpr const char* kProbeMsl = R"(
kernel void probe_cs(instance_acceleration_structure scene [[buffer(0)]],
                     const device RtMaterial* materials [[buffer(1)]],
                     const device float4* rays [[buffer(2)]], device float4* out [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  const float4 o = rays[2 * i], d = rays[2 * i + 1];
  const RtHit h = RtTraceClosest(scene, ray(o.xyz, d.xyz, 0.0, INFINITY), 0xFF, materials, intersection_params());
  if (!h.hit) {
    out[4 * i] = float4(-1.0);
    return;
  }
  const device RtHitPrim& p = *(const device RtHitPrim*)h.prim;
  const device RtMaterial& m = materials[h.instance];
  const float cone = o.w * h.distance;
  out[4 * i] = float4(h.distance, RtHitLod(h, d.xyz, cone, float2(d.w)), p.lod, float(RT_MATERIAL_VERSION));
  out[4 * i + 1] = float4(rt_oct_decode(p.normal), as_type<float>(m.flags));
  out[4 * i + 2] = float4(rt_hit_uv(p, float3(0.0, 1.0, 0.0)), m.albedo_max_level, as_type<float>(m.albedo_flags));
  out[4 * i + 3] = float4(float(sizeof(RtMaterial)), float(sizeof(RtHitPrim)), RtHitAlbedo(m, h, d.xyz, cone).r,
                          as_type<float>(m.albedo_border));
}
)";

bool Gpu::Init() {
  arena.buf = [dev newBufferWithLength:32u << 20 options:MTLResourceStorageModeShared];
  reflector = RtReflector::Make(dev);
  masker = RtShadowMasker::Make(dev);
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@((std::string(kRtMsl) + kProbeMsl).c_str()) options:nil error:&err];
  if (lib) probe = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe_cs"] error:&err];
  if (!probe) std::printf("FAIL probe kernel: %s\n", err.localizedDescription.UTF8String);
  return arena.buf && reflector && masker && probe;
}

// The BLAS memory hit data adds: a grid of 2 x 128 x 128 triangles with
// UVs in a scene without and with it (printed), the same without UVs (no
// more); another view's draw of it without UVs (another place) on the main
// view's BLAS (none built).
int MemoryChecks(Gpu& g) {
  constexpr uint32_t kN = 128;
  std::vector<uint8_t> vb(size_t(kN + 1) * (kN + 1) * 24, 0);
  std::vector<uint32_t> ib;
  for (uint32_t j = 0; j <= kN; j++)
    for (uint32_t i = 0; i <= kN; i++) {
      const float p[5] = {float(i), 0.0f, float(j), float(i) / kN, float(j) / kN};
      std::memcpy(&vb[(size_t(j) * (kN + 1) + i) * 24], p, 20);
    }
  for (uint32_t j = 0; j < kN; j++)
    for (uint32_t i = 0; i < kN; i++) {
      const uint32_t a = j * (kN + 1) + i, b = a + 1, c = a + kN + 1, d = c + 1;
      ib.insert(ib.end(), {a, c, b, b, c, d});
    }
  RtGeometry bare;
  bare.positions = vb.data();
  bare.size = vb.size();
  bare.stride = 24;
  bare.prim = 4;
  bare.primitives = kN * kN * 2;
  bare.indices = ib.data();
  bare.index_size = 4;
  RtMeshKey kb;
  kb.vertex_version = 1;
  kb.index_version = 2;
  kb.vertex_buffer = 1;
  kb.index_buffer = 2;
  kb.prim = 4;
  kb.primitives = bare.primitives;
  kb.stride = 24;
  kb.type = 2;
  kb.index_size = 4;
  RtGeometry uv = bare;
  uv.uv = {vb.data() + 12, vb.size() - 12, 24, 1, {0, 1}};
  RtMeshKey ku = kb;
  ku.attr_version = 1;
  ku.attr_buffer = 1;
  ku.attr_stride = 24;
  ku.uv_type = 1;
  ku.uv_offset = 12;
  ku.uv_lanes = 0 | 1 << 2;
  ku.alpha_type = 17;
  uint64_t bytes[3] = {};
  RtStats shared;
  for (int run = 0; run < 3; run++) {
    RtSceneOptions o;
    o.hit_data = run != 0;
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, o);
    g.arena.used = 0;
    s->SetAllocator(ArenaAlloc, &g.arena);
    s->BeginFrame(1, 0);
    Mat w, wvp, w2;
    Identity(w);
    Identity(wvp);
    Identity(w2);
    w2[3][0] = 500;
    if (run == 2) {
      s->Add(kb, bare, w, wvp, 1, 4);
    } else {
      s->Add(ku, uv, w, wvp, 1, 4);
    }
    bytes[run] = s->Stats().blas_bytes;
    if (run == 1) {
      s->Add(kb, bare, w2, wvp, 1, 4, nullptr, RtView::Other);
      shared = s->Stats();
    }
  }
  const double per = double(int64_t(bytes[1]) - int64_t(bytes[0])) / double(uv.primitives);
  std::printf("  BLAS of %u triangles: %llu bytes without hit data, %llu with (%.1f bytes a triangle more; "
              "RtHitPrim is %zu), %llu with hit data and no UVs\n",
              uv.primitives, (unsigned long long)bytes[0], (unsigned long long)bytes[1], per, sizeof(RtHitPrim),
              (unsigned long long)bytes[2]);
  int failures = Check(bytes[0] > 0 && bytes[1] > bytes[0] && bytes[2] == bytes[0],
                       "hit data: the BLASes of meshes with UVs larger by their RtHitPrim, of those without not");
  failures += Check(shared.shared_blas == 1 && shared.cached == 1 && shared.blas_bytes == bytes[1],
                    "hit data: another view's draw without UVs on the main view's BLAS (none built)");
  {
    // The other view drawing first (shadow maps before the scene): its own
    // BLAS in the first frame, the main view's from the next on; its own
    // ages out.
    RtSceneOptions o;
    o.hit_data = true;
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, o);
    s->SetAllocator(ArenaAlloc, &g.arena);
    Mat w, wvp, w2;
    Identity(w);
    Identity(wvp);
    Identity(w2);
    w2[3][0] = 500;
    uint64_t shared2 = 0, cached1 = 0;
    const uint64_t frames = RtScene::kEvictAfter + 40;
    for (uint64_t f = 1; f <= frames; f++) {
      g.arena.used = 0;
      s->BeginFrame(f, f - 1);
      s->Add(kb, bare, w2, wvp, 1, 4, nullptr, RtView::Other);
      s->Add(ku, uv, w, wvp, 1, 4);
      if (f == 1) cached1 = s->Stats().cached;
      if (f == 2) shared2 = s->Stats().shared_blas;
      s->EndFrame(true);
    }
    const RtStats end = s->Stats();
    std::printf("  other view first: %llu BLASes after frame 1, %llu shared by frame 2, %llu after %llu frames\n",
                (unsigned long long)cached1, (unsigned long long)shared2, (unsigned long long)end.cached,
                (unsigned long long)frames);
    failures += Check(cached1 == 2 && shared2 == 1 && end.cached == 1,
                      "hit data: an other view drawing before the main view takes its BLAS from the next frame on; "
                      "its own ages out");
  }
  return failures;
}

// One quad 8 x 4 at z = 0 (x 0..8, y 0..4), its texture 256 x 256 white
// with 3 levels, MAXMIPLEVEL 1, addressing CLAMP/MIRROR, a border colour;
// rays from z = -d straight at it: the hit data and table, and the albedo's
// level against 0.5 log2(1 / 32) + 8 + log2(cone / 1); the same quad scaled
// 2x (its world area 4x, a level less). Unscaled, beside it: the same quad
// captured (its normal and LOD constant made on the GPU from the captured
// positions and the halves of its UVs), captured with UVs that cannot be
// read and drawn static with such UVs (both still in the scene, without an
// albedo).
int ProbeChecks(Gpu& g) {
  int failures = 0;
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:256
                                                                               height:256
                                                                            mipmapped:YES];
  td.mipmapLevelCount = 3;
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> tex = [g.dev newTextureWithDescriptor:td];
  std::vector<uint32_t> white(256 * 256, 0xFFFFFFFFu);
  for (int l = 0; l < 3; l++)
    [tex replaceRegion:MTLRegionMake2D(0, 0, 256 >> l, 256 >> l) mipmapLevel:NSUInteger(l) withBytes:white.data()
           bytesPerRow:(256 >> l) * 4];
  auto quad = [](double x0, uint64_t buffer) {
    const V3 p[4] = {{x0, 0, 0}, {x0, 4, 0}, {x0 + 8, 0, 0}, {x0 + 8, 4, 0}};
    return MakeQuad(p, buffer, false);
  };
  Quad q = quad(0, 1), qc = quad(20, 2), qbad = quad(40, 3), qs = quad(60, 4);
  qbad.g.uv.size = 0;  // (UVs that cannot be read)
  qs.g.uv.size = 0;
  // The captured quads' keys (their UVs' and triangles' sources: kept from
  // one frame to the next).
  RtMeshKey kc = qc.key, kbad = qbad.key;
  for (RtMeshKey* k : {&kc, &kbad}) {
    k->vertex_version = k->vertex_buffer = 0;
    k->min_index = 0;
    k->vertex_count = 4;
  }
  double results[2][3] = {};
  bool layout = true, captured = false, bad = false;
  RtStats st;
  for (int scaled = 0; scaled < 2; scaled++) {
    RtSceneOptions o;
    o.hit_data = true;
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, o);
    s->SetAllocator(ArenaAlloc, &g.arena);
    Mat w, wvp;
    Identity(w, scaled ? 2.0 : 1.0);
    Identity(wvp);
    RtAlbedo a;
    a.texture = tex;
    a.max_level = 1;
    a.address_u = 3;
    a.address_v = 2;
    a.border = 0xFF336699u;
    RtGeometry gc = qc.g, gb = qbad.g;
    gc.positions = gb.positions = nullptr;
    // The captured quads' clip positions (the camera is the identity),
    // where their capture would write them.
    auto capture = [&](id<MTLCommandBuffer> cb, const RtCaptureSlot& slot, float x0) {
      const float clip[16] = {x0, 0, 0, 1, x0, 4, 0, 1, x0 + 8, 0, 0, 1, x0 + 8, 4, 0, 1};
      id<MTLBuffer> src = [g.dev newBufferWithBytes:clip length:sizeof clip options:MTLResourceStorageModeShared];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      [bl copyFromBuffer:src sourceOffset:0 toBuffer:slot.buf destinationOffset:slot.capture_off size:sizeof clip];
      [bl endEncoding];
    };
    uint64_t frame = 1;
    if (!scaled) {
      // A first frame drawing the same (the captured quads' data made then).
      g.arena.used = 0;
      s->BeginFrame(frame, frame - 1);
      s->Add(q.key, q.g, w, wvp, 1, 4, nullptr, RtView::Main, &a);
      const RtCaptureSlot c0 = s->AddCaptured(gc, 0, 4, 1, nullptr, &a, &kc);
      const RtCaptureSlot b0 = s->AddCaptured(gb, 0, 4, 1, nullptr, &a, &kbad);
      s->Add(qs.key, qs.g, w, wvp, 1, 4, nullptr, RtView::Main, &a);
      const RtFrame* end = s->EndFrame(true);
      if (!c0.buf || !b0.buf || !end) return Check(false, "probe: the first frame");
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      capture(cb, c0, 20);
      capture(cb, b0, 40);
      s->EncodeFrame(cb, end);
      if (!g.Commit(cb)) return 1;
      frame++;
    }
    g.arena.used = 0;
    s->BeginFrame(frame, frame - 1);
    s->Add(q.key, q.g, w, wvp, 1, 4, nullptr, RtView::Main, &a);
    RtCaptureSlot slot_c{}, slot_bad{};
    bool static_bad = false;
    if (!scaled) {
      slot_c = s->AddCaptured(gc, 0, 4, 1, nullptr, &a, &kc);
      slot_bad = s->AddCaptured(gb, 0, 4, 1, nullptr, &a, &kbad);
      static_bad = s->Add(qs.key, qs.g, w, wvp, 1, 4, nullptr, RtView::Main, &a);
    }
    const RtFrame* f = s->Snapshot();
    if (!f || !f->tlas || (!scaled && (!slot_c.buf || !slot_bad.buf || !static_bad)))
      return Check(false, "probe: a TLAS, the captured quads and the static one taken");
    const int n = scaled ? 3 : 6;
    float rays[6][8];
    for (int i = 0; i < n; i++) {
      const double d = i < 3 ? (i == 0 ? 10 : i == 1 ? 40 : 160) : 10;
      const float x = i < 3 ? 2.0f : float(2 + 20 * (i - 2));
      const float c[8] = {x, 1.0f, float(-d), 0.001f, 0, 0, 1, 256};
      std::memcpy(rays[i], c, sizeof c);
    }
    id<MTLBuffer> rb = [g.dev newBufferWithBytes:rays length:sizeof rays options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob = [g.dev newBufferWithLength:6 * 64 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    if (!scaled) {
      capture(cb, slot_c, 20);
      capture(cb, slot_bad, 40);
    }
    s->EncodeFrame(cb, f);
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g.probe];
    [e setAccelerationStructure:f->tlas atBufferIndex:0];
    [e setBuffer:f->materials offset:f->materials_offset atIndex:1];
    [e setBuffer:rb offset:0 atIndex:2];
    [e setBuffer:ob offset:0 atIndex:3];
    RtUseResources(e, f);
    [e dispatchThreads:MTLSizeMake(NSUInteger(n), 1, 1) threadsPerThreadgroup:MTLSizeMake(NSUInteger(n), 1, 1)];
    [e endEncoding];
    if (!g.Commit(cb)) return 1;
    if (!scaled) st = s->Stats();
    const auto* r = static_cast<const float*>(ob.contents);
    // The CPU's RtHitPrim of the triangle hit (the strip's first: p0, p1, p2).
    RtCutPrim cp{};
    cp.uv[1][1] = 1;
    cp.uv[2][0] = 1;
    cp.alpha = 0xFFFFFFu;
    const float a0[3] = {0, 0, 0}, a1[3] = {0, 4, 0}, a2[3] = {8, 0, 0};
    RtHitPrim want;
    RtHitTriangle(cp, a0, a1, a2, &want);
    float wn[3];
    RtOctDecode(want.normal, wn);
    auto same = [&](const float* x, uint32_t want_flags, float want_lod, bool lod_exact) {
      uint32_t flags, aflags, border;
      std::memcpy(&flags, &x[7], 4);
      std::memcpy(&aflags, &x[11], 4);
      std::memcpy(&border, &x[15], 4);
      const bool albedo = (want_flags & kRtMatAlbedo) != 0;
      return x[0] > 0 && x[3] == float(kRtMaterialVersion) &&
             (lod_exact ? x[2] == want_lod : std::fabs(x[2] - want_lod) < 1e-5f) &&
             std::fabs(x[4] - wn[0]) < 1e-4f && std::fabs(x[5] - wn[1]) < 1e-4f && std::fabs(x[6] - wn[2]) < 1e-4f &&
             flags == want_flags && (!albedo || (x[8] == 0 && x[9] == 1)) && x[10] == (albedo ? 1 : 0) &&
             aflags == (albedo ? (3u | 2u << 3) : 0u) && border == (albedo ? 0xFF336699u : 0u) &&
             x[12] == float(sizeof(RtMaterial)) && x[13] == float(sizeof(RtHitPrim)) &&
             std::fabs(x[14] - 1.0f) < 1e-3f;
    };
    for (int i = 0; i < 3; i++) {
      results[scaled][i] = r[i * 16 + 1];
      layout = layout && same(r + i * 16, kRtMatHitPrim | kRtMatAlbedo, want.lod, true);
    }
    if (!scaled) {
      captured = same(r + 3 * 16, kRtMatHitPrim | kRtMatAlbedo, want.lod, false) &&
                 std::fabs(r[3 * 16 + 1] - r[1]) < 1e-4f;
      bad = same(r + 4 * 16, kRtMatHitPrim, 0.0f, false) && same(r + 5 * 16, kRtMatHitPrim, 0.0f, true);
      std::printf("  captured: lod constant %.6f (CPU %.6f), level %.4f (static %.4f)\n", r[3 * 16 + 2], want.lod,
                  r[3 * 16 + 1], r[1]);
    }
  }
  failures += Check(layout, "RtMaterial version 2 and RtHitPrim as the GPU reads them: the flags, the albedo's "
                            "level, addressing and border, the normal, the UVs and the LOD constant the CPU made");
  failures += Check(captured && st.capture_prims_reused == 2 && st.capture_prim_bytes > 0,
                    "a captured draw's RtHitPrim: the UVs from the CPU (kept from the frame before), the normal and "
                    "LOD constant made on the GPU as the CPU makes them, its albedo and level as a static draw's");
  int skipped = 0;
  for (uint64_t k : st.capture_skipped) skipped += int(k);
  for (uint64_t k : st.skipped) skipped += int(k);
  failures += Check(bad && skipped == 0, "UVs that cannot be read (captured, static): the draw kept, without an "
                                         "albedo");
  bool lod = true;
  for (int scaled = 0; scaled < 2; scaled++)
    for (int i = 0; i < 3; i++) {
      const double d = (i == 0 ? 10 : i == 1 ? 40 : 160);
      const double want = 0.5 * std::log2(1.0 / 32.0) - scaled + 8.0 + std::log2(0.001 * d);
      std::printf("  level at %.0f%s: %.4f (analytic %.4f)\n", d, scaled ? " (scaled 2x)" : "", results[scaled][i],
                  want);
      lod = lod && std::fabs(results[scaled][i] - want) < 2e-3 &&
            (i == 0 || results[scaled][i] > results[scaled][i - 1]);
    }
  failures += Check(lod, "ray cones: the albedo's level rises with the distance, as the triangle's constant, the "
                         "texture's size and the cone's width say; a scaled instance a level lower");
  return failures;
}

// ---- the mirror and the box

constexpr uint32_t kW = 320, kH = 240;

// A rectangle of the analytic scene: p0 + u eu + v ev (u, v in 0..1).
struct Rect {
  V3 p0, eu, ev;
  int tex;    // 0 none (no albedo), 1 the box's (u < 0.5 colour A, else B), 2 the cut quad's
  bool hole;  // u < 0.5 lets rays through (the cut quad's)
};
struct Hit {
  int rect = -1;
  double t = 0, u = 0, v = 0;
};
Hit Trace(const std::vector<Rect>& rs, V3 o, V3 d, double tmin, double tmax) {
  Hit best;
  for (size_t i = 0; i < rs.size(); i++) {
    const Rect& r = rs[i];
    const V3 n = Cross(r.eu, r.ev);
    const double dn = Dot(d, n);
    if (std::fabs(dn) < 1e-12) continue;
    const double t = Dot(r.p0 - o, n) / dn;
    if (!(t > tmin && t < tmax) || (best.rect >= 0 && t >= best.t)) continue;
    const V3 p = o + d * t - r.p0;
    const double u = Dot(p, r.eu) / Dot(r.eu, r.eu), v = Dot(p, r.ev) / Dot(r.ev, r.ev);
    if (u < 0 || u > 1 || v < 0 || v > 1 || (r.hole && u < 0.5)) continue;
    best = {int(i), t, u, v};
  }
  return best;
}
Rect FromQuad(const V3 p[4], int tex, bool hole) { return {p[0], p[2] - p[0], p[1] - p[0], tex, hole}; }

struct Case {
  const char* what;
  bool reversed = true;   // depth stored reversed in a depth texture, else D3D's in a float one
  bool occluder = false;  // a quad over the box (no albedo): its faces in shadow
  bool cut = false;       // an alpha-cut quad in front of the box, its half u < 0.5 a hole
  double gloss = 1.0;     // the ground's
  bool wet = false;       // the ground's spec 0, gloss 0.3, wet where x >= 0 (the stencil 0)
  bool shadow = false;    // the shadow mask traced from the same snapshot too
  int front_cull = 1;     // the box's front face drawn with this D3DCULL (1 none)
  // The box's front face drawn by another view only (a shadow map's pass,
  // culling counterclockwise faces): 1 its projection as the main view's
  // kind, 2 one flipping y (it mirrors).
  int front_other = 0;
  // The ground's G-buffer normal tilted 45 degrees away from the camera
  // (its reflections aimed below it), a quad under it (red).
  bool tilted = false;
  // Traced into the last case's outputs, not new ones: what that trace
  // wrote must be gone where this one traces nothing (RtReflector clears
  // only the pixels its last trace listed).
  bool reuse = false;
  // The checkerboard (RtReflectionParams::checker) with this parity + 1 (0
  // off): half the listed pixels traced, the rest filled from them.
  int checker = 0;
  bool no_fill = false;  // with it, no neighbour lends (every pixel of the other half traced after)
  // Screen space first (RtSsrParams), the previous frame's lit scene kSsrLit
  // everywhere, drawn by the same camera: a ray's hit on the box the march
  // finds on screen is that light (times the scale).
  bool ssr = false;
  bool ssr_edge = false;  // with it, the march's results within 64 pixels of the edges mixed with the rays'
};
const double kSsrLit[3] = {0.8, 0.4, 0.2};

// What became of the reflections the box's front face would give
// (Case::front_cull): lit as the analytic trace says, stopped (a 0 at the
// face's distance), neither.
struct Outcome {
  int front_lit = 0, front_back = 0, front_wrong = 0;
  uint32_t back = 0, other_back = 0;  // the kernel's counts
};

constexpr double kScale = 0.1, kWrap = 0.25;
const V3 kLight = Unit({0.3, 1.0, -0.6});  // toward the light
const double kColour[3] = {2.0, 1.8, 1.6};
const double kAmb0[3] = {0.1, 0.1, 0.15}, kAmb1[3] = {0.05, 0.04, 0.03};
// The box's texture (4 x 4: columns 0-1 colour A, 2-3 B) and the cut quad's
// (2 x 1: a hole, then colour C).
const uint8_t kA[3] = {204, 102, 51}, kB[3] = {51, 153, 255}, kC[3] = {102, 255, 51};

int Run(Gpu& g, const Case& c, Outcome* outcome = nullptr) {
  int failures = 0;
  RtSceneOptions so;
  so.hit_data = true;
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, so);
  g.arena.used = 0;
  s->SetAllocator(ArenaAlloc, &g.arena);
  double vp[16], inv[16];
  const V3 eye{0.5, 2.5, -6};
  Camera(eye, {0.5, 0, 8}, 1.0, double(kW) / kH, 0.5, 500.0, vp);
  Invert4(vp, inv);
  // The scene: the ground (no albedo), the box's five faces, the occluder,
  // the cut quad.
  const V3 ground[4] = {{-60, 0, -60}, {-60, 0, 60}, {60, 0, -60}, {60, 0, 60}};
  const V3 faces[5][4] = {
      {{-4, 0, 10}, {-4, 4, 10}, {4, 0, 10}, {4, 4, 10}},  // front (u: x)
      {{4, 0, 12}, {4, 4, 12}, {-4, 0, 12}, {-4, 4, 12}},  // back
      {{-4, 0, 12}, {-4, 4, 12}, {-4, 0, 10}, {-4, 4, 10}},  // left
      {{4, 0, 10}, {4, 4, 10}, {4, 0, 12}, {4, 4, 12}},  // right
      {{-4, 4, 10}, {-4, 4, 12}, {4, 4, 10}, {4, 4, 12}},  // top
  };
  const V3 occl[4] = {{-8, 7, 5}, {-8, 7, 9}, {8, 7, 5}, {8, 7, 9}};
  const V3 cutq[4] = {{-4, 0, 8}, {-4, 4, 8}, {4, 0, 8}, {4, 4, 8}};
  std::vector<Rect> rects = {FromQuad(ground, 0, false)};
  for (const auto& f : faces) rects.push_back(FromQuad(f, 1, false));
  if (c.occluder) rects.push_back(FromQuad(occl, 0, false));
  if (c.cut) rects.push_back(FromQuad(cutq, 2, true));

  uint8_t box_texels[4 * 4 * 4];
  for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
      const uint8_t* col = i < 2 ? kA : kB;
      uint8_t* t = &box_texels[(j * 4 + i) * 4];
      t[0] = col[0], t[1] = col[1], t[2] = col[2], t[3] = 255;
    }
  const uint8_t cut_texels[8] = {0, 0, 0, 0, kC[0], kC[1], kC[2], 255};
  id<MTLTexture> box_tex = g.Texture(MTLPixelFormatRGBA8Unorm, 4, 4, box_texels, 16);
  id<MTLTexture> cut_tex = g.Texture(MTLPixelFormatRGBA8Unorm, 2, 1, cut_texels, 8);

  Mat w, mw;
  Identity(w);
  Wvp(w, vp, mw);
  std::vector<Quad> quads;
  quads.reserve(9);
  s->BeginFrame(1, 0);
  quads.push_back(MakeQuad(ground, 1, false));
  s->Add(quads.back().key, quads.back().g, w, mw, 1, 4);
  RtAlbedo box_albedo;
  box_albedo.texture = box_tex;
  for (int i = 0; i < 5; i++) {
    quads.push_back(MakeQuad(faces[i], 2 + uint64_t(i), false));
    if (i == 0 && c.front_other) {
      double lvp[16];
      Camera({0, 12, -2}, {0, 2, 11}, 1.2, 1.0, 1.0, 200.0, lvp);
      if (c.front_other == 2)
        for (int r = 0; r < 4; r++) lvp[r * 4 + 1] = -lvp[r * 4 + 1];
      Mat ow;
      Wvp(w, lvp, ow);
      s->Add(quads.back().key, quads.back().g, w, ow, 3, 4, nullptr, RtView::Other, &box_albedo);
      continue;
    }
    s->Add(quads.back().key, quads.back().g, w, mw, i == 0 ? uint32_t(c.front_cull) : 1u, 4, nullptr, RtView::Main,
           &box_albedo);
  }
  // Under the ground (tilted): red.
  const uint8_t red[4] = {255, 0, 0, 255};
  id<MTLTexture> red_tex = g.Texture(MTLPixelFormatRGBA8Unorm, 1, 1, red, 4);
  if (c.tilted) {
    const V3 under[4] = {{-60, -2, -60}, {-60, -2, 60}, {60, -2, -60}, {60, -2, 60}};
    RtAlbedo a;
    a.texture = red_tex;
    quads.push_back(MakeQuad(under, 12, false));
    s->Add(quads.back().key, quads.back().g, w, mw, 1, 4, nullptr, RtView::Main, &a);
  }
  if (c.occluder) {
    quads.push_back(MakeQuad(occl, 10, false));
    s->Add(quads.back().key, quads.back().g, w, mw, 1, 4);
  }
  if (c.cut) {
    RtCut cut;
    cut.texture = cut_tex;
    cut.material.alpha_scale = 1.0f;
    cut.material.flags = RtMaterialFlags(0, 7, 128, 3, 3, 3);  // GREATEREQUAL 128, clamp
    RtAlbedo a;
    a.texture = cut_tex;
    a.address_u = a.address_v = 3;
    quads.push_back(MakeQuad(cutq, 11, true));
    s->Add(quads.back().key, quads.back().g, w, mw, 1, 4, &cut, RtView::Main, &a);
  }
  const RtFrame* f = s->Snapshot();
  if (!f || !f->tlas) return Check(false, "a TLAS of the scene");

  // The synthetic G-buffer: each pixel's first hit from the camera
  // (D3D9's centres), its depth, normal, spec and gloss, the stencil.
  const size_t n = size_t(kW) * kH;
  std::vector<float> depth(n), normal(n * 4, 0.0f);
  std::vector<uint8_t> surface(n * 4, 0), stencil(n, 0);
  std::vector<V3> points(n);
  std::vector<int> prim(n, -1);
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const double x = 2.0 * i / kW - 1.0, y = 1.0 - 2.0 * j / kH;
      double a[4], b[4];
      for (int k = 0; k < 4; k++) {
        a[k] = x * inv[k] + y * inv[4 + k] + inv[12 + k];
        b[k] = x * inv[k] + y * inv[4 + k] + inv[8 + k] + inv[12 + k];
      }
      const V3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const size_t at = size_t(j) * kW + i;
      const Hit h = Trace(rects, pa, pb - pa, 0.0, 1.0);
      double z = 1.0;
      if (h.rect >= 0) {
        const V3 p = pa + (pb - pa) * h.t;
        double cc[4];
        for (int k = 0; k < 4; k++) cc[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
        z = cc[2] / cc[3];
        points[at] = p;
        prim[at] = h.rect;
        V3 nr = Unit(Cross(rects[h.rect].eu, rects[h.rect].ev));
        if (Dot(nr, p - eye) > 0) nr = nr * -1.0;
        if (c.tilted && h.rect == 0) nr = Unit({0, 1, 1});  // (away from the camera, which looks along +z)
        normal[at * 4] = float(nr.x * 0.5 + 0.5);
        normal[at * 4 + 1] = float(nr.y * 0.5 + 0.5);
        normal[at * 4 + 2] = float(nr.z * 0.5 + 0.5);
        if (h.rect == 0) {  // the ground: shiny but for x < -3
          const double spec = c.wet || p.x < -3 ? 0.0 : 0.5, gloss = c.wet ? 0.3 : c.gloss;
          surface[at * 4] = uint8_t(std::lround(spec * 255));
          surface[at * 4 + 1] = uint8_t(std::lround(gloss * 255));
          stencil[at] = c.wet && p.x < 0 ? 255 : 0;
        } else {
          stencil[at] = 255;
        }
      }
      depth[at] = float(c.reversed ? 1.0 - z : z);
    }
  id<MTLTexture> dt;
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  if (c.reversed) {
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:kW
                                                          height:kH
                                                       mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    dt = [g.dev newTextureWithDescriptor:td];
    id<MTLBuffer> src = [g.dev newBufferWithBytes:depth.data() length:n * 4 options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src sourceOffset:0 sourceBytesPerRow:kW * 4 sourceBytesPerImage:n * 4
             sourceSize:MTLSizeMake(kW, kH, 1) toTexture:dt destinationSlice:0 destinationLevel:0
      destinationOrigin:MTLOriginMake(0, 0, 0) options:MTLBlitOptionDepthFromDepthStencil];
    [bl endEncoding];
  } else {
    dt = g.Texture(MTLPixelFormatR32Float, kW, kH, depth.data(), kW * 4);
  }
  id<MTLTexture> nt = g.Texture(MTLPixelFormatRGBA32Float, kW, kH, normal.data(), kW * 16);
  id<MTLTexture> st = g.Texture(MTLPixelFormatRGBA8Unorm, kW, kH, surface.data(), kW * 4);
  id<MTLTexture> wt = g.Texture(MTLPixelFormatR8Unorm, kW, kH, stencil.data(), kW);

  // The draw's constants, as a game's lighting pass would have them.
  float consts[224][4] = {};
  for (int k = 0; k < 3; k++) {
    consts[17][k] = float(-Dot(kLight, k == 0 ? V3{1, 0, 0} : k == 1 ? V3{0, 1, 0} : V3{0, 0, 1}));
    consts[18][k] = float(kColour[k] / 2.0);
    consts[37][k] = float(kAmb0[k]);
    consts[38][k] = float(kAmb1[k]);
  }
  consts[18][3] = 2.0f;
  consts[72][0] = c.wet ? 1.0f / 256 : 1.0f / 512;
  consts[72][1] = 1.0f;
  RayReflection r;
  r.light_register = 17;
  r.colour_register = 18;
  r.ambient_registers[0] = 37;
  r.ambient_registers[1] = 38;
  r.blur_register = 72;
  r.up[0] = 0, r.up[1] = 1, r.up[2] = 0;
  r.diffuse_wrap = float(kWrap);
  r.output_scale = float(kScale);
  if (c.wet) {
    r.wet_slope = 4;
    r.wet_gain = 8;
    r.wet_max_exponent = 256;
    r.wet_register = 72;
    r.wet_component = 1;
  }
  const RtReflectionInputs in = RtReflectionInputsFor(r, consts);
  RtReflectionParams p;
  if (!RtReflectionParamsFor(*f, kW, kH, c.reversed, c.reversed, nullptr, r, in, c.wet, &p))
    return Check(false, "the reflections' parameters");
  if (c.checker) {
    p.checker = 1;
    p.parity = uint32_t(c.checker - 1);
    p.frame = uint32_t(c.checker * 5);
    if (c.no_fill) p.fill_cos = 2.0f;
  }
  id<MTLTexture> out = c.reuse && g.last_out ? g.last_out : g.reflector->NewOutput(kW, kH);
  id<MTLTexture> dist = c.reuse && g.last_dist ? g.last_dist : g.reflector->NewDistance(kW, kH);
  g.last_out = out;
  g.last_dist = dist;
  id<MTLBuffer> counts = [g.dev newBufferWithLength:sizeof(RtReflectionCounts) options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, sizeof(RtReflectionCounts));
  s->EncodeFrame(cb, f);
  RtSsrParams sp;
  id<MTLTexture> prev = nil;
  if (c.ssr) {
    if (!RtSsrParamsFor(p, vp, nullptr, kW, kH, float(kScale), &sp)) return Check(false, "the march's parameters");
    sp.edge = c.ssr_edge ? 64.0f : 1.0f;  // (1: no band mixed, each pixel the march's or the ray's)
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:kW
                                                                                 height:kH
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    prev = [g.dev newTextureWithDescriptor:td];
    std::vector<float> lit(n * 4);
    for (size_t at = 0; at < n; at++)
      for (int k = 0; k < 4; k++) lit[at * 4 + k] = k < 3 ? float(kSsrLit[k]) : 1.0f;
    [prev replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:lit.data() bytesPerRow:kW * 16];
  }
  g.reflector->Encode(cb, f, dt, nt, st, c.wet ? wt : nil, out, dist, p, counts, 0, c.ssr ? &sp : nullptr, prev);
  // The shadow mask from the same snapshot (one TLAS).
  id<MTLTexture> mask = nil;
  id<MTLBuffer> mcounts = [g.dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
  std::memset(mcounts.contents, 0, 8);
  if (c.shadow) {
    RtShadowParams sp;
    const float l[3] = {float(kLight.x), float(kLight.y), float(kLight.z)};
    if (!RtShadowParamsFor(*f, kW, kH, c.reversed, c.reversed, nullptr, l, &sp))
      return Check(false, "the mask's parameters");
    mask = g.masker->NewMask(kW, kH);
    g.masker->Encode(cb, f, dt, mask, sp, mcounts, 0);
  }
  if (!g.Commit(cb)) return 1;
  const RtStats stats = s->Stats();
  const std::vector<uint8_t> ob = g.Read(out, 8), db = g.Read(dist, 2);
  const std::vector<uint8_t> mb = mask ? g.Read(mask, 1) : std::vector<uint8_t>();
  if (ob.size() != n * 8 || db.size() != n * 2) return Check(false, "the reflections read back");
  auto half = [&](const std::vector<uint8_t>& b, size_t at) {
    uint16_t h;
    std::memcpy(&h, &b[at * 2], 2);
    return double(HalfToFloat(h));
  };

  // Each pixel against the analytic trace. Its class: not traced, a miss,
  // a hit on a rect with its texture's band and shadow; judged where its
  // four neighbours have the same (edges, the texture's seams, shadows').
  const double e = c.wet ? 256.0 : 512.0 * std::pow(std::lround(c.gloss * 255) / 255.0, 2.0);
  const double tmax = double(r.max_distance) * std::exp(-5.0 * std::sqrt(2.0 / (e + 2.0)));
  struct Want {
    int cls = 0;  // 0 not traced, 1 a miss, 2 a hit (with or without an albedo)
    int rect = -1, band = 0;
    bool lit = false, shadow = false;
    double t = 0, rgb[3] = {};
  };
  std::vector<Want> want(n);
  uint32_t want_traced = 0;
  auto blocked = [&](V3 q, double tmin) { return Trace(rects, q, kLight, tmin, 1e9).rect >= 0; };
  for (size_t at = 0; at < n; at++) {
    Want& wa = want[at];
    if (prim[at] >= 0) wa.shadow = blocked(points[at], 1e-4);
    if (prim[at] != 0) continue;  // the ground only is shiny
    const V3 q = points[at];
    const bool shiny = c.wet ? q.x >= 0 : q.x >= -3;
    if (!shiny || (!c.wet && c.gloss < 0.9)) continue;
    want_traced++;
    const V3 i = Unit(q - eye), rd{i.x, -i.y, i.z};
    const Hit h = Trace(rects, q, rd, 1e-4, tmax);
    if (h.rect < 0) {
      wa.cls = 1;
      continue;
    }
    wa.cls = 2;
    wa.rect = h.rect;
    wa.t = h.t;
    const Rect& rr = rects[h.rect];
    const double u = h.u;
    // (The box's texture wraps: bilinear across u = 0 and 1 too; pure texels
    // between the centres of columns 0 and 1, 2 and 3.)
    wa.band = rr.tex == 1   ? (u > 0.13 && u < 0.37 ? 1 : u > 0.63 && u < 0.87 ? 2 : 0)
              : rr.tex == 2 ? (u > 0.8 ? 3 : 0)
                            : 4;
    const V3 hp = q + rd * h.t;
    V3 hn = Unit(Cross(rr.eu, rr.ev));
    if (Dot(hn, rd) > 0) hn = hn * -1.0;
    wa.lit = !blocked(hp, 1e-4);
    const uint8_t* alb = wa.band == 1 ? kA : wa.band == 2 ? kB : kC;
    const double diffuse = Sat((Dot(hn, kLight) - kWrap) / (1.0 - kWrap)) * (wa.lit ? 1.0 : 0.0);
    for (int k = 0; k < 3; k++)
      wa.rgb[k] = alb[k] / 255.0 * (kColour[k] * diffuse + kAmb0[k] + kAmb1[k] * Sat(0.5 - 0.5 * hn.y)) * kScale;
  }
  auto same = [&](size_t a, size_t b) {
    const Want &x = want[a], &y = want[b];
    return x.cls == y.cls && x.rect == y.rect && x.band == y.band && x.lit == y.lit && x.shadow == y.shadow &&
           prim[a] == prim[b];
  };
  int judged = 0, wrong = 0, hits = 0, misses = 0, bare = 0, untraced = 0, dist_wrong = 0, mask_judged = 0,
      mask_wrong = 0, shown = 0, red_seen = 0, lit_seen = 0;
  double worst = 0;
  Outcome oc;
  if (c.tilted) {
    // Not judged against the trace (the rays straightened): none may reach
    // what lies under the ground.
    for (size_t at = 0; at < n; at++) {
      const double a = half(ob, at * 4 + 3), rr = half(ob, at * 4), gg = half(ob, at * 4 + 1);
      if (a == 1 && rr > 0 && gg == 0) red_seen++;
      if (a == 1 && gg > 0) lit_seen++;
    }
  }
  int ssr_seen = 0, ssr_false = 0;
  auto is_ssr = [&](const double* v) {
    bool same = true;
    for (int k = 0; k < 3; k++) same = same && std::fabs(v[k] - kSsrLit[k] * kScale) <= 2e-3 * kSsrLit[k] * kScale + 1e-4;
    return same;
  };
  for (uint32_t j = 1; j + 1 < kH && !c.tilted; j++)
    for (uint32_t i = 1; i + 1 < kW; i++) {
      const size_t at = size_t(j) * kW + i;
      if (!same(at, at - 1) || !same(at, at + 1) || !same(at, at - kW) || !same(at, at + kW)) continue;
      const Want& wa = want[at];
      const double a = half(ob, at * 4 + 3), d = half(db, at);
      double rgb[3];
      for (int k = 0; k < 3; k++) rgb[k] = half(ob, at * 4 + k);
      if (mask && prim[at] == 0) {
        mask_judged++;
        if ((mb[at] > 127) == wa.shadow) mask_wrong++;
      }
      if (wa.band == 0 && wa.cls == 2) continue;  // a seam
      if ((c.front_cull != 1 || c.front_other) && wa.cls == 2 && wa.rect == 1) {
        // The front face drawn culling: lit, or stopped at its distance.
        bool lit = a == 1;
        for (int k = 0; k < 3; k++) lit = lit && std::fabs(rgb[k] - wa.rgb[k]) <= 0.01 * wa.rgb[k] + 2e-4;
        const bool stopped = a == 0 && std::fabs(d - wa.t) <= 3e-3 * wa.t + 0.02;
        (lit ? oc.front_lit : stopped ? oc.front_back : oc.front_wrong)++;
        continue;
      }
      judged++;
      bool ok = true;
      // (Screen space and the checkerboard: a filled pixel mixes what its
      // neighbours have, the march's light (here a made-up one) and the
      // rays': only the untraced pixels and the counts judged.)
      if (c.ssr && (c.checker || c.ssr_edge) && wa.cls != 0) continue;
      // (Wet ground filled from up to 2 pixels away: judged where the 5 x 5
      // pixels around are alike.)
      if (c.wet && c.checker && wa.cls != 0) {
        bool alike = i >= 2 && j >= 2 && i + 2 < kW && j + 2 < kH;
        for (int dy = -2; dy <= 2 && alike; dy++)
          for (int dx = -2; dx <= 2 && alike; dx++) alike = same(at, size_t(int(at) + dy * int(kW) + dx));
        if (!alike) continue;
      }
      if (wa.cls == 0) {
        untraced++;
        ok = a == 0 && d == 0 && rgb[0] == 0 && rgb[1] == 0 && rgb[2] == 0;
      } else if (wa.cls == 1) {
        misses++;
        ok = a == 0 && d == double(kRtReflectionSkyDistance);
        if (c.ssr && !ok && a == 1 && is_ssr(rgb)) {
          ssr_false++;  // (a ray passing just behind what the screen shows)
          ok = true;
        }
      } else if (c.ssr && a == 1 && is_ssr(rgb)) {
        ssr_seen++;
        if (rects[wa.rect].tex == 0) bare++;
        else hits++;
      } else {
        if (std::fabs(d - wa.t) > 3e-3 * wa.t + 0.02) dist_wrong++;
        if (rects[wa.rect].tex == 0) {
          bare++;
          ok = a == 0;
        } else {
          hits++;
          ok = a == 1;
          for (int k = 0; k < 3; k++) {
            const double err = std::fabs(rgb[k] - wa.rgb[k]);
            worst = std::max(worst, err / (wa.rgb[k] + 1e-3));
            ok = ok && err <= 0.01 * wa.rgb[k] + 2e-4;
          }
          if (wa.rgb[0] > 0) shown++;
        }
      }
      if (!ok && wrong < 4)
        std::printf("  %s: pixel %u,%u class %d rect %d band %d lit %d: got a %.3f rgb %.4f %.4f %.4f dist %.3f, "
                    "want rgb %.4f %.4f %.4f dist %.3f\n",
                    c.what, i, j, wa.cls, wa.rect, wa.band, wa.lit, a, rgb[0], rgb[1], rgb[2], d, wa.rgb[0],
                    wa.rgb[1], wa.rgb[2], wa.t);
      if (!ok) wrong++;
    }
  const auto* cn = static_cast<const RtReflectionCounts*>(counts.contents);
  std::printf("  %s: %d pixels judged (%d hits, %d misses, %d on no albedo, %d not traced), %d wrong, %d distances "
              "off, worst colour error %.2f%%; counted %u rays (%u expected), %u hits, %u misses, %u without an "
              "albedo, mean distance %.2f%s\n",
              c.what, judged, hits, misses, bare, untraced, wrong, dist_wrong, worst * 100, cn->traced, want_traced,
              cn->hits, cn->misses, cn->bare, cn->hits ? cn->distance / cn->hits : 0.0,
              mask ? (" ; mask: " + std::to_string(mask_judged) + " ground pixels, " + std::to_string(mask_wrong) +
                      " wrong")
                         .c_str()
                   : "");
  // (The checkerboard: about half traced, the rest filled; a pixel no
  // neighbour can fill traced too.)
  if (c.ssr)
    std::printf("  %s: %u settled by the march (%d judged, %d of them where the ray misses), %u mixed\n", c.what,
                cn->screen, ssr_seen, ssr_false, cn->screen_mixed);
  const bool counted = (c.ssr ? cn->traced + cn->screen + cn->filled == want_traced &&
                                    (c.checker ? cn->filled > 0 && cn->screen > 1000
                                     : c.ssr_edge ? cn->filled == 0 && cn->screen_mixed > 100 && cn->screen > 500
                                                  : cn->filled == 0 && ssr_seen > 200 && cn->screen_mixed == 0) &&
                                    ssr_false <= ssr_seen / 50
                        : c.checker && c.no_fill ? cn->traced == want_traced && cn->filled == 0
                        : c.checker && c.wet ? cn->traced < want_traced * 0.4 && cn->traced > want_traced / 20 &&
                                                   cn->traced + cn->filled == want_traced
                        : c.checker ? cn->traced > want_traced * 0.45 && cn->traced < want_traced * 0.6 &&
                                        cn->traced + cn->filled == want_traced
                                  : cn->traced == want_traced && cn->filled == 0) &&
                       cn->hits + cn->misses + cn->bare + cn->back == cn->traced;
  if (c.checker) std::printf("  %s: %u filled from their neighbours\n", c.what, cn->filled);
  oc.back = cn->back;
  oc.other_back = cn->other_back;
  if (outcome) *outcome = oc;
  if (c.front_cull != 1 || c.front_other)
    std::printf("  %s: front face %d lit, %d stopped, %d neither; %u back faces stopped, %u of other views lit\n",
                c.what, oc.front_lit, oc.front_back, oc.front_wrong, cn->back, cn->other_back);
  if (c.tilted) {
    std::printf("  %s: %d pixels showing what is under the ground, %d lit by the box\n", c.what, red_seen, lit_seen);
    failures += Check(red_seen == 0 && lit_seen > 100 && counted,
                      (std::string(c.what) + ": the rays straightened above the surface, none under it").c_str());
    return failures;
  }
  if (c.gloss < 0.9 && !c.wet) {
    failures += Check(wrong == 0 && cn->traced == 0 && untraced > 1000 && hits == 0,
                      (std::string(c.what) + ": no ray, every output 0").c_str());
    return failures;
  }
  const bool front_case = c.front_cull != 1 || c.front_other;
  // (Untraced pixels only: the march's made-up light mixed in, or wet
  // ground filled from up to 5 x 5 around.)
  const bool mixed_judged = c.ssr && (c.checker || c.ssr_edge);
  failures += Check(wrong == 0 && dist_wrong == 0 && (hits > 200 || front_case || mixed_judged) &&
                        (misses > 200 || mixed_judged) &&
                        untraced > 200 && counted,
                    (std::string(c.what) + ": hits a 1, the texel lit as the scene's sun and ambient times the "
                                           "scale, at the hit's distance; misses a 0 at the sky's; the rest "
                                           "untouched; the counters")
                        .c_str());
  if (c.occluder)
    failures += Check(bare > 50 && cn->bare > 0, (std::string(c.what) + ": hits without an albedo a 0").c_str());
  if (c.shadow)
    failures += Check(mask_judged > 1000 && mask_wrong == 0 && stats.snapshots == 1 &&
                          static_cast<const uint32_t*>(mcounts.contents)[0] > 0,
                      (std::string(c.what) + ": the shadow mask from the same snapshot (one TLAS)").c_str());
  return failures;
}

int CullChecks(Gpu& g) {
  // The box's front face drawn culling either winding: one of the two culls
  // what the reflections see (the rays stop there, a 0), the other not (lit).
  Outcome o[2];
  int failures = 0;
  for (int k = 0; k < 2; k++) {
    Case c;
    c.what = k ? "mirror, the front face culling counterclockwise" : "mirror, the front face culling clockwise";
    c.front_cull = 2 + k;
    failures += Run(g, c, &o[k]);
  }
  auto lit = [](const Outcome& x) {
    return x.front_lit > 100 && x.front_back == 0 && x.front_wrong == 0 && x.back == 0;
  };
  auto back = [](const Outcome& x) {
    return x.front_back > 100 && x.front_lit == 0 && x.front_wrong == 0 && x.back > 0;
  };
  failures += Check((lit(o[0]) && back(o[1])) || (lit(o[1]) && back(o[0])),
                    "back faces of an instance that culls stop the ray (a 0, counted); front faces lit");
  return failures;
}

int OtherViewChecks(Gpu& g) {
  // The box's front face drawn only by another view, culling
  // counterclockwise faces, its projection mirroring or not: lit either
  // way (its front faces are not known); in one of the two its faces are
  // back faces to the rays (counted), never stopped.
  Outcome o[2];
  int failures = 0;
  for (int k = 0; k < 2; k++) {
    Case c;
    c.what = k ? "mirror, the front face another view's (its projection mirroring)"
               : "mirror, the front face another view's";
    c.front_other = 1 + k;
    failures += Run(g, c, &o[k]);
  }
  auto lit = [](const Outcome& x) {
    return x.front_lit > 100 && x.front_back == 0 && x.front_wrong == 0 && x.back == 0;
  };
  failures += Check(lit(o[0]) && lit(o[1]) && ((o[0].other_back > 0) != (o[1].other_back > 0)),
                    "another view's instance lit from either side (its back faces counted, not stopped)");
  return failures;
}

int GpuChecks(Gpu& g) {
  int failures = MemoryChecks(g) + ProbeChecks(g);
  Case a;
  a.what = "mirror, reversed depth";
  a.shadow = true;
  failures += Run(g, a);
  Case b;
  b.what = "mirror, float depth, an occluder over the box";
  b.reversed = false;
  b.occluder = true;
  failures += Run(g, b);
  Case cq;
  cq.what = "mirror, an alpha-cut quad in front";
  cq.cut = true;
  failures += Run(g, cq);
  Case d;
  d.what = "gloss blurrier than the cutoff";
  d.gloss = 0.5;
  failures += Run(g, d);
  // The same outputs traced again and again: the mirror, nothing, the
  // mirror, the wet ground (another set of pixels).
  Case r1 = a;
  r1.what = "mirror into used outputs";
  r1.shadow = false;
  failures += Run(g, r1);
  Case r2 = d;
  r2.what = "gloss blurrier than the cutoff, into the mirror's outputs";
  r2.reuse = true;
  failures += Run(g, r2);
  Case r3 = r1;
  r3.what = "mirror into the outputs of a trace of nothing";
  r3.reuse = true;
  failures += Run(g, r3);
  // The checkerboard, both halves, into the same outputs (the last one's
  // filled pixels traced now, and the reverse).
  for (int k = 1; k <= 2; k++) {
    Case cb = a;
    cb.what = k == 1 ? "mirror, the checkerboard's even half traced" : "mirror, the checkerboard's odd half traced";
    cb.shadow = false;
    cb.checker = k;
    cb.reuse = k == 2;
    failures += Run(g, cb);
  }
  Case ss = a;
  ss.what = "mirror, screen space first";
  ss.shadow = false;
  ss.ssr = true;
  failures += Run(g, ss);
  Case sse = ss;
  sse.what = "mirror, screen space first, mixed with the rays along the edges";
  sse.ssr_edge = true;
  failures += Run(g, sse);
  Case ssc = cq;
  ssc.what = "mirror, an alpha-cut quad in front, screen space first, the checkerboard";
  ssc.ssr = true;
  ssc.checker = 2;
  failures += Run(g, ssc);
  Case cbn = a;
  cbn.what = "mirror, the checkerboard, no neighbour lending (the other half traced after)";
  cbn.shadow = false;
  cbn.checker = 1;
  cbn.no_fill = true;
  cbn.reuse = true;
  failures += Run(g, cbn);
  Case cbc = cq;
  cbc.what = "mirror, an alpha-cut quad in front, the checkerboard";
  cbc.checker = 1;
  failures += Run(g, cbc);
  Case cbo = b;
  cbo.what = "mirror, float depth, an occluder over the box, the checkerboard";
  cbo.checker = 2;
  failures += Run(g, cbo);
  Case wet;
  wet.what = "wet ground (spec 0), dry where the stencil is set";
  wet.wet = true;
  failures += Run(g, wet);
  for (int k = 1; k <= 2; k++) {
    Case ws = wet;
    ws.what = k == 1 ? "wet ground, sparse (1 of 2 x 2, 1 of 4 x 4 seen from above)"
                     : "wet ground, sparse, the next places, into the same outputs";
    ws.checker = k;
    ws.reuse = k == 2;
    failures += Run(g, ws);
  }
  Case r4 = wet;
  r4.what = "wet ground into the mirror's outputs";
  r4.reuse = true;
  Case r5 = r3;
  r5.what = "mirror into the wet ground's outputs";
  failures += Run(g, r5);
  failures += Run(g, r4);
  Case tilt;
  tilt.what = "normals tilted below the ground's plane";
  tilt.tilted = true;
  failures += Run(g, tilt);
  failures += CullChecks(g);
  failures += OtherViewChecks(g);
  return failures;
}

}  // namespace

int main() {
  @autoreleasepool {
    // The far light's mask at every pixel: it is judged pixel by pixel
    // against the reflections' snapshot (its half-resolution form has its own
    // checks, rt_shadow_test).
    setenv("RECOMP_RT_SHADOW_FULL", "1", 1);
    int failures = CpuChecks();
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!RtSupported(dev)) {
      std::printf("skipped: no GPU that traces rays in hardware (%d CPU failures)\n", failures);
      return failures ? 1 : 77;
    }
    Gpu g;
    if (!g.Init()) return Check(false, "the reflections' and shadows' kernels");
    failures += GpuChecks(g);
    if (failures) {
      std::printf("%d failed\n", failures);
      return 1;
    }
    std::puts("rt_reflection: ok");
    return 0;
  }
}
