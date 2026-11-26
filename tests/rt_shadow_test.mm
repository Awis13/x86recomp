// Ray-traced shadow masks (gfx/rt_shadow.h) and the draws of other passes
// in the ray-traced scene (gfx/rt_scene.h, RtView). On the CPU: which
// view a draw belongs to (a captured draw of another pass none, a pass
// writing colour none unless asked for), points reconstructed from stored
// depth 4 km from the origin (reversed and not, with the scene's double
// camera), the parameters refused without a light or a camera, and the
// backend's choices (RtShadowNext: traced again after a depth write, for
// another depth texture or light, after the frame; all lit without a
// depth, light or camera, for another view's depth or the last frame's
// camera; a pending depth clear run first only where the trace reads that
// depth; the mask bound at the claimed sampler only). On a GPU with
// hardware ray tracing: the scene's other views (a mirrored world left
// out, a cut main-view draw keeping its material over an opaque other
// view's, an other view's draw promoting a losing camera's instance, other
// views not carried into the next frame); a ground plane drawn by the main
// view and a square above it drawn only by another pass (a shadow map's
// caster), the ground drawn by that pass too (one instance), a synthetic
// depth buffer of the ground from a known camera, as the backend stores it
// (Depth32Float_Stencil8, reversed) and as a float texture (not
// reversed): every pixel's mask against the shadow the square casts (lit,
// in shadow, at the predicted edge, the sky lit), the same 4 km from the
// origin with a low light (no pixel of the plane shadowing itself), an
// alpha-cut square whose hole lets light through, the all-lit mask; and a
// box on the ground drawn through the backend's own vertex shader prelude
// (its half pixel and a jitter, depth stored reversed, a camera that keeps
// D3D's depth and one that reverses it): each pixel's point where the
// camera's ray meets the scene, lit and dark sides, and the box's shadow
// across depth edges; and a street drawn the same way over the jitter's
// eight phases (StreetRun: the sun at 60, 25 and 8 degrees, at the origin
// and 4 km from it, normals from a G-buffer, from the depth, tilted as a
// normal map tilts them): no lit surface shadowing itself, no shadow lost,
// the shadowed pixels the same in every phase but at the shadows' edges;
// the camera turned all the way round (every 30 degrees) over normal-mapped
// normals, the sun at 15 degrees, lit surfaces lit at every heading; and
// the same street traced from the pixels' centres instead of where the
// depth was drawn, which shadows itself and shimmers.
//
//   rt_shadow_test        (skipped, exit 77, without a GPU that traces rays)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/reversed_z.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"
#include "runtime/gfx/shader_compile.h"
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

// D3D's left-handed camera at `eye` looking at `at` (y up), its
// perspective (depth 0..1 at near..far), as world to clip rows.
void Camera(V3 eye, V3 at, double fov_y, double aspect, double n, double f, double vp[16], bool reversed = false) {
  const V3 fw = Unit(at - eye), rt = Unit(Cross({0, 1, 0}, fw)), up = Cross(fw, rt);
  const double view[16] = {rt.x, up.x, fw.x, 0, rt.y, up.y, fw.y, 0, rt.z, up.z, fw.z, 0,
                           -Dot(rt, eye), -Dot(up, eye), -Dot(fw, eye), 1};
  const double ys = 1.0 / std::tan(fov_y * 0.5);
  double proj[16] = {};
  proj[0] = ys / aspect;
  proj[5] = ys;
  proj[10] = reversed ? n / (n - f) : f / (f - n);  // reversed: depth 1 at near, 0 at far
  proj[11] = 1;
  proj[14] = reversed ? -n * f / (n - f) : -n * f / (f - n);
  Mul4(view, proj, vp);
}

void Translate(Mat m, V3 o) {
  std::memset(m, 0, sizeof(Mat));
  m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
  m[3][0] = float(o.x), m[3][1] = float(o.y), m[3][2] = float(o.z);
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

// A quad as a strip of 2 triangles, FLOAT3 positions then UVs (24-byte
// vertices: position, uv, D3DCOLOR), in buffer number `buffer`; alpha-cut
// keys and attributes when `cut`.
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
  if (cut) {
    g.uv = {q.vb.data() + 12, q.vb.size() - 12, 24, 1, {0, 1}};
    k.cut = 1;
    k.attr_version = k.vertex_version;
    k.attr_buffer = buffer;
    k.attr_stride = 24;
    k.uv_type = 1;
    k.uv_offset = 12;
    k.uv_lanes = 0 | 1 << 2;
    k.alpha_type = 17;
  }
  return q;
}

// ---- CPU

int CpuChecks() {
  int failures = 0;
  {
    RtDrawPass main{true, true, true, true, true};
    RtDrawPass shadow{true, true, true, false, false};  // a depth-only target of another size
    RtDrawPass prepass{true, true, true, false, true};  // depth only, the back buffer's size
    RtDrawPass reflection{true, true, true, true, false};  // writes colour, another size
    RtDrawPass cascade{true, true, true, true, false, true};  // writes its depth in R32F colour too
    RtDrawPass no_write{true, true, false, true, true};
    RtDrawPass no_depth{false, false, false, true, true};
    const bool ok = RtViewOf(main, false, true) == RtView::Main && RtViewOf(main, true, true) == RtView::Main &&
                    RtViewOf(shadow, false, true) == RtView::Other && RtViewOf(prepass, false, true) == RtView::Other &&
                    RtViewOf(reflection, false, true) == RtView::None &&
                    RtViewOf(reflection, false, true, true) == RtView::Other &&
                    RtViewOf(cascade, false, true) == RtView::Other && RtViewOf(cascade, true, true) == RtView::None &&
                    RtViewOf(shadow, true, true) == RtView::None && RtViewOf(shadow, false, false) == RtView::None &&
                    RtViewOf(no_write, false, true) == RtView::None && RtViewOf(no_depth, false, true) == RtView::None;
    failures += Check(ok, "views: the main view's draws; other passes' static depth-only draws (not captured ones, "
                          "nor with other passes off; passes writing colour only when asked, or to single-channel "
                          "float targets: a depth map's); none without depth test and write");
  }
  {
    // The backend's choice of mask, step by step.
    int depth_a = 0, depth_b = 0;
    RtShadowWant w;
    w.depth = &depth_a;
    w.epoch = 7;
    w.depth_camera = w.scene_camera = 3;
    w.main_size = true;
    w.light[1] = -1;
    RtShadowMemo m;
    bool ok = RtShadowNext(&m, w) == RtShadowStep::Trace && !m.white && RtShadowNext(&m, w) == RtShadowStep::Reuse;
    RtShadowWant x = w;
    x.epoch = 8;  // depth written since
    ok = ok && RtShadowNext(&m, x) == RtShadowStep::Trace && RtShadowNext(&m, x) == RtShadowStep::Reuse;
    x.depth = &depth_b;  // another depth texture
    ok = ok && RtShadowNext(&m, x) == RtShadowStep::Trace;
    x.light[0] = 0.5f;  // the light moved
    ok = ok && RtShadowNext(&m, x) == RtShadowStep::Trace;
    m = {};  // a presented frame's end
    ok = ok && RtShadowNext(&m, x) == RtShadowStep::Trace;
    m.white = true;  // the trace could not be made: all lit until something changes
    ok = ok && RtShadowNext(&m, x) == RtShadowStep::Reuse && m.white;
    const auto white = [&](RtShadowWant v) {
      RtShadowMemo n;
      return RtShadowNext(&n, v) == RtShadowStep::White && n.white && RtShadowNext(&n, v) == RtShadowStep::Reuse &&
             n.white;
    };
    RtShadowWant other_view = w, last_frame = w, small = w, dark = w, none = w, partial = w, black = w;
    black.dark = true;  // the light's colour black (the sun set)
    other_view.depth_camera = 2;  // written last by another camera
    last_frame.scene_camera = 0;  // no vote yet this frame: the last frame's camera
    small.main_size = false;
    dark.light[1] = 0;
    none.depth = nullptr;
    partial.depth_camera = 0;  // its last main-view draw did not cover the viewport
    ok = ok && white(other_view) && white(last_frame) && white(small) && white(dark) && white(none) && white(partial) &&
         white(black);
    // The light turning black within a frame, and back: all lit, then traced.
    {
      RtShadowMemo n;
      ok = ok && RtShadowNext(&n, w) == RtShadowStep::Trace && RtShadowNext(&n, black) == RtShadowStep::White &&
           RtShadowNext(&n, w) == RtShadowStep::Trace;
    }
    failures += Check(ok, "the mask chosen: traced once, again after a depth write, for another depth texture or "
                          "light and after the frame; all lit when the trace fails, without a depth, light or camera, "
                          "for another view's depth, a partial viewport or the last frame's camera, and where the "
                          "light's colour is black");
    // Why all lit.
    ok = RtShadowWhyWhite(w) == RtShadowWhite::None && RtShadowWhyWhite(none) == RtShadowWhite::NoDepth &&
         RtShadowWhyWhite(small) == RtShadowWhite::WrongSize && RtShadowWhyWhite(dark) == RtShadowWhite::NoLight &&
         RtShadowWhyWhite(last_frame) == RtShadowWhite::NoCamera &&
         RtShadowWhyWhite(other_view) == RtShadowWhite::OtherCamera &&
         RtShadowWhyWhite(partial) == RtShadowWhite::OtherCamera && RtShadowWhyWhite(black) == RtShadowWhite::Dark;
    // The last writer's camera: a voting draw over the whole viewport names
    // it, over part of it forgets it, a draw that did not vote keeps it, a
    // clear forgets it.
    ok = ok && RtShadowWriter(0, false, 5, true) == 5 && RtShadowWriter(5, false, 6, true) == 6 &&
         RtShadowWriter(5, false, 6, false) == 0 && RtShadowWriter(5, false, 0, false) == 5 &&
         RtShadowWriter(5, false, 0, true) == 5 && RtShadowWriter(5, true, 6, true) == 0;
    failures += Check(ok, "why all lit, by reason; a target's camera after a write: a voting draw over the whole "
                          "viewport names it, a partial one or a clear forgets it, others keep it");
    ok = RtShadowSettlesClear(2, true) && RtShadowSettlesClear(4, true) && RtShadowSettlesClear(7, true) &&
         !RtShadowSettlesClear(1, true) && !RtShadowSettlesClear(2, false) && !RtShadowSettlesClear(0, true);
    ok = ok && RtShadowSampler(true, 15, true) == 15 && RtShadowSampler(false, 15, true) == -1 &&
         RtShadowSampler(true, 15, false) == -1 && RtShadowSampler(true, 16, true) == -1;
    failures += Check(ok, "a pending clear run first only if it clears the depth traced; the mask bound only at a "
                          "claimed sampler");
    // Draws of shaders still reading a game's own shadow map: all counted,
    // the first 16 shaders said once each.
    RtOwnMaps om;
    int said = 0;
    for (uint64_t h = 1; h <= 20; h++)
      for (int k = 0; k < 3; k++) said += RtOwnMapDraw(&om, h) ? 1 : 0;
    const bool again = RtOwnMapDraw(&om, 3) || RtOwnMapDraw(&om, 17);
    failures += Check(said == int(kRtOwnMapsSaid) && !again && om.draws == 62 && om.count == kRtOwnMapsSaid &&
                          om.said[0] == 1 && om.said[15] == 16,
                      "own shadow maps' draws: every one counted, the first 16 shaders said once each");
  }
  {
    // A camera 4 km out: a known point back from its pixel and depth,
    // stored reversed and not.
    const V3 o{4000, 30, -4000};
    double vp[16], inv[16];
    Camera(o + V3{0, 12, -8}, o + V3{1, 0, 3}, 1.0, 4.0 / 3.0, 0.5, 3000.0, vp);
    Invert4(vp, inv);
    RtFrame f, rounded;
    for (int k = 0; k < 16; k++) rounded.inv_view_proj[k] = f.inv_view_proj[k] = float(inv[k]);
    std::memcpy(f.inv_view_proj_exact, inv, sizeof inv);
    const V3 p = o + V3{1.37, 0, 2.11};
    double c[4];
    for (int k = 0; k < 4; k++) c[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
    const uint32_t w = 1600, h = 1200;
    const double i = (c[0] / c[3] + 1) * 0.5 * w, j = (1 - c[1] / c[3]) * 0.5 * h, z = c[2] / c[3];
    const float light[3] = {0, 1, 0};
    RtShadowParams rev, fwd;
    const float jitter[2] = {0.25f, -0.375f};
    bool ok = RtShadowParamsFor(f, w, h, true, true, nullptr, light, &rev) &&
              RtShadowParamsFor(f, w, h, false, false, jitter, light, &fwd);
    double a[3], b[3];
    RtShadowPoint(rev, i, j, double(float(1.0 - z)), a);
    RtShadowPoint(fwd, i + 0.25, j - 0.375, double(float(z)), b);
    const double ea = Len(V3{a[0], a[1], a[2]} - p), eb = Len(V3{b[0], b[1], b[2]} - p);
    ok = ok && ea < 1e-3 && eb < 1e-3 && rev.sky == 0.0f && fwd.sky == 1.0f && rev.depth_texture == 1 &&
         fwd.depth_texture == 0 && std::fabs(rev.light[1] - 1.0f) < 1e-6f && rev.pixel > 0.0f;
    // With the float camera alone: its translation rows' rounding over a
    // far point's small w (centimetres 100 m out).
    RtShadowParams fl;
    ok = ok && RtShadowParamsFor(rounded, w, h, true, true, nullptr, light, &fl);
    double e[3];
    RtShadowPoint(fl, i, j, double(float(1.0 - z)), e);
    const double ef = Len(V3{e[0], e[1], e[2]} - p);
    std::printf("  4 km: %.3f mm reversed, %.3f mm not; %.2f mm from the float camera\n", ea * 1e3, eb * 1e3, ef * 1e3);
    const float none[3] = {0, 0, 0};
    RtFrame singular;
    ok = ok && !RtShadowParamsFor(f, w, h, true, true, nullptr, none, &rev) &&
         !RtShadowParamsFor(singular, w, h, true, true, nullptr, light, &rev);
    failures += Check(ok, "a point 4 km from the origin back from its pixel and depth within 1 mm (reversed; not "
                          "reversed, jittered) with the scene's double camera; no parameters without a light or a "
                          "camera");
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
  std::unique_ptr<RtShadowMasker> masker;

  bool Init() {
    arena.buf = [dev newBufferWithLength:16u << 20 options:MTLResourceStorageModeShared];
    masker = RtShadowMasker::Make(dev);
    return arena.buf && masker;
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
};

constexpr uint32_t kW = 320, kH = 240;
constexpr double kSize = 4;  // the square's side

// A scene around `o`: the ground (y = o.y, 120 x 120) drawn by the main
// view, a 4 x 4 square 2 above it drawn only by another pass (twice: two
// cascades), the ground by that pass too; the camera above and behind,
// looking down at it; depth stored reversed in a depth texture or not in a
// float one, drawn `jitter` pixels right and down. The mask for light `l`
// against the shadow the square casts.
struct Case {
  const char* what;
  V3 o;
  V3 l;
  bool reversed;
  bool cut;  // the square alpha-cut, its half x < 2 a hole
  float jitter[2];
  bool diag = false;  // with diagnostics (RtShadowDiag, the TLAS summarised) and the mask's view
  bool linear = false;  // the square's alpha bilinear (MAGFILTER linear): 0 to 1 between the texels' centres
};

struct Result {
  bool ran = false;
  int robust = 0, wrong = 0, shadowed = 0, lit = 0, edge = 0, sky = 0, sky_wrong = 0;
  int ground_shadowed_where_lit = 0;
  uint32_t instances = 0;
  uint32_t rays = 0, hits = 0;  // the kernel's counters
  RtStats stats;
  bool diag_ok = false;  // (Case::diag) the diagnostics as the scene and the mask say
};

// How far above the ground (normal +y) a ray toward `l` starts with offset
// `off` (rt_shadow.h): off / max(N.L, min_slope) along l, and along the
// normal what that leaves short of off.
double Rise(double off, V3 l, double min_slope) {
  l = Unit(l);
  return off * std::clamp(1.0 - l.y / min_slope, 0.0, 1.0) + l.y * off / std::max(l.y, min_slope);
}

Result Run(Gpu& g, const Case& c) {
  Result res;
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
  if (!s) return res;
  g.arena.used = 0;
  s->SetAllocator(ArenaAlloc, &g.arena);
  const V3 o = c.o;
  double vp[16], inv[16], lvp[16];
  const V3 eye = o + V3{1, 12, -8};
  Camera(eye, o + V3{1, 0, 3}, 1.0, double(kW) / kH, 0.5, 3000.0, vp);
  Invert4(vp, inv);
  Camera(o + c.l * 50.0, o, 0.8, 1.0, 1.0, 200.0, lvp);  // the other pass's camera (a light's)
  const V3 ground[4] = {{-60, 0, -60}, {-60, 0, 60}, {60, 0, -60}, {60, 0, 60}};
  const V3 square[4] = {{0, 2, 0}, {0, 2, kSize}, {kSize, 2, 0}, {kSize, 2, kSize}};
  Quad gq = MakeQuad(ground, 1, false), sq = MakeQuad(square, 2, c.cut);
  // A 2 x 1 texture: texel 0 (u < 0.5) alpha 0, texel 1 alpha 255.
  id<MTLTexture> tex = nil;
  RtCut cut;
  if (c.cut) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:2
                                                                                 height:1
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    tex = [g.dev newTextureWithDescriptor:td];
    const uint32_t texels[2] = {0x00808080u, 0xFF808080u};
    [tex replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0 withBytes:texels bytesPerRow:8];
    cut.texture = tex;
    cut.material.alpha_scale = 1.0f;
    cut.material.flags = RtMaterialFlags(c.linear ? kRtMatLinear : 0, 7, 128, 3, 3, 3);  // GREATEREQUAL 128, clamp
  }
  Mat w, mw, ow;
  Translate(w, o);
  Wvp(w, vp, mw);
  Wvp(w, lvp, ow);
  s->BeginFrame(1, 0);
  // The other pass first (shadow maps before the scene), each cascade.
  s->Add(sq.key, sq.g, w, ow, 3, 4, c.cut ? &cut : nullptr, RtView::Other);
  s->Add(sq.key, sq.g, w, ow, 3, 4, c.cut ? &cut : nullptr, RtView::Other);
  s->Add(gq.key, gq.g, w, ow, 3, 4, nullptr, RtView::Other);
  s->Add(gq.key, gq.g, w, mw, 3, 4, nullptr, RtView::Main);  // already placed: votes
  std::vector<RtInstanceSummary> summary;
  if (c.diag) s->DescribeNextTlas(&summary);
  const RtFrame* f = s->Snapshot();
  res.instances = f ? f->instances : 0;

  // The depth buffer of the ground: per pixel (D3D9's centres) the ray
  // from the eye, its hit on y = o.y; the far plane where there is none.
  std::vector<float> depth(size_t(kW) * kH);
  std::vector<V3> points(depth.size());
  std::vector<bool> hit(depth.size());
  const double sky = 1.0;
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const double x = 2.0 * (i - c.jitter[0]) / kW - 1.0, y = 1.0 - 2.0 * (j - c.jitter[1]) / kH;
      double a[4], b[4];
      for (int k = 0; k < 4; k++) {
        a[k] = x * inv[k] + y * inv[4 + k] + 0.0 * inv[8 + k] + inv[12 + k];
        b[k] = x * inv[k] + y * inv[4 + k] + 1.0 * inv[8 + k] + inv[12 + k];
      }
      const V3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const V3 d = pb - pa;
      const size_t at = size_t(j) * kW + i;
      double z = sky;
      if (d.y < 0) {
        const double t = (o.y - pa.y) / d.y;
        if (t > 0 && t < 1) {
          const V3 p = pa + d * t;
          double cc[4];
          for (int k = 0; k < 4; k++) cc[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
          z = cc[2] / cc[3];
          points[at] = p;
          hit[at] = true;
        }
      }
      depth[at] = float(c.reversed ? 1.0 - z : z);
    }
  // Row 0 the sky whatever the ground (a sky dome drawn at the far plane).
  for (uint32_t i = 0; i < kW; i++) {
    depth[i] = float(c.reversed ? 1.0 - sky : sky);
    hit[i] = false;
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
    id<MTLBuffer> src = [g.dev newBufferWithBytes:depth.data() length:depth.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:kW * 4
        sourceBytesPerImage:kW * kH * 4
                 sourceSize:MTLSizeMake(kW, kH, 1)
                  toTexture:dt
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)
                    options:MTLBlitOptionDepthFromDepthStencil];
    [bl endEncoding];
  } else {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                  width:kW
                                                                                 height:kH
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    dt = [g.dev newTextureWithDescriptor:td];
    [dt replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:depth.data() bytesPerRow:kW * 4];
  }
  RtShadowParams p;
  const float l[3] = {float(c.l.x), float(c.l.y), float(c.l.z)};
  if (!f || !f->tlas || !RtShadowParamsFor(*f, kW, kH, c.reversed, c.reversed, c.jitter, l, &p)) return res;
  id<MTLTexture> mask = g.masker->NewMask(kW, kH);
  id<MTLBuffer> out = [g.dev newBufferWithLength:kW * kH options:MTLResourceStorageModeShared];
  s->EncodeFrame(cb, f);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 8);
  id<MTLBuffer> diag = nil;
  if (c.diag) {
    diag = [g.dev newBufferWithLength:sizeof(RtShadowDiag) + kRtShadowDiagCounts * 4 options:MTLResourceStorageModeShared];
    std::memset(diag.contents, 0, diag.length);
    p.diag = kRtShadowDiagCounts;
  }
  g.masker->Encode(cb, f, dt, mask, p, counts, 0, diag);
  // The mask's view over a target of another size (white lit).
  constexpr uint32_t kVw = 64, kVh = 48;
  MTLTextureDescriptor* vd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                width:kVw
                                                                               height:kVh
                                                                            mipmapped:NO];
  vd.storageMode = MTLStorageModePrivate;
  vd.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> view = [g.dev newTextureWithDescriptor:vd];
  id<MTLBuffer> vout = [g.dev newBufferWithLength:kVw * kVh * 4 options:MTLResourceStorageModeShared];
  if (c.diag) g.masker->EncodeView(cb, mask, view);
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromTexture:mask
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kW, kH, 1)
                    toBuffer:out
           destinationOffset:0
      destinationBytesPerRow:kW
    destinationBytesPerImage:kW * kH];
  if (c.diag)
    [bl copyFromTexture:view sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(kVw, kVh, 1) toBuffer:vout destinationOffset:0 destinationBytesPerRow:kVw * 4
        destinationBytesPerImage:kVw * kVh * 4];
  [bl endEncoding];
  if (!g.Commit(cb)) return res;
  // The frame ends as the backend's do (its TLAS of everything).
  const RtFrame* end = s->EndFrame(true);
  id<MTLCommandBuffer> cb2 = [g.queue commandBuffer];
  s->EncodeFrame(cb2, end);
  if (!g.Commit(cb2)) return res;
  res.stats = s->Stats();
  res.ran = true;
  res.rays = static_cast<const uint32_t*>(counts.contents)[0];
  res.hits = static_cast<const uint32_t*>(counts.contents)[1];

  // Each pixel against the square's shadow: the ray from the ground point
  // toward the light, where it crosses the square's plane. Pixels within a
  // margin of an edge (the reconstruction's error and the ray's offset)
  // are not judged.
  const V3 lu = Unit(c.l);
  const auto* m = static_cast<const uint8_t*>(out.contents);
  const double hole_x = c.cut ? kSize / 2 : 0.0;
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const size_t at = size_t(j) * kW + i;
      const bool got_lit = m[at] > 127;
      if (!hit[at]) {
        res.sky++;
        if (!got_lit) res.sky_wrong++;
        continue;
      }
      const V3 pw = points[at];
      double foot = 0;
      if (i + 1 < kW && hit[at + 1]) foot = std::max(foot, Len(points[at + 1] - pw));
      if (j + 1 < kH && hit[at + kW]) foot = std::max(foot, Len(points[at + kW] - pw));
      if (foot == 0) continue;
      const double dist = Len(pw - eye);
      const double off = dist * (p.offset_pixels * p.pixel + p.offset_view) + Len(pw) * p.offset_world;
      const double margin = 2.0 * foot + 2.0 * off * std::sqrt(lu.x * lu.x + lu.z * lu.z) / lu.y + 1e-3;
      const V3 q = (pw - o) + lu * ((2.0 - (pw.y - o.y)) / lu.y);  // relative to o
      const bool inside = q.x > hole_x && q.x < kSize && q.z > 0 && q.z < kSize;
      const double edge = std::min({std::fabs(q.x), std::fabs(q.x - kSize), std::fabs(q.z), std::fabs(q.z - kSize),
                                    c.cut ? std::fabs(q.x - hole_x) : 1e9});
      const bool near_square = q.x > -margin && q.x < kSize + margin && q.z > -margin && q.z < kSize + margin;
      if (near_square && edge < margin) continue;
      res.robust++;
      if (inside) {
        res.shadowed++;
      } else {
        res.lit++;
        if (!got_lit) res.ground_shadowed_where_lit++;
      }
      if (near_square && edge < margin + 3.0 * foot) res.edge++;
      if (got_lit == inside) res.wrong++;
    }
  if (c.diag) {
    // The centre pixel's record against its ground point and mask; the
    // counts by instance against the counters; the summaries: the square
    // (another view's, its box) the only blocker; the view the mask's
    // nearest texels.
    const auto* r = static_cast<const RtShadowDiag*>(diag.contents);
    const auto* bc = reinterpret_cast<const uint32_t*>(r + 1);
    const size_t centre = size_t(kH / 2) * kW + kW / 2;
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kRtShadowDiagCounts; i++) sum += bc[i];
    const V3 cp = points[centre];
    bool ok = hit[centre] && Len(V3{r->point[0], r->point[1], r->point[2]} - cp) < 1e-2 &&
              r->state == (m[centre] > 127 ? 2u : 3u) && std::fabs(r->normal[1] - 1.0f) < 1e-3f &&
              std::fabs(r->origin[1] - r->point[1] - Rise(r->origin[3], c.l, p.min_slope)) < 1e-4f &&
              sum == res.hits;
    uint32_t square = UINT32_MAX;
    for (uint32_t i = 0; i < summary.size(); i++)
      if (summary[i].view == 1 && summary[i].box && std::fabs(summary[i].lo[1] - 2.0f) < 1e-3f &&
          std::fabs(summary[i].hi[0] - float(kSize)) < 1e-3f)
        square = i;
    ok = ok && summary.size() == 2 && square != UINT32_MAX && bc[square] == sum && summary[square].triangles == 2;
    const auto* vp = static_cast<const uint8_t*>(vout.contents);
    for (uint32_t j = 0; j < kVh && ok; j++)
      for (uint32_t i = 0; i < kVw && ok; i++) {
        const uint32_t mi = (2 * i + 1) * kW / (2 * kVw), mj = (2 * j + 1) * kH / (2 * kVh);
        const uint8_t want = m[size_t(mj) * kW + mi];
        const uint8_t* px = vp + (size_t(j) * kVw + i) * 4;
        ok = px[0] == want && px[1] == want && px[2] == want && px[3] == 255;
      }
    res.diag_ok = ok;
    std::printf("  diagnostics: centre state %u point (%.2f %.2f %.2f), %llu blocked counted, square #%u\n", r->state,
                r->point[0], r->point[1], r->point[2], (unsigned long long)sum, square);
  }
  std::printf("  %s: %d pixels judged (%d in shadow, %d lit, %d next to an edge), %d wrong, %d sky (%d not lit); "
              "%u rays, %u blocked\n",
              c.what, res.robust, res.shadowed, res.lit, res.edge, res.wrong, res.sky, res.sky_wrong, res.rays,
              res.hits);
  return res;
}

int GpuChecks(Gpu& g) {
  int failures = 0;
  const V3 tilt = Unit({-0.35, 1.0, 0.25});
  const Case kCases[] = {
      {"reversed depth texture", {0, 0, 0}, tilt, true, false, {0, 0}, true},
      {"float texture, not reversed, jittered", {0, 0, 0}, tilt, false, false, {0.3f, -0.4f}},
      {"4 km from the origin, a low light", {4000, 30, -4000}, Unit({-0.3, 0.18, -1.0}), true, false, {0, 0}},
      {"alpha-cut square, half a hole", {0, 0, 0}, tilt, true, true, {0, 0}},
      {"alpha-cut square, bilinear, half a hole", {0, 0, 0}, tilt, true, true, {0, 0}, false, true},
  };
  Result r[5];
  for (int k = 0; k < 5; k++) r[k] = Run(g, kCases[k]);
  bool ok = true;
  for (const Result& x : r)
    ok = ok && x.ran && x.wrong == 0 && x.shadowed > 100 && x.lit > 4000 && x.edge > 40 && x.sky >= int(kW) &&
         x.sky_wrong == 0 && x.rays == kW * kH - uint32_t(x.sky) && x.hits >= uint32_t(x.shadowed) &&
         x.hits < x.rays / 4;
  failures += Check(ok, "every judged pixel as the square's shadow predicts, both sides of its edges; the sky lit; "
                        "a ray for every other pixel, counted with those blocked");
  failures += Check(r[0].diag_ok, "diagnostics: the centre pixel's point, normal, ray and outcome; rays blocked by "
                                  "instance; the TLAS's summaries (another view's square the blocker); the mask's "
                                  "view");
  failures += Check(r[0].ran && r[1].ran && std::abs(r[0].shadowed - r[1].shadowed) * 20 < r[0].shadowed,
                    "reversed depth texture and float texture (jittered) alike");
  failures += Check(r[2].ran && r[2].ground_shadowed_where_lit == 0 && r[2].lit > 4000,
                    "4 km from the origin with a low light: no pixel of the plane shadows itself");
  failures += Check(r[3].ran && r[3].wrong == 0 && r[3].shadowed < r[0].shadowed && r[3].shadowed > 40,
                    "the alpha-cut square: light through its hole, shadow under the rest");
  failures += Check(r[4].ran && r[4].wrong == 0 && std::abs(r[4].shadowed - r[3].shadowed) * 20 < r[3].shadowed,
                    "the alpha-cut square bilinear (its alpha 0.5 where nearest changes texel): the same hole");
  // The scene: the square from the other pass once (two cascades), the
  // ground drawn by both once; the main view's draw still sets the camera.
  const RtStats& st = r[0].stats;
  failures += Check(r[0].instances == 2 && st.deduped == 2 && st.other == 2 && st.instances == 2,
                    "another pass's draws in the TLAS; the same mesh and world matrix from two passes one instance");
  // The all-lit mask.
  {
    id<MTLTexture> white = g.masker->White();
    uint8_t v = 0;
    [white getBytes:&v bytesPerRow:1 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    failures += Check(white && white.pixelFormat == MTLPixelFormatR8Unorm && v == 255, "the all-lit mask");
  }
  return failures;
}

// The scene's other views: a mirrored world left out; the same geometry
// drawn opaque by another view and alpha-cut by the main view one
// instance, the main view's (cut); a losing camera's instance that an
// other view drew too listed; other views' instances after a snapshot not
// carried into the next frame's.
int SceneChecks(Gpu& g) {
  int failures = 0;
  const V3 ground[4] = {{-10, 0, -10}, {-10, 0, 10}, {10, 0, -10}, {10, 0, 10}};
  const V3 square[4] = {{0, 2, 0}, {0, 2, 1}, {1, 2, 0}, {1, 2, 1}};
  Quad gq = MakeQuad(ground, 1, false), sq = MakeQuad(square, 2, false), cq = MakeQuad(square, 2, true);
  double vp[16], vp2[16], lvp[16];
  Camera({1, 12, -8}, {1, 0, 3}, 1.0, 4.0 / 3.0, 0.5, 3000.0, vp);
  Camera({30, 5, 0}, {0, 0, 0}, 0.7, 1.0, 0.5, 500.0, vp2);
  Camera({0, 50, 0}, {0, 0, 0.01}, 0.8, 1.0, 1.0, 200.0, lvp);
  Mat w, mw, mw2, ow, mir;
  Translate(w, {0, 0, 0});
  Wvp(w, vp, mw);
  Wvp(w, vp2, mw2);
  Wvp(w, lvp, ow);
  std::memcpy(mir, w, sizeof(Mat));
  mir[1][1] = -1;  // mirrored in y
  // A texture for the cut square (opaque texels: the material is what is checked).
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> tex = [g.dev newTextureWithDescriptor:td];
  const uint32_t texel = 0xFFFFFFFFu;
  [tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&texel bytesPerRow:4];
  RtCut cut;
  cut.texture = tex;
  cut.material.alpha_scale = 1.0f;
  cut.material.flags = RtMaterialFlags(0, 7, 128, 3, 3, 3);
  auto frame = [&](std::unique_ptr<RtScene>& s, uint64_t serial, auto adds) {
    g.arena.used = 0;
    s->BeginFrame(serial, serial - 1);
    adds();
    const RtFrame* f = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, f);
    g.Commit(cb);
    return f;
  };
  {
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
    s->SetAllocator(ArenaAlloc, &g.arena);
    uint64_t vote_main = 0, vote_other = 1;
    const RtFrame* f = frame(s, 1, [&] {
      s->Add(sq.key, sq.g, mir, ow, 3, 4, nullptr, RtView::Other);    // mirrored: left out
      s->Add(sq.key, sq.g, w, ow, 3, 4, nullptr, RtView::Other);      // opaque, another view
      vote_other = s->TakeVote();
      s->Add(cq.key, cq.g, w, mw, 3, 4, &cut, RtView::Main);          // cut, the main view: takes its place
      vote_main = s->TakeVote();
      s->Add(gq.key, gq.g, w, mw, 3, 4, nullptr, RtView::Main);
    });
    const RtStats st = s->Stats();
    failures += Check(f && f->instances == 2 && st.skipped[size_t(RtSkip::OtherMirrored)] == 1 && st.deduped == 1 &&
                          st.cut == 1 && vote_other == 0 && vote_main != 0 && f->camera_id == vote_main &&
                          s->CameraId() == 0,
                      "another view's mirrored world left out; its opaque draw and the main view's cut one of the "
                      "same geometry one instance, cut; the main view's votes name the frame's camera");
  }
  {
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
    s->SetAllocator(ArenaAlloc, &g.arena);
    const RtFrame* f = frame(s, 1, [&] {
      s->Add(gq.key, gq.g, w, mw, 3, 4, nullptr, RtView::Main);  // camera 1 (wins)
      Mat w2, m2;
      Translate(w2, {0, -1, 0});
      Wvp(w2, vp, m2);
      s->Add(gq.key, gq.g, w2, m2, 3, 4, nullptr, RtView::Main);
      s->Add(sq.key, sq.g, w, mw2, 3, 4, nullptr, RtView::Main);  // camera 2 (loses)
      s->Add(sq.key, sq.g, w, ow, 3, 4, nullptr, RtView::Other);  // drawn by another view too: listed
    });
    failures += Check(f && f->instances == 3 && s->Stats().deduped == 1 &&
                          s->Stats().skipped[size_t(RtSkip::OtherCamera)] == 0,
                      "a losing camera's instance another view drew too: listed with any camera");
  }
  {
    // Frame 1: a snapshot, then another view's draw; frame 2's snapshot
    // carries the main view's draws after frame 1's, not the other's.
    // (Without retained instances, which would keep the other's: carrying
    // alone.)
    RtSceneOptions no_retain;
    no_retain.retain_frames = 0;
    std::unique_ptr<RtScene> s = MakeRtScene(g.dev, no_retain);
    s->SetAllocator(ArenaAlloc, &g.arena);
    g.arena.used = 0;
    s->BeginFrame(1, 0);
    s->Add(gq.key, gq.g, w, mw, 3, 4, nullptr, RtView::Main);
    const RtFrame* a = s->Snapshot();
    s->Add(sq.key, sq.g, w, ow, 3, 4, nullptr, RtView::Other);
    const RtFrame* b = s->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, a);
    s->EncodeFrame(cb, b);
    g.Commit(cb);
    s->BeginFrame(2, 1);
    s->Add(gq.key, gq.g, w, mw, 3, 4, nullptr, RtView::Main);
    const RtFrame* c = s->Snapshot();
    const uint32_t carried = c ? c->instances : 0;
    s->EndFrame(true);
    failures += Check(a && b && b->instances == 2 && carried == 1, "other views' instances not carried into the next "
                                                                    "frame's snapshots");
  }
  return failures;
}

// Captured draws block a ray only beyond the gap (kRtShadowCapturedGap of
// the distance) where the point is on one (the camera's line through it
// meets a captured instance within the gap): a captured square 3 cm above
// the ground leaves it lit (as it would its own surface: the line through
// the ground under it meets the square), one 2 m up shadows it; a static
// square 3 cm up shadows it. A captured wall standing on the ground (a
// person's leg) shadows the ground on its far side from the light right up
// to its foot (the pixels within 12 cm of it: within the gap, the line
// through those points meeting nothing captured).
int CapturedChecks(Gpu& g) {
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
  s->SetAllocator(ArenaAlloc, &g.arena);
  g.arena.used = 0;
  double vp[16], inv[16];
  const V3 eye{1, 12, -8};
  Camera(eye, {1, 0, 3}, 1.0, double(kW) / kH, 0.5, 3000.0, vp);
  Invert4(vp, inv);
  const V3 l = Unit({-0.35, 1.0, 0.25});
  const V3 ground[4] = {{-60, 0, -60}, {-60, 0, 60}, {60, 0, -60}, {60, 0, 60}};
  auto square = [](V3 c, double h) {
    return std::array<V3, 4>{V3{c.x - 0.5, h, c.z - 0.5}, V3{c.x - 0.5, h, c.z + 0.5}, V3{c.x + 0.5, h, c.z - 0.5},
                             V3{c.x + 0.5, h, c.z + 0.5}};
  };
  const V3 c_low{-3, 0, 4}, c_high{1, 0, 6}, c_static{4.5, 0, 3};
  const auto low = square(c_low, 0.03), high = square(c_high, 2.0), fixed = square(c_static, 0.03);
  // The wall: x = -2, z 0.5 to 1.5, 1.8 high (the light toward -x: its
  // shadow on the +x side, toward the camera).
  const std::array<V3, 4> wall = {V3{-2, 0, 0.5}, V3{-2, 0, 1.5}, V3{-2, 1.8, 0.5}, V3{-2, 1.8, 1.5}};
  Quad gq = MakeQuad(ground, 1, false), lq = MakeQuad(low.data(), 2, false), hq = MakeQuad(high.data(), 3, false),
       sq = MakeQuad(fixed.data(), 4, false), wq = MakeQuad(wall.data(), 5, false);
  Mat w, mw;
  Translate(w, {0, 0, 0});
  Wvp(w, vp, mw);
  s->BeginFrame(1, 0);
  s->Add(gq.key, gq.g, w, mw, 3, 4);
  s->Add(sq.key, sq.g, w, mw, 3, 4);
  // The captured squares: their clip positions written where the capture
  // would.
  RtGeometry cl = lq.g, ch = hq.g, cw = wq.g;
  cl.positions = ch.positions = cw.positions = nullptr;
  const RtCaptureSlot slot_low = s->AddCaptured(cl, 0, 4, 1), slot_high = s->AddCaptured(ch, 0, 4, 1),
                      slot_wall = s->AddCaptured(cw, 0, 4, 1);
  auto clip = [&](const std::array<V3, 4>& q) {
    std::vector<float> c;
    for (const V3& v : q)
      for (int k = 0; k < 4; k++)
        c.push_back(float(v.x * vp[k] + v.y * vp[4 + k] + v.z * vp[8 + k] + vp[12 + k]));
    return c;
  };
  const std::vector<float> clip_low = clip(low), clip_high = clip(high), clip_wall = clip(wall);
  const RtFrame* f = s->Snapshot();
  // The depth: the ground's (D3D9's centres, reversed), as Run makes it.
  std::vector<float> depth(size_t(kW) * kH);
  std::vector<V3> pts(depth.size());
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const double x = 2.0 * i / kW - 1.0, y = 1.0 - 2.0 * j / kH;
      double a[4], b[4];
      for (int k = 0; k < 4; k++) {
        a[k] = x * inv[k] + y * inv[4 + k] + inv[12 + k];
        b[k] = x * inv[k] + y * inv[4 + k] + inv[8 + k] + inv[12 + k];
      }
      const V3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const V3 d = pb - pa;
      const double t = -pa.y / d.y;
      const V3 p = pa + d * t;
      double cc[4];
      for (int k = 0; k < 4; k++) cc[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
      depth[size_t(j) * kW + i] = float(1.0 - cc[2] / cc[3]);
      pts[size_t(j) * kW + i] = p;
    }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                                width:kW
                                                                               height:kH
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  id<MTLTexture> dt = [g.dev newTextureWithDescriptor:td];
  RtShadowParams p;
  const float lf[3] = {float(l.x), float(l.y), float(l.z)};
  if (!slot_low.buf || !slot_high.buf || !slot_wall.buf || !f ||
      !RtShadowParamsFor(*f, kW, kH, true, true, nullptr, lf, &p))
    return Check(false, "captured: the scene and parameters");
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  id<MTLBuffer> src = [g.dev newBufferWithBytes:depth.data() length:depth.size() * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cs0 = [g.dev newBufferWithBytes:clip_low.data() length:clip_low.size() * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cs1 = [g.dev newBufferWithBytes:clip_high.data() length:clip_high.size() * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cs2 = [g.dev newBufferWithBytes:clip_wall.data()
                                         length:clip_wall.size() * 4
                                        options:MTLResourceStorageModeShared];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromBuffer:src sourceOffset:0 sourceBytesPerRow:kW * 4 sourceBytesPerImage:kW * kH * 4
          sourceSize:MTLSizeMake(kW, kH, 1) toTexture:dt destinationSlice:0 destinationLevel:0
   destinationOrigin:MTLOriginMake(0, 0, 0) options:MTLBlitOptionDepthFromDepthStencil];
  [bl copyFromBuffer:cs0 sourceOffset:0 toBuffer:slot_low.buf destinationOffset:slot_low.capture_off size:cs0.length];
  [bl copyFromBuffer:cs1 sourceOffset:0 toBuffer:slot_high.buf destinationOffset:slot_high.capture_off size:cs1.length];
  [bl copyFromBuffer:cs2 sourceOffset:0 toBuffer:slot_wall.buf destinationOffset:slot_wall.capture_off size:cs2.length];
  [bl endEncoding];
  id<MTLTexture> mask = g.masker->NewMask(kW, kH);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 8);
  id<MTLBuffer> out = [g.dev newBufferWithLength:kW * kH options:MTLResourceStorageModeShared];
  s->EncodeFrame(cb, f);
  g.masker->Encode(cb, f, dt, mask, p, counts, 0);
  bl = [cb blitCommandEncoder];
  [bl copyFromTexture:mask sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(kW, kH, 1) toBuffer:out destinationOffset:0 destinationBytesPerRow:kW
      destinationBytesPerImage:kW * kH];
  [bl endEncoding];
  if (!g.Commit(cb)) return Check(false, "captured: traced");
  s->EndFrame(true);
  // Under each square's middle (its shadow's: along the light down to the
  // ground), the 5 x 5 pixels there.
  const auto* m = static_cast<const uint8_t*>(out.contents);
  auto lit_under = [&](V3 c, double h, int* lit, int* dark) {
    const V3 gp = V3{c.x, h, c.z} - l * (h / l.y);
    double cc[4];
    for (int k = 0; k < 4; k++) cc[k] = gp.x * vp[k] + gp.y * vp[4 + k] + gp.z * vp[8 + k] + vp[12 + k];
    const int i = int(std::lround((cc[0] / cc[3] + 1) * 0.5 * kW)), j = int(std::lround((1 - cc[1] / cc[3]) * 0.5 * kH));
    *lit = *dark = 0;
    for (int dj = -2; dj <= 2; dj++)
      for (int di = -2; di <= 2; di++) (m[size_t(j + dj) * kW + size_t(i + di)] > 127 ? *lit : *dark)++;
  };
  int ll, ld, hl, hd, sl, sd;
  lit_under(c_low, 0.03, &ll, &ld);
  lit_under(c_high, 2.0, &hl, &hd);
  lit_under(c_static, 0.03, &sl, &sd);
  std::printf("  captured 3 cm up: %d lit of 25; captured 2 m up: %d dark of 25; static 3 cm up: %d dark of 25\n", ll,
              hd, sd);
  int failures = Check(ll == 25 && hd == 25 && sd == 25,
                       "captured draws block beyond the gap only (one 3 cm up leaves the ground lit, one 2 m up "
                       "shadows it); static ones from the surface on");
  // The ground at the wall's foot, on its shadow's side, at its middle: the
  // pixels whose points lie 1 to 12 cm from it (each its distance printed;
  // nearer, the ray's start may lie past it),
  // and those 0.9 to 1.1 m off.
  int near_n = 0, near_dark = 0, far_n = 0, far_lit = 0;
  for (size_t k = 0; k < pts.size(); k++) {
    const V3 q = pts[k];
    if (q.z < 0.8 || q.z > 1.2) continue;
    const double off = q.x + 2.0;
    if (off > 0.01 && off < 0.12) {
      near_n++;
      near_dark += m[k] < 128;
      std::printf("    %.1f cm: %u\n", off * 100, m[k]);
    }
    if (off > 0.9 && off < 1.1) far_n++, far_lit += m[k] > 127;
  }
  std::printf("  the wall's foot: %d of %d pixels within 12 cm dark; %d of %d 1 m off lit\n", near_dark, near_n,
              far_lit, far_n);
  failures += Check(near_n > 0 && near_dark == near_n && far_n > 0 && far_lit == far_n,
                    "a captured wall standing on the ground shadows it up to its foot, not 1 m off");
  return failures;
}

// ---- drawn through the backend's prelude

using Tokens = std::vector<uint32_t>;
enum Reg : uint32_t { R = 0, V = 1, C = 2, O = 6, OC = 8 };
uint32_t TypeBits(uint32_t type) { return ((type & 7) << 28) | ((type & 0x18) << 8); }
uint32_t Dst(uint32_t type, uint32_t n) { return 0x80000000u | n | TypeBits(type) | 0xFu << 16; }
uint32_t Src(uint32_t type, uint32_t n, uint32_t swz = 0xE4) { return 0x80000000u | n | TypeBits(type) | swz << 16; }

// vs_3_0: the position times the matrix in c8-c11 (rows), as the HLSL
// compiler makes mul(float4(pos, 1), M); ps_3_0: c0.
Tokens WvpVs() {
  Tokens t = {0xFFFE0300u};
  auto ins = [&](uint32_t op, std::initializer_list<uint32_t> p) {
    t.push_back(op | uint32_t(p.size()) << 24);
    t.insert(t.end(), p);
  };
  ins(31, {0x80000000u, Dst(V, 0)});  // dcl_position v0
  ins(31, {0x80000000u, Dst(O, 0)});  // dcl_position o0
  ins(5, {Dst(R, 0), Src(V, 0, 0x55), Src(C, 9)});
  ins(4, {Dst(R, 0), Src(V, 0, 0x00), Src(C, 8), Src(R, 0)});
  ins(4, {Dst(R, 0), Src(V, 0, 0xAA), Src(C, 10), Src(R, 0)});
  ins(2, {Dst(O, 0), Src(R, 0), Src(C, 11)});
  t.push_back(0x0000FFFF);
  return t;
}
Tokens ConstPs() {
  Tokens t = {0xFFFF0300u, 0x02000001, Dst(OC, 0), Src(C, 0), 0x0000FFFF};
  return t;
}

// The prelude's state blocks (std140), as metal_device.mm fills them.
struct VsState {
  int32_t in[16][4];
  float pixel[4];
  float clip[6][4];
  int32_t clip_mask[4];
  float depth[4];
};
struct PsState {
  int32_t alpha[4];
  float bias[16];
  float vscale[4];
};

id<MTLFunction> Compile(Gpu& g, const Tokens& tokens, std::vector<hlsl::Semantic>* inputs) {
  hlsl::Program p;
  std::string err;
  if (!hlsl::TranslateBytecode(tokens.data(), tokens.size(), &p, &err)) {
    std::printf("FAIL translate: %s\n", err.c_str());
    return nil;
  }
  const bool vertex = p.stage == hlsl::Stage::Vertex;
  if (vertex) *inputs = p.inputs;
  const std::string glsl = "#version 450\n" + MetalShaderPrelude(p.stage) + p.glsl;
  std::vector<uint32_t> spirv;
  std::string msl, log;
  if (!GlslToSpirv(glsl, vertex ? ShaderStage::Vertex : ShaderStage::Fragment, &spirv, &log) ||
      !SpirvToMsl(spirv, &msl, &log)) {
    std::printf("FAIL compile: %s\n", log.c_str());
    return nil;
  }
  MTLCompileOptions* o = nil;
  if (MetalLeanMacros(&msl, vertex)) {
    o = [MTLCompileOptions new];
    o.preprocessorMacros = @{@"R_ALPHA_TEST" : @"true"};
  }
  NSError* e = nil;
  id<MTLLibrary> lib = [g.dev newLibraryWithSource:@(msl.c_str()) options:o error:&e];
  if (!lib) {
    std::printf("FAIL MSL: %s\n", e.localizedDescription.UTF8String);
    return nil;
  }
  return [lib newFunctionWithName:@"main0"];
}

// Faces: 0 none (sky), 1 the ground (y = 0), 2-7 the box [0, 2]^3's +x, -x,
// +y, -y, +z, -z.
constexpr double kBox = 2;
const V3 kFaceNormal[8] = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

// Where the ray o + t d (t > 0) first meets the scene: t and the face.
int Hit(V3 o, V3 d, double* t_out) {
  int face = 0;
  double best = 1e30;
  if (d.y < 0) {
    const double t = -o.y / d.y;
    if (t > 0) best = t, face = 1;
  }
  // The box: slabs.
  double t0 = 0, t1 = 1e30;
  int enter = -1;
  const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
  bool miss = false;
  for (int a = 0; a < 3 && !miss; a++) {
    if (std::fabs(dd[a]) < 1e-12) {
      if (oo[a] < 0 || oo[a] > kBox) miss = true;
      continue;
    }
    double ta = (0 - oo[a]) / dd[a], tb = (kBox - oo[a]) / dd[a];
    int fa = 2 + a * 2 + 1, fb = 2 + a * 2;  // entering at 0: the -axis face; at kBox: the +axis face
    if (ta > tb) std::swap(ta, tb), std::swap(fa, fb);
    if (ta > t0) t0 = ta, enter = fa;
    t1 = std::min(t1, tb);
    if (t0 > t1) miss = true;
  }
  if (!miss && enter >= 0 && t0 > 0 && t0 < best) best = t0, face = enter;
  *t_out = best;
  return face;
}

struct BoxCase {
  const char* what;
  bool reversed_camera;  // depth 1 at near (D3DCMP_GREATER)
  float jitter[2];
};

int BoxRun(Gpu& g, const BoxCase& c) {
  std::vector<hlsl::Semantic> inputs;
  id<MTLFunction> vs = Compile(g, WvpVs(), &inputs), ps = Compile(g, ConstPs(), &inputs);
  if (!vs || !ps) return Check(false, "box: the shaders made");
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = vs;
  pd.fragmentFunction = ps;
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  NSError* err = nil;
  id<MTLRenderPipelineState> pipe = [g.dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pipe) return Check(false, "box: the pipeline made");
  // Stored reversed (1 - d): the game's test mirrored, the clear 1 - d.
  const uint32_t zfunc = c.reversed_camera ? 5 : 4;  // GREATER, LESSEQUAL
  MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
  dd.depthCompareFunction = c.reversed_camera ? MTLCompareFunctionLess : MTLCompareFunctionGreaterEqual;
  dd.depthWriteEnabled = YES;
  id<MTLDepthStencilState> ds = [g.dev newDepthStencilStateWithDescriptor:dd];
  const double clear = c.reversed_camera ? 1.0 : 0.0;

  const V3 eye{-4, 9, -8}, at{1, 0.5, 1};
  double vp[16], inv[16];
  Camera(eye, at, 1.0, double(kW) / kH, 0.5, 3000.0, vp, c.reversed_camera);
  Invert4(vp, inv);
  // The ground and the box, world space (the identity world matrix).
  std::vector<V3> tri = {{-60, 0, -60}, {-60, 0, 60}, {60, 0, -60}, {60, 0, -60}, {-60, 0, 60}, {60, 0, 60}};
  const V3 b[8] = {{0, 0, 0}, {kBox, 0, 0}, {0, kBox, 0}, {kBox, kBox, 0},
                   {0, 0, kBox}, {kBox, 0, kBox}, {0, kBox, kBox}, {kBox, kBox, kBox}};
  const int faces[6][4] = {{1, 3, 5, 7}, {0, 4, 2, 6}, {2, 6, 3, 7}, {0, 1, 4, 5}, {4, 5, 6, 7}, {0, 2, 1, 3}};
  for (const auto& f : faces)
    for (int k : {0, 1, 2, 2, 1, 3}) tri.push_back(b[f[k]]);
  std::vector<float> verts;
  for (const V3& v : tri) verts.insert(verts.end(), {float(v.x), float(v.y), float(v.z), 1.0f});
  verts.insert(verts.end(), 4, 0.0f);  // (room for the shader's unaligned reads)
  float vc[12][4] = {};
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) vc[8 + r][k] = float(vp[r * 4 + k]);
  VsState vst{};
  for (auto& in : vst.in) in[2] = -1;
  for (size_t i = 0; i < inputs.size(); i++) {
    vst.in[i][0] = 0;
    vst.in[i][1] = 16;
    vst.in[i][2] = 3;  // FLOAT4
  }
  // As metal_device.mm's PixelState and ClipState: a reversed buffer, the jitter.
  vst.pixel[0] = 1.0f / kW;
  vst.pixel[1] = 1.0f / kH;
  vst.pixel[2] = 2.0f * c.jitter[0] / kW;
  vst.pixel[3] = 2.0f * c.jitter[1] / kH;
  vst.clip_mask[1] = 1;
  PsState pst{};
  pst.alpha[0] = 8;
  pst.alpha[3] = 1;
  pst.vscale[0] = pst.vscale[1] = 1;
  const float color[2][4] = {{1, 1, 1, 1}, {}};

  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                width:kW
                                                                               height:kH
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [g.dev newTextureWithDescriptor:td];
  td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  id<MTLTexture> depth = [g.dev newTextureWithDescriptor:td];

  // The scene: both meshes the main view's (identity world).
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
  s->SetAllocator(ArenaAlloc, &g.arena);
  g.arena.used = 0;
  const V3 gp[4] = {{-60, 0, -60}, {-60, 0, 60}, {60, 0, -60}, {60, 0, 60}};
  Quad gq = MakeQuad(gp, 1, false);
  struct Mesh {
    std::vector<float> pos;
    RtGeometry g;
    RtMeshKey key;
  } box;
  for (size_t i = 6; i < tri.size(); i++) box.pos.insert(box.pos.end(), {float(tri[i].x), float(tri[i].y), float(tri[i].z)});
  box.g.positions = reinterpret_cast<const uint8_t*>(box.pos.data());
  box.g.size = box.pos.size() * 4;
  box.g.stride = 12;
  box.g.type = 2;
  box.g.prim = 4;
  box.g.primitives = 12;
  box.key.vertex_version = 3 * 16 + 1;
  box.key.vertex_buffer = 3;
  box.key.prim = 4;
  box.key.primitives = 12;
  box.key.stride = 12;
  box.key.type = 2;
  Mat w, mw;
  Translate(w, {0, 0, 0});
  Wvp(w, vp, mw);
  s->BeginFrame(1, 0);
  s->Add(gq.key, gq.g, w, mw, 1, zfunc);
  s->Add(box.key, box.g, w, mw, 1, zfunc);
  const RtFrame* f = s->Snapshot();
  RtShadowParams p;
  const V3 l = Unit({-0.5, 0.8, 0.35});
  const float lf[3] = {float(l.x), float(l.y), float(l.z)};
  if (!f || !f->tlas || f->depth_reversed != c.reversed_camera ||
      !RtShadowParamsFor(*f, kW, kH, true, true, c.jitter, lf, &p))
    return Check(false, "box: the scene's camera and parameters");

  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = target;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
  rp.depthAttachment.texture = depth;
  rp.depthAttachment.loadAction = MTLLoadActionClear;
  rp.depthAttachment.clearDepth = clear;
  rp.depthAttachment.storeAction = MTLStoreActionStore;
  rp.stencilAttachment.texture = depth;
  rp.stencilAttachment.loadAction = MTLLoadActionClear;
  rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:pipe];
  [e setDepthStencilState:ds];
  [e setFrontFacingWinding:MTLWindingClockwise];
  [e setCullMode:MTLCullModeNone];
  const DepthRange zr = ReversedViewportDepth(0.0f, 1.0f);
  [e setViewport:(MTLViewport){0, 0, double(kW), double(kH), double(zr.min_z), double(zr.max_z)}];
  [e setVertexBytes:vc length:sizeof vc atIndex:0];
  [e setVertexBytes:&vst length:sizeof vst atIndex:2];
  id<MTLBuffer> vb = [g.dev newBufferWithBytes:verts.data() length:verts.size() * 4 options:MTLResourceStorageModeShared];
  [e setVertexBuffer:vb offset:0 atIndex:3];
  [e setFragmentBytes:color length:sizeof color atIndex:0];
  [e setFragmentBytes:&pst length:sizeof pst atIndex:2];
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:NSUInteger(tri.size())];
  [e endEncoding];
  id<MTLTexture> mask = g.masker->NewMask(kW, kH);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 8);
  id<MTLBuffer> zout = [g.dev newBufferWithLength:kW * kH * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> mout = [g.dev newBufferWithLength:kW * kH options:MTLResourceStorageModeShared];
  s->EncodeFrame(cb, f);
  g.masker->Encode(cb, f, depth, mask, p, counts, 0);
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(kW, kH, 1) toBuffer:zout destinationOffset:0 destinationBytesPerRow:kW * 4
      destinationBytesPerImage:kW * kH * 4 options:MTLBlitOptionDepthFromDepthStencil];
  [bl copyFromTexture:mask sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(kW, kH, 1) toBuffer:mout destinationOffset:0 destinationBytesPerRow:kW
      destinationBytesPerImage:kW * kH];
  [bl endEncoding];
  if (!g.Commit(cb)) return Check(false, "box: drawn and traced");
  const RtFrame* end = s->EndFrame(true);
  id<MTLCommandBuffer> cb2 = [g.queue commandBuffer];
  s->EncodeFrame(cb2, end);
  g.Commit(cb2);

  // Per pixel, the camera's ray (unjittered, at D3D9's centre less the
  // jitter): its face and point.
  std::vector<int> face(size_t(kW) * kH);
  std::vector<V3> pt(face.size());
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const double x = 2.0 * (i - c.jitter[0]) / kW - 1.0, y = 1.0 - 2.0 * (j - c.jitter[1]) / kH;
      double hv[4];
      for (int k = 0; k < 4; k++) hv[k] = x * inv[k] + y * inv[4 + k] + 0.5 * inv[8 + k] + inv[12 + k];
      const V3 q{hv[0] / hv[3], hv[1] / hv[3], hv[2] / hv[3]};
      const V3 d = Unit(q - eye);
      double t;
      const size_t at = size_t(j) * kW + i;
      face[at] = Hit(eye, d, &t);
      if (face[at]) pt[at] = eye + d * t;
    }
  const auto* z = static_cast<const float*>(zout.contents);
  const auto* m = static_cast<const uint8_t*>(mout.contents);
  int judged = 0, wrong = 0, misplaced = 0, placed = 0, sky_wrong = 0, edges = 0;
  int ground_lit = 0, ground_dark = 0, box_lit = 0, box_dark = 0;
  double worst = 0;
  for (uint32_t j = 1; j + 1 < kH; j++)
    for (uint32_t i = 1; i + 1 < kW; i++) {
      const size_t at = size_t(j) * kW + i;
      const bool lit = m[at] > 127;
      if (!face[at]) {
        if (!lit) sky_wrong++;
        continue;
      }
      const V3 pw = pt[at];
      const double dist = Len(pw - eye);
      // Its neighbours: on its face, or across a depth edge (far from it);
      // a crease (another face close by) is not judged.
      bool crease = false, across = false;
      double foot = 0;
      for (size_t n : {at - 1, at + 1, at - kW, at + kW}) {
        if (face[n] == face[at]) {
          foot = std::max(foot, Len(pt[n] - pw));
          continue;
        }
        if (!face[n] || Len(pt[n] - pw) > 0.05 * dist)
          across = true;
        else
          crease = true;
      }
      if (crease || foot == 0) continue;
      // The point the depth says, against the ray's.
      double r[3];
      RtShadowPoint(p, i, j, z[at], r);
      const double e = Len(V3{r[0], r[1], r[2]} - pw);
      if (!across) {
        placed++;
        worst = std::max(worst, e / dist);
        if (e > 5e-4 * dist) misplaced++;
      }
      // Lit: a box face toward the light; the ground where the ray to the
      // light misses the box (unless near its shadow's edge).
      bool want;
      if (face[at] >= 2) {
        const double nl = Dot(kFaceNormal[face[at]], l);
        want = nl > 0;
        (want ? box_lit : box_dark)++;
      } else {
        const double off = dist * (p.offset_pixels * p.pixel + p.offset_view) + Len(pw) * p.offset_world;
        const double margin = 2.0 * foot + 2.0 * off / l.y + 1e-3;
        int votes = 0;
        for (const V3 s2 : {V3{0, 0, 0}, V3{margin, 0, 0}, V3{-margin, 0, 0}, V3{0, 0, margin}, V3{0, 0, -margin}}) {
          double t;
          const int hf = Hit(pw + s2 + V3{0, 1e-6, 0}, l, &t);
          votes += hf >= 2 ? 1 : 0;
        }
        if (votes != 0 && votes != 5) continue;
        want = votes == 0;
        (want ? ground_lit : ground_dark)++;
      }
      judged++;
      if (across) edges++;
      if (lit != want) wrong++;
    }
  const uint32_t rays = static_cast<const uint32_t*>(counts.contents)[0];
  std::printf("  %s: %d pixels judged (ground %d lit, %d in shadow; box %d lit, %d dark; %d across a depth edge), %d "
              "wrong, %d sky not lit; %d points %d off (worst %.1e of the distance); %u rays\n",
              c.what, judged, ground_lit, ground_dark, box_lit, box_dark, edges, wrong, sky_wrong, placed, misplaced,
              worst, rays);
  return Check(wrong == 0 && sky_wrong == 0 && misplaced == 0 && placed > 10000 && ground_lit > 1000 &&
                   ground_dark > 200 && box_lit > 200 && box_dark > 200 && edges > 50 && rays > 0,
               (std::string(c.what) + ": the backend's rasterized depth back to the camera ray's points; lit and dark "
                                      "box sides, its shadow on the ground, across depth edges")
                   .c_str());
}

int BoxChecks(Gpu& g) {
  int failures = 0;
  failures += BoxRun(g, {"box, D3D's depth", false, {0, 0}});
  failures += BoxRun(g, {"box, a reversing camera, jittered", true, {0.3f, -0.25f}});
  return failures;
}

// ---- a street, drawn through the backend's prelude, over the jitter's phases

// A street (y up): the ground, a building's side, a car-sized box and a
// pillar (axis-aligned boxes, closed), the camera at head height looking
// along it (points 5 to 120 away), all `o` from the origin (the meshes
// local, placed by a world matrix as a game places them). The sun at
// `elevation_deg`, ahead of the camera on its left: the building's side
// toward the street lit, the boxes' sides toward the camera dark, their
// shadows toward the camera, across the street and onto the building. Depth drawn through the prelude, stored
// reversed (Depth32Float_Stencil8), for each of `phases` jitter phases (the
// backend's Halton (2, 3) sequence). Per phase and pixel the truth: the
// camera ray through the jittered sample (where the rasterizer took the
// depth) and whether its point sees the sun. Judged:
// - lit pixels in shadow (a surface shadowing itself) away from the true
//   shadow's edges (no shadowed truth within 2 pixels): none;
// - shadowed pixels lit away from the edges (2 pixels): none (the shadows
//   are still there, hard, where they belong);
// - the mask's shadowed pixels the same in every phase but within a pixel
//   of a true shadow edge (in any phase): no shimmer under the jitter.
// The pixels a crease runs through are not judged: the rays start along
// the normal the depth makes, both surfaces' planes predict their depth,
// and the depth cannot tell which they are on. And the camera turned all
// the way round over normal-mapped normals with the sun low: lit surfaces
// lit at every heading.
struct StreetBox {
  V3 lo, hi;
};
const StreetBox kStreet[] = {
    {{8, 0, 10}, {14, 12, 120}},      // a building's side, along the street (its face x = 8)
    {{-3, 0, 16}, {-1, 1.6, 20.5}},   // a car
    {{-6, 0, 40}, {-5, 5, 41}},       // a pillar
    {{-1, 0, 70}, {1.5, 2.5, 73}},    // a box far down the street
    {{5, 3.5, 25}, {8, 3.8, 33}},     // an awning on the building
};
constexpr int kStreetBoxes = sizeof(kStreet) / sizeof(kStreet[0]);
constexpr double kStreetGround[4] = {-160, 160, -160, 160};  // x0, x1, z0, z1 (y = 0)

// Where the ray o + t d (t > 0) first meets the street (local coordinates):
// t, the face (0 none, 1 the ground, 2 + 6 box + k box `box`'s face k: +x,
// -x, +y, -y, +z, -z) and its normal. `skip`: a face not to hit (the
// point's own).
int StreetHit(V3 o, V3 d, double* t_out, V3* n_out, int skip = -1) {
  int face = 0;
  double best = 1e30;
  if (d.y < 0 && skip != 1) {
    const double t = -o.y / d.y;
    const V3 p = o + d * t;
    if (t > 0 && p.x >= kStreetGround[0] && p.x <= kStreetGround[1] && p.z >= kStreetGround[2] &&
        p.z <= kStreetGround[3])
      best = t, face = 1, *n_out = {0, 1, 0};
  }
  for (int b = 0; b < kStreetBoxes; b++) {
    const double lo[3] = {kStreet[b].lo.x, kStreet[b].lo.y, kStreet[b].lo.z};
    const double hi[3] = {kStreet[b].hi.x, kStreet[b].hi.y, kStreet[b].hi.z};
    const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
    double t0 = -1e30, t1 = 1e30;
    int enter = -1;
    bool miss = false;
    for (int a = 0; a < 3 && !miss; a++) {
      if (std::fabs(dd[a]) < 1e-15) {
        if (oo[a] < lo[a] || oo[a] > hi[a]) miss = true;
        continue;
      }
      double ta = (lo[a] - oo[a]) / dd[a], tb = (hi[a] - oo[a]) / dd[a];
      int fa = a * 2 + 1, fb = a * 2;  // entering at lo: the -axis face
      if (ta > tb) std::swap(ta, tb), std::swap(fa, fb);
      if (ta > t0) t0 = ta, enter = fa;
      t1 = std::min(t1, tb);
      if (t0 > t1) miss = true;
    }
    if (miss || enter < 0 || !(t0 > 0) || t0 >= best || 2 + 6 * b + enter == skip) continue;
    best = t0, face = 2 + 6 * b + enter;
    const int a = enter / 2;
    V3 n{0, 0, 0};
    (a == 0 ? n.x : a == 1 ? n.y : n.z) = (enter & 1) ? -1 : 1;
    *n_out = n;
  }
  *t_out = best;
  return face;
}

double StreetHalton(int i, int b) {
  double f = 1, r = 0;
  for (; i > 0; i /= b) {
    f /= b;
    r += f * (i % b);
  }
  return r;
}

struct StreetResult {
  int self_shadowed = 0, lost = 0, unstable = 0, lit = 0, dark = 0, judged = 0, sky_wrong = 0;
  int self_at_face_edge = 0;      // of self_shadowed: another face (or the sky) within a pixel
  int unstable_at_face_edge = 0;  // of unstable: likewise (in any phase)
  double blocked = 0, truth_dark = 0;  // fractions of the traced rays and of the pixels not sky
  uint64_t rays = 0, map_lit = 0;  // texels traced to the light, and lit by the cascade map (a short ray)
};

// The surfaces' normals the mask is given: none (made from the depth), a
// G-buffer's (each pixel's face's, xyz as unorm), or a G-buffer's as a
// normal map tilts them: up to 25 degrees, which way and how much fixed on
// the surface (by its point, in 5 cm cells), so that some face away from
// the camera or the sun.
enum class StreetNormals { Depth, GBuffer, Bumped };

uint32_t StreetHash(int64_t a, int64_t b, int64_t c) {
  uint64_t h = uint64_t(a) * 0x9E3779B97F4A7C15ull ^ uint64_t(b) * 0xC2B2AE3D27D4EB4Full ^ uint64_t(c) * 0x165667B19E3779F9ull;
  h ^= h >> 31;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 29;
  return uint32_t(h);
}

// `yaw_deg` turns the camera about the vertical through its eye.
StreetResult StreetRun(Gpu& g, double elevation_deg, V3 o, int phases, StreetNormals normals, bool mask_jitter,
                       bool* ok, double yaw_deg = 0, uint32_t step = 2, int judged_edge = 2, bool cascade = false,
                       double cascade_offset = 0, double shadow_gap = 0) {
  StreetResult res;
  *ok = false;
  constexpr uint32_t W = 1920, H = 1080;  // (about a pixel's angle of a game's)
  std::vector<hlsl::Semantic> inputs;
  id<MTLFunction> vs = Compile(g, WvpVs(), &inputs), ps = Compile(g, ConstPs(), &inputs);
  if (!vs || !ps) return res;
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = vs;
  pd.fragmentFunction = ps;
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  NSError* err = nil;
  id<MTLRenderPipelineState> pipe = [g.dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pipe) return res;
  // D3D's depth (LESSEQUAL), stored reversed: the test mirrored, cleared to 0.
  MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
  dd.depthCompareFunction = MTLCompareFunctionGreaterEqual;
  dd.depthWriteEnabled = YES;
  id<MTLDepthStencilState> ds = [g.dev newDepthStencilStateWithDescriptor:dd];

  // The camera at head height (local coordinates; world = local + o).
  const V3 eye_l{0.5, 2.2, 0};
  const double yaw = yaw_deg * M_PI / 180.0;
  const V3 ahead{1.0, -1.2, 30};
  const V3 at_l = eye_l + V3{ahead.x * std::cos(yaw) + ahead.z * std::sin(yaw), ahead.y,
                             -ahead.x * std::sin(yaw) + ahead.z * std::cos(yaw)};
  double vp[16], inv[16];
  Camera(eye_l + o, at_l + o, 0.9, double(W) / H, 0.3, 3000.0, vp);
  Invert4(vp, inv);
  const double e = elevation_deg * M_PI / 180.0;
  const V3 hz = Unit({-0.6, 0, 1});
  const V3 l = Unit({std::cos(e) * hz.x, std::sin(e), std::cos(e) * hz.z});

  // The meshes (local): the ground's two triangles and the boxes'.
  std::vector<V3> tri = {{kStreetGround[0], 0, kStreetGround[2]}, {kStreetGround[0], 0, kStreetGround[3]},
                         {kStreetGround[1], 0, kStreetGround[2]}, {kStreetGround[1], 0, kStreetGround[2]},
                         {kStreetGround[0], 0, kStreetGround[3]}, {kStreetGround[1], 0, kStreetGround[3]}};
  const int faces[6][4] = {{1, 3, 5, 7}, {0, 4, 2, 6}, {2, 6, 3, 7}, {0, 1, 4, 5}, {4, 5, 6, 7}, {0, 2, 1, 3}};
  for (const StreetBox& sb : kStreet) {
    V3 c[8];
    for (int k = 0; k < 8; k++)
      c[k] = {(k & 1) ? sb.hi.x : sb.lo.x, (k & 2) ? sb.hi.y : sb.lo.y, (k & 4) ? sb.hi.z : sb.lo.z};
    for (const auto& f : faces)
      for (int k : {0, 1, 2, 2, 1, 3}) tri.push_back(c[f[k]]);
  }
  std::vector<float> verts, pos;
  for (const V3& v : tri) {
    verts.insert(verts.end(), {float(v.x), float(v.y), float(v.z), 1.0f});
    pos.insert(pos.end(), {float(v.x), float(v.y), float(v.z)});
  }
  verts.insert(verts.end(), 4, 0.0f);
  Mat w, mw;
  Translate(w, o);
  Wvp(w, vp, mw);
  float vc[12][4] = {};
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) vc[8 + r][k] = mw[r][k];
  id<MTLBuffer> vb = [g.dev newBufferWithBytes:verts.data() length:verts.size() * 4 options:MTLResourceStorageModeShared];

  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
  s->SetAllocator(ArenaAlloc, &g.arena);
  g.arena.used = 0;
  RtGeometry geo;
  geo.positions = reinterpret_cast<const uint8_t*>(pos.data());
  geo.size = pos.size() * 4;
  geo.stride = 12;
  geo.type = 2;
  geo.prim = 4;
  geo.primitives = uint32_t(tri.size() / 3);
  RtMeshKey key;
  key.vertex_version = 5 * 16 + 1;
  key.vertex_buffer = 5;
  key.prim = 4;
  key.primitives = geo.primitives;
  key.stride = 12;
  key.type = 2;
  s->BeginFrame(1, 0);
  s->Add(key, geo, w, mw, 1, 4);
  const RtFrame* f = s->Snapshot();
  if (!f || !f->tlas) return res;
  {
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    s->EncodeFrame(cb, f);
    if (!g.Commit(cb)) return res;
  }

  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:W
                                                                               height:H
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [g.dev newTextureWithDescriptor:td];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> gbuf = [g.dev newTextureWithDescriptor:td];  // the G-buffer's normals
  std::vector<uint32_t> gtexels(size_t(W) * H);
  td.storageMode = MTLStorageModePrivate;
  td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  id<MTLTexture> depth = [g.dev newTextureWithDescriptor:td];
  id<MTLTexture> mask = g.masker->NewMask(W, H);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  id<MTLBuffer> mout = [g.dev newBufferWithLength:W * H options:MTLResourceStorageModeShared];

  // A game's cascaded shadow map (`cascade`), made on the CPU as a game's
  // shadow pass draws one: one square cascade, the left of a 4096 x 2048
  // atlas, 200 x 200 around the camera (texels of 0.1), each texel the
  // street's surface nearest the light (z = its distance along it / 1000 +
  // 0.5: larger nearer the light; -1 where there is none), the lookup's
  // registers as such a game's shader has them.
  RtShadowCascade cas{};
  id<MTLTexture> cmap = nil;
  if (cascade) {
    constexpr uint32_t CW = 4096, CH = 2048;
    const double R = 100, Dz = 1000;
    const V3 u = Unit(Cross(V3{0, 1, 0}, l)), v = Cross(l, u), cw = eye_l + o;
    std::vector<float> map(size_t(CW) * CH, -1.0f);
    for (uint32_t j = 0; j < CH; j++)
      for (uint32_t i = 0; i < CH; i++) {
        const double x = 2.0 * (i + 0.5) / CH - 1.0, y = 1.0 - 2.0 * (j + 0.5) / CH;
        const V3 from = eye_l + u * (x * R) + v * (y * R) + l * 500.0;
        double t;
        V3 nrm;
        if (StreetHit(from, l * -1.0, &t, &nrm))
          map[size_t(j) * CW + i] = float(Dot(from - l * t - eye_l, l) / Dz + 0.5 + cascade_offset);
      }
    MTLTextureDescriptor* cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                  width:CW
                                                                                 height:CH
                                                                              mipmapped:NO];
    cd.storageMode = MTLStorageModeShared;
    cd.usage = MTLTextureUsageShaderRead;
    cmap = [g.dev newTextureWithDescriptor:cd];
    [cmap replaceRegion:MTLRegionMake2D(0, 0, CW, CH) mipmapLevel:0 withBytes:map.data() bytesPerRow:CW * 4];
    const double uc[3] = {u.x, u.y, u.z}, vc3[3] = {v.x, v.y, v.z}, lc[3] = {l.x, l.y, l.z};
    for (int k = 0; k < 3; k++) {
      cas.matrix[k][0] = float(uc[k] / R);
      cas.matrix[k][1] = float(vc3[k] / R);
      cas.matrix[k][3] = float(lc[k] / Dz);
    }
    cas.matrix[3][0] = float(-Dot(cw, u) / R);
    cas.matrix[3][1] = float(-Dot(cw, v) / R);
    cas.matrix[3][3] = float(-Dot(cw, l) / Dz + 0.5);
    // uv = (x / 4 + 1 / 4, 1 / 2 - y / 2): (1, s) . select (one cascade:
    // the view depth's bounds never passed).
    cas.select[0][0] = 0.25f;
    cas.select[1][0] = -0.5f;
    cas.select[2][0] = 0.25f;
    cas.select[3][0] = 0.5f;
    cas.splits[0] = cas.splits[1] = cas.splits[2] = 1.0f;
    cas.range = float(R * 0.95);
    cas.max_texel = 1.0f;
    cas.shadow_gap = float(shadow_gap);
    cas.on = 1;
  }

  const size_t n = size_t(W) * H;
  // Per phase: the mask (1 lit) and the truth (0 sky, 1 lit, 2 in shadow).
  std::vector<std::vector<uint8_t>> masks(phases, std::vector<uint8_t>(n)), truth(phases, std::vector<uint8_t>(n)),
      faceof(phases, std::vector<uint8_t>(n));
  uint64_t rays = 0, hits = 0, not_sky = 0, truth_dark = 0;
  for (int ph = 0; ph < phases; ph++) {
    const float jitter[2] = {float(StreetHalton(ph + 1, 2) - 0.5), float(StreetHalton(ph + 1, 3) - 0.5)};
    VsState vst{};
    for (auto& in : vst.in) in[2] = -1;
    for (size_t i = 0; i < inputs.size(); i++) {
      vst.in[i][0] = 0;
      vst.in[i][1] = 16;
      vst.in[i][2] = 3;  // FLOAT4
    }
    vst.pixel[0] = 1.0f / W;
    vst.pixel[1] = 1.0f / H;
    vst.pixel[2] = 2.0f * jitter[0] / W;
    vst.pixel[3] = 2.0f * jitter[1] / H;
    vst.clip_mask[1] = 1;
    PsState pst{};
    pst.alpha[0] = 8;
    pst.alpha[3] = 1;
    pst.vscale[0] = pst.vscale[1] = 1;
    // The truth: the camera's ray through the jittered sample (unjittered
    // camera, D3D9's centre less the jitter), its point and the sun.
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        const double x = 2.0 * (i - jitter[0]) / W - 1.0, y = 1.0 - 2.0 * (j - jitter[1]) / H;
        double hv[4];
        for (int k = 0; k < 4; k++) hv[k] = x * inv[k] + y * inv[4 + k] + 0.5 * inv[8 + k] + inv[12 + k];
        const V3 q = V3{hv[0] / hv[3], hv[1] / hv[3], hv[2] / hv[3]} - o;
        const V3 d = Unit(q - eye_l);
        double t;
        V3 nrm;
        const int face = StreetHit(eye_l, d, &t, &nrm);
        uint8_t v = 0;
        if (face) {
          const V3 pt = eye_l + d * t;
          double t2;
          V3 n2;
          v = Dot(nrm, l) <= 0 || StreetHit(pt, l, &t2, &n2, face) ? 2 : 1;
          not_sky++;
          truth_dark += v == 2;
        }
        truth[ph][size_t(j) * W + i] = v;
        faceof[ph][size_t(j) * W + i] = uint8_t(face);
        // The G-buffer: the face's normal, tilted where a normal map is.
        V3 gn = face ? nrm : V3{0, 0, 0};
        if (face && normals == StreetNormals::Bumped) {
          const V3 pt = eye_l + d * t;
          const uint32_t hh = StreetHash(int64_t(std::floor(pt.x * 20)), int64_t(std::floor(pt.y * 20)),
                                         int64_t(std::floor(pt.z * 20)) * 64 + face);
          const double a = 2 * M_PI * (hh & 0xFFFF) / 65536.0, tilt = 25.0 * M_PI / 180.0 * (hh >> 16) / 65536.0;
          const V3 t1 = Unit(Cross(nrm, std::fabs(nrm.y) > 0.5 ? V3{1, 0, 0} : V3{0, 1, 0})), t2 = Cross(nrm, t1);
          gn = Unit(nrm * std::cos(tilt) + (t1 * std::cos(a) + t2 * std::sin(a)) * std::sin(tilt));
        }
        auto unorm = [](double c) { return uint32_t(std::lround(std::clamp(0.5 * c + 0.5, 0.0, 1.0) * 255.0)); };
        gtexels[size_t(j) * W + i] = unorm(gn.x) | unorm(gn.y) << 8 | unorm(gn.z) << 16 | 0xFFu << 24;
      }
    [gbuf replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:gtexels.data() bytesPerRow:W * 4];
    RtShadowParams p;
    const float lf[3] = {float(l.x), float(l.y), float(l.z)};
    if (!RtShadowParamsFor(*f, W, H, true, true, mask_jitter ? jitter : nullptr, lf, &p)) return res;
    std::memset(counts.contents, 0, 16);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    rp.depthAttachment.texture = depth;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 0.0;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.stencilAttachment.texture = depth;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
    [re setRenderPipelineState:pipe];
    [re setDepthStencilState:ds];
    [re setFrontFacingWinding:MTLWindingClockwise];
    [re setCullMode:MTLCullModeNone];
    const DepthRange zr = ReversedViewportDepth(0.0f, 1.0f);
    [re setViewport:(MTLViewport){0, 0, double(W), double(H), double(zr.min_z), double(zr.max_z)}];
    [re setVertexBytes:vc length:sizeof vc atIndex:0];
    [re setVertexBytes:&vst length:sizeof vst atIndex:2];
    [re setVertexBuffer:vb offset:0 atIndex:3];
    [re setFragmentBytes:&pst length:sizeof pst atIndex:2];
    const float color[2][4] = {{1, 1, 1, 1}, {}};
    [re setFragmentBytes:color length:sizeof color atIndex:0];
    [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:NSUInteger(tri.size())];
    [re endEncoding];
    g.masker->Encode(cb, f, depth, mask, p, counts, 0, nil, nullptr,
                     normals == StreetNormals::Depth ? nil : gbuf, step, cascade ? &cas : nullptr, cmap);
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:mask sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(W, H, 1) toBuffer:mout destinationOffset:0 destinationBytesPerRow:W
        destinationBytesPerImage:W * H];
    [bl endEncoding];
    if (!g.Commit(cb)) return res;
    rays += static_cast<const uint32_t*>(counts.contents)[0];
    res.map_lit += static_cast<const uint32_t*>(counts.contents)[3];
    hits += static_cast<const uint32_t*>(counts.contents)[1];
    const auto* m = static_cast<const uint8_t*>(mout.contents);
    for (size_t k = 0; k < n; k++) masks[ph][k] = m[k] > 127 ? 1 : 0;
  }
  {
    const RtFrame* end = s->EndFrame(true);
    id<MTLCommandBuffer> cb2 = [g.queue commandBuffer];
    s->EncodeFrame(cb2, end);
    g.Commit(cb2);
  }
  // Within r pixels of a true shadow edge in phase ph: the truth's lit and
  // shadowed both there (the sky counts as lit).
  auto near_edge = [&](int ph, uint32_t i, uint32_t j, int r) {
    bool any_lit = false, any_dark = false;
    for (int y = int(j) - r; y <= int(j) + r; y++)
      for (int x = int(i) - r; x <= int(i) + r; x++) {
        if (x < 0 || y < 0 || x >= int(W) || y >= int(H)) continue;
        (truth[ph][size_t(y) * W + x] == 2 ? any_dark : any_lit) = true;
      }
    return any_lit && any_dark;
  };
  for (int ph = 0; ph < phases; ph++)
    for (uint32_t j = 2; j + 2 < H; j++)
      for (uint32_t i = 2; i + 2 < W; i++) {
        const size_t at = size_t(j) * W + i;
        const uint8_t t = truth[ph][at];
        const bool lit = masks[ph][at];
        if (!t) {
          res.sky_wrong += !lit;
          continue;
        }
        if (near_edge(ph, i, j, judged_edge)) continue;
        res.judged++;
        (t == 1 ? res.lit : res.dark)++;
        if (t == 1 && !lit) {
          res.self_shadowed++;
          bool other = false;
          for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) other = other || faceof[ph][at + y * int(W) + x] != faceof[ph][at];
          res.self_at_face_edge += other;
        }
        if (t == 2 && lit) res.lost++;
      }
  for (uint32_t j = 2; j + 2 < H; j++)
    for (uint32_t i = 2; i + 2 < W; i++) {
      const size_t at = size_t(j) * W + i;
      bool edge = false, differs = false;
      for (int ph = 0; ph < phases; ph++) {
        edge = edge || near_edge(ph, i, j, 1);
        differs = differs || masks[ph][at] != masks[0][at];
      }
      if (differs && !edge) {
        res.unstable++;
        bool other = false;
        for (int ph = 0; ph < phases; ph++)
          for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) other = other || faceof[ph][at + y * int(W) + x] != faceof[0][at];
        res.unstable_at_face_edge += other;
      }
    }
  res.blocked = rays ? double(hits) / double(rays) : 0;
  res.rays = rays;
  res.truth_dark = not_sky ? double(truth_dark) / double(not_sky) : 0;
  *ok = rays > 0;
  return res;
}

int StreetChecks(Gpu& g) {
  int failures = 0;
  struct Run {
    StreetNormals normals;
    const char* name;
  };
  const Run runs[] = {{StreetNormals::GBuffer, "G-buffer normals"},
                      {StreetNormals::Depth, "normals from the depth"},
                      {StreetNormals::Bumped, "normal-mapped G-buffer normals"}};
  for (const Run& run : runs)
    for (const V3 o : {V3{0, 0, 0}, V3{2900, 25, 2700}})
      for (double elev : {60.0, 25.0, 8.0}) {
        bool ran;
        const StreetResult r = StreetRun(g, elev, o, 8, run.normals, true, &ran);
        char what[200];
        std::snprintf(what, sizeof what, "street, %s, sun at %.0f degrees, %.0f from the origin", run.name, elev,
                      Len(o));
        std::printf("  %s: %d pixel-phases judged (%d lit, %d in shadow): %d lit ones shadowed (%d next to another "
                    "face), %d shadowed ones lit, %d pixels differing between phases off the edges (%d next to "
                    "another face), %d sky not lit; blocked rays %.2f%% (true shadow %.2f%%)\n",
                    what, r.judged, r.lit, r.dark, r.self_shadowed, r.self_at_face_edge, r.lost, r.unstable,
                    r.unstable_at_face_edge, r.sky_wrong, 100.0 * r.blocked, 100.0 * r.truth_dark);
        const int self = r.self_shadowed - r.self_at_face_edge;
        const int unstable = r.unstable - r.unstable_at_face_edge;
        failures += Check(ran && self == 0 && r.lost == 0 && unstable == 0 && r.sky_wrong == 0 &&
                              r.lit > 1000000 && r.dark > 200000,
                          (std::string(what) + ": over 8 jitter phases no lit surface shadows itself, no shadow is "
                                               "lost, the shadowed pixels the same in every phase off the edges")
                              .c_str());
      }
  // The camera turned all the way round, the sun low (15 degrees), the
  // normals a normal map's: every heading the same lit surfaces lit.
  for (int yaw = 0; yaw < 360; yaw += 30) {
    bool ran;
    const StreetResult r = StreetRun(g, 15.0, {0, 0, 0}, 2, StreetNormals::Bumped, true, &ran, yaw);
    char what[160];
    std::snprintf(what, sizeof what, "street, normal-mapped G-buffer normals, sun at 15 degrees, heading %d", yaw);
    std::printf("  %s: %d lit ones shadowed (%d next to another face), %d shadowed ones lit; blocked rays %.2f%% "
                "(true shadow %.2f%%)\n",
                what, r.self_shadowed, r.self_at_face_edge, r.lost, 100.0 * r.blocked, 100.0 * r.truth_dark);
    failures += Check(ran && r.self_shadowed == r.self_at_face_edge && r.lost == 0 && r.lit > 1000000,
                      (std::string(what) + ": no lit surface shadows itself, no shadow lost").c_str());
  }
  // The point at the pixel's centre, not the jittered sample the depth was
  // drawn at: off the surface by up to half a pixel's footprint, shadowing
  // itself where that is below it, differently every phase.
  bool ran;
  const StreetResult r = StreetRun(g, 25.0, {0, 0, 0}, 8, StreetNormals::GBuffer, false, &ran);
  std::printf("  street, the points at the pixels' centres (the jitter left out), sun at 25 degrees: %d lit ones "
              "shadowed, %d pixels differing between phases off the edges; blocked rays %.2f%% (true shadow %.2f%%)\n",
              r.self_shadowed, r.unstable, 100.0 * r.blocked, 100.0 * r.truth_dark);
  failures += Check(ran && r.self_shadowed + r.unstable > 0,
                    "street: the points at the pixels' centres less the jitter (where the depth was drawn), not "
                    "at the centres (which shimmer)");
  return failures;
}

}  // namespace

// The far light's mask at half the resolution, upsampled by depth: the
// streets again, every phase: no lit surface shadowed, no shadow lost, the
// same pixels in every phase off the edges, at most a couple of pixels at
// the sky's edge; a quarter of the rays, blocked as often as the true
// shadow covers.
int HalfChecks(Gpu& g) {
  int failures = 0;
  for (const V3 o : {V3{0, 0, 0}, V3{2900, 25, 2700}})
    for (double elev : {60.0, 25.0, 8.0}) {
      bool ran;
      const StreetResult r = StreetRun(g, elev, o, 8, StreetNormals::GBuffer, true, &ran);
      char what[200];
      std::snprintf(what, sizeof what, "half-resolution mask, street, sun at %.0f degrees, %.0f from the origin", elev,
                    Len(o));
      std::printf("  %s: %d pixel-phases judged: %d lit ones shadowed (%d next to another face), %d shadowed ones "
                  "lit, %d pixels differing between phases off the edges (%d next to another face), %d sky not lit; "
                  "blocked rays %.2f%% (true shadow %.2f%%)\n",
                  what, r.judged, r.self_shadowed, r.self_at_face_edge, r.lost, r.unstable, r.unstable_at_face_edge,
                  r.sky_wrong, 100.0 * r.blocked, 100.0 * r.truth_dark);
      const int self = r.self_shadowed - r.self_at_face_edge;
      const int unstable = r.unstable - r.unstable_at_face_edge;
      failures += Check(ran && self == 0 && r.lost == 0 && unstable == 0 && r.sky_wrong <= 2 &&
                            r.lit > 1000000 && r.dark > 200000 && std::fabs(r.blocked - r.truth_dark) < 0.02,
                        (std::string(what) + ": no lit surface shadowed, no shadow lost, stable over 8 phases, the "
                                             "blocked rays the true shadow's share")
                            .c_str());
    }
  // A dim light's mask (kRtShadowDimStep: a sixteenth of the pixels
  // traced): every shadow still there and no lit surface shadowed farther
  // than its step from a true shadow's edge.
  for (double elev : {60.0, 25.0}) {
    bool ran;
    const StreetResult r = StreetRun(g, elev, {0, 0, 0}, 2, StreetNormals::GBuffer, true, &ran, 0,
                                     kRtShadowDimStep, int(kRtShadowDimStep));
    char what[200];
    std::snprintf(what, sizeof what, "a dim light's mask, street, sun at %.0f degrees", elev);
    std::printf("  %s: %d pixel-phases judged: %d lit ones shadowed (%d next to another face), %d shadowed ones "
                "lit, %d sky not lit; blocked rays %.2f%% (true shadow %.2f%%)\n",
                what, r.judged, r.self_shadowed, r.self_at_face_edge, r.lost, r.sky_wrong, 100.0 * r.blocked,
                100.0 * r.truth_dark);
    failures += Check(ran && r.self_shadowed == r.self_at_face_edge && r.lost == 0 && r.sky_wrong <= 4 &&
                          r.lit > 200000 && r.dark > 40000 && std::fabs(r.blocked - r.truth_dark) < 0.03,
                      (std::string(what) + ": no lit surface shadowed, no shadow lost off its edges, the blocked "
                                           "rays the true shadow's share")
                          .c_str());
  }
  return failures;
}

// The far light's mask with a game's cascaded shadow map (RtShadowCascade):
// the streets again, every phase as without it (no lit surface shadowed,
// no shadow lost, stable), most texels lit by the map and a short ray.
int HybridChecks(Gpu& g) {
  int failures = 0;
  for (const V3 o : {V3{0, 0, 0}, V3{2900, 25, 2700}})
    for (double elev : {60.0, 25.0}) {
      bool ran;
      const StreetResult r = StreetRun(g, elev, o, 8, StreetNormals::GBuffer, true, &ran, 0, 2, 2, true);
      char what[200];
      std::snprintf(what, sizeof what, "with a cascade map, street, sun at %.0f degrees, %.0f from the origin", elev,
                    Len(o));
      std::printf("  %s: %d pixel-phases judged: %d lit ones shadowed (%d next to another face), %d shadowed ones "
                  "lit, %d pixels differing between phases off the edges (%d next to another face), %d sky not lit; "
                  "%llu texels traced to the light, %llu lit by the map with a short ray\n",
                  what, r.judged, r.self_shadowed, r.self_at_face_edge, r.lost, r.unstable, r.unstable_at_face_edge,
                  r.sky_wrong, (unsigned long long)r.rays, (unsigned long long)r.map_lit);
      const int self = r.self_shadowed - r.self_at_face_edge;
      const int unstable = r.unstable - r.unstable_at_face_edge;
      failures += Check(ran && self == 0 && r.lost == 0 && unstable == 0 && r.sky_wrong <= 2 &&
                            r.lit > 1000000 && r.dark > 200000 && r.map_lit > r.rays,
                        (std::string(what) + ": no lit surface shadowed, no shadow lost, stable over 8 phases, "
                                             "most texels lit by the map")
                            .c_str());
    }
  // The map deciding shadows too where its casters are 3 or more above the
  // point along the light (the building, the pillar: the street's tall
  // casters): as before, and more texels decided by it.
  for (double elev : {60.0, 25.0}) {
    bool ran, ran0;
    const StreetResult r0 = StreetRun(g, elev, {0, 0, 0}, 8, StreetNormals::GBuffer, true, &ran0, 0, 2, 2, true);
    const StreetResult r = StreetRun(g, elev, {0, 0, 0}, 8, StreetNormals::GBuffer, true, &ran, 0, 2, 2, true, 0, 3.0);
    char what[200];
    std::snprintf(what, sizeof what, "with a cascade map deciding shadows, street, sun at %.0f degrees", elev);
    std::printf("  %s: %d lit ones shadowed (%d next to another face), %d shadowed ones lit, %d differing between "
                "phases off the edges; %llu traced, %llu decided by the map (%llu without shadows)\n",
                what, r.self_shadowed, r.self_at_face_edge, r.lost, r.unstable - r.unstable_at_face_edge,
                (unsigned long long)r.rays, (unsigned long long)r.map_lit, (unsigned long long)r0.map_lit);
    failures += Check(ran && ran0 && r.self_shadowed == r.self_at_face_edge && r.lost == 0 &&
                          r.unstable == r.unstable_at_face_edge && r.map_lit > r0.map_lit,
                      (std::string(what) + ": no lit surface shadowed, no shadow lost, stable, more texels decided "
                                           "by the map")
                          .c_str());
  }
  // A map whose depth is not the one the lookup expects (offset as far as
  // 50 along the light: every texel's caster nearer the light, or none
  // seen): nothing taken from it, the rays as without it.
  for (const double off : {0.05, -0.05}) {
    bool ran;
    const StreetResult r = StreetRun(g, 25.0, {0, 0, 0}, 2, StreetNormals::GBuffer, true, &ran, 0, 2, 2, true, off);
    char what[160];
    std::snprintf(what, sizeof what, "a cascade map %s the lookup's depth by 50", off > 0 ? "above" : "below");
    std::printf("  %s: %d lit ones shadowed, %d shadowed ones lit; %llu lit by the map\n", what, r.self_shadowed,
                r.lost, (unsigned long long)r.map_lit);
    failures += Check(ran && r.self_shadowed == r.self_at_face_edge && r.lost == 0 && r.map_lit == 0,
                      (std::string(what) + ": nothing taken from it, every texel traced").c_str());
  }
  return failures;
}

int main() {
  @autoreleasepool {
    int failures = CpuChecks();
    Gpu g;
    if (!g.dev || !RtSupported(g.dev)) {
      std::printf("skipped: %s\n", g.dev ? "the GPU does not trace rays in hardware (Apple9)" : "no Metal device");
      return failures ? 1 : 77;
    }
    if (!g.Init()) return Check(false, "shadow kernel made");
    // The far light's mask traced at every pixel (RECOMP_RT_SHADOW_FULL)
    // for the exact checks; at half the resolution (the default) after.
    setenv("RECOMP_RT_SHADOW_FULL", "1", 1);
    failures += SceneChecks(g);
    failures += CapturedChecks(g);
    failures += GpuChecks(g);
    failures += BoxChecks(g);
    failures += StreetChecks(g);
    unsetenv("RECOMP_RT_SHADOW_FULL");
    failures += HalfChecks(g);
    failures += HybridChecks(g);
    return failures ? 1 : 0;
  }
}
