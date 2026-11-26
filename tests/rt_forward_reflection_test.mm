// Ray-traced reflections in forward pixel shaders (gfx/rt_forward_reflection.h).
// On the CPU: the MSL patch on hand-written text (the scene's MSL put
// before it, the macro before the entry point, its four parameters added,
// an empty parameter list, a second patch a no-op) and refused (no main0,
// no stub, the stub only after main0, a fragment buffer of its own taken,
// main0's list not closed before a body), the text then untouched; why a
// draw traces nothing (RtForwardWhyOff: the draw's own camera where its
// matrices give it, whatever camera wrote the depth last; else the
// snapshot's target or the depth's camera); a draw's parameters
// (RtForwardParamsFor: the camera's position, the light's sign, the wrap
// clamped, the gap and offsets of the shadows). The patch on SPIRV-Cross's
// MSL of a translated ps_3_0 whose read of s3 became the call (hlsl::
// RayReflectionCall), the backend's prelude before it.
// On a GPU that traces rays from render pipelines (Apple9): the library as
// the backend makes it (RtForwardCompile: the plain, cached text patched
// with this build's scene MSL and left untouched; the plain text compiled
// where the patched one does not compile or the patch refuses it, or
// without the option); that shader
// drawn as a full-screen quad (the origin a constant input, the direction
// from the screen's position) against a scene of one textured quad, every
// judged pixel's colour against an analytic trace (a 1 and the albedo times
// the light (wrapped diffuse, a shadow ray) plus the ambient, times the
// scale where the ray hits the quad; 0 where it misses); the quad culling
// either winding (one side stops the rays with a 0, the other is lit);
// with the parameters' "on" 0, nothing anywhere, also with the null scene
// bound (RtForwardNullScene, what such draws bind without a snapshot).
// Shadow calls (hlsl::RayShadowCall): the patch on text with the shadow
// stub alone (its macro only) and with both; the parameters' shadow bits
// (traced, the light's sign) and the minimum slope; a translated ps_3_0
// whose read of s15 became the call drawn over points behind the quad
// (from the light): 0 where the quad blocks the light, 1 elsewhere; all 1
// with the shadow bit clear.
//
//   rt_forward_reflection_test   (the GPU part skipped, exit 77, without such a GPU)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/metal_prelude.h"
#include "runtime/gfx/rt_forward_reflection.h"
#include "runtime/gfx/rt_reflection.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"
#include "runtime/gfx/rt_support.h"
#include "runtime/gfx/shader_compile.h"
#include "runtime/hlsl/hlsl.h"

namespace {

using namespace recomp::gfx;
namespace hlsl = recomp::hlsl;

int failures = 0;

void Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) failures++;
}

bool Has(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

// ---- the patch on the CPU

constexpr const char* kShape = R"(#include <metal_stdlib>
using namespace metal;
struct main0_out { float4 r_c0 [[color(0)]]; };
static inline __attribute__((always_inline))
float4 r_rt_reflect(thread const float4& pixel, thread const float4& origin)
{
    return float4(0.0);
}

fragment main0_out main0(main0_in in [[stage_in]], constant r_float_regs& _29 [[buffer(0)]], float4 gl_FragCoord [[position]])
{
    main0_out out = {};
    float4 param = gl_FragCoord;
    out.r_c0 = r_rt_reflect(param, in.r_var1);
    return out;
}
)";

std::string Replace(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

void PatchChecks() {
  std::string m = kShape;
  const bool ok = RtForwardReflectionPatch(&m);
  const size_t def = m.find("#define r_rt_reflect(...) r_rt_reflect_rt(r_rt_scene, r_rt_mat, r_rt_info, r_rt_params, "
                            "__VA_ARGS__)\n"),
               entry = m.find("\nfragment main0_out main0(");
  Check(ok && RtForwardReflectionPatched(m) && Has(m, "#include <metal_raytracing>") && Has(m, "RtReflectRay(") &&
            Has(m, "float4 r_rt_reflect_rt(") && def != std::string::npos && entry != std::string::npos &&
            def + std::strlen("#define r_rt_reflect(...) r_rt_reflect_rt(r_rt_scene, r_rt_mat, r_rt_info, "
                              "r_rt_params, __VA_ARGS__)") == entry &&
            Has(m, "float4 gl_FragCoord [[position]], instance_acceleration_structure r_rt_scene [[buffer(3)]], "
                   "const device RtMaterial* r_rt_mat [[buffer(4)]], const device RtInfo* r_rt_info "
                   "[[buffer(5)]], constant RtForwardParams& r_rt_params [[buffer(6)]])\n{"),
        "patch: the scene's MSL first, the macro just before main0, its four parameters last");
  std::string again = m;
  Check(RtForwardReflectionPatch(&again) && again == m, "patch: a patched text left as it is");
  std::string empty = Replace(kShape, "main0(main0_in in [[stage_in]], constant r_float_regs& _29 [[buffer(0)]], "
                                      "float4 gl_FragCoord [[position]])",
                              "main0( )");
  Check(RtForwardReflectionPatch(&empty) && Has(empty, "main0( instance_acceleration_structure r_rt_scene"),
        "patch: an empty parameter list takes the four without a comma");
  const std::string bad[] = {
      Replace(kShape, "main0(", "other("),
      Replace(Replace(kShape, "float4 r_rt_reflect(thread", "float4 r_other(thread"), "r_rt_reflect(param",
              "r_other(param"),
      Replace(kShape, "float4 r_rt_reflect(thread", "float4 r_other(thread"),
      Replace(kShape, "out.r_c0 = r_rt_reflect(param, in.r_var1);", "out.r_c0 = in.r_var1;"),
      Replace(kShape, "[[buffer(0)]]", "[[buffer(5)]]"),
      Replace(kShape, "[[position]])\n{", "[[position]]\n{"),
      Replace(kShape, "[[position]])\n{", "[[position]]);\n"),
  };
  bool left = true;
  for (const std::string& b : bad) {
    std::string x = b;
    if (RtForwardReflectionPatch(&x) || x != b || RtForwardReflectionPatched(x)) left = false;
  }
  Check(left, "patch: refused, the text untouched, without main0, without the stub or its call, the stub defined "
              "only after main0, a buffer of its own taken, main0's list not closed before a body");
}

// The shadow stub and its call in main0, alone and beside the reflection's.
void ShadowPatchChecks() {
  const std::string shadow_only =
      Replace(Replace(kShape, "float4 r_rt_reflect(thread const float4& pixel, thread const float4& origin)",
                      "float4 r_rt_shadow(thread const float4& pixel, thread const float4& world)"),
              "r_rt_reflect(param, in.r_var1)", "r_rt_shadow(param, in.r_var1)");
  std::string m = shadow_only;
  const bool ok = RtForwardReflectionPatch(&m);
  Check(ok && RtForwardReflectionPatched(m) && Has(m, "float4 r_rt_shadow_rt(") && Has(m, "rt_shadowed(") &&
            Has(m, "#define r_rt_shadow(...) r_rt_shadow_rt(r_rt_scene, r_rt_mat, r_rt_info, r_rt_params, "
                   "r_rt_lmask, r_rt_ldist, __VA_ARGS__)\nfragment main0_out main0(") &&
            !Has(m, "#define r_rt_reflect(") &&
            Has(m, "constant RtForwardParams& r_rt_params [[buffer(6)]], texture2d_array<half, access::read> "
                   "r_rt_lmask [[texture(16)]], texture2d<float, access::read> r_rt_ldist [[texture(17)]])\n{"),
        "shadow patch: the stub alone, its macro only, the four parameters and the local lights' two textures");
  std::string taken = Replace(shadow_only, "[[buffer(0)]]", "[[buffer(0)]], texture2d<float> x [[texture(16)]]");
  const std::string taken_kept = taken;
  Check(!RtForwardReflectionPatch(&taken) && taken == taken_kept,
        "shadow patch: refused with the local lights' texture slots taken");
  std::string both = Replace(kShape, "static inline __attribute__((always_inline))\nfloat4 r_rt_reflect(",
                             "static inline __attribute__((always_inline))\nfloat4 r_rt_shadow(thread const float4& "
                             "pixel, thread const float4& world)\n{\n    return float4(1.0);\n}\n\nstatic inline "
                             "__attribute__((always_inline))\nfloat4 r_rt_reflect(");
  both = Replace(both, "out.r_c0 = r_rt_reflect(param, in.r_var1);",
                 "out.r_c0 = r_rt_reflect(param, in.r_var1) * r_rt_shadow(param, in.r_var1);");
  Check(RtForwardReflectionPatch(&both) && Has(both, "#define r_rt_reflect(...) r_rt_reflect_rt(") &&
            Has(both, "#define r_rt_shadow(...) r_rt_shadow_rt("),
        "shadow patch: beside the reflection's, both macros");
  std::string uncalled = Replace(shadow_only, "out.r_c0 = r_rt_shadow(param, in.r_var1);", "out.r_c0 = in.r_var1;");
  const std::string kept = uncalled;
  Check(!RtForwardReflectionPatch(&uncalled) && uncalled == kept, "shadow patch: refused without the call in main0");
}

void WhyChecks() {
  RtForwardWant w;
  w.config = w.snapshot = true;
  w.snapshot_camera = w.scene_camera = w.depth_camera = 7;
  // The draw's own camera decides where its matrices give it: the main
  // view's traced whatever its depth's camera (another of the frame's
  // cameras voted last for the depth buffer, a target of its own), another
  // camera's (an environment map's pass) not.
  {
    RtForwardWant own = w, own_other_depth = w, other = w, unmatched = w, target = w, neither = w;
    own.draw_known = own_other_depth.draw_known = other.draw_known = unmatched.draw_known = true;
    own.draw_camera = own_other_depth.draw_camera = 7;
    own_other_depth.depth_camera = 8;
    other.draw_camera = 8;
    unmatched.draw_camera = 0;
    target.depth_camera = 8;
    target.snapshot_target = true;
    neither.depth_camera = 8;
    Check(RtForwardWhyOff(own) == RtForwardOff::None && RtForwardWhyOff(own_other_depth) == RtForwardOff::None &&
              RtForwardWhyOff(other) == RtForwardOff::OtherView &&
              RtForwardWhyOff(unmatched) == RtForwardOff::OtherView && RtForwardWhyOff(target) == RtForwardOff::None &&
              RtForwardWhyOff(neither) == RtForwardOff::OtherView,
          "why off: the draw's own camera where known (the depth's then not asked; none of the frame's or another: "
          "not traced), else the snapshot's target or the depth's camera");
  }
  RtForwardWant no_config = w, no_snap = w, no_cam = w, old_cam = w, other = w, no_depth = w;
  no_config.config = false;
  no_snap.snapshot = false;
  no_cam.scene_camera = 0;
  old_cam.snapshot_camera = 6;
  other.depth_camera = 9;
  no_depth.depth_camera = 0;
  Check(RtForwardWhyOff(w) == RtForwardOff::None && RtForwardWhyOff(no_config) == RtForwardOff::NoConfig &&
            RtForwardWhyOff(no_snap) == RtForwardOff::NoSnapshot &&
            RtForwardWhyOff(no_cam) == RtForwardOff::NoCamera && RtForwardWhyOff(old_cam) == RtForwardOff::NoCamera &&
            RtForwardWhyOff(other) == RtForwardOff::OtherView &&
            RtForwardWhyOff(no_depth) == RtForwardOff::OtherView,
        "why off: traced with the scene's camera's snapshot and depth; else not taken, no snapshot, no camera or "
        "a snapshot of another, another view's depth (or none)");
}

// ---- scene and camera helpers

struct V3 {
  double x, y, z;
};
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 Cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 Unit(V3 a) {
  const double l = std::sqrt(Dot(a, a));
  return {a.x / l, a.y / l, a.z / l};
}

// D3D's left-handed camera at `eye` looking at `at` (y up), as world to
// clip rows.
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

void ParamsChecks() {
  double vp[16], inv[16];
  Camera({1, 2, -6}, {1, 2, 10}, 1.0, 4.0 / 3.0, 0.5, 500.0, vp);
  Invert4(vp, inv);
  RtFrame f;
  for (int k = 0; k < 16; k++) {
    f.inv_view_proj[k] = float(inv[k]);
    f.inv_view_proj_exact[k] = inv[k];
  }
  RayForwardReflection r;
  r.toward_light = true;
  r.up[0] = 0, r.up[1] = 2, r.up[2] = 0;
  r.diffuse_wrap = 2.0f;
  r.output_scale = 0.1f;
  r.max_distance = 50.0f;
  RtForwardParams p;
  const bool ok = RtForwardParamsFor(f, 64, 48, r, &p);
  Check(ok && std::fabs(p.eye[0] - 1) < 1e-4 && std::fabs(p.eye[1] - 2) < 1e-4 && std::fabs(p.eye[2] + 6) < 1e-4 &&
            p.up[1] == 1.0f && p.on == 1 && p.toward_light == 1 && p.scale == 0.1f && p.wrap < 1.0f &&
            p.max_distance == 50.0f && p.pixel > 0.0f && p.pixel < 0.05f && p.captured_gap == kRtShadowCapturedGap &&
            p.offsets[0] == kRtShadowOffsetPixels && p.offsets[1] == kRtShadowOffsetView &&
            p.offsets[2] == kRtShadowOffsetWorld,
        "params: the camera's position, up made unit, the light's sign, the wrap below 1, the shadows' gap and "
        "offsets");
  RtFrame bad;
  Check(!RtForwardParamsFor(bad, 64, 48, r, &p) && p.on == 0, "params: refused without a camera, nothing traced");
  Check(RtForwardParamsFor(f, 64, 48, r, &p) && p.shadow == 0 && p.offsets[3] == kRtShadowMinSlope,
        "params: a reflection alone traces no shadow calls (lit); the minimum slope");
  RayForwardShadow sh;
  RtForwardParams q, t;
  sh.toward_light = false;
  const bool a = RtForwardParamsFor(f, 64, 48, nullptr, &sh, &q);
  sh.toward_light = true;
  const bool b = RtForwardParamsFor(f, 64, 48, &r, &sh, &t);
  Check(a && q.on == 0 && q.shadow == 1u && std::fabs(q.eye[2] + 6) < 1e-4 && q.pixel > 0.0f && b && t.on == 1 &&
            t.shadow == 3u,
        "params: shadow calls traced (bit 0), the light toward it (bit 1); with the reflection's or alone (on 0)");
}

// ps_3_0: texld r0, v4, s15 (v4 TEXCOORD2: the point); mov oC0, r0. Its
// reads of s15 the shadow call with c0.
bool TranslateShadow(hlsl::Program* p) {
  const std::vector<uint32_t> t = {
      0xFFFF0300u,
      0x0200001Fu, 0x80020005u, 0x90070004u,  // dcl_texcoord2 v4.xyz
      0x0200001Fu, 0x90000000u, 0xA00F080Fu,  // dcl_2d s15
      0x03000042u, 0x800F0000u, 0x90E40004u, 0xA0E4080Fu,  // texld r0, v4, s15
      0x02000001u, 0x800F0800u, 0x80E40000u,  // mov oC0, r0
      0x0000FFFFu,
  };
  hlsl::TranslateOptions o;
  o.ray_shadow.sampler = 15;
  o.ray_shadow.light_register = 0;
  std::string err;
  if (!hlsl::TranslateBytecode(t.data(), t.size(), p, &err, o)) {
    std::printf("FAIL translate the shadow shader: %s\n", err.c_str());
    return false;
  }
  return p->ray_shadow_sampler == 15;
}

// ---- the translated shader

// ps_3_0: texld r0, v1, s3 (v1 TEXCOORD1: the direction, v4 TEXCOORD2: the
// origin); mov oC0, r0. Its reads of s3 the call with c0-c3.
bool Translate(hlsl::Program* p) {
  const std::vector<uint32_t> t = {
      0xFFFF0300u,
      0x0200001Fu, 0x80010005u, 0x900F0001u,  // dcl_texcoord1 v1
      0x0200001Fu, 0x80020005u, 0x90070004u,  // dcl_texcoord2 v4.xyz
      0x0200001Fu, 0x90000000u, 0xA00F0803u,  // dcl_2d s3
      0x03000042u, 0x800F0000u, 0x90E40001u, 0xA0E40803u,  // texld r0, v1, s3
      0x02000001u, 0x800F0800u, 0x80E40000u,  // mov oC0, r0
      0x0000FFFFu,
  };
  hlsl::TranslateOptions o;
  o.ray_reflection.sampler = 3;
  o.ray_reflection.origin = 4;
  for (uint32_t k = 0; k < 4; k++) o.ray_reflection.registers[k] = k;
  std::string err;
  if (!hlsl::TranslateBytecode(t.data(), t.size(), p, &err, o)) {
    std::printf("FAIL translate: %s\n", err.c_str());
    return false;
  }
  return p->ray_reflection_sampler == 3;
}

// Its MSL as the Metal backend makes it, patched; false if a step fails.
bool Msl(const hlsl::Program& p, std::string* msl) {
  const std::string glsl = "#version 450\n" + MetalShaderPrelude(hlsl::Stage::Pixel) + p.glsl;
  std::vector<uint32_t> spirv;
  std::string log;
  if (!GlslToSpirv(glsl, ShaderStage::Fragment, &spirv, &log) || !SpirvToMsl(spirv, msl, &log)) {
    std::printf("FAIL the shader's MSL: %s\n", log.c_str());
    return false;
  }
  const bool patched = RtForwardReflectionPatch(msl);
  MetalLeanMacros(msl, false);
  return patched;
}

// ---- RtForwardCompile: the plain text in, ray queries put in, the plain
// text where they cannot be or do not compile.

struct FakeCompile {
  id<MTLDevice> dev;
  bool reject_patched = false;
  int calls = 0;
  std::vector<std::string> texts;
};

id<MTLLibrary> Compile(void* ctx, const std::string& msl, bool lean, NSError** error) {
  auto* c = static_cast<FakeCompile*>(ctx);
  c->calls++;
  c->texts.push_back(msl);
  if (c->reject_patched && RtForwardReflectionPatched(msl)) return nil;
  MTLCompileOptions* co = [MTLCompileOptions new];
  if (lean) co.preprocessorMacros = @{@"R_ALPHA_TEST" : @"true"};
  return [c->dev newLibraryWithSource:@(msl.c_str()) options:co error:error];
}

void CompileChecks(id<MTLDevice> dev, const std::string& plain) {
  const std::string kept = plain;
  FakeCompile ok{dev};
  const RtForwardBuild a = RtForwardCompile(plain, true, Compile, &ok);
  Check(a.lib && a.patched && !a.refused && !a.patch_error && ok.calls == 1 && RtForwardReflectionPatched(a.msl) &&
            Has(a.msl, std::string(kRtMsl).substr(0, 2000)) && Has(a.msl, kRtReflectRayMsl) && plain == kept &&
            !RtForwardReflectionPatched(plain),
        "compile: the plain text (as cached) patched now, with this build's scene MSL; the plain text untouched");
  // (The scene MSL's rays are out of line in a patched shader: its pipeline
  // compiles each traversal once, rt_forward_cost_test.)
  Check(Has(a.msl, "__attribute__((noinline))\nRtHit RtTraceClosest(") &&
            Has(a.msl, "__attribute__((noinline))\nbool RtTraceAny(") &&
            Has(a.msl, "__attribute__((noinline))\nbool rt_shadowed("),
        "compile: the scene MSL's closest hit, any hit and shadow out of line in the patched text");
  FakeCompile bad{dev, true};
  const RtForwardBuild b = RtForwardCompile(plain, true, Compile, &bad);
  Check(b.lib && !b.patched && b.patch_error == nil && bad.calls == 2 && !RtForwardReflectionPatched(b.msl) &&
            RtForwardReflectionPatched(bad.texts[0]) && !RtForwardReflectionPatched(bad.texts[1]),
        "compile: a patched text that does not compile: the plain one compiled instead");
  FakeCompile off{dev};
  const RtForwardBuild c = RtForwardCompile(plain, false, Compile, &off);
  std::string odd = Replace(plain, "fragment main0_out main0(", "fragment main0_out other0(");
  FakeCompile refused{dev};
  const RtForwardBuild e = RtForwardCompile(odd, true, Compile, &refused);
  Check(c.lib && !c.patched && off.calls == 1 && !e.patched && e.refused && refused.calls == 1,
        "compile: without the option the plain text; a text the patch refuses compiled plain");
}

// ---- the GPU

constexpr uint32_t kW = 64, kH = 48;
const V3 kOrigin{0, 2, 0};
// The direction at clip position (x, y): (x kDx, y kDy, 1).
constexpr double kDx = 0.8, kDy = 0.5;
// The quad: z = 10, x in [-4, 4], y in [0, 4]; a 1 x 1 texture.
constexpr double kZ = 10, kX0 = -4, kX1 = 4, kY0 = 0, kY1 = 4;
const uint8_t kTexel[4] = {200, 100, 50, 255};
// The registers: c0 the light's direction (away from it), c1 its colour
// (times w), c2 and c3 the ambient's a and b.
const float kConsts[4][4] = {{0, 0, 1, 0}, {0.5f, 0.25f, 0.125f, 2}, {0.1f, 0.05f, 0.02f, 0}, {0.2f, 0.1f, 0.3f, 0}};
constexpr float kScale = 0.5f, kWrap = 0.25f;

struct Gpu {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  id<MTLBuffer> arena = nil;
  size_t used = 0;
  id<MTLRenderPipelineState> pipe = nil;
  RtForwardNullScene null_scene;
};

RtUpload ArenaAlloc(void* ctx, size_t bytes, size_t align) {
  auto* g = static_cast<Gpu*>(ctx);
  const size_t off = (g->used + align - 1) & ~(align - 1);
  if (off + bytes > g->arena.length) return {nil, 0, nullptr};
  g->used = off + bytes;
  return {g->arena, off, static_cast<uint8_t*>(g->arena.contents) + off};
}

// The pipeline: a full-screen strip whose varyings at the shader's
// locations are the direction and the origin, and the patched shader.
bool Pipeline(Gpu& g, const hlsl::Program& p, const std::string& msl) {
  const int dir = hlsl::VaryingLocation({hlsl::kTexcoord, 1}), org = hlsl::VaryingLocation({hlsl::kTexcoord, 2});
  const std::string vs = R"(#include <metal_stdlib>
using namespace metal;
struct VOut {
  float4 pos [[position]];
  float4 dir [[user(locn)" + std::to_string(dir) + R"()]];
  float4 org [[user(locn)" + std::to_string(org) + R"()]];
};
vertex VOut quad_vs(uint vid [[vertex_id]], constant float4* p [[buffer(0)]]) {
  const float2 c = float2((vid & 1u) != 0u ? 1.0 : -1.0, (vid & 2u) != 0u ? -1.0 : 1.0);
  VOut o;
  o.pos = float4(c, 0.5, 1.0);
  o.dir = float4(c.x * p[0].x, c.y * p[0].y, p[0].z, 0.0);
  o.org = p[1];
  return o;
}
)";
  NSError* err = nil;
  id<MTLLibrary> vl = [g.dev newLibraryWithSource:@(vs.c_str()) options:nil error:&err];
  MTLCompileOptions* co = [MTLCompileOptions new];
  co.preprocessorMacros = @{@"R_ALPHA_TEST" : @"false"};
  id<MTLLibrary> fl = [g.dev newLibraryWithSource:@(msl.c_str()) options:co error:&err];
  if (!vl || !fl) {
    std::printf("FAIL the pipeline's libraries: %s\n", err.localizedDescription.UTF8String);
    return false;
  }
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [vl newFunctionWithName:@"quad_vs"];
  pd.fragmentFunction = [fl newFunctionWithName:@"main0"];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  g.pipe = [g.dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!g.pipe) std::printf("FAIL the pipeline: %s\n", err.localizedDescription.UTF8String);
  (void)p;
  return g.pipe != nil;
}

// The quad's mesh: a strip, 24-byte vertices (position, uv, colour).
struct Quad {
  std::vector<uint8_t> vb;
  std::vector<uint16_t> ib = {0, 1, 2, 3};
  RtGeometry g;
  RtMeshKey key;
};
Quad MakeQuad(uint64_t buffer) {
  const V3 p[4] = {{kX0, kY0, kZ}, {kX0, kY1, kZ}, {kX1, kY0, kZ}, {kX1, kY1, kZ}};
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
  k.attr_version = k.vertex_version;
  k.attr_buffer = buffer;
  k.attr_stride = 24;
  k.uv_type = 1;
  k.uv_offset = 12;
  k.uv_lanes = 0 | 1 << 2;
  k.alpha_type = 17;
  return q;
}

// One draw of the quad's scene (D3DCULL `cull`) with the parameters' "on"
// `on` (and the null scene bound in its place with `bind` false): its
// pixels (RGBA32F).
bool Draw(Gpu& g, uint32_t cull, bool on, bool bind, std::vector<float>* px) {
  RtSceneOptions so;
  so.hit_data = true;
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, so);
  g.used = 0;
  s->SetAllocator(ArenaAlloc, &g);
  double vp[16];
  Camera({0, 2, -6}, {0, 2, 10}, 1.0, double(kW) / kH, 0.5, 500.0, vp);
  float w[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}, mw[4][4];
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) mw[r][k] = float(vp[r * 4 + k]);
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:1
                                                                               height:1
                                                                            mipmapped:NO];
  td.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> tex = [g.dev newTextureWithDescriptor:td];
  [tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:kTexel bytesPerRow:4];
  RtAlbedo albedo;
  albedo.texture = tex;
  Quad q = MakeQuad(1);
  s->BeginFrame(1, 0);
  s->Add(q.key, q.g, w, mw, cull, 4, nullptr, RtView::Main, &albedo);
  const RtFrame* f = s->Snapshot();
  if (!f || !f->tlas) {
    std::printf("FAIL a TLAS of the scene\n");
    return false;
  }
  RayForwardReflection r;
  r.up[0] = 0, r.up[1] = 1, r.up[2] = 0;
  r.diffuse_wrap = kWrap;
  r.output_scale = kScale;
  RtForwardParams p;
  if (!RtForwardParamsFor(*f, kW, kH, r, &p)) {
    std::printf("FAIL the parameters\n");
    return false;
  }
  if (!on) p = RtForwardParams{};
  MTLTextureDescriptor* od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:kW
                                                                               height:kH
                                                                            mipmapped:NO];
  od.usage = MTLTextureUsageRenderTarget;
  od.storageMode = MTLStorageModeShared;
  id<MTLTexture> out = [g.dev newTextureWithDescriptor:od];
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  s->EncodeFrame(cb, f);
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = out;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:g.pipe];
  const float vin[2][4] = {{float(kDx), float(kDy), 1, 0}, {float(kOrigin.x), float(kOrigin.y), float(kOrigin.z), 1}};
  [e setVertexBytes:vin length:sizeof vin atIndex:0];
  [e setFragmentBytes:kConsts length:sizeof kConsts atIndex:0];
  const uint8_t zeros[256] = {};
  [e setFragmentBytes:zeros length:sizeof zeros atIndex:1];
  [e setFragmentBytes:zeros length:sizeof zeros atIndex:2];
  if (bind) {
    RtUseResources(e, f, MTLRenderStageFragment);
    [e setFragmentAccelerationStructure:f->tlas atBufferIndex:kRtForwardSceneBuffer];
    [e setFragmentBuffer:f->materials offset:f->materials_offset atIndex:kRtForwardMaterialBuffer];
    [e setFragmentBuffer:f->info offset:f->info_offset atIndex:kRtForwardInfoBuffer];
  } else {
    [e useResource:g.null_scene.blas usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    [e setFragmentAccelerationStructure:g.null_scene.tlas atBufferIndex:kRtForwardSceneBuffer];
    [e setFragmentBuffer:g.null_scene.materials offset:0 atIndex:kRtForwardMaterialBuffer];
    [e setFragmentBuffer:g.null_scene.info offset:0 atIndex:kRtForwardInfoBuffer];
  }
  [e setFragmentBytes:&p length:sizeof p atIndex:kRtForwardParamsBuffer];
  [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) {
    std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
    return false;
  }
  px->assign(size_t(kW) * kH * 4, 0.0f);
  [out getBytes:px->data() bytesPerRow:kW * 16 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
  return true;
}

// What pixel (i, j) should hold: -1 where it is not judged (near the
// quad's edges), else 1 where its ray hits the quad, 0 where it misses.
int Expected(uint32_t i, uint32_t j) {
  const double x = 2.0 * (i + 0.5) / kW - 1.0, y = 1.0 - 2.0 * (j + 0.5) / kH;
  const double t = (kZ - kOrigin.z) / 1.0;
  const double hx = kOrigin.x + x * kDx * t, hy = kOrigin.y + y * kDy * t;
  const double m = 0.15;
  const bool in = hx > kX0 + m && hx < kX1 - m && hy > kY0 + m && hy < kY1 - m;
  const bool out = hx < kX0 - m || hx > kX1 + m || hy < kY0 - m || hy > kY1 + m;
  return in ? 1 : out ? 0 : -1;
}

// The colour of a hit: the texel times the light (the face toward the
// ray, -z, has N.L 1: diffuse 1, nothing in the shadow ray's way) plus the
// ambient (N.up 0: a + b / 2), times the scale.
void HitColour(float out[3]) {
  for (int k = 0; k < 3; k++) {
    const double albedo = kTexel[k] / 255.0;
    const double colour = double(kConsts[1][k]) * kConsts[1][3];
    const double amb = double(kConsts[2][k]) + double(kConsts[3][k]) * 0.5;
    out[k] = float(albedo * (colour + amb) * kScale);
  }
}

// Every judged pixel: hits the colour and a 1, misses 0; `lit`: hits are
// expected (else 0 everywhere). Counts the judged hits.
bool Judge(const std::vector<float>& px, bool lit, int* hits, const char* what, bool quiet = false) {
  float want[3];
  HitColour(want);
  int bad = 0;
  *hits = 0;
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const int e = Expected(i, j);
      if (e < 0) continue;
      const float* p = &px[(size_t(j) * kW + i) * 4];
      const bool hit = lit && e == 1;
      bool ok = std::fabs(p[3] - (hit ? 1.0f : 0.0f)) < 1e-6f;
      for (int k = 0; k < 3; k++) ok = ok && std::fabs(p[k] - (hit ? want[k] : 0.0f)) < 2e-3f;
      if (hit) (*hits)++;
      if (!ok && bad++ < 4 && !quiet)
        std::printf("  %s: pixel (%u, %u): %.4f %.4f %.4f %.4f, want %s (%.4f %.4f %.4f)\n", what, i, j, p[0], p[1],
                    p[2], p[3], hit ? "a hit" : "nothing", hit ? want[0] : 0.0f, hit ? want[1] : 0.0f,
                    hit ? want[2] : 0.0f);
    }
  return bad == 0;
}

// ---- shadow calls on the GPU: points on z = kShadowZ behind the quad
// (from the light, which shines along +z: c0 = (0, 0, 1), away from it),
// at clip (x, y) the point (x kShadowSx, kShadowY + y kShadowSy, kShadowZ).
constexpr double kShadowZ = 20, kShadowSx = 8, kShadowY = 2, kShadowSy = 4;

id<MTLRenderPipelineState> ShadowPipeline(Gpu& g, const std::string& msl) {
  const int org = hlsl::VaryingLocation({hlsl::kTexcoord, 2});
  const std::string vs = R"(#include <metal_stdlib>
using namespace metal;
struct VOut {
  float4 pos [[position]];
  float4 org [[user(locn)" + std::to_string(org) + R"()]];
};
vertex VOut shadow_vs(uint vid [[vertex_id]], constant float4* p [[buffer(0)]]) {
  const float2 c = float2((vid & 1u) != 0u ? 1.0 : -1.0, (vid & 2u) != 0u ? -1.0 : 1.0);
  VOut o;
  o.pos = float4(c, 0.5, 1.0);
  o.org = float4(c.x * p[0].x, p[0].y + c.y * p[0].z, p[0].w, 1.0);
  return o;
}
)";
  NSError* err = nil;
  id<MTLLibrary> vl = [g.dev newLibraryWithSource:@(vs.c_str()) options:nil error:&err];
  MTLCompileOptions* co = [MTLCompileOptions new];
  co.preprocessorMacros = @{@"R_ALPHA_TEST" : @"false"};
  id<MTLLibrary> fl = vl ? [g.dev newLibraryWithSource:@(msl.c_str()) options:co error:&err] : nil;
  if (!vl || !fl) {
    std::printf("FAIL the shadow pipeline's libraries: %s\n", err.localizedDescription.UTF8String);
    return nil;
  }
  MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
  pd.vertexFunction = [vl newFunctionWithName:@"shadow_vs"];
  pd.fragmentFunction = [fl newFunctionWithName:@"main0"];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  id<MTLRenderPipelineState> pipe = [g.dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pipe) std::printf("FAIL the shadow pipeline: %s\n", err.localizedDescription.UTF8String);
  return pipe;
}

// The local lights' masks for a draw (kRtForwardLightMasksTexture,
// kRtForwardLightDistTexture): 1 x 1 stand-ins, or (`pattern`) a layer 1
// of kW x kH holding Pattern(i, j), every distance 1.
uint8_t Pattern(uint32_t i, uint32_t j) { return ((i / 7 + j / 5) & 1) ? 255 : 0; }
void LightTextures(Gpu& g, bool pattern, id<MTLTexture>* masks, id<MTLTexture>* dist) {
  const uint32_t w = pattern ? kW : 1, h = pattern ? kH : 1;
  MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
  md.textureType = MTLTextureType2DArray;
  md.arrayLength = 2;
  md.storageMode = MTLStorageModeShared;
  md.usage = MTLTextureUsageShaderRead;
  *masks = [g.dev newTextureWithDescriptor:md];
  MTLTextureDescriptor* dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
  dd.storageMode = MTLStorageModeShared;
  dd.usage = MTLTextureUsageShaderRead;
  *dist = [g.dev newTextureWithDescriptor:dd];
  std::vector<uint8_t> layer(size_t(w) * h, 0x40), other(size_t(w) * h, 0x40);
  std::vector<float> d(size_t(w) * h, 1.0f);
  for (uint32_t j = 0; j < h && pattern; j++)
    for (uint32_t i = 0; i < w; i++) layer[size_t(j) * w + i] = Pattern(i, j);
  [*masks replaceRegion:MTLRegionMake2D(0, 0, w, h)
            mipmapLevel:0
                  slice:0
              withBytes:other.data()
            bytesPerRow:w
          bytesPerImage:w * h];
  [*masks replaceRegion:MTLRegionMake2D(0, 0, w, h)
            mipmapLevel:0
                  slice:1
              withBytes:layer.data()
            bytesPerRow:w
          bytesPerImage:w * h];
  [*dist replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:d.data() bytesPerRow:w * 4];
}

// One draw of the shadow shader over the quad's scene, the parameters'
// shadow bit `on`: its pixels' red. With `masked`, the light a local one
// whose masks' layer 1 (LightTextures' pattern) the draw reads.
bool DrawShadow(Gpu& g, id<MTLRenderPipelineState> pipe, bool on, std::vector<float>* red, bool masked = false) {
  RtSceneOptions so;
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, so);
  g.used = 0;
  s->SetAllocator(ArenaAlloc, &g);
  double vp[16];
  Camera({0, 2, -6}, {0, 2, 10}, 1.0, double(kW) / kH, 0.5, 500.0, vp);
  float w[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}, mw[4][4];
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) mw[r][k] = float(vp[r * 4 + k]);
  Quad q = MakeQuad(2);
  s->BeginFrame(1, 0);
  s->Add(q.key, q.g, w, mw, 1, 4, nullptr, RtView::Main, nullptr);
  const RtFrame* f = s->Snapshot();
  if (!f || !f->tlas) {
    std::printf("FAIL a TLAS of the shadow scene\n");
    return false;
  }
  RayForwardShadow sh;
  RtForwardParams p;
  if (!RtForwardParamsFor(*f, kW, kH, nullptr, &sh, &p)) {
    std::printf("FAIL the shadow parameters\n");
    return false;
  }
  if (!on) p.shadow = 0;
  if (masked) {
    p.shadow |= 4u;  // (the register then a position: never traced here)
    p.light_slot = 1;
    p.light_step = 1;
  }
  id<MTLTexture> lmask, ldist;
  LightTextures(g, masked, &lmask, &ldist);
  MTLTextureDescriptor* od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:kW
                                                                               height:kH
                                                                            mipmapped:NO];
  od.usage = MTLTextureUsageRenderTarget;
  od.storageMode = MTLStorageModeShared;
  id<MTLTexture> out = [g.dev newTextureWithDescriptor:od];
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  s->EncodeFrame(cb, f);
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = out;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:pipe];
  const float vin[4] = {float(kShadowSx), float(kShadowY), float(kShadowSy), float(kShadowZ)};
  [e setVertexBytes:vin length:sizeof vin atIndex:0];
  const float consts[4][4] = {{0, 0, 1, 0}};
  [e setFragmentBytes:consts length:sizeof consts atIndex:0];
  const uint8_t zeros[256] = {};
  [e setFragmentBytes:zeros length:sizeof zeros atIndex:1];
  [e setFragmentBytes:zeros length:sizeof zeros atIndex:2];
  RtUseResources(e, f, MTLRenderStageFragment);
  [e setFragmentAccelerationStructure:f->tlas atBufferIndex:kRtForwardSceneBuffer];
  [e setFragmentBuffer:f->materials offset:f->materials_offset atIndex:kRtForwardMaterialBuffer];
  [e setFragmentBuffer:f->info offset:f->info_offset atIndex:kRtForwardInfoBuffer];
  [e setFragmentBytes:&p length:sizeof p atIndex:kRtForwardParamsBuffer];
  [e setFragmentTexture:lmask atIndex:kRtForwardLightMasksTexture];
  [e setFragmentTexture:ldist atIndex:kRtForwardLightDistTexture];
  [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) {
    std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
    return false;
  }
  std::vector<float> px(size_t(kW) * kH * 4, 0.0f);
  [out getBytes:px.data() bytesPerRow:kW * 16 fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
  red->resize(size_t(kW) * kH);
  for (size_t k = 0; k < red->size(); k++) (*red)[k] = px[k * 4];
  return true;
}

// Every judged pixel (not near the quad's shadow's edges): 0 where the quad
// blocks the light, else 1 (`on` false: 1 everywhere); counts the shadowed.
bool JudgeShadow(const std::vector<float>& red, bool on, int* shadowed, const char* what) {
  int bad = 0;
  *shadowed = 0;
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const double x = 2.0 * (i + 0.5) / kW - 1.0, y = 1.0 - 2.0 * (j + 0.5) / kH;
      const double hx = x * kShadowSx, hy = kShadowY + y * kShadowSy, m = 0.2;
      const bool in = hx > kX0 + m && hx < kX1 - m && hy > kY0 + m && hy < kY1 - m;
      const bool out = hx < kX0 - m || hx > kX1 + m || hy < kY0 - m || hy > kY1 + m;
      if (!in && !out) continue;
      const float want = on && in ? 0.0f : 1.0f;
      if (want == 0.0f) (*shadowed)++;
      const float v = red[size_t(j) * kW + i];
      if (v != want && bad++ < 4) std::printf("  %s: pixel (%u, %u): %.4f, want %.0f\n", what, i, j, v, want);
    }
  return bad == 0;
}

void ShadowGpuChecks(Gpu& g) {
  if (!g.arena) g.arena = [g.dev newBufferWithLength:8u << 20 options:MTLResourceStorageModeShared];
  hlsl::Program prog;
  if (!TranslateShadow(&prog)) {
    Check(false, "shadow: the translated shader");
    return;
  }
  std::string msl;
  const bool patched = Msl(prog, &msl);
  Check(patched && Has(msl, "r_rt_shadow_rt(r_rt_scene"), "shadow: the translated shader's MSL patched");
  id<MTLRenderPipelineState> pipe = patched ? ShadowPipeline(g, msl) : nil;
  if (!pipe) {
    Check(false, "shadow: the patched shader's pipeline");
    return;
  }
  std::vector<float> red;
  int shadowed = 0;
  if (DrawShadow(g, pipe, true, &red))
    Check(JudgeShadow(red, true, &shadowed, "shadow") && shadowed > 100,
          "shadow: 0 behind the quad from the light, 1 elsewhere");
  if (DrawShadow(g, pipe, false, &red))
    Check(JudgeShadow(red, false, &shadowed, "shadow off"), "shadow: the shadow bit clear, lit everywhere");
  // A local light in the frame's masks: no ray, its layer's texel at each
  // pixel (the neighbours' weight only a thousandth at a pixel's own texel).
  if (DrawShadow(g, pipe, true, &red, true)) {
    int bad = 0;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++) {
        const float want = Pattern(i, j) / 255.0f, v = red[size_t(j) * kW + i];
        if (std::fabs(v - want) > 0.01f && bad++ < 4) std::printf("  masked: pixel (%u, %u): %.4f, want %.0f\n", i, j, v, want);
      }
    Check(bad == 0, "shadow: a local light in the frame's masks reads its layer at the pixel, no ray");
  }
}

void GpuChecks(Gpu& g, const hlsl::Program& prog, const std::string& msl) {
  g.arena = [g.dev newBufferWithLength:8u << 20 options:MTLResourceStorageModeShared];
  if (!g.arena || !Pipeline(g, prog, msl)) {
    Check(false, "the patched shader's pipeline");
    return;
  }
  Check(true, "the patched shader's pipeline");
  std::vector<float> px;
  int hits = 0;
  if (Draw(g, 1, true, true, &px))
    Check(Judge(px, true, &hits, "no culling") && hits > 100,
          "hits: the texel times the light plus the ambient, times the scale, a 1; misses 0");
  // One winding's culling makes the face the rays meet a back face: 0.
  std::vector<float> a, b;
  int ha = 0, hb = 0;
  if (Draw(g, 2, true, true, &a) && Draw(g, 3, true, true, &b)) {
    const bool la = Judge(a, true, &ha, "cull 2", true), lb = Judge(b, true, &hb, "cull 3", true);
    const bool za = Judge(a, false, &ha, "cull 2 (stopped)", true), zb = Judge(b, false, &hb, "cull 3 (stopped)", true);
    Check((la && zb) || (lb && za), "culling: one winding lit, the other's back face stops the rays with a 0");
  }
  if (Draw(g, 1, false, true, &px)) Check(Judge(px, false, &hits, "off"), "off: nothing anywhere");
  if (!RtForwardMakeNullScene(g.dev, g.queue, &g.null_scene)) {
    Check(false, "the null scene");
    return;
  }
  if (Draw(g, 1, false, false, &px))
    Check(Judge(px, false, &hits, "off, the null scene"), "off with the null scene bound: nothing");
}

}  // namespace

int main() {
  @autoreleasepool {
    PatchChecks();
    ShadowPatchChecks();
    WhyChecks();
    ParamsChecks();
    hlsl::Program prog;
    std::string msl;
    const bool translated = Translate(&prog);
    const bool patched = translated && Msl(prog, &msl);
    Check(translated && patched && Has(msl, "r_rt_reflect_rt(r_rt_scene") && Has(msl, "[[buffer(6)]]"),
          "a translated shader's MSL (SPIRV-Cross, the prelude) patched");
    Gpu g;
    if (!RtSupportedFromRender(g.dev)) {
      std::printf("skipped the GPU: no ray tracing from render pipelines (Apple9)\n");
      return failures ? 1 : 77;
    }
    if (patched) {
      std::string plain;
      const std::string glsl = "#version 450\n" + MetalShaderPrelude(hlsl::Stage::Pixel) + prog.glsl;
      std::vector<uint32_t> spirv;
      std::string log;
      if (GlslToSpirv(glsl, ShaderStage::Fragment, &spirv, &log) && SpirvToMsl(spirv, &plain, &log))
        CompileChecks(g.dev, plain);
      else
        Check(false, "the plain MSL");
      GpuChecks(g, prog, msl);
    }
    ShadowGpuChecks(g);
  }
  if (failures) return 1;
  std::puts("rt_forward_reflection: ok");
  return 0;
}
