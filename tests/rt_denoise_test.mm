// MetalFX's denoiser of the lit scene (gfx/rt_denoise.h) and the soft
// shadows' cone (gfx/rt_shadow.h: rt_cone_dir). On the CPU: a camera split
// into view and projection (rigid view, view * proj == vp, an off-centre
// projection, an orthographic one refused); the guides' roughness and
// Fresnel. On a GPU that traces rays: the cone's directions stay within the
// cone, spread over it and average to the light's direction (a pixel's over
// frames, and all), none for a cone of 0. Where MetalFX has the denoiser
// (macOS 26, Apple9; else exit 77 after the rest): the scaler made at 1:1
// (its scales logged); the guides' kernel (albedo, decoded normal,
// roughness, specular albedo with the Fresnel of the pixel's view
// direction) and the masks' (reactive where the depth changed, counted; the
// hit distance, the sky's where none) against CPU references; a static
// scene with noise denoised over frames (its mean kept, its variance down);
// a pattern panning by a known motion followed without a ghost; a reset
// dropping the history.
//
//   rt_denoise_test        (exit 77 without MetalFX's denoiser)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/rt_denoise.h"
#include "runtime/gfx/rt_reflection.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"
#include "runtime/gfx/rt_support.h"

namespace {

using namespace recomp::gfx;

int Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  return ok ? 0 : 1;
}

struct V3 {
  double x, y, z;
};
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 Cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 Unit(V3 a) { return a * (1.0 / std::sqrt(Dot(a, a))); }

// D3D's left-handed camera at `eye` looking at `at` (z up), its perspective
// (depth 0..1 at near..far), off centre by (cx, cy), as rows.
void Camera(V3 eye, V3 at, double fov_y, double aspect, double n, double f, double cx, double cy, double view[16],
            double proj[16]) {
  const V3 fw = Unit(at - eye), rt = Unit(Cross({0, 0, 1}, fw)), up = Cross(fw, rt);
  const double v[16] = {rt.x, up.x, fw.x, 0, rt.y, up.y, fw.y, 0, rt.z, up.z, fw.z, 0,
                        -Dot(rt, eye), -Dot(up, eye), -Dot(fw, eye), 1};
  const double ys = 1.0 / std::tan(fov_y / 2), xs = ys / aspect;
  double p[16] = {};
  p[0] = xs;
  p[5] = ys;
  p[8] = cx;
  p[9] = cy;
  p[10] = f / (f - n);
  p[11] = 1;
  p[14] = -n * f / (f - n);
  std::memcpy(view, v, sizeof v);
  std::memcpy(proj, p, sizeof p);
}

double MaxDiff(const double* a, const double* b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int CpuChecks() {
  int fails = 0;
  for (int k = 0; k < 2; k++) {
    double v[16], p[16], vp[16], sv[16], sp[16], back[16];
    Camera({1200.5, -830.25, 35}, {1180, -790, 30}, 0.9, 16.0 / 9.0, 0.25, 3000, k ? 0.1 : 0, k ? -0.05 : 0, v, p);
    Mul4(v, p, vp);
    const bool ok = RtSplitCamera(vp, sv, sp);
    Mul4(sv, sp, back);
    // The view's axes: unit and orthogonal.
    double ortho = 0;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        double d = 0;
        for (int r = 0; r < 3; r++) d += sv[r * 4 + i] * sv[r * 4 + j];
        ortho = std::max(ortho, std::fabs(d - (i == j ? 1.0 : 0.0)));
      }
    char what[160];
    std::snprintf(what, sizeof what,
                  "camera split%s: view %.2g and projection %.2g from the originals, view * proj %.2g from vp, axes "
                  "%.2g from orthonormal",
                  k ? " (off centre)" : "", MaxDiff(sv, v, 16), MaxDiff(sp, p, 16), MaxDiff(back, vp, 16), ortho);
    fails += Check(ok && MaxDiff(sv, v, 16) < 1e-9 && MaxDiff(sp, p, 16) < 1e-9 && MaxDiff(back, vp, 16) < 1e-6 &&
                       ortho < 1e-12,
                   what);
  }
  const double ortho_vp[16] = {0.01, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0.001, 0, 0, 0, 0.5, 1};
  double a[16], b[16];
  fails += Check(!RtSplitCamera(ortho_vp, a, b), "camera split: an orthographic camera refused");
  fails += Check(RtDenoiseRoughness(0.0f, 512.0f) == 1.0f &&
                     std::fabs(RtDenoiseRoughness(1.0f, 512.0f) - std::sqrt(2.0f / 514.0f)) < 1e-7f,
                 "roughness: 1 at gloss 0, sqrt(2 / (e + 2)) at gloss 1");
  fails += Check(std::fabs(RtDenoiseSpecular(0.5f, 1.0f, 2.0f, 0.25f, 4.0f) - 0.25f) < 1e-7f &&
                     std::fabs(RtDenoiseSpecular(0.5f, 0.0f, 2.0f, 0.25f, 4.0f) - 1.0f) < 1e-7f &&
                     RtDenoiseSpecular(1.0f, 0.0f, 2.0f, 0.25f, 4.0f) == 1.0f,
                 "specular albedo: the Fresnel's minimum facing, all at grazing, clamped to 1");
  return fails;
}

// ---- the GPU

struct Gpu {
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
};

id<MTLTexture> Texture(const Gpu& g, MTLPixelFormat pf, uint32_t w, uint32_t h, MTLTextureUsage usage) {
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  td.usage = usage;
  return [g.dev newTextureWithDescriptor:td];
}

void Upload(const Gpu& g, id<MTLTexture> t, const void* data, size_t bpp) {
  const size_t row = t.width * bpp, size = row * t.height;
  id<MTLBuffer> b = [g.dev newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromBuffer:b
             sourceOffset:0
        sourceBytesPerRow:row
      sourceBytesPerImage:size
               sourceSize:MTLSizeMake(t.width, t.height, 1)
                toTexture:t
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

std::vector<uint8_t> Download(const Gpu& g, id<MTLTexture> t, size_t bpp) {
  const size_t row = t.width * bpp, size = row * t.height;
  id<MTLBuffer> b = [g.dev newBufferWithLength:size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  [bl copyFromTexture:t
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(t.width, t.height, 1)
                      toBuffer:b
             destinationOffset:0
        destinationBytesPerRow:row
      destinationBytesPerImage:size];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  std::vector<uint8_t> out(size);
  std::memcpy(out.data(), b.contents, size);
  return out;
}

std::vector<float> Halves(const std::vector<uint8_t>& d) {
  std::vector<float> out(d.size() / 2);
  for (size_t i = 0; i < out.size(); i++) {
    __fp16 h;
    std::memcpy(&h, d.data() + 2 * i, 2);
    out[i] = float(h);
  }
  return out;
}
std::vector<__fp16> ToHalves(const std::vector<float>& v) {
  std::vector<__fp16> out(v.size());
  for (size_t i = 0; i < v.size(); i++) out[i] = __fp16(v[i]);
  return out;
}

// The cone's directions (rt_cone_dir) for 16 x 16 pixels over 64 frames.
int ConeChecks(const Gpu& g) {
  int fails = 0;
  const std::string msl = std::string(kRtMsl) + kRtShadowMsl + R"(
kernel void cone_test(device float4* out [[buffer(0)]], constant float4& l [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
  for (uint f = 0; f < 64u; f++) out[(gid.y * 16u + gid.x) * 64u + f] = float4(rt_cone_dir(l.xyz, l.w, gid, f), 0.0);
}
)";
  NSError* err = nil;
  id<MTLLibrary> lib = [g.dev newLibraryWithSource:@(msl.c_str()) options:nil error:&err];
  id<MTLComputePipelineState> pipe =
      lib ? [g.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cone_test"] error:&err] : nil;
  if (!pipe) {
    std::printf("FAIL the cone's test kernel: %s\n", err.localizedDescription.UTF8String);
    return 1;
  }
  const V3 l = Unit({0.3, -0.5, 0.81});
  for (float cone : {kRtShadowCone, 0.0f}) {
    const float lv[4] = {float(l.x), float(l.y), float(l.z), cone};
    id<MTLBuffer> out = [g.dev newBufferWithLength:16 * 16 * 64 * 16 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipe];
    [e setBuffer:out offset:0 atIndex:0];
    [e setBytes:lv length:sizeof lv atIndex:1];
    [e dispatchThreads:MTLSizeMake(16, 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* d = static_cast<const float*>(out.contents);
    double widest = 0, worst_pixel_mean = 0, sum[3] = {};
    bool exact = true;
    for (int p = 0; p < 256; p++) {
      double m[3] = {};
      for (int f = 0; f < 64; f++) {
        const float* v = d + (p * 64 + f) * 4;
        const V3 dir = {v[0], v[1], v[2]};
        const V3 c = Cross(dir, l);
        widest = std::max(widest, std::asin(std::min(1.0, std::sqrt(Dot(c, c)))));
        exact = exact && v[0] == lv[0] && v[1] == lv[1] && v[2] == lv[2];
        for (int k = 0; k < 3; k++) m[k] += v[k] / 64.0, sum[k] += v[k] / (64.0 * 256.0);
      }
      const V3 c = Cross(Unit({m[0], m[1], m[2]}), l);
      worst_pixel_mean = std::max(worst_pixel_mean, std::asin(std::min(1.0, std::sqrt(Dot(c, c)))));
    }
    const V3 c = Cross(Unit({sum[0], sum[1], sum[2]}), l);
    const double mean = std::asin(std::min(1.0, std::sqrt(Dot(c, c))));
    char what[200];
    if (cone > 0) {
      std::snprintf(what, sizeof what,
                    "the cone (%.4f rad): directions at most %.5f from the light's, a pixel's mean over 64 frames at "
                    "most %.5f, all pixels' %.6f",
                    cone, widest, worst_pixel_mean, mean);
      fails += Check(widest <= cone * 1.01 + 1e-5 && widest >= cone * 0.8 && worst_pixel_mean < 0.2 * cone &&
                         mean < 0.05 * cone,
                     what);
    } else {
      fails += Check(exact, "a cone of 0: the light's direction itself");
    }
  }
  return fails;
}

// The denoiser's tests at 64 x 64.
constexpr uint32_t kW = 64, kH = 64;

struct Scene {
  double view[16], proj[16], vp[16], inv[16];
};
Scene MakeScene() {
  Scene s;
  Camera({10, 20, 5}, {10, 40, 4}, 1.0, 1.0, 0.5, 1000, 0, 0, s.view, s.proj);
  Mul4(s.view, s.proj, s.vp);
  Invert4(s.vp, s.inv);
  return s;
}

RtDenoiseGuideParams GuideParams(const Scene& s, float jx, float jy) {
  RtDenoiseGuideParams p{};
  for (int i = 0; i < 16; i++) p.inv_view_proj[i] = float(s.inv[i]);
  p.jitter[0] = jx;
  p.jitter[1] = jy;
  p.exponent_scale = 512.0f;
  p.spec_scale = 2.0f;
  p.fresnel_min = 0.25f;
  p.fresnel_power = 4.0f;
  p.spec_channel = 0;
  p.gloss_channel = 1;
  return p;
}

V3 World(const Scene& s, double x, double y, double z) {
  double o[4] = {};
  const double v[4] = {x, y, z, 1};
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < 4; i++) o[k] += v[i] * s.inv[i * 4 + k];
  return {o[0] / o[3], o[1] / o[3], o[2] / o[3]};
}

struct GBuffer {
  id<MTLTexture> albedo, normal, surface, depth;
  std::vector<uint8_t> a, n, su;
};

// A G-buffer: varying albedo, normals, spec (y) and gloss (x) where
// `varied`, else flat (grey, up, half spec, half gloss); depth `z` everywhere.
GBuffer MakeGBuffer(const Gpu& g, bool varied, float z) {
  GBuffer b;
  b.a.resize(kW * kH * 4);
  b.n.resize(kW * kH * 4);
  b.su.resize(kW * kH * 4);
  for (uint32_t y = 0; y < kH; y++)
    for (uint32_t x = 0; x < kW; x++) {
      uint8_t* a = &b.a[(y * kW + x) * 4];
      uint8_t* n = &b.n[(y * kW + x) * 4];
      uint8_t* s = &b.su[(y * kW + x) * 4];
      if (varied) {
        const V3 nv = Unit({std::sin(x * 0.3) * 0.8, std::cos(y * 0.2) * 0.8, 0.4 + 0.5 * ((x + y) % 3)});
        a[0] = uint8_t(x * 4), a[1] = uint8_t(y * 4), a[2] = 200, a[3] = 255;
        n[0] = uint8_t(std::lround((nv.x * 0.5 + 0.5) * 255)), n[1] = uint8_t(std::lround((nv.y * 0.5 + 0.5) * 255)),
        n[2] = uint8_t(std::lround((nv.z * 0.5 + 0.5) * 255)), n[3] = 255;
        s[0] = uint8_t(y * 4), s[1] = uint8_t(x * 4), s[2] = 0, s[3] = 0;
      } else {
        a[0] = a[1] = a[2] = 128, a[3] = 255;
        n[0] = n[1] = 128, n[2] = 255, n[3] = 255;
        s[0] = s[1] = 128, s[2] = s[3] = 0;
      }
    }
  b.albedo = Texture(g, MTLPixelFormatRGBA8Unorm, kW, kH, MTLTextureUsageShaderRead);
  b.normal = Texture(g, MTLPixelFormatRGBA8Unorm, kW, kH, MTLTextureUsageShaderRead);
  b.surface = Texture(g, MTLPixelFormatRGBA8Unorm, kW, kH, MTLTextureUsageShaderRead);
  Upload(g, b.albedo, b.a.data(), 4);
  Upload(g, b.normal, b.n.data(), 4);
  Upload(g, b.surface, b.su.data(), 4);
  b.depth = Texture(g, MTLPixelFormatDepth32Float, kW, kH, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
  const std::vector<float> d(kW * kH, z);
  Upload(g, b.depth, d.data(), 4);
  return b;
}

void Run(const Gpu& g, void (^encode)(id<MTLCommandBuffer>)) {
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  encode(cb);
  [cb commit];
  [cb waitUntilCompleted];
}

int GuideChecks(const Gpu& g, RtDenoiser* dn) {
  int fails = 0;
  const Scene sc = MakeScene();
  GBuffer b = MakeGBuffer(g, true, 0.5f);
  const RtDenoiser::Set set = dn->Current();
  const RtDenoiseGuideParams p = GuideParams(sc, 0.25f, -0.125f);
  Run(g, ^(id<MTLCommandBuffer> cb) {
    dn->EncodeGuides(cb, set, b.albedo, b.normal, b.surface, b.depth, p);
  });
  const std::vector<float> al = Halves(Download(g, set.albedo, 8)), nm = Halves(Download(g, set.normal, 8)),
                           ro = Halves(Download(g, set.roughness, 2)), sp = Halves(Download(g, set.specular, 8));
  double ea = 0, en = 0, er = 0, es = 0;
  for (uint32_t y = 0; y < kH; y++)
    for (uint32_t x = 0; x < kW; x++) {
      const size_t i = y * kW + x;
      for (int k = 0; k < 3; k++) ea = std::max(ea, std::fabs(al[i * 4 + k] - b.a[i * 4 + k] / 255.0));
      const V3 n = Unit({b.n[i * 4] / 255.0 * 2 - 1, b.n[i * 4 + 1] / 255.0 * 2 - 1, b.n[i * 4 + 2] / 255.0 * 2 - 1});
      en = std::max({en, std::fabs(nm[i * 4] - n.x), std::fabs(nm[i * 4 + 1] - n.y), std::fabs(nm[i * 4 + 2] - n.z)});
      const float spec = b.su[i * 4] / 255.0f, gloss = b.su[i * 4 + 1] / 255.0f;
      er = std::max(er, double(std::fabs(ro[i] - RtDenoiseRoughness(gloss, 512.0f))));
      const double cx = 2.0 * (x - 0.25) / kW - 1, cy = 1.0 - 2.0 * (y + 0.125) / kH;
      const V3 view = Unit(World(sc, cx, cy, 1) - World(sc, cx, cy, 0));
      const float ref = RtDenoiseSpecular(spec, float(Dot(view, n)), 2.0f, 0.25f, 4.0f);
      for (int k = 0; k < 3; k++) es = std::max(es, std::fabs(sp[i * 4 + k] - double(ref)));
    }
  char what[200];
  std::snprintf(what, sizeof what,
                "guides against the CPU: albedo %.4f, normal %.4f, roughness %.4f, specular albedo %.4f at most off",
                ea, en, er, es);
  fails += Check(ea < 2e-3 && en < 2e-3 && er < 2e-3 && es < 4e-3, what);

  // The masks: the depth moved in a 10 x 10 square; hit distances in the
  // right half, none traced in the left.
  std::vector<float> d(kW * kH, 0.5f);
  for (uint32_t y = 20; y < 30; y++)
    for (uint32_t x = 10; x < 20; x++) d[y * kW + x] = 0.25f;
  id<MTLTexture> moved = Texture(g, MTLPixelFormatDepth32Float, kW, kH, MTLTextureUsageShaderRead);
  Upload(g, moved, d.data(), 4);
  std::vector<float> hd(kW * kH, 0.0f);
  for (uint32_t y = 0; y < kH; y++)
    for (uint32_t x = kW / 2; x < kW; x++) hd[y * kW + x] = 5.0f;
  id<MTLTexture> hit = Texture(g, MTLPixelFormatR16Float, kW, kH, MTLTextureUsageShaderRead);
  const std::vector<__fp16> hh = ToHalves(hd);
  Upload(g, hit, hh.data(), 2);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 16);
  Run(g, ^(id<MTLCommandBuffer> cb) {
    dn->EncodeMasks(cb, set, moved, hit, counts, 0);
  });
  const std::vector<uint8_t> re = Download(g, set.reactive, 1);
  const std::vector<float> ho = Halves(Download(g, set.hit, 2));
  int wrong = 0, hit_wrong = 0;
  for (uint32_t y = 0; y < kH; y++)
    for (uint32_t x = 0; x < kW; x++) {
      const bool in = x >= 10 && x < 20 && y >= 20 && y < 30;
      wrong += re[y * kW + x] != (in ? 255 : 0);
      const float want = x >= kW / 2 ? 5.0f : kRtReflectionSkyDistance;
      hit_wrong += ho[y * kW + x] != want;
    }
  const uint32_t counted = *static_cast<const uint32_t*>(counts.contents);
  std::snprintf(what, sizeof what,
                "masks: reactive where the depth changed (%d pixels wrong, %u counted of 100), the hit distance or "
                "the sky's (%d wrong)",
                wrong, counted, hit_wrong);
  fails += Check(wrong == 0 && counted == 100 && hit_wrong == 0, what);
  return fails;
}

// A run of frames: colour(k) each frame, motion `mv` (texture coordinates),
// the depth constant, flat guides; the output of the last.
struct Runner {
  const Gpu& g;
  RtDenoiser* dn;
  Scene sc = MakeScene();
  GBuffer b;
  id<MTLTexture> color, out;
  id<MTLBuffer> counts;
  Runner(const Gpu& gpu, RtDenoiser* d) : g(gpu), dn(d) {
    b = MakeGBuffer(g, false, 0.5f);
    color = Texture(g, MTLPixelFormatRGBA16Float, kW, kH, dn->ColorUsage() | MTLTextureUsageShaderRead);
    out = dn->NewOutput(MTLTextureUsageShaderRead);
    counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  }
  void SetMotion(float mx, float my) {
    std::vector<float> m(kW * kH * 2);
    for (size_t i = 0; i < m.size(); i += 2) m[i] = mx, m[i + 1] = my;
    const std::vector<__fp16> h = ToHalves(m);
    Upload(g, dn->Current().motion, h.data(), 4);
    const std::vector<float> d(kW * kH, 0.5f);
    Upload(g, dn->Current().depth, d.data(), 4);
  }
  // One frame of grey values `v` (kW x kH), jitter phase k; the output's
  // reds.
  std::vector<float> Frame(const std::vector<float>& v, int k, bool reset) {
    std::vector<float> rgba(kW * kH * 4);
    for (size_t i = 0; i < v.size(); i++) rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v[i], rgba[i * 4 + 3] = 1;
    const std::vector<__fp16> h = ToHalves(rgba);
    Upload(g, color, h.data(), 8);
    RtDenoiseFrame f;
    auto halton = [](int i, int base) {
      float fr = 1, r = 0;
      for (; i > 0; i /= base) fr /= float(base), r += fr * float(i % base);
      return r;
    };
    f.jitter[0] = halton(k % 8 + 1, 2) - 0.5f;
    f.jitter[1] = halton(k % 8 + 1, 3) - 0.5f;
    f.motion_scale[0] = float(kW);
    f.motion_scale[1] = float(kH);
    std::memcpy(f.view, sc.view, sizeof f.view);
    std::memcpy(f.proj, sc.proj, sizeof f.proj);
    f.depth_reversed = false;
    f.reset = reset;
    const RtDenoiser::Set set = dn->Current();
    const RtDenoiseGuideParams p = GuideParams(sc, f.jitter[0], f.jitter[1]);
    id<MTLTexture> c = color, o = out;
    GBuffer& gb = b;
    id<MTLBuffer> cn = counts;
    RtDenoiser* d = dn;
    Run(g, ^(id<MTLCommandBuffer> cb) {
      d->EncodeGuides(cb, set, gb.albedo, gb.normal, gb.surface, gb.depth, p);
      d->EncodeMasks(cb, set, gb.depth, nil, cn, 0);
      RtDenoiser::EncodeDenoise(cb, set, c, o, f);
    });
    const std::vector<float> all = Halves(Download(g, out, 8));
    std::vector<float> red(kW * kH);
    for (size_t i = 0; i < red.size(); i++) red[i] = all[i * 4];
    return red;
  }
};

void Stats(const std::vector<float>& v, double* mean, double* var) {
  double s = 0, s2 = 0;
  for (float x : v) s += x, s2 += double(x) * x;
  *mean = s / double(v.size());
  *var = s2 / double(v.size()) - *mean * *mean;
}

float Noise(uint32_t i, uint32_t k) {
  uint32_t h = i * 0x9E3779B1u ^ (k + 1) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return float(h & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
}

int DenoiseChecks(const Gpu& g, RtDenoiser* dn) {
  int fails = 0;
  char what[200];
  {
    // A static scene with noise: the mean kept, the variance down.
    Runner r(g, dn);
    r.SetMotion(0, 0);
    std::vector<float> in(kW * kH), out;
    double in_mean = 0, in_var = 0;
    for (int k = 0; k < 48; k++) {
      for (uint32_t i = 0; i < kW * kH; i++) in[i] = 0.5f + 0.2f * Noise(i, uint32_t(k));
      out = r.Frame(in, k, k == 0);
    }
    Stats(in, &in_mean, &in_var);
    double m, v;
    Stats(out, &m, &v);
    std::snprintf(what, sizeof what,
                  "static noise over 48 frames: mean %.4f (input %.4f), variance %.5f (input %.5f, %.0f%%)", m,
                  in_mean, v, in_var, 100.0 * v / in_var);
    fails += Check(std::fabs(m - 0.5) < 0.03 && v < 0.25 * in_var, what);
  }
  {
    // A pattern panning left by 2 pixels a frame (each pixel was 2 to the
    // right last frame: the motion +2 / kW), against the same pattern
    // standing still (what the denoiser makes of it without motion) and
    // panning with the motion wrong (none, or the other way): followed, no
    // ghost beyond the still pattern's error, and well under the wrong ones'.
    auto pan = [&](float speed, float motion) {
      Runner r(g, dn);
      r.SetMotion(motion / float(kW), 0.0f);
      std::vector<float> in(kW * kH), out;
      for (int k = 0; k < 32; k++) {
        for (uint32_t y = 0; y < kH; y++)
          for (uint32_t x = 0; x < kW; x++)
            in[y * kW + x] = 0.5f + 0.3f * float(std::sin(2.0 * M_PI * (double(x) + speed * k) / 16.0));
        out = r.Frame(in, k, k == 0);
      }
      // (Away from the left and right edges, where the content enters.)
      double err = 0;
      int n = 0;
      for (uint32_t y = 0; y < kH; y++)
        for (uint32_t x = 8; x < kW - 8; x++, n++) err += std::fabs(out[y * kW + x] - in[y * kW + x]);
      return err / n;
    };
    const double still = pan(0, 0), right = pan(2, 2), none = pan(2, 0), back = pan(2, -2);
    std::snprintf(what, sizeof what,
                  "a pattern panning 2 pixels a frame for 32 frames: mean error %.4f with its motion (%.4f standing "
                  "still; %.4f without motion, %.4f with it reversed)",
                  right, still, none, back);
    fails += Check(right < still + 0.01 && right < 0.5 * std::min(none, back), what);
  }
  {
    // A reset drops the history: after 24 frames of 0.2, one of 0.8.
    Runner r(g, dn);
    r.SetMotion(0, 0);
    std::vector<float> lo(kW * kH, 0.2f), hi(kW * kH, 0.8f), out;
    for (int k = 0; k < 24; k++) out = r.Frame(lo, k, k == 0);
    double before, v;
    Stats(out, &before, &v);
    out = r.Frame(hi, 24, true);
    double after;
    Stats(out, &after, &v);
    std::snprintf(what, sizeof what, "a reset: %.4f after 24 frames of 0.2, %.4f the first of 0.8 with a reset",
                  before, after);
    fails += Check(std::fabs(before - 0.2) < 0.02 && std::fabs(after - 0.8) < 0.02, what);
  }
  return fails;
}

}  // namespace

int main() {
  @autoreleasepool {
    int fails = CpuChecks();
    Gpu g{MTLCreateSystemDefaultDevice(), nil};
    if (!g.dev) {
      std::printf("no GPU: the GPU parts skipped\n");
      return fails ? 1 : 77;
    }
    g.queue = [g.dev newCommandQueue];
    if (RtSupported(g.dev))
      fails += ConeChecks(g);
    else
      std::printf("no hardware ray tracing: the cone's checks skipped\n");
    std::string why;
    float lo = 0, hi = 0;
    if (!RtDenoiseSupported(g.dev, &why, &lo, &hi)) {
      std::printf("MetalFX's denoiser is not here (%s): its checks skipped\n", why.c_str());
      return fails ? 1 : 77;
    }
    std::printf("MetalFX's denoiser takes input scales %.3g to %.3g\n", lo, hi);
    std::unique_ptr<RtDenoiser> dn = RtDenoiser::Make(g.dev, &why);
    fails += Check(dn != nullptr, ("the denoiser's kernels" + (dn ? std::string() : ": " + why)).c_str());
    if (!dn) return 1;
    const bool made = dn->Ensure(kW, kH, MTLPixelFormatRGBA16Float, true);
    fails += Check(made && dn->Current().w == kW && dn->Current().h == kH, "the scaler at 1:1, 64 x 64");
    if (!made) return 1;
    fails += GuideChecks(g, dn.get());
    fails += DenoiseChecks(g, dn.get());
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
  }
}
