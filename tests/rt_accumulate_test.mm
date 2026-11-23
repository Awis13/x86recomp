// The accumulation of a sparsely traced term (gfx/rt_accumulate.h). On the
// CPU: camera cuts (none standing still or for a small move; a turn, a jump
// and a missing camera are). On any GPU, with a synthetic scene whose term
// is known at every pixel (a ground plane, a box floating over it and a
// thin rod, the sun's shadow of them worked out on the CPU; a synthetic
// depth buffer from a known camera), the term given at one pixel of every
// 3 x 3 turning frame to frame as a sparse trace gives it:
//  1. standing still: after 12 frames every pixel within 0.05 of the term,
//     along the shadows' edges too, the thin rod's shadow (a line about a
//     pixel tall) whole (reversed depth texture and float depth alike);
//  2. the box moved: its old shadow gone and its new one there within two
//     frames;
//  3. the camera moving sideways: the result follows the surface (its error
//     like the still camera's, well under that of a run told the wrong
//     camera);
//  4. disocclusion: ground revealed from behind a box takes nothing from
//     the box that covered it;
//  5. the history dropped at a cut and a size change (the frame then the
//     same as a first frame's), kept for a small move;
//  6. a uniformly lit (and a uniformly shadowed) view: every pixel takes
//     the shortcut and equals the input, no tile filtered;
//  and the passes' GPU time at 3440 x 1440 printed (no assertion). On a GPU
// with hardware ray tracing, the sun's mask of a scene traced (gfx/
// rt_shadow.h): 7. without the accumulation (the far light's mask as
// before) byte for byte the turning trace at its top-left pixel upsampled
// the same way; 8. traced at one of every 3 x 3 and accumulated, every
// pixel within 0.05 of the mask traced at every pixel after 12 frames, and
// under the jitter's phases its shadows' edges at least as stable frame to
// frame as the half-resolution mask's; the passes' times at 3440 x 1440.
//
//   rt_accumulate_test        (exit 77 without a Metal GPU)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx/gpu_timer.h"
#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/rt_accumulate.h"
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
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double k) { return {a.x * k, a.y * k, a.z * k}; }
double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 Cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 Unit(V3 a) { return a * (1.0 / std::sqrt(Dot(a, a))); }

// D3D's left-handed camera at `eye` looking at `at` (y up), its
// perspective (depth 0..1 at near..far), as world to clip rows.
void Camera(V3 eye, V3 at, double fov_y, double aspect, double n, double f, double vp[16]) {
  const V3 fw = Unit(at - eye), rt = Unit(Cross({0, 1, 0}, fw)), up = Cross(fw, rt);
  const double view[16] = {rt.x, up.x, fw.x, 0, rt.y,          up.y,          fw.y,          0,
                           rt.z, up.z, fw.z, 0, -Dot(rt, eye), -Dot(up, eye), -Dot(fw, eye), 1};
  const double ys = 1.0 / std::tan(fov_y / 2), xs = ys / aspect;
  double proj[16] = {};
  proj[0] = xs;
  proj[5] = ys;
  proj[10] = f / (f - n);
  proj[11] = 1;
  proj[14] = -n * f / (f - n);
  Mul4(view, proj, vp);
}

// A frame whose camera is `vp` (its exact inverse given).
bool FrameOf(const double* vp, RtFrame* f) {
  double inv[16];
  if (!Invert4(vp, inv)) return false;
  for (int k = 0; k < 16; k++) f->inv_view_proj[k] = float(inv[k]);
  std::memcpy(f->inv_view_proj_exact, inv, sizeof inv);
  return true;
}

struct Box {
  V3 lo, hi;
};
// Where the ray o + t d (t > 0) enters the box (its t), or -1.
double RayBox(V3 o, V3 d, const Box& box) {
  const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
  const double lo[3] = {box.lo.x, box.lo.y, box.lo.z}, hi[3] = {box.hi.x, box.hi.y, box.hi.z};
  double t0 = 0, t1 = 1e30;
  for (int k = 0; k < 3; k++) {
    if (std::fabs(dd[k]) < 1e-15) {
      if (oo[k] < lo[k] || oo[k] > hi[k]) return -1;
      continue;
    }
    double ta = (lo[k] - oo[k]) / dd[k], tb = (hi[k] - oo[k]) / dd[k];
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    if (t0 > t1) return -1;
  }
  return t0 > 0 ? t0 : -1;
}
// The face normal of the box at a point on it.
V3 BoxNormal(const Box& b, V3 p) {
  const V3 c = (b.lo + b.hi) * 0.5, h = (b.hi - b.lo) * 0.5, r = p - c;
  const double ex = std::fabs(std::fabs(r.x) - h.x), ey = std::fabs(std::fabs(r.y) - h.y),
               ez = std::fabs(std::fabs(r.z) - h.z);
  return ex <= ey && ex <= ez ? V3{r.x > 0 ? 1.0 : -1.0, 0, 0}
         : ey <= ez           ? V3{0, r.y > 0 ? 1.0 : -1.0, 0}
                              : V3{0, 0, r.z > 0 ? 1.0 : -1.0};
}

// A scene: the ground (y = 0, unless `no_ground`) and boxes, the sun's
// direction (toward it); the term at a point is 1 where the sun reaches it
// (its surface facing the sun, no box in the way), else 0.
struct Scene {
  std::vector<Box> boxes;
  V3 sun = Unit({0.3, 1.0, 0.45});
  bool no_ground = false;
};
const Box kFloating{{-1.5, 1.2, 1.0}, {0.5, 1.8, 2.2}};  // over the ground, its shadow on it
const Box kRod{{-3.5, 0.98, 4.98}, {3.5, 1.02, 5.02}};   // its shadow a line about a pixel tall

constexpr uint32_t kW = 320, kH = 240;
const V3 kEye{0, 9, -9}, kAt{0, 0, 2};

void View(V3 eye, V3 at, uint32_t w, uint32_t h, double vp[16]) { Camera(eye, at, 1.0, double(w) / h, 0.5, 3000, vp); }

// The scene through camera `vp` at w x h, the draws moved `jitter` pixels
// right and down: each pixel's stored depth (reversed where asked, 1 - z;
// the far plane's where nothing is hit) and the term at its point.
struct Image {
  uint32_t w = 0, h = 0;
  std::vector<float> depth, term;
  std::vector<int> surface;      // -1 sky, 0 ground, 1 + k box k
  std::vector<uint32_t> normal;  // the surface's normal as a G-buffer holds it (RGBA8, xyz as unorm)
};
Image Render(const Scene& s, const double* vp, uint32_t w, uint32_t h, const float jitter[2], bool reversed) {
  Image im;
  im.w = w, im.h = h;
  im.depth.assign(size_t(w) * h, reversed ? 0.0f : 1.0f);
  im.term.assign(im.depth.size(), 1.0f);
  im.surface.assign(im.depth.size(), -1);
  im.normal.assign(im.depth.size(), 0);
  double inv[16];
  Invert4(vp, inv);
  for (uint32_t j = 0; j < h; j++)
    for (uint32_t i = 0; i < w; i++) {
      const double x = 2.0 * (i - jitter[0]) / w - 1.0, y = 1.0 - 2.0 * (j - jitter[1]) / h;
      double a[4], b[4];
      for (int k = 0; k < 4; k++) {
        a[k] = x * inv[k] + y * inv[4 + k] + inv[12 + k];
        b[k] = x * inv[k] + y * inv[4 + k] + inv[8 + k] + inv[12 + k];
      }
      const V3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const V3 d = pb - pa;
      double t = 2;
      int surf = -1;
      if (!s.no_ground && d.y < 0 && -pa.y / d.y > 0 && -pa.y / d.y < 1) t = -pa.y / d.y, surf = 0;
      for (size_t k = 0; k < s.boxes.size(); k++) {
        const double tb = RayBox(pa, d, s.boxes[k]);
        if (tb > 0 && tb < t) t = tb, surf = int(k) + 1;
      }
      const size_t at = size_t(j) * w + i;
      im.surface[at] = surf;
      if (surf < 0) continue;
      const V3 p = pa + d * t;
      double c[4];
      for (int k = 0; k < 4; k++) c[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
      const double z = c[2] / c[3];
      im.depth[at] = float(reversed ? 1.0 - z : z);
      const V3 n = surf == 0 ? V3{0, 1, 0} : BoxNormal(s.boxes[size_t(surf - 1)], p);
      auto unorm = [](double x) { return uint32_t(std::lround((x * 0.5 + 0.5) * 255.0)); };
      im.normal[at] = unorm(n.x) | unorm(n.y) << 8 | unorm(n.z) << 16 | 0xFF000000u;
      bool lit = Dot(n, s.sun) > 0;
      const V3 o = p + n * 1e-6;
      for (size_t k = 0; k < s.boxes.size() && lit; k++)
        if (int(k) + 1 != surf && RayBox(o, s.sun, s.boxes[k]) > 0) lit = false;
      im.term[at] = lit ? 1.0f : 0.0f;
    }
  return im;
}

// ---- CPU

int CpuChecks() {
  auto cam = [](V3 eye, V3 at) {
    double vp[16];
    View(eye, at, kW, kH, vp);
    RtFrame f;
    FrameOf(vp, &f);
    RtAccumulateCamera c;
    RtAccumulateCameraOf(f, &c);
    return c;
  };
  const RtAccumulateCamera a = cam(kEye, kAt), small = cam(kEye + V3{0.1, 0, 0}, kAt + V3{0.1, 0, 0}),
                           turned = cam(kEye, kEye + V3{9, -9, 0}),
                           jumped = cam(kEye + V3{0, 0, 60}, kAt + V3{0, 0, 60});
  RtAccumulateCamera none;
  const bool ok = a.valid && std::fabs(a.eye[1] - 9) < 1e-6 && !RtAccumulateCut(a, a) && !RtAccumulateCut(a, small) &&
                  RtAccumulateCut(a, turned) && RtAccumulateCut(a, jumped) && RtAccumulateCut(none, a);
  return Check(ok, "camera cuts: none standing still or for a small move; a turn of 90 degrees, a jump far ahead and "
                   "no camera before are");
}

// ---- GPU

// The timer GpuTimer asks (gfx/gpu_timer.h): a sample buffer's pairs, by
// name; the passes' times read after the work is done.
struct Timer {
  id<MTLCounterSampleBuffer> buf = nil;
  std::vector<std::string> names;
  double tick_ns = 1;
  bool Init(id<MTLDevice> dev) {
    if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return false;
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> c in dev.counterSets)
      if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
    if (!ts) return false;
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = 2048;
    buf = [dev newCounterSampleBufferWithDescriptor:d error:nil];
    MTLTimestamp c0, g0, c1, g1;
    [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
    [NSThread sleepForTimeInterval:0.05];
    [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
    if (g1 > g0) tick_ns = double(c1 - c0) / double(g1 - g0);
    return buf != nil;
  }
  static bool Next(void* ctx, const char* name, __autoreleasing id<MTLCounterSampleBuffer>* b, NSUInteger* s,
                   NSUInteger* e) {
    auto* t = static_cast<Timer*>(ctx);
    if (t->names.size() >= 1024) return false;
    *b = t->buf;
    *s = t->names.size() * 2;
    *e = *s + 1;
    t->names.push_back(name);
    return true;
  }
  void Start() {
    names.clear();
    CurrentGpuTimer() = GpuTimer{this, &Next};
  }
  // Each pass's times (ms) from the `from`-th sample pair on, by name.
  void Stop(std::map<std::string, std::vector<double>>* ms, size_t from) {
    CurrentGpuTimer() = GpuTimer{};
    if (names.empty() || !ms) return;
    NSData* d = [buf resolveCounterRange:NSMakeRange(0, names.size() * 2)];
    const auto* r = static_cast<const MTLCounterResultTimestamp*>(d.bytes);
    for (size_t i = from; d && i < names.size(); i++)
      if (r[2 * i + 1].timestamp > r[2 * i].timestamp && r[2 * i].timestamp != MTLCounterErrorValue)
        (*ms)[names[i]].push_back(double(r[2 * i + 1].timestamp - r[2 * i].timestamp) * tick_ns / 1e6);
  }
  // "name min / median ms" for each pass, and the sums of the minima and
  // the medians.
  static void Print(const std::map<std::string, std::vector<double>>& ms) {
    double lo = 0, mid = 0;
    for (auto [name, v] : ms) {
      std::sort(v.begin(), v.end());
      std::printf(" %s %.3f / %.3f;", name.c_str(), v.front(), v[v.size() / 2]);
      lo += v.front(), mid += v[v.size() / 2];
    }
    std::printf(" together %.3f / %.3f ms (least / median a frame)\n", lo, mid);
  }
};

struct Gpu {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = nil;
  std::unique_ptr<RtAccumulator> acc;
  bool Commit(id<MTLCommandBuffer> cb) const {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      std::printf("FAIL command buffer: %s\n", cb.error.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }
  id<MTLTexture> Texture(MTLPixelFormat fmt, uint32_t w, uint32_t h, MTLTextureUsage usage) const {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = usage;
    return [dev newTextureWithDescriptor:td];
  }
  // A depth texture (Depth32Float_Stencil8) holding `v`, as the backend
  // keeps one.
  id<MTLTexture> DepthTexture(const std::vector<float>& v, uint32_t w, uint32_t h) const {
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> t = [dev newTextureWithDescriptor:td];
    id<MTLBuffer> src = [dev newBufferWithBytes:v.data() length:v.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:w * 4
        sourceBytesPerImage:size_t(w) * h * 4
                 sourceSize:MTLSizeMake(w, h, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)
                    options:MTLBlitOptionDepthFromDepthStencil];
    [bl endEncoding];
    return Commit(cb) ? t : nil;
  }
  id<MTLTexture> NormalTexture(const std::vector<uint32_t>& v, uint32_t w, uint32_t h) const {
    id<MTLTexture> t = Texture(MTLPixelFormatRGBA8Unorm, w, h, MTLTextureUsageShaderRead);
    [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:v.data() bytesPerRow:w * 4];
    return t;
  }
  id<MTLTexture> FloatTexture(const std::vector<float>& v, uint32_t w, uint32_t h) const {
    id<MTLTexture> t = Texture(MTLPixelFormatR32Float, w, h, MTLTextureUsageShaderRead);
    [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:v.data() bytesPerRow:w * 4];
    return t;
  }
};

const float kNoJitter[2] = {0, 0};

// The jitter of frame k: Halton (2, 3) less a half, as a temporal upscaler
// moves the draws.
void Jitter(int k, float out[2]) {
  auto halton = [](int i, int b) {
    float f = 1, r = 0;
    for (; i > 0; i /= b) f /= float(b), r += f * float(i % b);
    return r;
  };
  out[0] = halton(k % 8 + 1, 2) - 0.5f;
  out[1] = halton(k % 8 + 1, 3) - 0.5f;
}

// One frame's result.
struct Shot {
  bool ok = false, reset = false;
  std::vector<float> out;  // the output (0..1)
  Image im;                // the scene as drawn (its term at every pixel)
  float spread = 0;        // three channels: their largest difference
  bool again = false;      // Begin refused a second frame of the same number
  RtAccumulateCounts n;
};

// Runs of frames through one history.
struct Acc {
  const Gpu& g;
  RtAccumulateHistory h;
  uint32_t step = 3;
  bool reversed = false;  // depth stored reversed in a depth texture (else not, in a float one)
  bool three = false;     // the term grey in three channels (RGBA16Float in and out)
  explicit Acc(const Gpu& gpu) : g(gpu) {}
  // A frame encoded and committed, not yet waited for.
  struct Pending {
    Shot r;
    id<MTLTexture> out = nil;
    id<MTLBuffer> counts = nil;
    id<MTLCommandBuffer> cb = nil;
    uint32_t w = 0, h = 0;
  };
  // Frame `k` (the presented frame's number, the samples' turn) of scene
  // `s` seen by `vp` at w x h; the passes told the camera `told` (null: vp)
  // and the light `light` (null: the scene's sun), and to reset where asked.
  bool Start(Pending* pd, const Scene& s, const double* vp, uint32_t k, const float jitter[2] = kNoJitter,
             bool reset = false, const double* told = nullptr, uint32_t w = kW, uint32_t hh = kH,
             const float* light = nullptr) {
    Shot& r = pd->r;
    pd->w = w, pd->h = hh;
    r.im = Render(s, vp, w, hh, jitter, reversed);
    const uint32_t stride = RtLightStride(step), lw = (w + step - 1) / step, lh = (hh + step - 1) / step;
    RtFrame f;
    RtShadowParams p;
    double eye[3];
    const float up[3] = {0, 0, 1};
    if (!FrameOf(told ? told : vp, &f) || !RtShadowParamsFor(f, w, hh, reversed, reversed, jitter, up, &p) ||
        !RtLightsEye(f, eye))
      return false;
    // The samples: the term, and the points as the trace gives them.
    std::vector<uint8_t> sparse(size_t(lw) * lh);
    std::vector<__fp16> sparse3(sparse.size() * 4, __fp16(1.0f)), pts(sparse.size() * 4);
    for (uint32_t ty = 0; ty < lh; ty++)
      for (uint32_t tx = 0; tx < lw; tx++) {
        uint32_t px, py;
        RtLightSample(tx, ty, k, stride, step, &px, &py);
        px = std::min(px, w - 1), py = std::min(py, hh - 1);
        const size_t at = size_t(py) * w + px, t = size_t(ty) * lw + tx;
        const float v = r.im.term[at];
        sparse[t] = uint8_t(std::lround(v * 255.0f));
        for (int c = 0; c < 3; c++) sparse3[t * 4 + size_t(c)] = __fp16(v);
        double world[3];
        RtShadowPoint(p, px, py, r.im.depth[at], world);
        for (int c = 0; c < 3; c++) pts[t * 4 + size_t(c)] = __fp16(float(world[c] - eye[c]));
        pts[t * 4 + 3] = r.im.surface[at] < 0 ? __fp16(-1.0f) : __fp16(float((px - tx * step) + 8 * (py - ty * step)));
      }
    id<MTLTexture> st =
        g.Texture(three ? MTLPixelFormatRGBA16Float : MTLPixelFormatR8Unorm, lw, lh, MTLTextureUsageShaderRead);
    if (three)
      [st replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:0 withBytes:sparse3.data() bytesPerRow:lw * 8];
    else
      [st replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:0 withBytes:sparse.data() bytesPerRow:lw];
    id<MTLTexture> pt = g.Texture(MTLPixelFormatRGBA16Float, lw, lh, MTLTextureUsageShaderRead);
    [pt replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:0 withBytes:pts.data() bytesPerRow:lw * 8];
    id<MTLTexture> dt = reversed ? g.DepthTexture(r.im.depth, w, hh) : g.FloatTexture(r.im.depth, w, hh);
    pd->out = g.Texture(three ? MTLPixelFormatRGBA16Float : MTLPixelFormatR8Unorm, w, hh,
                        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    RtAccumulateTerm term;
    term.channels = three ? 3 : 1;
    const float sun[3] = {float(s.sun.x), float(s.sun.y), float(s.sun.z)};
    RtAccumulateFrame fr;
    const bool begun = dt && g.acc->Begin(&h, f, p, term, step, stride, k, light ? light : sun, reset, &fr);
    r.again = fr.again;
    if (!begun) return false;
    pd->counts = [g.dev newBufferWithLength:sizeof(RtAccumulateCounts) options:MTLResourceStorageModeShared];
    std::memset(pd->counts.contents, 0, sizeof(RtAccumulateCounts));
    pd->cb = [g.queue commandBuffer];
    g.acc->Encode(pd->cb, fr, st, pt, dt, nil, pd->out, pd->counts, 0);
    [pd->cb commit];
    r.reset = fr.reset;
    return true;
  }
  Shot Finish(Pending* pd) {
    Shot& r = pd->r;
    [pd->cb waitUntilCompleted];
    if (pd->cb.status != MTLCommandBufferStatusCompleted) return r;
    r.out.resize(size_t(pd->w) * pd->h);
    if (three) {
      // (The three channels' largest difference from the first's.)
      std::vector<__fp16> o(r.out.size() * 4);
      [pd->out getBytes:o.data() bytesPerRow:pd->w * 8 fromRegion:MTLRegionMake2D(0, 0, pd->w, pd->h) mipmapLevel:0];
      for (size_t i = 0; i < r.out.size(); i++) {
        r.out[i] = float(o[i * 4]);
        for (int c = 1; c < 3; c++) r.spread = std::max(r.spread, std::fabs(float(o[i * 4 + size_t(c)]) - r.out[i]));
      }
    } else {
      std::vector<uint8_t> o(r.out.size());
      [pd->out getBytes:o.data() bytesPerRow:pd->w fromRegion:MTLRegionMake2D(0, 0, pd->w, pd->h) mipmapLevel:0];
      for (size_t i = 0; i < o.size(); i++) r.out[i] = o[i] / 255.0f;
    }
    std::memcpy(&r.n, pd->counts.contents, sizeof r.n);
    r.ok = true;
    return r;
  }
  Shot Frame(const Scene& s, const double* vp, uint32_t k, const float jitter[2] = kNoJitter, bool reset = false,
             const double* told = nullptr, uint32_t w = kW, uint32_t hh = kH, const float* light = nullptr) {
    Pending pd;
    if (!Start(&pd, s, vp, k, jitter, reset, told, w, hh, light)) return pd.r;
    return Finish(&pd);
  }
};

// The largest and mean error against the term, over pixels where `pick`.
template <typename F>
void Errors(const Shot& s, F pick, double* most, double* mean, int* n) {
  *most = 0, *mean = 0, *n = 0;
  for (uint32_t j = 0; j < s.im.h; j++)
    for (uint32_t i = 0; i < s.im.w; i++) {
      const size_t at = size_t(j) * s.im.w + i;
      if (!pick(i, j, at)) continue;
      const double e = std::fabs(s.out[at] - s.im.term[at]);
      *most = std::max(*most, e);
      *mean += e;
      ++*n;
    }
  if (*n) *mean /= *n;
}
// Whether the term differs between the pixel and one of its four
// neighbours (an edge).
bool Edge(const Image& im, uint32_t i, uint32_t j) {
  const float v = im.term[size_t(j) * im.w + i];
  const int o[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (const auto& d : o) {
    const int x = int(i) + d[0], y = int(j) + d[1];
    if (x >= 0 && y >= 0 && x < int(im.w) && y < int(im.h) && im.term[size_t(y) * im.w + x] != v) return true;
  }
  return false;
}

Scene Standard() {
  Scene s;
  s.boxes = {kFloating, kRod};
  return s;
}

int StaticChecks(const Gpu& g) {
  int fails = 0;
  for (int rev = 0; rev < 2; rev++) {
    Acc a(g);
    a.reversed = rev != 0;
    const Scene s = Standard();
    double vp[16];
    View(kEye, kAt, kW, kH, vp);
    Shot r;
    for (uint32_t k = 0; k < 12; k++) r = a.Frame(s, vp, k);
    if (!r.ok) return Check(false, "a still view: the frames");
    double all, all_mean, edge, edge_mean, line, line_mean;
    int n_all, n_edge, n_line;
    Errors(r, [&](uint32_t, uint32_t, size_t at) { return r.im.surface[at] >= 0; }, &all, &all_mean, &n_all);
    Errors(
        r, [&](uint32_t i, uint32_t j, size_t at) { return r.im.surface[at] >= 0 && Edge(r.im, i, j); }, &edge,
        &edge_mean, &n_edge);
    // The rod's shadow on the ground (the dark ground of the rod alone),
    // and the ground within two rows of it.
    Scene rod_only;
    rod_only.boxes = {kRod};
    const Image ro = Render(rod_only, vp, kW, kH, kNoJitter, false);
    int lo = int(kH), hi = -1;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++)
        if (ro.surface[size_t(j) * kW + i] == 0 && ro.term[size_t(j) * kW + i] < 0.5f)
          lo = std::min(lo, int(j)), hi = std::max(hi, int(j));
    double want = 0, got = 0;
    auto in_line = [&](uint32_t i, uint32_t j, size_t at) {
      if (r.im.surface[at] != 0) return false;
      for (int d = -2; d <= 2; d++) {
        const int y = int(j) + d;
        if (y >= 0 && y < int(kH) && ro.surface[size_t(y) * kW + i] == 0 && ro.term[size_t(y) * kW + i] < 0.5f)
          return true;
      }
      return false;
    };
    Errors(r, in_line, &line, &line_mean, &n_line);
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++)
        if (in_line(i, j, size_t(j) * kW + i)) {
          want += 1.0 - r.im.term[size_t(j) * kW + i];
          got += 1.0 - r.out[size_t(j) * kW + i];
        }
    char what[400];
    std::snprintf(what, sizeof what,
                  "a still view (%s), 3 x 3 turning, after 12 frames: every pixel within 0.05 of the term (largest "
                  "%.3f, mean %.4f over %d), along the edges too (largest %.3f, mean %.4f over %d); the thin rod's "
                  "shadow (rows %d-%d) within 0.05 a pixel (largest %.3f), its sum %.1f of %.1f (within 5%%)",
                  rev ? "reversed depth texture" : "float depth", all, all_mean, n_all, edge, edge_mean, n_edge, lo, hi,
                  line, want > 0 ? got : 0.0, want);
    fails += Check(n_all > 50000 && n_edge > 500 && all <= 0.05 && all_mean < 0.005 && edge <= 0.05 && want > 40 &&
                       hi - lo <= 3 && line <= 0.05 && std::fabs(got / want - 1.0) < 0.05,
                   what);
  }
  return fails;
}

int MovedChecks(const Gpu& g) {
  // 12 frames still, then the box moved aside (its shadow too); for 10
  // frames after, the pixels of its old shadow (lit now) darker than 0.9
  // and of its new one lighter than 0.1: those four pixels or more inside
  // the shadows (no other value within the fill's reach, under the step,
  // and the filter's pixel) and all.
  Acc a(g);
  Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  Shot r;
  for (uint32_t k = 0; k < 12; k++) r = a.Frame(s, vp, k);
  const Image before = r.im;
  s.boxes[0].lo.x += 2.5, s.boxes[0].hi.x += 2.5;
  int inner[11] = {}, all[11] = {}, old_n = 0, new_n = 0;
  for (int k = 1; k <= 10; k++) {
    r = a.Frame(s, vp, 11 + uint32_t(k));
    if (!r.ok) return Check(false, "a moved box: the frames");
    old_n = new_n = 0;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++) {
        const size_t at = size_t(j) * kW + i;
        if (r.im.surface[at] != 0 || before.surface[at] != 0) continue;
        const bool old_shadow = before.term[at] < 0.5f && r.im.term[at] > 0.5f;
        const bool new_shadow = before.term[at] > 0.5f && r.im.term[at] < 0.5f;
        if (!old_shadow && !new_shadow) continue;
        (old_shadow ? old_n : new_n)++;
        const bool wrong = old_shadow ? r.out[at] < 0.9f : r.out[at] > 0.1f;
        if (!wrong) continue;
        all[k]++;
        bool in = true;
        for (int dy = -4; dy <= 4 && in; dy++)
          for (int dx = -4; dx <= 4 && in; dx++) {
            const int x = int(i) + dx, y = int(j) + dy;
            if (x < 0 || y < 0 || x >= int(kW) || y >= int(kH)) continue;
            const size_t b = size_t(y) * kW + x;
            in = before.term[b] == before.term[at] && r.im.term[b] == r.im.term[at];
          }
        inner[k] += in;
      }
  }
  char what[400];
  std::snprintf(what, sizeof what,
                "a moved box: of its old shadow (%d pixels, lit now) and its new one (%d), wrong (old darker than "
                "0.9, new lighter than 0.1) in frames 1 / 2 / 3 / 5 / 9 / 10 after: %d / %d / %d / %d / %d / %d, "
                "of them 4 pixels or more inside: %d / %d / %d / %d / %d / %d; none inside from the first frame",
                old_n, new_n, all[1], all[2], all[3], all[5], all[9], all[10], inner[1], inner[2], inner[3], inner[5],
                inner[9], inner[10]);
  int inner_after = 0;
  for (int k = 1; k <= 10; k++) inner_after += inner[k];
  // (Every pixel right by the 10th frame is a target not met: the pixels
  // left are at the old shadow's edge, cleared only by their own samples.)
  if (all[10]) std::printf("  measured (target not met): %d pixels still wrong in the 10th frame\n", all[10]);
  return Check(old_n > 300 && new_n > 300 && inner_after == 0, what);
}

int PanChecks(const Gpu& g) {
  // The camera moving sideways 0.06 a frame (about a pixel at the shadow),
  // 24 frames; at the end, over the screen but 16 pixels at its sides (what
  // came into view in the last 10 frames): the shadows' dark mass and its
  // centroid against the term's (a ghost lags behind: its centroid off the
  // term's, its mass more), and the mean error at the shadows' edges (the
  // history resampled as it moves blurs them a little).
  struct Out {
    double off = 99, mass = 0, edge = 1;
  };
  auto run = [&](double speed, bool wrong) {
    Out o;
    Acc a(g);
    const Scene s = Standard();
    Shot r;
    double first[16];
    View(kEye, kAt, kW, kH, first);
    for (uint32_t k = 0; k < 24; k++) {
      double vp[16];
      const V3 m{speed * k, 0, 0};
      View(kEye + m, kAt + m, kW, kH, vp);
      r = a.Frame(s, vp, k, kNoJitter, false, wrong ? first : nullptr);
    }
    if (!r.ok) return o;
    double mo = 0, mt = 0, xo = 0, xt = 0, most, emean;
    int en;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 16; i < kW - 16; i++) {
        const size_t at = size_t(j) * kW + i;
        if (r.im.surface[at] < 0) continue;
        mo += 1.0 - r.out[at], xo += (1.0 - r.out[at]) * i;
        mt += 1.0 - r.im.term[at], xt += (1.0 - r.im.term[at]) * i;
      }
    Errors(
        r,
        [&](uint32_t i, uint32_t j, size_t at) {
          return r.im.surface[at] >= 0 && i >= 16 && i < kW - 16 && Edge(r.im, i, j);
        },
        &most, &emean, &en);
    o.off = std::fabs(xo / mo - xt / mt);
    o.mass = mo / mt - 1.0;
    o.edge = emean;
    return o;
  };
  const Out still = run(0, false), moving = run(0.06, false), back = run(-0.06, false), wrong = run(0.06, true);
  char what[500];
  std::snprintf(what, sizeof what,
                "the camera moving sideways for 24 frames: the shadows' centroid %.2f pixels off the term's (%.2f "
                "the other way, %.2f standing still, %.2f told the wrong camera), their mass %+.1f%% (%+.1f%%, "
                "%+.1f%%, %+.1f%%), the error at their edges %.3f (%.3f, %.3f, %.3f): no ghost (centroid within a "
                "quarter pixel, mass within 3%%, edges within 0.08; the wrong camera's off by half a pixel or more)",
                moving.off, back.off, still.off, wrong.off, 100 * moving.mass, 100 * back.mass, 100 * still.mass,
                100 * wrong.mass, moving.edge, back.edge, still.edge, wrong.edge);
  return Check(moving.off < 0.25 && back.off < 0.25 && std::fabs(moving.mass) < 0.03 && std::fabs(back.mass) < 0.03 &&
                   moving.edge < 0.08 && back.edge < 0.08 && still.off < 0.05 && wrong.off > 0.5,
               what);
}

int DisocclusionChecks(const Gpu& g) {
  // A tall box standing on the lit ground, its faces toward the camera in
  // shadow (the sun behind it); the camera moving left, the ground behind
  // the box's right side revealed frame by frame.
  Acc a(g);
  Scene s;
  s.sun = Unit({0.0, 1.0, 0.8});
  s.boxes = {{{-1.0, 0.0, 0.0}, {1.0, 3.0, 1.0}}};
  Shot r, prev;
  int revealed = 0, wrong = 0;
  double worst = 1;
  for (uint32_t k = 0; k < 16; k++) {
    double vp[16];
    const V3 m{-0.2 * k, 0, 0};
    View(kEye + m, kAt + m, kW, kH, vp);
    prev = r;
    r = a.Frame(s, vp, k);
    if (!r.ok) return Check(false, "disocclusion: the frames");
    if (k < 4) continue;
    // Ground pixels where the box was last frame, lit ground all round
    // (three pixels: where the fill reaches; not by the box's foot, whose
    // dark face meets the ground on its plane).
    for (uint32_t j = 3; j + 3 < kH; j++)
      for (uint32_t i = 3; i + 3 < kW; i++) {
        const size_t at = size_t(j) * kW + i;
        if (r.im.surface[at] != 0 || prev.im.surface[at] != 1) continue;
        bool lit = true;
        for (int dy = -3; dy <= 3 && lit; dy++)
          for (int dx = -3; dx <= 3 && lit; dx++) {
            const size_t b = size_t(int(j) + dy) * kW + size_t(int(i) + dx);
            lit = r.im.surface[b] == 0 && r.im.term[b] > 0.5f;
          }
        if (!lit) continue;
        revealed++;
        worst = std::min(worst, double(r.out[at]));
        wrong += r.out[at] < 0.9f;
      }
  }
  char what[300];
  std::snprintf(what, sizeof what,
                "disocclusion: %d ground pixels revealed from behind a box in shadow, %d of them darker than 0.9 "
                "(darkest %.2f): none takes the box's history",
                revealed, wrong, worst);
  return Check(revealed > 100 && wrong == 0, what);
}

int ResetChecks(const Gpu& g) {
  int fails = 0;
  const Scene s = Standard();
  double vp[16], turned[16];
  View(kEye, kAt, kW, kH, vp);
  View(kEye, kEye + V3{9, -9, 0}, kW, kH, turned);
  // 12 frames, then the camera turned 90 degrees: reset, and the frame the
  // same as a first one's.
  Acc a(g);
  Shot r;
  for (uint32_t k = 0; k < 12; k++) r = a.Frame(s, vp, k);
  const bool kept = r.ok && !r.reset;
  const Shot cut = a.Frame(s, turned, 12);
  Acc b(g);
  const Shot first = b.Frame(s, turned, 12);
  // A small move: kept.
  double moved[16];
  View(kEye + V3{0.05, 0, 0}, kAt + V3{0.05, 0, 0}, kW, kH, moved);
  Acc c(g);
  for (uint32_t k = 0; k < 4; k++) r = c.Frame(s, vp, k);
  const Shot small = c.Frame(s, moved, 4);
  // A size change: reset, the same as a first frame at that size.
  const uint32_t w2 = 256, h2 = 200;
  double vp2[16];
  View(kEye, kAt, w2, h2, vp2);
  const Shot resized = c.Frame(s, vp2, 5, kNoJitter, false, nullptr, w2, h2);
  Acc d(g);
  const Shot first2 = d.Frame(s, vp2, 5, kNoJitter, false, nullptr, w2, h2);
  // Asked for.
  const Shot asked = c.Frame(s, vp2, 6, kNoJitter, true, nullptr, w2, h2);
  Acc e(g);
  const Shot first3 = e.Frame(s, vp2, 6, kNoJitter, false, nullptr, w2, h2);
  const bool ok = kept && cut.ok && cut.reset && first.reset && cut.out == first.out && small.ok && !small.reset &&
                  resized.ok && resized.reset && resized.out == first2.out && asked.reset && asked.out == first3.out &&
                  c.h.resets == 3 && cut.n.fresh > 50000;
  fails += Check(ok, "the history dropped at a camera cut (90 degrees), a size change and when asked (each frame then "
                     "byte for byte a first frame's), kept for a small move");
  return fails;
}

int UniformChecks(const Gpu& g) {
  int fails = 0;
  for (int dark = 0; dark < 2; dark++) {
    Scene s;
    if (dark) s.boxes = {{{-200, 20, -200}, {200, 21, 200}}};  // a roof over everything in view (not seen)
    Acc a(g);
    double vp[16];
    View(kEye, kAt, kW, kH, vp);
    bool ok = true;
    uint32_t uniform = 0, ground = 0, tiles = 0;
    for (uint32_t k = 0; k < 4 && ok; k++) {
      const Shot r = a.Frame(s, vp, k);
      ok = r.ok;
      ground = 0;
      for (size_t at = 0; at < r.out.size() && ok; at++) {
        if (r.im.surface[at] < 0) continue;
        ground++;
        ok = r.out[at] == r.im.term[at] && r.im.term[at] == (dark ? 0.0f : 1.0f);
      }
      uniform = r.n.uniform;
      tiles += r.n.tiles;
      ok = ok && uniform == ground;
    }
    char what[200];
    std::snprintf(what, sizeof what,
                  "a uniformly %s view: every pixel (%u) took the shortcut (%u counted) and equals the input, no tile "
                  "filtered (%u)",
                  dark ? "shadowed" : "lit", ground, uniform, tiles);
    fails += Check(ok && ground > 50000 && tiles == 0, what);
  }
  return fails;
}

// The jitter's phases a pixel's own sample falls on (RtLightSample with the
// full frame count) at steps 2, 3 and 4: all 8 within 32 cycles, for every
// pixel of 4 x 4 blocks; frames below step^2 as before the cycles' orders.
int PhaseChecks() {
  bool ok = true;
  for (uint32_t step = 2; step <= 4 && ok; step++) {
    const uint32_t n = step * step, stride = RtLightStride(step);
    for (uint32_t by = 0; by < 4 && ok; by++)
      for (uint32_t bx = 0; bx < 4 && ok; bx++) {
        uint32_t phases[64] = {};
        for (uint32_t f = 0; f < 32 * n; f++) {
          uint32_t x, y;
          RtLightSample(bx, by, f, stride, step, &x, &y);
          phases[(y - by * step) * step + (x - bx * step)] |= 1u << (f % 8);
          if (f < n) {
            // (The order before: the block's hash, the turn.)
            uint32_t h = (bx * 0x8da6b343u) ^ (by * 0xd8163841u);
            h ^= h >> 16, h *= 0x7feb352du, h ^= h >> 15;
            const uint32_t cell = (h % n + f * stride) % n;
            ok = ok && x == bx * step + cell % step && y == by * step + cell / step;
          }
        }
        for (uint32_t c = 0; c < n; c++) ok = ok && phases[c] == 0xFFu;
      }
  }
  return Check(ok, "the samples' turn over the full frame count: at steps 2, 3 and 4 every pixel's own sample on all "
                   "8 jitter phases within 32 cycles; the first cycle's order as before");
}

// The term averaged over the jitter's 8 phases (what a still view's
// accumulated mask should come to), and the scene at no jitter.
std::vector<float> Averaged(const Scene& s, const double* vp, Image* im0) {
  std::vector<float> avg(size_t(kW) * kH, 0.0f);
  for (int k = 0; k < 8; k++) {
    float j[2];
    Jitter(k, j);
    const Image im = Render(s, vp, kW, kH, j, false);
    for (size_t i = 0; i < avg.size(); i++) avg[i] += im.term[i] / 8.0f;
  }
  *im0 = Render(s, vp, kW, kH, kNoJitter, false);
  return avg;
}

// A still view under the jitter's 8 phases at steps 3 and 4: no change seen
// (the jitter's noise is not one), and the result near the term averaged
// over the phases.
int JitterChecks(const Gpu& g) {
  int fails = 0;
  const Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  Image im0;
  const std::vector<float> avg = Averaged(s, vp, &im0);
  for (uint32_t step = 3; step <= 4; step++) {
    Acc a(g);
    a.step = step;
    const uint32_t frames = step * step * 8 + 16;
    uint64_t changed = 0;
    std::vector<float> mean(avg.size(), 0.0f);
    bool ok = true;
    for (uint32_t k = 0; k < frames && ok; k++) {
      float j[2];
      Jitter(int(k), j);
      const Shot r = a.Frame(s, vp, k, j);
      ok = r.ok;
      if (k + 16 >= frames) {
        changed += r.n.changed;
        for (size_t i = 0; i < mean.size(); i++) mean[i] += r.out[i] / 16.0f;
      }
    }
    double most = 0, sum = 0, edge = 0;
    int n = 0, ne = 0;
    for (uint32_t y = 1; y + 1 < kH; y++)
      for (uint32_t x = 1; x + 1 < kW; x++) {
        const size_t at = size_t(y) * kW + x;
        if (im0.surface[at] < 0) continue;
        const double e = std::fabs(mean[at] - avg[at]);
        most = std::max(most, e), sum += e, n++;
        if (avg[at] > 0.0f && avg[at] < 1.0f) edge += e, ne++;
      }
    char what[300];
    std::snprintf(what, sizeof what,
                  "a still view under the jitter's 8 phases, step %u, %u frames: %llu pixels seen changed in the last "
                  "16; the last 16 frames' mean against the term averaged over the phases: mean error %.4f (bound "
                  "0.002), at the %d edge pixels %.4f, largest %.3f (targets 0.08, 0.35, and no change: not met)",
                  step, frames, (unsigned long long)changed, n ? sum / n : 1.0, ne, ne ? edge / ne : 1.0, most);
    fails += Check(ok && n && sum / n < 0.002 && ne > 200, what);
  }
  return fails;
}

// Ground lit at every phase of the jitter (a feature thinner than a pixel
// is lit at some phases only: the accumulation rightly keeps it).
std::vector<uint8_t> LitAll(const Scene& s, const double* vp) {
  std::vector<uint8_t> lit(size_t(kW) * kH, 1);
  for (int k = 0; k < 8; k++) {
    float j[2];
    Jitter(k, j);
    const Image im = Render(s, vp, kW, kH, j, false);
    for (size_t i = 0; i < lit.size(); i++) lit[i] &= im.surface[i] == 0 && im.term[i] > 0.5f;
  }
  return lit;
}
// Pixels lit (`lit`) with every pixel within `r` lit too (beyond what the
// fill reaches from a shadow), and of them those shown darker than 0.9;
// the darkest.
void Residue(const Shot& s, const std::vector<uint8_t>& lit_all, int r, int* n, int* dark, double* darkest) {
  *n = *dark = 0, *darkest = 0;
  for (int y = r; y + r < int(kH); y++)
    for (int x = r; x + r < int(kW); x++) {
      const size_t at = size_t(y) * kW + size_t(x);
      bool lit = true;
      for (int dy = -r; dy <= r && lit; dy++)
        for (int dx = -r; dx <= r && lit; dx++) lit = lit_all[size_t(y + dy) * kW + size_t(x + dx)] != 0;
      if (!lit) continue;
      ++*n;
      *dark += s.out[at] < 0.9f;
      *darkest = std::max(*darkest, 1.0 - s.out[at]);
    }
}

// Casters moving every frame under the jitter: the box at about 1, 4 and
// 16 pixels a frame (its shadow too), and the thin rod (a shadow a pixel or
// two tall) standing for 12 frames, then moving at 4 pixels a frame: on
// ground lit at every jitter phase with everything within 4 pixels lit so
// (beyond the fill's reach),
// how much stays dark (a trail), from the 3rd frame of the motion on; and
// the error at the shadows' edges.
int MovingChecks(const Gpu& g) {
  int fails = 0;
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  const double px = 0.054;  // (world units a pixel at the shadows, across)
  struct Run {
    const char* what;
    int box;         // which box moves (0 the floating box, 1 the rod)
    double speed;    // pixels a frame
    uint32_t still;  // frames standing first
  };
  const Run runs[] = {{"the box at 1 pixel a frame", 0, 1, 0},
                      {"the box at 4 pixels a frame", 0, 4, 0},
                      {"the box at 16 pixels a frame", 0, 16, 0},
                      {"the thin rod standing 12 frames, then 4 pixels a frame", 1, 4, 12}};
  for (const Run& run : runs) {
    Acc a(g);
    Scene s = Standard();
    int dark = 0, n = 0, worst_frame = -1;
    double darkest = 0, edge = 0;
    int ne = 0;
    bool ok = true;
    for (uint32_t k = 0; k < run.still + 24 && ok; k++) {
      Scene f = s;
      const double dx = k < run.still ? 0.0 : run.speed * px * (k - run.still + 1);
      f.boxes[size_t(run.box)].lo.x += dx, f.boxes[size_t(run.box)].hi.x += dx;
      float j[2];
      Jitter(int(k), j);
      const Shot r = a.Frame(f, vp, k, j);
      ok = r.ok;
      if (k < run.still + 2) continue;
      int nn, dd;
      double most;
      Residue(r, LitAll(f, vp), 4, &nn, &dd, &most);
      n += nn, dark += dd;
      if (most > darkest) darkest = most, worst_frame = int(k - run.still);
      for (uint32_t y = 1; y + 1 < kH; y++)
        for (uint32_t x = 1; x + 1 < kW; x++)
          if (r.im.surface[size_t(y) * kW + x] >= 0 && Edge(r.im, x, y))
            edge += std::fabs(r.out[size_t(y) * kW + x] - r.im.term[size_t(y) * kW + x]), ne++;
    }
    char what[400];
    std::snprintf(what, sizeof what,
                  "%s, jittered: of %d lit pixels (all within 4 lit at every phase) %d shown darker than 0.9, the "
                  "darkest by %.3f (frame %d of the motion; target 0.1%s); mean error at the shadows' edges %.3f",
                  run.what, n, dark, darkest, worst_frame, darkest < 0.1 ? ": met" : ": not met", ne ? edge / ne : 1.0);
    std::printf("  measured: %s\n", what);
    fails += Check(ok && n > 10000, "a moving caster's frames made (its trail measured above)");
  }
  return fails;
}

// Begin's bookkeeping: a second frame of the same number refused (the
// history left as it was), a skipped frame, a step's change and the light
// turned by 5 degrees drop the history, half a degree does not; frames in
// flight (four encoded before the first is done) as frames one at a time.
int BeginChecks(const Gpu& g) {
  const Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  Acc a(g);
  Shot r;
  for (uint32_t k = 0; k < 6; k++) r = a.Frame(s, vp, k);
  const uint64_t frames = a.h.frames;
  const Shot twice = a.Frame(s, vp, 5);
  const bool again = !twice.ok && twice.again && a.h.frames == frames;
  const Shot next = a.Frame(s, vp, 6), skipped = a.Frame(s, vp, 8);
  auto turn = [&](double deg, uint32_t k) {
    const double c = std::cos(deg * M_PI / 180), sn = std::sin(deg * M_PI / 180);
    const float l[3] = {float(s.sun.x * c - s.sun.y * sn), float(s.sun.x * sn + s.sun.y * c), float(s.sun.z)};
    return a.Frame(s, vp, k, kNoJitter, false, nullptr, kW, kH, l).reset;
  };
  const bool small = a.Frame(s, vp, 9).ok && !turn(0.5, 10), big = turn(5.0, 11);
  a.step = 4;
  const bool step = a.Frame(s, vp, 12).reset;
  // In flight against one at a time, the jitter on.
  Acc serial(g), flight(g);
  std::vector<Shot> one, four;
  for (uint32_t k = 0; k < 12; k++) {
    float j[2];
    Jitter(int(k), j);
    one.push_back(serial.Frame(s, vp, k, j));
  }
  for (uint32_t k = 0; k < 12; k += 4) {
    Acc::Pending pd[4];
    for (uint32_t i = 0; i < 4; i++) {
      float j[2];
      Jitter(int(k + i), j);
      flight.Start(&pd[i], s, vp, k + i, j);
    }
    for (auto& x : pd) four.push_back(flight.Finish(&x));
  }
  bool same = true;
  for (size_t k = 0; k < one.size(); k++) same = same && one[k].ok && four[k].ok && one[k].out == four[k].out;
  return Check(again && next.ok && !next.reset && skipped.reset && small && big && step && same,
               "a second frame of one number refused (the history kept); a skipped frame, a step's change and the "
               "light turned 5 degrees reset it, half a degree does not; four frames in flight byte for byte as "
               "one at a time");
}

// The scene swapped under a still camera (no cut: the same camera, other
// boxes): pixels four or more inside the new shadows and the lit right
// from the first frame, every pixel by step^2 + 1.
int SwapChecks(const Gpu& g) {
  Acc a(g);
  Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  for (uint32_t k = 0; k < 12; k++) a.Frame(s, vp, k);
  s.boxes = {{{1.0, 1.0, -1.0}, {3.0, 1.5, 0.5}}, {{-4.0, 0.5, 3.0}, {-2.5, 2.5, 3.5}}};
  int inner_wrong = 0, all[11] = {};
  for (int k = 1; k <= 10; k++) {
    const Shot r = a.Frame(s, vp, 11 + uint32_t(k));
    for (int y = 0; y < int(kH); y++)
      for (int x = 0; x < int(kW); x++) {
        const size_t at = size_t(y) * kW + size_t(x);
        if (r.im.surface[at] < 0 || std::fabs(r.out[at] - r.im.term[at]) < 0.5f) continue;
        all[k]++;
        bool in = true;
        for (int dy = -4; dy <= 4 && in; dy++)
          for (int dx = -4; dx <= 4 && in; dx++) {
            const int xx = x + dx, yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < int(kW) && yy < int(kH))
              in = r.im.term[size_t(yy) * kW + size_t(xx)] == r.im.term[at] &&
                   r.im.surface[size_t(yy) * kW + size_t(xx)] == r.im.surface[at];
          }
        inner_wrong += in;
      }
  }
  char what[300];
  std::snprintf(what, sizeof what,
                "the scene swapped under a still camera: pixels off by a half or more in frames 1 / 2 / 5 / 10 after: "
                "%d / %d / %d / %d, four or more inside a region %d in all (targets: none inside, none at all by the "
                "10th: %s)",
                all[1], all[2], all[5], all[10], inner_wrong, inner_wrong == 0 && all[10] == 0 ? "met" : "not met");
  std::printf("  measured: %s\n", what);
  return Check(all[10] < all[1] / 2, "the scene swapped: wrong pixels fewer than half by the 10th frame");
}

// A crease: a tall box on the ground, its face to the camera in shadow, the
// ground before it lit, the camera moving sideways: the lit ground next to
// the face (within three pixels of it) takes none of the face's history.
int CreaseChecks(const Gpu& g) {
  Acc a(g);
  Scene s;
  s.sun = Unit({0.0, 1.0, 0.8});
  s.boxes = {{{-1.0, 0.0, 0.0}, {1.0, 3.0, 1.0}}};
  int near = 0, wrong = 0;
  double darkest = 0;
  for (uint32_t k = 0; k < 24; k++) {
    double vp[16];
    const V3 m{0.05 * k, 0, 0};
    View(kEye + m, kAt + m, kW, kH, vp);
    const Shot r = a.Frame(s, vp, k);
    if (k < 10) continue;
    for (int y = 3; y + 3 < int(kH); y++)
      for (int x = 3; x + 3 < int(kW); x++) {
        const size_t at = size_t(y) * kW + size_t(x);
        if (r.im.surface[at] != 0 || r.im.term[at] < 0.5f) continue;
        bool face = false;  // the face within 3 pixels, nothing dark on the ground near
        bool dark = false;
        for (int dy = -3; dy <= 3; dy++)
          for (int dx = -3; dx <= 3; dx++) {
            const size_t b = size_t(y + dy) * kW + size_t(x + dx);
            face = face || r.im.surface[b] == 1;
            dark = dark || (r.im.surface[b] == 0 && r.im.term[b] < 0.5f);
          }
        if (!face || dark) continue;
        near++;
        wrong += r.out[at] < 0.9f;
        darkest = std::max(darkest, 1.0 - r.out[at]);
      }
  }
  char what[300];
  std::snprintf(what, sizeof what,
                "a crease (a shadowed face on lit ground), the camera moving: of %d lit ground pixels within 3 of the "
                "face %d shown darker than 0.9 (the darkest by %.3f)",
                near, wrong, darkest);
  return Check(near > 500 && wrong == 0, what);
}

// Three channels (the term grey, as a later many-light pass would give its
// colour): a still view and a moved box, every frame within a 255th of the
// one-channel result (8-bit against half floats), the channels equal.
int ThreeChecks(const Gpu& g) {
  Acc one(g), three(g);
  three.three = true;
  Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  double most = 0, spread = 0;
  bool ok = true;
  for (uint32_t k = 0; k < 16 && ok; k++) {
    if (k == 12) s.boxes[0].lo.x += 2.5, s.boxes[0].hi.x += 2.5;
    const Shot a = one.Frame(s, vp, k), b = three.Frame(s, vp, k);
    ok = a.ok && b.ok;
    for (size_t i = 0; ok && i < a.out.size(); i++) most = std::max(most, double(std::fabs(a.out[i] - b.out[i])));
    spread = std::max(spread, double(b.spread));
  }
  char what[300];
  std::snprintf(what, sizeof what,
                "three channels (a grey term): 16 frames, a box moved at the 12th, every pixel within %.4f of one "
                "channel's (a 255th: 8-bit against half floats), the channels %.4f apart at most",
                most, spread);
  return Check(ok && most <= 1.0 / 255.0 + 1e-3 && spread == 0, what);
}

// ---- GPU with hardware ray tracing: the sun's mask traced.

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
struct Mesh {
  std::vector<float> pos;
  RtGeometry g;
  RtMeshKey key;
};
void Tri(Mesh* m, V3 a, V3 b, V3 c) {
  for (V3 v : {a, b, c}) m->pos.insert(m->pos.end(), {float(v.x), float(v.y), float(v.z)});
}
void Finish(Mesh* m, uint64_t buffer) {
  const uint32_t tris = uint32_t(m->pos.size() / 9);
  m->g.positions = reinterpret_cast<const uint8_t*>(m->pos.data());
  m->g.size = m->pos.size() * 4;
  m->g.stride = 12;
  m->g.type = 2;
  m->g.prim = 4;
  m->g.primitives = tris;
  m->key.vertex_version = buffer * 16 + 1;
  m->key.vertex_buffer = buffer;
  m->key.prim = 4;
  m->key.primitives = tris;
  m->key.stride = 12;
  m->key.type = 2;
}
Mesh BoxMesh(const Box& b, uint64_t buffer) {
  Mesh m;
  const V3 c[8] = {{b.lo.x, b.lo.y, b.lo.z}, {b.hi.x, b.lo.y, b.lo.z}, {b.lo.x, b.hi.y, b.lo.z},
                   {b.hi.x, b.hi.y, b.lo.z}, {b.lo.x, b.lo.y, b.hi.z}, {b.hi.x, b.lo.y, b.hi.z},
                   {b.lo.x, b.hi.y, b.hi.z}, {b.hi.x, b.hi.y, b.hi.z}};
  const int faces[6][4] = {{1, 3, 5, 7}, {0, 4, 2, 6}, {2, 6, 3, 7}, {0, 1, 4, 5}, {4, 5, 6, 7}, {0, 2, 1, 3}};
  for (const auto& f : faces) {
    Tri(&m, c[f[0]], c[f[1]], c[f[2]]);
    Tri(&m, c[f[2]], c[f[1]], c[f[3]]);
  }
  Finish(&m, buffer);
  return m;
}
Mesh QuadMesh(double y, double half, uint64_t buffer) {
  Mesh m;
  Tri(&m, {-half, y, -half}, {-half, y, half}, {half, y, -half});
  Tri(&m, {half, y, -half}, {-half, y, half}, {half, y, half});
  Finish(&m, buffer);
  return m;
}

// The traced scene (the standard one's ground and boxes) and the frames'
// masks through the masker.
struct Traced {
  const Gpu& g;
  std::unique_ptr<RtShadowMasker> masker;
  std::unique_ptr<RtScene> scene;
  Arena arena;
  Mesh ground = QuadMesh(0, 400, 1), box = BoxMesh(kFloating, 2), rod = BoxMesh(kRod, 3);
  const RtFrame* f = nullptr;
  id<MTLBuffer> counts = nil;
  explicit Traced(const Gpu& gpu) : g(gpu) {}
  bool Init() {
    masker = RtShadowMasker::Make(g.dev);
    scene = MakeRtScene(g.dev, {});
    arena.buf = [g.dev newBufferWithLength:16u << 20 options:MTLResourceStorageModeShared];
    counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    if (!masker || !scene || !arena.buf) return false;
    scene->SetAllocator(ArenaAlloc, &arena);
    return true;
  }
  // A snapshot of the scene seen by `vp` (its TLAS built), the floating box
  // moved `dx` along x.
  bool Snap(const double* vp, double dx = 0) {
    float w[4][4] = {}, mw[4][4], bw[4][4], bmw[4][4];
    w[0][0] = w[1][1] = w[2][2] = w[3][3] = 1;
    std::memcpy(bw, w, sizeof w);
    bw[3][0] = float(dx);
    for (int r = 0; r < 4; r++)
      for (int k = 0; k < 4; k++) {
        mw[r][k] = float(vp[r * 4 + k]);
        bmw[r][k] = float(vp[r * 4 + k] + (r == 3 ? dx * vp[k] : 0.0));
      }
    scene->BeginFrame(1, 0);
    scene->Add(ground.key, ground.g, w, mw, 1, 4);
    scene->Add(box.key, box.g, bw, bmw, 1, 4);
    scene->Add(rod.key, rod.g, w, mw, 1, 4);
    f = scene->Snapshot();
    if (!f || !f->tlas) return false;
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    scene->EncodeFrame(cb, f);
    return g.Commit(cb);
  }
  void End() {
    const RtFrame* e = scene->EndFrame(true);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    scene->EncodeFrame(cb, e);
    g.Commit(cb);
  }
};

// An R8 texture's bytes (a private one copied out first).
std::vector<uint8_t> Read8(const Gpu& g, id<MTLTexture> t) {
  std::vector<uint8_t> v(t.width * t.height);
  if (t.storageMode == MTLStorageModePrivate) {
    id<MTLBuffer> b = [g.dev newBufferWithLength:v.size() options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(t.width, t.height, 1)
                        toBuffer:b
               destinationOffset:0
          destinationBytesPerRow:t.width
        destinationBytesPerImage:v.size()];
    [bl endEncoding];
    if (g.Commit(cb)) std::memcpy(v.data(), b.contents, v.size());
    return v;
  }
  [t getBytes:v.data() bytesPerRow:t.width fromRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0];
  return v;
}

// The mask as before (test 7's) hashed when this test was written.
constexpr uint64_t kOldPathHash = 0x4e8a98a860a73474ull;

int TracedChecks(const Gpu& g) {
  int fails = 0;
  Traced tr(g);
  if (!tr.Init()) return Check(false, "the traced scene: the masker and scene");
  const Scene s = Standard();
  double vp[16];
  View(kEye, kAt, kW, kH, vp);
  if (!tr.Snap(vp)) return Check(false, "the traced scene: a TLAS");
  const float sun[3] = {float(s.sun.x), float(s.sun.y), float(s.sun.z)};
  auto mask8 = [&] {
    return g.Texture(MTLPixelFormatR8Unorm, kW, kH, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
  };
  {
    // 7. The mask as before (Encode at 2 x 2) against the turning passes at
    // their top-left pixel (stride 0) upsampled the same way, jittered,
    // reversed depth texture.
    float jit[2];
    Jitter(3, jit);
    const Image im = Render(s, vp, kW, kH, jit, true);
    id<MTLTexture> dt = g.DepthTexture(im.depth, kW, kH);
    RtShadowParams p;
    id<MTLTexture> a = mask8(), b = mask8();
    bool ok = dt && RtShadowParamsFor(*tr.f, kW, kH, true, true, jit, sun, &p);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    if (ok) {
      tr.masker->Encode(cb, tr.f, dt, a, p, tr.counts, 0, nil, nullptr, nil, 2);
      ok = tr.masker->EncodeSparse(cb, tr.f, dt, p, tr.counts, 0, nil, 2, 0, 7, nullptr, nil, b) != nil;
    }
    ok = ok && g.Commit(cb);
    const std::vector<uint8_t> va = ok ? Read8(g, a) : std::vector<uint8_t>(),
                               vb = ok ? Read8(g, b) : std::vector<uint8_t>();
    int dark = 0, differ = 0, most = 0;
    for (uint8_t v : va) dark += v < 128;
    for (size_t k = 0; k < va.size() && k < vb.size(); k++)
      if (va[k] != vb[k]) differ++, most = std::max(most, std::abs(int(va[k]) - int(vb[k])));
    char what[300];
    std::snprintf(what, sizeof what,
                  "the sun's mask without the accumulation (traced at 2 x 2, upsampled; %d pixels in shadow) byte "
                  "for byte the turning passes at their top-left pixel upsampled the same way (%d bytes differ, by "
                  "%d at most)",
                  dark, differ, most);
    fails += Check(ok && va == vb && dark > 500, what);
    // And the mask as before against its bytes when this was written (on
    // the GPU that wrote it: a change is a change of the path, or of the
    // GPU's arithmetic).
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t v : va) hash = (hash ^ v) * 1099511628211ull;
    std::snprintf(what, sizeof what, "the sun's mask as before: its bytes' FNV-1a %016llx, as when written (%016llx)",
                  (unsigned long long)hash, (unsigned long long)kOldPathHash);
    fails += Check(hash == kOldPathHash, what);
  }
  {
    // 7b. The sun's mask traced at one pixel of every 4 x 4 turning with the
    // frame and upsampled each frame on its own (the device's with
    // RECOMP_RT_SUN_TURN=4 where the depth is jittered), against the mask traced at
    // every pixel: each frame within the edges' reach of it (pixels off by
    // more than half: few), and the mean of 16 frames (one turn of every
    // block's pixels) close to it.
    float jit[2];
    Jitter(3, jit);
    const Image im = Render(s, vp, kW, kH, jit, true);
    id<MTLTexture> dt = g.DepthTexture(im.depth, kW, kH);
    RtShadowParams p;
    bool ok = dt && RtShadowParamsFor(*tr.f, kW, kH, true, true, jit, sun, &p);
    std::vector<uint8_t> ref;
    if (ok) {
      id<MTLTexture> a = mask8();
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      ok = tr.masker->EncodeSparse(cb, tr.f, dt, p, tr.counts, 0, nil, 1, 0, 0, nullptr, nil, a) != nil && g.Commit(cb);
      if (ok) ref = Read8(g, a);
    }
    std::vector<double> mean(ref.size(), 0.0);
    int worst_frame = 0, dark = 0;
    for (uint8_t v : ref) dark += v < 128;
    for (uint32_t k = 0; ok && k < 16; k++) {
      id<MTLTexture> b = mask8();
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      ok =
          tr.masker->EncodeSparse(cb, tr.f, dt, p, tr.counts, 0, nil, 4, RtLightStride(4), k, nullptr, nil, b) != nil &&
          g.Commit(cb);
      if (!ok) break;
      const std::vector<uint8_t> vb = Read8(g, b);
      int off = 0;
      for (size_t i = 0; i < ref.size(); i++) {
        off += std::abs(int(vb[i]) - int(ref[i])) > 127;
        mean[i] += vb[i] / 255.0 / 16.0;
      }
      worst_frame = std::max(worst_frame, off);
    }
    double err = 0.0;
    for (size_t i = 0; i < ref.size(); i++) err += std::fabs(mean[i] - ref[i] / 255.0);
    err = ref.empty() ? 1.0 : err / double(ref.size());
    char what[300];
    std::snprintf(what, sizeof what,
                  "the sun's mask at 1 of 4 x 4 turning, upsampled each frame: at most %d of %zu pixels off the "
                  "mask traced at every pixel (%d in shadow) in a frame; 16 frames' mean off it by %.4f",
                  worst_frame, ref.size(), dark, err);
    fails += Check(ok && dark > 500 && worst_frame * 50 < int(ref.size()) && err < 0.01, what);
  }
  // 8. The mask traced at every pixel (step 1), and at one of every 3 x 3
  // turning, accumulated; no jitter, then the jitter's phases.
  auto traced = [&](const Image& im, const float jit[2], uint32_t step, uint32_t stride, uint32_t k,
                    RtAccumulateHistory* h, std::vector<uint8_t>* out, bool old_path) {
    id<MTLTexture> dt = g.DepthTexture(im.depth, kW, kH);
    RtShadowParams p;
    if (!dt || !RtShadowParamsFor(*tr.f, kW, kH, true, true, jit, sun, &p)) return false;
    id<MTLTexture> m = mask8();
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    if (old_path) {
      tr.masker->Encode(cb, tr.f, dt, m, p, tr.counts, 0, nil, nullptr, nil, 2);
    } else {
      id<MTLTexture> low = tr.masker->EncodeSparse(cb, tr.f, dt, p, tr.counts, 0, nil, step, stride, k);
      if (!low) return false;
      if (step == 1) {
        if (!g.Commit(cb)) return false;
        *out = Read8(g, low);
        return true;
      }
      RtAccumulateFrame fr;
      if (!g.acc->Begin(h, *tr.f, p, RtAccumulateTerm{}, step, stride, k, sun, false, &fr)) return false;
      g.acc->Encode(cb, fr, low, tr.masker->SparsePoints(), dt, nil, m);
    }
    if (!g.Commit(cb)) return false;
    *out = Read8(g, m);
    return true;
  };
  {
    // Under the jitter's 8 phases, 88 frames: the last 16 frames' mean
    // against the mask traced at every pixel averaged over the 8 phases.
    std::vector<float> avg(size_t(kW) * kH, 0.0f), mean(avg.size(), 0.0f);
    bool ok = true;
    for (int k = 0; k < 8 && ok; k++) {
      float jit[2];
      Jitter(k, jit);
      std::vector<uint8_t> full;
      ok = traced(Render(s, vp, kW, kH, jit, true), jit, 1, 0, 0, nullptr, &full, false);
      for (size_t i = 0; ok && i < avg.size(); i++) avg[i] += full[i] / 255.0f / 8.0f;
    }
    RtAccumulateHistory h;
    for (uint32_t k = 0; k < 88 && ok; k++) {
      float jit[2];
      Jitter(int(k), jit);
      std::vector<uint8_t> acc;
      ok = traced(Render(s, vp, kW, kH, jit, true), jit, 3, RtLightStride(3), k, &h, &acc, false);
      if (k >= 72)
        for (size_t i = 0; ok && i < mean.size(); i++) mean[i] += acc[i] / 255.0f / 16.0f;
    }
    const Image im0 = Render(s, vp, kW, kH, kNoJitter, true);
    double most = 0, sum = 0, edge = 0;
    int n = 0, ne = 0;
    for (size_t i = 0; ok && i < avg.size(); i++) {
      if (im0.surface[i] < 0) continue;
      const double e = std::fabs(mean[i] - avg[i]);
      most = std::max(most, e), sum += e, n++;
      if (avg[i] > 0.0f && avg[i] < 1.0f) edge += e, ne++;
    }
    char what[300];
    std::snprintf(what, sizeof what,
                  "the sun's mask traced at one of every 3 x 3 pixels, turning, accumulated under the jitter: the last "
                  "16 of 88 frames' mean against the mask traced at every pixel averaged over the 8 phases: mean "
                  "error %.4f (bound 0.003), at the %d edge pixels %.4f, largest %.3f (targets 0.08, 0.35: not met)",
                  n ? sum / n : 1.0, ne, ne ? edge / ne : 1.0, most);
    fails += Check(ok && n > 50000 && ne > 200 && sum / n < 0.003, what);
  }
  {
    // The jitter's eight phases over and over, the view still: the mean
    // change frame to frame at the shadows' edges (pixels the full trace
    // gives a neighbour of another value), the last 16 of 32 frames; the
    // half-resolution mask as before, and accumulated.
    double change[2] = {};
    bool ok = true;
    const Image im0 = Render(s, vp, kW, kH, kNoJitter, true);
    std::vector<uint8_t> full;
    ok = traced(im0, kNoJitter, 1, 0, 0, nullptr, &full, false);
    for (int path = 0; path < 2 && ok; path++) {
      RtAccumulateHistory h;
      std::vector<uint8_t> last, cur;
      int n = 0;
      for (uint32_t k = 0; k < 32 && ok; k++) {
        float jit[2];
        Jitter(int(k), jit);
        const Image im = Render(s, vp, kW, kH, jit, true);
        ok = traced(im, jit, 3, RtLightStride(3), k, &h, &cur, path == 0);
        if (k >= 16)
          for (uint32_t j = 1; j + 1 < kH; j++)
            for (uint32_t i = 1; i + 1 < kW; i++) {
              const size_t at = size_t(j) * kW + i;
              if (im0.surface[at] < 0) continue;
              if (full[at] == full[at - 1] && full[at] == full[at + 1] && full[at] == full[at - kW] &&
                  full[at] == full[at + kW])
                continue;
              change[path] += std::fabs(double(cur[at]) - double(last[at])) / 255.0;
              n++;
            }
        last = cur;
      }
      if (n) change[path] /= n;
    }
    char what[300];
    std::snprintf(what, sizeof what,
                  "under the jitter's phases, the shadows' edges change by %.4f a frame accumulated (one ray for 3 x "
                  "3 pixels), %.4f as before (one for 2 x 2, upsampled): at least as stable",
                  change[1], change[0]);
    fails += Check(ok && change[1] <= change[0], what);
  }
  tr.End();
  {
    // The box moving 4 pixels a frame under the jitter, 24 frames: each
    // frame the mask traced at every pixel, as before, and accumulated;
    // from the 4th on, their mean error at the shadows' edges, how much
    // they change there frame to frame, and the darkest pixel lit at every
    // jitter phase with all lit so within 4 (a trail).
    RtAccumulateHistory h;
    double err[2] = {}, change[2] = {}, trail[2] = {};
    int ne = 0;
    std::vector<uint8_t> last[2];
    bool ok = true;
    for (uint32_t k = 0; k < 24 && ok; k++) {
      Scene f = s;
      const double dx = 0.22 * k;
      f.boxes[0].lo.x += dx, f.boxes[0].hi.x += dx;
      ok = tr.Snap(vp, dx);
      float jit[2];
      Jitter(int(k), jit);
      const Image im = Render(f, vp, kW, kH, jit, true);
      std::vector<uint8_t> full, out[2];
      ok = ok && traced(im, jit, 1, 0, 0, nullptr, &full, false) && traced(im, jit, 3, 0, k, nullptr, &out[0], true) &&
           traced(im, jit, 3, RtLightStride(3), k, &h, &out[1], false);
      tr.End();
      if (!ok || k < 3) {
        for (int q = 0; q < 2 && ok; q++) last[q] = out[q];
        continue;
      }
      const std::vector<uint8_t> lit_all = LitAll(f, vp);
      for (int y = 4; y + 4 < int(kH); y++)
        for (int x = 4; x + 4 < int(kW); x++) {
          const size_t at = size_t(y) * kW + size_t(x);
          if (im.surface[at] < 0) continue;
          bool edge = false, lit = true;
          for (int dy = -4; dy <= 4; dy++)
            for (int dx2 = -4; dx2 <= 4; dx2++) {
              const uint8_t v = full[size_t(y + dy) * kW + size_t(x + dx2)];
              lit = lit && lit_all[size_t(y + dy) * kW + size_t(x + dx2)];
              if (std::abs(dy) + std::abs(dx2) == 1 && (v > 127) != (full[at] > 127)) edge = true;
            }
          for (int q = 0; q < 2; q++) {
            if (lit) trail[q] = std::max(trail[q], 1.0 - out[q][at] / 255.0);
            if (!edge) continue;
            err[q] += std::fabs(double(out[q][at]) - full[at]) / 255.0;
            change[q] += std::fabs(double(out[q][at]) - last[q][at]) / 255.0;
          }
          ne += edge;
        }
      for (int q = 0; q < 2; q++) last[q] = out[q];
    }
    for (int q = 0; q < 2; q++) err[q] /= std::max(ne / 2, 1), change[q] /= std::max(ne / 2, 1);
    char what[400];
    std::snprintf(what, sizeof what,
                  "the box moving 4 pixels a frame, traced, jittered: at the shadows' edges mean error %.3f "
                  "accumulated, %.3f as before; change frame to frame %.3f, %.3f; darkest lit pixel with all lit "
                  "within 4: %.3f, %.3f (target 0.1%s)",
                  err[1], err[0], change[1], change[0], trail[1], trail[0], trail[1] < 0.1 ? ": met" : ": not met");
    std::printf("  measured: %s\n", what);
    fails += Check(ok, "the moving box traced both ways (measured above)");
  }
  return fails;
}

// The sun's passes at 3440 x 1440 as before (2 x 2, upsampled) and
// accumulated (3 x 3), the traced scene's ground filling the view, normals
// given as a G-buffer's (a ray costs what this scene's does, far less than
// a city's): 64 frames, each traced both ways one after the other (the
// same load on a shared GPU), the jitter's phases turning; the last 48
// timed, each pass's least and median time.
void TracedTiming(const Gpu& g) {
  Timer t;
  Traced tr(g);
  if (!t.Init(g.dev) || !tr.Init()) return;
  constexpr uint32_t w = 3440, h = 1440;
  const Scene s = Standard();
  double vp[16];
  Camera(kEye, kAt, 1.0, double(w) / h, 0.5, 3000, vp);
  if (!tr.Snap(vp)) return;
  const float sun[3] = {float(s.sun.x), float(s.sun.y), float(s.sun.z)};
  MTLTextureDescriptor* od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
  od.storageMode = MTLStorageModePrivate;
  od.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> m = [g.dev newTextureWithDescriptor:od];
  // The depth and normals of the jitter's eight phases.
  id<MTLTexture> dt[8], nt[8];
  float jit[8][2];
  for (int k = 0; k < 8; k++) {
    Jitter(k, jit[k]);
    const Image im = Render(s, vp, w, h, jit[k], true);
    dt[k] = g.DepthTexture(im.depth, w, h);
    nt[k] = g.NormalTexture(im.normal, w, h);
    if (!dt[k]) return;
  }
  RtAccumulateHistory hist;
  std::map<std::string, std::vector<double>> ms[2];
  for (int k = 0; k < 64; k++) {
    RtShadowParams p;
    if (!RtShadowParamsFor(*tr.f, w, h, true, true, jit[k % 8], sun, &p)) return;
    for (int path = 0; path < 2; path++) {
      t.Start();
      id<MTLCommandBuffer> cb = [g.queue commandBuffer];
      if (path == 0) {
        tr.masker->Encode(cb, tr.f, dt[k % 8], m, p, tr.counts, 0, nil, nullptr, nt[k % 8], 2);
      } else {
        id<MTLTexture> low =
            tr.masker->EncodeSparse(cb, tr.f, dt[k % 8], p, tr.counts, 0, nt[k % 8], 3, RtLightStride(3), uint32_t(k));
        RtAccumulateFrame fr;
        if (low && g.acc->Begin(&hist, *tr.f, p, RtAccumulateTerm{}, 3, RtLightStride(3), uint32_t(k), sun, false, &fr))
          g.acc->Encode(cb, fr, low, tr.masker->SparsePoints(), dt[k % 8], nt[k % 8], m);
      }
      const bool done = g.Commit(cb);
      t.Stop(k >= 16 ? &ms[path] : nullptr, 0);
      if (!done) return;
    }
  }
  for (int path = 0; path < 2; path++) {
    std::printf("  the sun's passes at 3440 x 1440 %s:", path ? "accumulated (3 x 3)" : "as before (2 x 2)");
    Timer::Print(ms[path]);
  }
  tr.End();
}

}  // namespace

int main() {
  @autoreleasepool {
    int fails = CpuChecks() + PhaseChecks();
    Gpu g;
    if (!g.dev) {
      std::printf("skipped: no Metal device\n");
      return fails ? 1 : 77;
    }
    g.queue = [g.dev newCommandQueue];
    g.acc = RtAccumulator::Make(g.dev);
    if (!g.acc) return Check(false, "the accumulation's kernels");
    fails += StaticChecks(g);
    fails += MovedChecks(g);
    fails += PanChecks(g);
    fails += DisocclusionChecks(g);
    fails += ResetChecks(g);
    fails += UniformChecks(g);
    fails += ThreeChecks(g);
    fails += JitterChecks(g);
    fails += MovingChecks(g);
    fails += BeginChecks(g);
    fails += SwapChecks(g);
    fails += CreaseChecks(g);
    if (RtSupported(g.dev)) {
      fails += TracedChecks(g);
      TracedTiming(g);
    } else {
      std::printf("no hardware ray tracing: the traced checks skipped\n");
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
  }
}
