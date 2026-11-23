// A frame's many local lights lit with ray-traced shadows (gfx/
// rt_many_lights.h). On the CPU: the bins (each light in exactly the tiles
// its bounds on screen touch, strongest first, a full tile's weakest
// dropped and counted, lights off screen counted, the model and class in
// the kernels' flags) and a sample's decoding. On a GPU with hardware ray
// tracing, over a synthetic G-buffer of a ground plane and boxes seen from
// a known camera: one light without an occluder equal at every pixel to the
// model's formula (a point and a spot light, both models, a depth texture
// stored reversed and a float one); a wall between a light and the ground
// passing exactly nothing behind it and the full light beside it; a pixel
// class's light only on its pixels; pixels no light reaches adding nothing
// and tracing no ray; an occluder moved every frame shadowing in each frame
// where it is then, not where it was (no lag); many lights' mean over frames
// against each light's own visibility (the error and the per-frame noise
// printed); the groups nothing lights skipped (RECOMP_RT_MANY_EMPTY) against
// every group shaded and added, the same value at every pixel over frames
// whose lights move, through both ways of adding the light, and the groups
// counted; the passes' GPU time at 3440 x 1440 with 100 and 380 lights and
// with a day's dozen, with those groups skipped and without (printed, not
// judged).
//
//   rt_many_lights_test        (GPU part skipped, exit 77, without a GPU that traces rays)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "runtime/gfx/matrix4.h"
#include "runtime/gfx/rt_many_lights.h"
#include "runtime/gfx/rt_occlusion.h"
#include "runtime/gfx/rt_scene.h"
#include "runtime/gfx/rt_shadow.h"

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

// D3D's left-handed camera at `eye` looking at `at` (y up), as world to
// clip rows (depth 0..1 at near..far).
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

struct Box {
  V3 lo, hi;
};
Box Moved(const Box& b, V3 by) { return {b.lo + by, b.hi + by}; }
// Whether segment a -> b crosses box `box` grown by `grow` (negative: shrunk).
bool SegmentHits(V3 a, V3 b, const Box& box, double grow) {
  const double o[3] = {a.x, a.y, a.z}, d[3] = {b.x - a.x, b.y - a.y, b.z - a.z};
  const double lo[3] = {box.lo.x - grow, box.lo.y - grow, box.lo.z - grow};
  const double hi[3] = {box.hi.x + grow, box.hi.y + grow, box.hi.z + grow};
  double t0 = 0, t1 = 1;
  for (int k = 0; k < 3; k++) {
    if (lo[k] > hi[k]) return false;
    if (std::fabs(d[k]) < 1e-15) {
      if (o[k] < lo[k] || o[k] > hi[k]) return false;
      continue;
    }
    double ta = (lo[k] - o[k]) / d[k], tb = (hi[k] - o[k]) / d[k];
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    if (t0 > t1) return false;
  }
  return true;
}
// Where the ray o + t d (t > 0) enters the box, and the face's normal; -1: none.
double RayBox(V3 o, V3 d, const Box& box, V3* n) {
  const double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
  const double lo[3] = {box.lo.x, box.lo.y, box.lo.z}, hi[3] = {box.hi.x, box.hi.y, box.hi.z};
  double t0 = -1e30, t1 = 1e30;
  int axis = -1;
  double sign = 0;
  for (int k = 0; k < 3; k++) {
    if (std::fabs(dd[k]) < 1e-15) {
      if (oo[k] < lo[k] || oo[k] > hi[k]) return -1;
      continue;
    }
    double ta = (lo[k] - oo[k]) / dd[k], tb = (hi[k] - oo[k]) / dd[k];
    double s = -1;
    if (ta > tb) std::swap(ta, tb), s = 1;
    if (ta > t0) t0 = ta, axis = k, sign = s;
    t1 = std::min(t1, tb);
  }
  if (t0 > t1 || t0 <= 0 || axis < 0) return -1;
  double nn[3] = {0, 0, 0};
  nn[axis] = sign;
  *n = {nn[0], nn[1], nn[2]};
  return t0;
}

// ---- the model on the CPU (device.h: RayLightModel), as the kernels read
// the G-buffer's bytes

RayLightModel Regular() {
  RayLightModel m;
  m.falloff_power = 4, m.falloff_bias = 0.1f, m.falloff_scale = 1.1111112f;
  m.specular = true, m.spec_scale = 2, m.exponent_scale = 512;
  m.factor_classes = 0b0101;
  m.normal_low_bits = true;
  return m;
}
RayLightModel Filler() {
  RayLightModel m;
  m.falloff_power = 2, m.falloff_bias = 1.0f / 3.0f, m.falloff_scale = 1.5f;
  m.wrap = 0.2f, m.wrap_scale = 1.25f;
  m.specular = false;
  return m;
}

struct Texel {
  float albedo[3], normal[4], surface[3];
  uint32_t value;
};
V3 NormalOf(const Texel& t, bool low_bits) {
  if (!low_bits) return Unit({t.normal[0] * 2.0 - 0.99999, t.normal[1] * 2.0 - 0.99999, t.normal[2] * 2.0 - 0.99999});
  double e[3];
  const double k[3] = {0.99609375, 7.96875, 63.75};
  for (int i = 0; i < 3; i++) {
    const double v = t.normal[3] * k[i];
    e[i] = v - std::floor(v);
  }
  double r[3] = {2 * e[0] - 0.25 * e[1], 2 * e[1] - 0.25 * e[2], 2 * e[2]};
  return Unit({t.normal[0] * 256.0 + r[0] - 127.99999, t.normal[1] * 256.0 + r[1] - 127.99999,
               t.normal[2] * 256.0 + r[2] - 127.99999});
}
bool InClass(const RayManyLights& c, uint32_t k, uint32_t v) {
  return k == 0 || ((c.classes[k - 1][v >> 5] >> (v & 31)) & 1u);
}
// Light l's at point p (world) of texel t, the camera at `eye`, unshadowed.
V3 ModelLight(const RayManyLights& c, const RtLight& l, V3 p, V3 eye, const Texel& t) {
  const uint32_t mi = RtLightModel(l.flags), ci = RtLightClass(l.flags);
  const RayLightModel& m = c.models[mi];
  const V3 lp{l.position[0], l.position[1], l.position[2]};
  const V3 to = lp - p;
  const double a = std::clamp(1.0 - Dot(to, to) / (double(l.radius) * l.radius), 0.0, 1.0);
  if (!(a > 0)) return {0, 0, 0};
  const double f = std::pow(a, m.falloff_power) - m.falloff_bias;
  if (f < 0 || !InClass(c, ci, t.value)) return {0, 0, 0};
  const V3 L = Unit(to + V3{1e-5, 1e-5, 1e-5});
  double k = 1;
  if (l.kind == RtLightKind::Spot) {
    const V3 dir{l.direction[0], l.direction[1], l.direction[2]};
    k = std::clamp((Dot(L, dir * -1.0) - l.cone_offset) * l.cone_scale, 0.0, 1.0);
  }
  const double s = f * m.falloff_scale * k;
  const V3 col{l.color[0] * s, l.color[1] * s, l.color[2] * s};
  const V3 N = NormalOf(t, m.normal_low_bits);
  const double diff = std::clamp((Dot(L, N) - m.wrap) * m.wrap_scale, 0.0, 1.0);
  V3 r{col.x * diff * t.albedo[0], col.y * diff * t.albedo[1], col.z * diff * t.albedo[2]};
  if (m.specular) {
    const V3 I = Unit(p - eye + V3{1e-5, 1e-5, 1e-5});
    const V3 R = I - N * (2 * Dot(I, N));
    const double e = double(t.surface[1]) * t.surface[1] * m.exponent_scale;
    const double sa = std::clamp(Dot(L, R), 0.0, 1.0);
    r = r + col * (m.spec_scale * t.surface[0] * (e > 0 ? std::pow(sa, e) : 1.0));
  }
  if ((m.factor_classes >> ci) & 1u) r = r * std::clamp(double(t.surface[2]) * t.surface[2] + 0.5, 0.0, 1.0);
  return r;
}
double Lum(V3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

RtLight Point(V3 p, double r, V3 colour, uint32_t model = 0, uint32_t cls = 0) {
  RtLight l{};
  l.position[0] = float(p.x), l.position[1] = float(p.y), l.position[2] = float(p.z);
  l.radius = float(r);
  l.color[0] = float(colour.x), l.color[1] = float(colour.y), l.color[2] = float(colour.z);
  l.kind = RtLightKind::Point;
  l.flags = (model << kRtLightModelShift) | (cls << kRtLightClassShift);
  return l;
}
RtLight Spot(V3 p, double r, V3 colour, V3 dir, double cos_inner, double cos_outer) {
  RtLight l = Point(p, r, colour);
  l.kind = RtLightKind::Spot;
  const V3 d = Unit(dir);
  l.direction[0] = float(d.x), l.direction[1] = float(d.y), l.direction[2] = float(d.z);
  l.cone_offset = float(cos_outer);
  l.cone_scale = float(1.0 / std::max(cos_inner - cos_outer, 0.01));
  return l;
}

RayManyLights Config() {
  RayManyLights c;
  c.models[0] = Regular();
  c.models[1] = Filler();
  // Class 1: values 8 and up; 2: below 8 (as interior and exterior pixels
  // might be told apart).
  for (uint32_t v = 0; v < 256; v++) (v >= 8 ? c.classes[0] : c.classes[1])[v >> 5] |= 1u << (v & 31);
  return c;
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
  // (ml: as RECOMP_RT_MANY_EMPTY says; ml_skip and ml_all: the groups
  // nothing lights skipped, and every group shaded and added.)
  std::unique_ptr<RtManyLights> ml, ml_skip, ml_all;
  std::unique_ptr<RtOccluder> ao;
  bool Init() {
    arena.buf = [dev newBufferWithLength:64u << 20 options:MTLResourceStorageModeShared];
    ml = RtManyLights::Make(dev);
    ml_skip = RtManyLights::Make(dev, 1);
    ml_all = RtManyLights::Make(dev, 0);
    ao = RtOccluder::Make(dev);
    return arena.buf && ml && ml_skip && ml_all && ao;
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

// A view: the scene (the ground and `boxes`) traced and drawn from a camera
// into a w x h G-buffer; per pixel its point and texel (none: the sky).
struct View {
  uint32_t w = 0, h = 0;
  V3 eye;
  double vp[16];
  std::unique_ptr<RtScene> scene;
  const RtFrame* f = nullptr;
  std::vector<Mesh> meshes;
  std::vector<V3> points;
  std::vector<uint8_t> hit;
  std::vector<Texel> texels;
  id<MTLTexture> depth = nil, albedo = nil, normal = nil, surface = nil, cls = nil, target = nil;
  RtShadowParams p;
  double e[3];
};
uint8_t Unorm(double v) { return uint8_t(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); }

// `cls_of`: the class value of a pixel (i).
template <typename C>
bool MakeView(Gpu& g, uint32_t w, uint32_t h, V3 eye, V3 at, const std::vector<Box>& boxes, bool reversed, C&& cls_of,
              View* v) {
  v->w = w, v->h = h, v->eye = eye;
  v->scene = MakeRtScene(g.dev, {});
  if (!v->scene) return false;
  g.arena.used = 0;
  v->scene->SetAllocator(ArenaAlloc, &g.arena);
  Camera(eye, at, 1.0, double(w) / h, 0.5, 3000.0, v->vp);
  double inv[16];
  Invert4(v->vp, inv);
  float wm[4][4] = {}, mw[4][4];
  wm[0][0] = wm[1][1] = wm[2][2] = wm[3][3] = 1;
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) mw[r][k] = float(v->vp[r * 4 + k]);
  v->meshes.clear();
  v->meshes.reserve(boxes.size() + 1);
  v->meshes.push_back(QuadMesh(0, 200, 1));
  for (size_t k = 0; k < boxes.size(); k++) v->meshes.push_back(BoxMesh(boxes[k], 2 + k));
  v->scene->BeginFrame(1, 0);
  for (Mesh& m : v->meshes) v->scene->Add(m.key, m.g, wm, mw, 1, 4);
  v->f = v->scene->Snapshot();
  if (!v->f || !v->f->tlas) return false;
  const size_t n = size_t(w) * h;
  v->points.assign(n, {});
  v->hit.assign(n, 0);
  v->texels.assign(n, {});
  std::vector<float> depth(n);
  std::vector<uint8_t> albedo(n * 4), normal(n * 4), surface(n * 4), cls(n);
  for (uint32_t j = 0; j < h; j++)
    for (uint32_t i = 0; i < w; i++) {
      const double x = 2.0 * i / w - 1.0, y = 1.0 - 2.0 * j / h;
      double a[4], b[4];
      for (int k = 0; k < 4; k++) {
        a[k] = x * inv[k] + y * inv[4 + k] + inv[12 + k];
        b[k] = x * inv[k] + y * inv[4 + k] + inv[8 + k] + inv[12 + k];
      }
      const V3 pa{a[0] / a[3], a[1] / a[3], a[2] / a[3]}, pb{b[0] / b[3], b[1] / b[3], b[2] / b[3]};
      const V3 d = pb - pa;
      const size_t at_ = size_t(j) * w + i;
      double t = 2, z = 1;
      V3 nrm{0, 1, 0};
      if (d.y < 0 && -pa.y / d.y > 0 && -pa.y / d.y < 1) t = -pa.y / d.y;
      for (const Box& bx : boxes) {
        V3 bn;
        const double tb = RayBox(pa, d, bx, &bn);
        if (tb > 0 && tb < t) t = tb, nrm = bn;
      }
      Texel& tx = v->texels[at_];
      if (t < 2) {
        const V3 p = pa + d * t;
        double cc[4];
        for (int k = 0; k < 4; k++) cc[k] = p.x * v->vp[k] + p.y * v->vp[4 + k] + p.z * v->vp[8 + k] + v->vp[12 + k];
        z = cc[2] / cc[3];
        v->points[at_] = p;
        v->hit[at_] = 1;
        const uint8_t al[3] = {Unorm(0.55), Unorm(0.45), Unorm(0.35)};
        const uint8_t nn[4] = {Unorm(nrm.x * 0.5 + 0.5), Unorm(nrm.y * 0.5 + 0.5), Unorm(nrm.z * 0.5 + 0.5), 0};
        const uint8_t sf[3] = {Unorm(0.25), Unorm(0.6), Unorm(0.7)};
        for (int k = 0; k < 3; k++) {
          albedo[at_ * 4 + k] = al[k];
          surface[at_ * 4 + k] = sf[k];
          tx.albedo[k] = al[k] / 255.0f;
          tx.surface[k] = sf[k] / 255.0f;
        }
        for (int k = 0; k < 4; k++) {
          normal[at_ * 4 + k] = nn[k];
          tx.normal[k] = nn[k] / 255.0f;
        }
        cls[at_] = uint8_t(cls_of(i));
        tx.value = cls[at_];
      }
      depth[at_] = float(reversed ? 1.0 - z : z);
    }
  auto tex = [&](MTLPixelFormat pf, const void* data, uint32_t bpp, bool target) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                  width:w
                                                                                 height:h
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | (target ? MTLTextureUsageRenderTarget : 0);
    id<MTLTexture> t = [g.dev newTextureWithDescriptor:td];
    if (data) [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data bytesPerRow:w * bpp];
    return t;
  };
  v->albedo = tex(MTLPixelFormatRGBA8Unorm, albedo.data(), 4, false);
  v->normal = tex(MTLPixelFormatRGBA8Unorm, normal.data(), 4, false);
  v->surface = tex(MTLPixelFormatRGBA8Unorm, surface.data(), 4, false);
  v->cls = tex(MTLPixelFormatR8Unorm, cls.data(), 1, false);
  v->target = tex(MTLPixelFormatRGBA16Float, nullptr, 8, true);
  if (reversed) {
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    v->depth = [g.dev newTextureWithDescriptor:td];
    id<MTLBuffer> src = [g.dev newBufferWithBytes:depth.data() length:n * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:w * 4
        sourceBytesPerImage:n * 4
                 sourceSize:MTLSizeMake(w, h, 1)
                  toTexture:v->depth
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)
                    options:MTLBlitOptionDepthFromDepthStencil];
    [bl endEncoding];
    if (!g.Commit(cb)) return false;
  } else {
    v->depth = tex(MTLPixelFormatR32Float, depth.data(), 4, false);
  }
  const float up[3] = {0, 0, 1};
  return RtShadowParamsFor(*v->f, w, h, reversed, reversed, nullptr, up, &v->p) && RtLightsEye(*v->f, v->e);
}
bool MakeView(Gpu& g, uint32_t w, uint32_t h, V3 eye, V3 at, const std::vector<Box>& boxes, bool reversed, View* v) {
  return MakeView(g, w, h, eye, at, boxes, reversed, [](uint32_t) { return 0u; }, v);
}

struct Lit {
  std::vector<V3> rgb;
  uint32_t rays = 0, blocked = 0;
  // (Where the groups nothing lights are skipped: those groups, and of them
  // the ones whose gather kept no light.)
  uint32_t unlit = 0, bare = 0;
  double gpu_ms = 0, part_ms[2] = {};
  RtManyBins bins;
};

// One frame's pass over view `v` with `lights` (the target cleared first),
// by `with` (null: g.ml); `into`: the light drawn into a render pass opened
// here (EncodeAddInto) instead of the passes' own (EncodeAddPass).
bool Pass(Gpu& g, View& v, const RayManyLights& c, const std::vector<RtLight>& lights, uint32_t frame, bool turn,
          Lit* out, bool read = true, const RtManyLights* with = nullptr, bool into = false) {
  const RtManyLights& ml = with ? *with : *g.ml;
  RtManyBins& bins = out->bins;
  RtManyLightsBin(*v.f, v.w, v.h, nullptr, lights, v.e, c.lamp_gap, &bins);
  RtManyParams mp{};
  mp.frame = frame;
  mp.stride = turn ? RtLightStride(kRtManyBlock) : 0;
  mp.rest_share = kRtManyRestShare;
  for (int k = 0; k < 4; k++) mp.models[k] = RtManyModelFor(c.models[k]);
  std::memcpy(mp.classes, c.classes, sizeof mp.classes);
  RtManyGBuffer gb;
  gb.depth = v.depth, gb.albedo = v.albedo, gb.normal = v.normal, gb.surface = v.surface, gb.cls = v.cls;
  id<MTLBuffer> counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 16);
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = v.target;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
  v.scene->EncodeFrame(cb, v.f);
  if (!g.Commit(cb)) return false;
  // The trace and the shading in command buffers of their own (timed apart).
  for (uint32_t part = 1; part <= 2; part++) {
    cb = [g.queue commandBuffer];
    ml.Encode(cb, int(frame % RtManyLights::kSlots), v.f, gb, v.p, mp, bins, v.target, 0, counts, 0,
              part | (into ? RtManyLights::kNoAdd : 0u));
    if (into && part == 2) {
      MTLRenderPassDescriptor* ap = [MTLRenderPassDescriptor renderPassDescriptor];
      ap.colorAttachments[0].texture = v.target;
      ap.colorAttachments[0].loadAction = MTLLoadActionLoad;
      ap.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:ap];
      const MTLPixelFormat colors[4] = {v.target.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid,
                                        MTLPixelFormatInvalid};
      const bool drawn = ml.EncodeAddInto(e, colors, 1, MTLPixelFormatInvalid, MTLPixelFormatInvalid, 1);
      [e endEncoding];
      if (!drawn) return false;
    }
    if (!g.Commit(cb)) return false;
    out->part_ms[part - 1] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  }
  out->gpu_ms = out->part_ms[0] + out->part_ms[1];
  const uint32_t* n = static_cast<const uint32_t*>(counts.contents);
  out->rays = n[0], out->blocked = n[1];
  out->unlit = n[2], out->bare = n[3];
  if (!read) return true;
  std::vector<__fp16> px(size_t(v.w) * v.h * 4);
  [v.target getBytes:px.data() bytesPerRow:v.w * 8 fromRegion:MTLRegionMake2D(0, 0, v.w, v.h) mipmapLevel:0];
  out->rgb.resize(size_t(v.w) * v.h);
  for (size_t k = 0; k < out->rgb.size(); k++)
    out->rgb[k] = {double(px[k * 4]), double(px[k * 4 + 1]), double(px[k * 4 + 2])};
  return true;
}

bool Near(V3 a, V3 b, double rel, double abs_) {
  auto one = [&](double x, double y) { return std::fabs(x - y) <= abs_ + rel * std::fabs(y); };
  return one(a.x, b.x) && one(a.y, b.y) && one(a.z, b.z);
}

// Whether pixel k's point sees light l past `boxes` grown by `grow` (as the
// rays leave the surface: a little above it).
bool Sees(const View& v, size_t k, const RtLight& l, const std::vector<Box>& boxes, double grow) {
  const V3 lp{l.position[0], l.position[1], l.position[2]};
  const V3 p = v.points[k] + Unit(lp - v.points[k]) * 0.01;
  for (const Box& b : boxes)
    if (SegmentHits(p, lp, b, grow)) return false;
  return true;
}

// ---- CPU

int CpuChecks() {
  int failures = 0;
  // A frame looking down the z axis from above.
  double vp[16], inv[16];
  Camera({0, 9, -9}, {0, 0, 1}, 1.0, 640.0 / 480.0, 0.5, 3000.0, vp);
  Invert4(vp, inv);
  RtFrame f;
  for (int k = 0; k < 16; k++) f.inv_view_proj[k] = float(inv[k]);
  std::memcpy(f.inv_view_proj_exact, inv, sizeof inv);
  const double eye[3] = {0, 9, -9};
  std::vector<RtLight> lights = {Point({0, 1, 0}, 3, {1, 1, 1}), Point({3, 1, 2}, 6, {4, 2, 1}, 1, 2),
                                 Point({0, 1, -400}, 3, {1, 1, 1})};  // (the last behind the camera)
  RtManyBins b;
  RtManyLightsBin(f, 640, 480, nullptr, lights, eye, 0.5f, &b);
  failures += Check(b.tiles[0] == 20 && b.tiles[1] == 15, "bins: 32 x 32 pixel tiles");
  failures += Check(b.lights.size() == 2 && b.off_screen == 1, "bins: a light behind the camera reaches no tile");
  // Strongest first: the second light (power 16 x 36 against 9).
  failures += Check(b.lights.size() == 2 && b.lights[0].colour[0] == 4.0f && b.lights[0].flags == (1u | (2u << 2)) &&
                        b.lights[1].flags == 0 && b.lights[0].lamp_gap == 0.5f,
                    "bins: strongest first, its model and class in the flags, the lamp gap");
  failures += Check(b.lights.size() == 2 && std::fabs(b.lights[1].position[1] - (1 - 9)) < 1e-5 &&
                        b.lights[1].direction[3] == -2.0f && b.lights[1].colour[3] == 1.0f &&
                        std::fabs(b.lights[1].position[3] - 1.0f / 9.0f) < 1e-7,
                    "bins: a point light relative to the camera, its cone everywhere, 1 / r^2");
  // Each light in exactly the tiles its bounds touch.
  bool exact = true;
  for (uint32_t li = 0; li < b.lights.size(); li++) {
    const RtLight& l = li == 0 ? lights[1] : lights[0];
    RtLocalLight ll;
    for (int i = 0; i < 3; i++) ll.position[i] = l.position[i];
    ll.inv_sq_radius = 1.0f / (l.radius * l.radius);
    const RtRect r = RtLocalLightBounds(f, 640, 480, nullptr, ll);
    for (uint32_t ty = 0; ty < b.tiles[1]; ty++)
      for (uint32_t tx = 0; tx < b.tiles[0]; tx++) {
        const size_t t = size_t(ty) * b.tiles[0] + tx;
        bool in = false;
        for (uint32_t k = 0; k < b.heads[2 * t + 1]; k++) in = in || b.index[b.heads[2 * t] + k] == li;
        const bool want =
            !r.Empty() && tx * 32 < r.x1 && (tx + 1) * 32 > r.x0 && ty * 32 < r.y1 && (ty + 1) * 32 > r.y0;
        exact = exact && in == want;
      }
  }
  failures += Check(exact, "bins: each light in exactly the tiles its bounds on screen touch");
  // A full tile: 130 lights over the same pixels keep the 128 strongest.
  std::vector<RtLight> many;
  for (int k = 0; k < 130; k++) many.push_back(Point({0, 1, 0}, 2, {1.0 + k, 1, 1}));
  RtManyLightsBin(f, 640, 480, nullptr, many, eye, 0, &b);
  bool full = b.dropped > 0, kept = true;
  for (size_t t = 0; t < size_t(b.tiles[0]) * b.tiles[1]; t++)
    if (b.heads[2 * t + 1]) {
      full = full && b.heads[2 * t + 1] == kRtManyTileCap;
      for (uint32_t k = 0; k < b.heads[2 * t + 1]; k++) kept = kept && b.index[b.heads[2 * t] + k] < kRtManyTileCap;
    }
  failures += Check(full && kept, "bins: a full tile keeps the strongest 128, the others dropped and counted");
  const RtManySample s = RtManySampleOf(5u | (1u << 11) | (7u << 12));
  failures += Check(s.first == 5 && s.first_seen && s.second == 7 && !s.second_seen, "a sample's decoding");
  return failures;
}

// ---- GPU checks

int GpuChecks(Gpu& g) {
  int failures = 0;
  const RayManyLights c = Config();
  const V3 eye{0, 9, -9}, at{0, 0, 1};
  constexpr uint32_t W = 640, H = 480;

  // One light, nothing in its way: the formula at every pixel.
  for (int reversed = 0; reversed < 2; reversed++) {
    View v;
    if (!MakeView(g, W, H, eye, at, {}, reversed != 0, &v)) return failures + Check(false, "a view");
    const std::vector<std::vector<RtLight>> cases = {
        {Point({0.5, 1.5, 1}, 6, {3, 2, 1})},
        {Spot({-1, 3, 2}, 9, {2, 3, 4}, {0.2, -1, 0.1}, 0.9, 0.6)},
        {Point({1, 1, 3}, 5, {1, 2, 3}, 1)},
    };
    const char* const names[] = {"a point light", "a spot light", "the second model's point light"};
    for (size_t k = 0; k < cases.size(); k++) {
      Lit r;
      if (!Pass(g, v, c, cases[k], 3, true, &r)) return failures + Check(false, "a pass");
      size_t bad = 0, lit = 0;
      double worst = 0;
      for (size_t i = 0; i < r.rgb.size(); i++) {
        if (!v.hit[i]) continue;
        const V3 want = ModelLight(c, cases[k][0], v.points[i], eye, v.texels[i]);
        // (Judged where it is not faint: at the very edge of its reach a
        // pixel whose neighbours' samples it does not reach gets nothing.)
        if (!(Lum(want) > 1e-2)) continue;
        lit++;
        if (!Near(r.rgb[i], want, 2e-3, 2e-3)) {
          if (!bad)
            std::printf("  off: pixel %zu %zu, point %.3f %.3f %.3f, %.5f %.5f %.5f against %.5f %.5f %.5f\n", i % W,
                        i / W, v.points[i].x, v.points[i].y, v.points[i].z, r.rgb[i].x, r.rgb[i].y, r.rgb[i].z, want.x,
                        want.y, want.z);
          bad++, worst = std::max(worst, std::fabs(Lum(r.rgb[i]) - Lum(want)));
        }
      }
      char what[256];
      std::snprintf(what, sizeof what, "%s, %s depth: the model's light at every pixel (%zu lit, %zu off, worst %.4f)",
                    names[k], reversed ? "a reversed" : "a float", lit, bad, worst);
      failures += Check(lit > 1000 && bad == 0, what);
    }
  }

  // A wall between the light and the ground beyond it: nothing at all
  // behind it, the full light beside it.
  {
    const std::vector<Box> wall = {{{-1.5, 0, 1.0}, {1.5, 3.0, 1.3}}};
    View v;
    if (!MakeView(g, W, H, eye, at, wall, false, &v)) return failures + Check(false, "the wall's view");
    const std::vector<RtLight> l = {Point({0, 1.5, -0.5}, 12, {3, 3, 3})};
    size_t behind = 0, behind_lit = 0, beside = 0, beside_off = 0;
    for (uint32_t frame = 0; frame < 4; frame++) {
      Lit r;
      if (!Pass(g, v, c, l, frame, true, &r)) return failures + Check(false, "the wall's pass");
      for (size_t i = 0; i < r.rgb.size(); i++) {
        if (!v.hit[i]) continue;
        const V3 want = ModelLight(c, l[0], v.points[i], eye, v.texels[i]);
        if (!(Lum(want) > 1e-2)) continue;
        if (!Sees(v, i, l[0], wall, -0.1)) {
          behind++;
          if (r.rgb[i].x != 0 || r.rgb[i].y != 0 || r.rgb[i].z != 0) {
            if (behind_lit++ < 3) {
              std::printf("  lit behind the wall: pixel %zu %zu, point %.2f %.2f %.2f, %.4f of %.4f, frame %u\n", i % W,
                          i / W, v.points[i].x, v.points[i].y, v.points[i].z, Lum(r.rgb[i]), Lum(want), frame);
              id<MTLTexture> st = g.ml->Samples();
              const uint32_t bw = uint32_t(st.width), bh = uint32_t(st.height);
              id<MTLBuffer> buf = [g.dev newBufferWithLength:size_t(bw) * bh * 8 options:MTLResourceStorageModeShared];
              id<MTLCommandBuffer> cb = [g.queue commandBuffer];
              id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
              [bl copyFromTexture:st
                               sourceSlice:0
                               sourceLevel:0
                              sourceOrigin:MTLOriginMake(0, 0, 0)
                                sourceSize:MTLSizeMake(bw, bh, 1)
                                  toBuffer:buf
                         destinationOffset:0
                    destinationBytesPerRow:bw * 8
                  destinationBytesPerImage:size_t(bw) * bh * 8];
              [bl endEncoding];
              g.Commit(cb);
              const uint32_t* sp = static_cast<const uint32_t*>(buf.contents);
              const double dp = std::sqrt(Dot(v.points[i] - eye, v.points[i] - eye));
              const int bx = int(i % W) / 2, by = int(i / W) / 2;
              for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                  const size_t at = size_t(by + dy) * bw + size_t(bx + dx);
                  float dd;
                  std::memcpy(&dd, &sp[at * 2 + 1], 4);
                  const RtManySample q = RtManySampleOf(sp[at * 2]);
                  std::printf("    block %+d %+d: light %u seen %d, distance %.3f (pixel's %.3f)\n", dx, dy, q.first,
                              q.first_seen, dd, dp);
                }
            }
          }
        } else if (Sees(v, i, l[0], wall, 0.1)) {
          beside++;
          if (!Near(r.rgb[i], want, 2e-3, 2e-3)) beside_off++;
        }
      }
    }
    char what[256];
    std::snprintf(what, sizeof what,
                  "a wall: exactly nothing behind it (%zu of %zu lit), the light beside it (%zu of %zu off)",
                  behind_lit, behind, beside_off, beside);
    failures += Check(behind > 1000 && behind_lit == 0 && beside > 1000 && beside_off == 0, what);
  }

  // A pixel class: a light of class 2 (values below 8) lights the left
  // half (value 0) and nothing of the right (value 20).
  {
    View v;
    if (!MakeView(g, W, H, eye, at, {}, false, [](uint32_t i) { return i < W / 2 ? 0u : 20u; }, &v))
      return failures + Check(false, "the class view");
    const std::vector<RtLight> l = {Point({0, 1.5, 1}, 8, {2, 2, 2}, 0, 2)};
    Lit r;
    if (!Pass(g, v, c, l, 1, true, &r)) return failures + Check(false, "the class pass");
    size_t left = 0, left_off = 0, right_lit = 0;
    for (uint32_t j = 0; j < H; j++)
      for (uint32_t i = 0; i < W; i++) {
        const size_t k = size_t(j) * W + i;
        if (!v.hit[k]) continue;
        const V3 want = ModelLight(c, l[0], v.points[k], eye, v.texels[k]);
        if (i >= W / 2) {
          if (Lum(r.rgb[k]) != 0) right_lit++;
        } else if (Lum(want) > 1e-2) {
          left++;
          if (!Near(r.rgb[k], want, 2e-3, 2e-3)) left_off++;
        }
      }
    char what[200];
    std::snprintf(what, sizeof what, "a pixel class: its pixels lit (%zu of %zu off), the others not (%zu lit)",
                  left_off, left, right_lit);
    failures += Check(left > 1000 && left_off == 0 && right_lit == 0, what);
  }

  // Pixels no light reaches: nothing added, no ray.
  {
    View v;
    if (!MakeView(g, W, H, eye, at, {}, false, &v)) return failures + Check(false, "the small light's view");
    const std::vector<RtLight> l = {Point({0, 0.5, 2}, 1.0, {2, 2, 2})};
    Lit r;
    if (!Pass(g, v, c, l, 0, false, &r)) return failures + Check(false, "the small light's pass");
    size_t reached = 0, outside_lit = 0;
    for (size_t k = 0; k < r.rgb.size(); k++) {
      const bool in = v.hit[k] && Lum(ModelLight(c, l[0], v.points[k], eye, v.texels[k])) > 0;
      reached += in;
      if (!in && Lum(r.rgb[k]) != 0) outside_lit++;
    }
    // (At most a ray a block whose sample the light reaches.)
    char what[200];
    std::snprintf(what, sizeof what,
                  "no light: nothing added (%zu pixels), rays only where it reaches (%u for %zu pixels)", outside_lit,
                  r.rays, reached);
    failures += Check(outside_lit == 0 && r.rays > 0 && r.rays <= reached / 4 + 64, what);
  }

  // An occluder moving every frame: in each frame its shadow where it is
  // then (pixels sure of the shadow dark, those sure of the light lit).
  {
    const std::vector<RtLight> l = {Point({0, 3, 1}, 14, {3, 3, 3})};
    const Box box{{-0.5, 0.8, 0.5}, {0.5, 1.8, 1.5}};
    size_t wrong = 0, judged = 0;
    for (uint32_t frame = 0; frame < 8; frame++) {
      const std::vector<Box> now = {Moved(box, {-2.0 + 0.5 * frame, 0, 0})};
      View v;
      if (!MakeView(g, W, H, eye, at, now, false, &v)) return failures + Check(false, "the moving view");
      Lit r;
      if (!Pass(g, v, c, l, frame, true, &r)) return failures + Check(false, "the moving pass");
      for (size_t k = 0; k < r.rgb.size(); k++) {
        if (!v.hit[k] || v.points[k].y > 1e-6) continue;  // (the ground's pixels)
        const V3 want = ModelLight(c, l[0], v.points[k], eye, v.texels[k]);
        if (!(Lum(want) > 1e-2)) continue;
        const bool dark = !Sees(v, k, l[0], now, -0.1), lit = !dark && Sees(v, k, l[0], now, 0.1);
        if (dark || lit) {
          judged++;
          if (dark ? Lum(r.rgb[k]) != 0 : !Near(r.rgb[k], want, 2e-3, 2e-3)) {
            if (wrong++ < 4)
              std::printf("  wrong (%s): frame %u pixel %zu %zu, point %.3f %.3f %.3f, %.5f of %.5f\n",
                          dark ? "in shadow" : "lit", frame, k % W, k / W, v.points[k].x, v.points[k].y, v.points[k].z,
                          Lum(r.rgb[k]), Lum(want));
          }
        }
      }
    }
    char what[200];
    std::snprintf(what, sizeof what, "a moving occluder: its shadow where it is in every frame (%zu of %zu wrong)",
                  wrong, judged);
    failures += Check(judged > 10000 && wrong == 0, what);
  }

  // Many lights: the mean over frames against each one's own visibility.
  {
    std::vector<Box> boxes;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    for (int k = 0; k < 6; k++) {
      const V3 lo{-6 + 12 * u(rng), 0, -1 + 10 * u(rng)};
      boxes.push_back({lo, lo + V3{0.4 + u(rng), 0.5 + 1.5 * u(rng), 0.4 + u(rng)}});
    }
    std::vector<RtLight> l;
    for (int k = 0; k < 40; k++)
      l.push_back(Point({-7 + 14 * u(rng), 0.3 + 2.5 * u(rng), -2 + 13 * u(rng)}, 2 + 5 * u(rng),
                        {0.5 + u(rng), 0.5 + u(rng), 0.5 + u(rng)}, k % 4 == 3 ? 1 : 0));
    View v;
    if (!MakeView(g, W, H, eye, at, boxes, false, &v)) return failures + Check(false, "the many lights' view");
    constexpr uint32_t kFrames = 64;
    std::vector<V3> sum(size_t(W) * H, {0, 0, 0});
    std::vector<double> sq(sum.size(), 0);
    uint64_t rays = 0;
    for (uint32_t frame = 0; frame < kFrames; frame++) {
      Lit r;
      if (!Pass(g, v, c, l, frame, true, &r)) return failures + Check(false, "the many lights' pass");
      rays += r.rays;
      for (size_t k = 0; k < sum.size(); k++) {
        sum[k] = sum[k] + r.rgb[k];
        sq[k] += Lum(r.rgb[k]) * Lum(r.rgb[k]);
      }
    }
    // Judged where every light's visibility is sure (none of its rays near
    // an edge), its sum of shadowed light against the mean.
    double err = 0, ref = 0, noise = 0;
    size_t judged = 0;
    for (size_t k = 0; k < sum.size(); k++) {
      if (!v.hit[k]) continue;
      V3 want{0, 0, 0};
      bool sure = true;
      for (const RtLight& li : l) {
        const V3 c1 = ModelLight(c, li, v.points[k], eye, v.texels[k]);
        if (!(Lum(c1) > 0)) continue;
        const bool far_ = Sees(v, k, li, boxes, 0.15), near_ = Sees(v, k, li, boxes, -0.15);
        if (far_ != near_) sure = false;
        if (far_) want = want + c1;
      }
      if (!sure || !(Lum(want) > 1e-2)) continue;
      const double mean = Lum(sum[k]) / kFrames;
      judged++;
      err += std::fabs(mean - Lum(want));
      ref += Lum(want);
      noise += std::sqrt(std::max(0.0, sq[k] / kFrames - mean * mean));
    }
    const double rel = ref > 0 ? err / ref : 1;
    std::printf(
        "  many lights: %zu pixels judged, mean error %.2f%% of the light, per-frame noise %.2f%%, %.0f rays a frame\n",
        judged, 100 * rel, ref > 0 ? 100 * noise / ref : 0.0, double(rays) / kFrames);
    char what[200];
    std::snprintf(what, sizeof what,
                  "many lights: the mean over %u frames within 5%% of each one's shadowed light (%.2f%%)", kFrames,
                  100 * rel);
    failures += Check(judged > 5000 && rel < 0.05, what);
  }

  // The groups nothing lights skipped (RECOMP_RT_MANY_EMPTY) against every
  // group shaded and added: the same value at every pixel, through either
  // way of adding the light, over frames whose lights change (a group lit in
  // one frame and not in the next keeps that frame's texels: none may show).
  // A small light walking over the ground; with it in even frames a light of
  // class 2 whose reach takes in pixels of value 20 (reached, not lit), in
  // odd ones a far light behind a wall (in its shadow: reached, lit by
  // nothing). (At most two lights a pixel: a group gathers its lights in no
  // fixed order, and three or more would be summed and picked in it.)
  {
    const std::vector<Box> wall = {{{-1.5, 0, 1.0}, {1.5, 3.0, 1.3}}};
    View v;
    if (!MakeView(g, W, H, eye, at, wall, false, [](uint32_t i) { return i < W / 2 ? 0u : 20u; }, &v))
      return failures + Check(false, "the skipped groups' view");
    const uint32_t gw = (W + kRtManyGroup - 1) / kRtManyGroup, gh = (H + kRtManyGroup - 1) / kRtManyGroup;
    constexpr uint32_t kFrames = 8;
    auto is_lit = [](const V3& x) { return x.x != 0 || x.y != 0 || x.z != 0; };
    auto same = [](const V3& x, const V3& y) { return x.x == y.x && x.y == y.y && x.z == y.z; };
    size_t lit = 0, differ = 0, differ_into = 0;
    // (Groups: counted nothing lit, of them with no light kept, without a
    // lit pixel in the picture, and counted with every group shaded.)
    uint64_t unlit = 0, bare = 0, dark = 0, counted_full = 0;
    bool same_counts = true;
    for (uint32_t frame = 0; frame < kFrames; frame++) {
      std::vector<RtLight> l = {Point({-3.5 + frame, 0.6, 3.0}, 2.5, {2, 2, 2})};
      l.push_back(frame % 2 == 0 ? Point({1, 1.5, 5}, 6, {2, 3, 1}, 0, 2) : Point({0, 1.5, -0.5}, 12, {3, 3, 3}));
      Lit full, skip, into;
      if (!Pass(g, v, c, l, frame, true, &full, true, g.ml_all.get()) ||
          !Pass(g, v, c, l, frame, true, &skip, true, g.ml_skip.get()) ||
          !Pass(g, v, c, l, frame, true, &into, true, g.ml_skip.get(), true))
        return failures + Check(false, "the skipped groups' passes");
      for (size_t k = 0; k < full.rgb.size(); k++) {
        lit += is_lit(full.rgb[k]);
        if (!same(full.rgb[k], skip.rgb[k]) && !differ++)
          std::printf("  differs: frame %u pixel %zu %zu, %.6f %.6f %.6f skipping, %.6f %.6f %.6f not\n", frame, k % W,
                      k / W, skip.rgb[k].x, skip.rgb[k].y, skip.rgb[k].z, full.rgb[k].x, full.rgb[k].y, full.rgb[k].z);
        differ_into += !same(full.rgb[k], into.rgb[k]);
      }
      for (uint32_t gy = 0; gy < gh; gy++)
        for (uint32_t gx = 0; gx < gw; gx++) {
          bool any = false;
          for (uint32_t j = gy * kRtManyGroup; j < std::min(H, (gy + 1) * kRtManyGroup) && !any; j++)
            for (uint32_t i = gx * kRtManyGroup; i < std::min(W, (gx + 1) * kRtManyGroup) && !any; i++)
              any = is_lit(full.rgb[size_t(j) * W + i]);
          dark += !any;
        }
      unlit += skip.unlit, bare += skip.bare, counted_full += full.unlit + full.bare;
      same_counts = same_counts && into.unlit == skip.unlit && into.bare == skip.bare;
    }
    char what[300];
    std::snprintf(what, sizeof what,
                  "the groups nothing lights skipped: every pixel as with all shaded (%zu lit over %u frames; %zu "
                  "differ added in a pass of their own, %zu drawn into an open pass)",
                  lit, kFrames, differ, differ_into);
    failures += Check(lit > 10000 && differ == 0 && differ_into == 0, what);
    // (A group counted has no lit pixel; one without a lit pixel may still
    // not be counted: light too faint for the texture's 16 bits.)
    std::snprintf(what, sizeof what,
                  "the groups nothing lights counted: %llu of %u (%llu without a lit pixel), %llu of them with no "
                  "light kept; none counted with all shaded",
                  (unsigned long long)unlit, gw * gh * kFrames, (unsigned long long)dark, (unsigned long long)bare);
    failures += Check(unlit > 0 && unlit <= dark && bare > 0 && bare < unlit && same_counts && counted_full == 0, what);
  }
  return failures;
}

// The passes' time at 3440 x 1440 with n lights like a street's at night
// in the game's light list (street lamps: spots down, 14 m; cars' head
// lamps, spots of 10 m, and tail lamps of 4 m; small points of 3-8 m; one
// long head lamp of 40 m), along a street 30 m wide and 300 m long between
// buildings, with cars on it, seen from 2 m up (printed); without `head`
// the long head lamp left out (with a dozen lights: a day's, most of the
// picture reached by none). Each with the groups nothing lights skipped
// (RECOMP_RT_MANY_EMPTY) and with every group shaded and added.
void Timing(Gpu& g, uint32_t n, bool head = true) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<Box> boxes;
  for (int side = -1; side <= 1; side += 2)
    for (int k = 0; k < 15; k++) {
      const double x0 = side < 0 ? -25.0 : 15.0, z0 = k * 20.0 + 1;
      boxes.push_back({{x0, 0, z0}, {x0 + 10, 8 + 20 * u(rng), z0 + 18}});
    }
  std::vector<V3> cars;
  for (int k = 0; k < 40; k++) {
    const V3 c{-9 + 18 * u(rng), 0, 5 + 290 * u(rng)};
    cars.push_back(c);
    boxes.push_back({c - V3{0.9, 0, 2.2}, c + V3{0.9, 1.4, 2.2}});
  }
  std::vector<RtLight> l;
  const uint32_t lamps = n * 3 / 10, car_lamps = n * 5 / 10;
  for (uint32_t k = 0; k < lamps; k++) {
    const double z = 300.0 * k / std::max(lamps / 2, 1u);
    l.push_back(Spot({k % 2 ? 13.5 : -13.5, 6, std::fmod(z, 300.0)}, 14, {2.0, 1.4, 0.7}, {0, -1, 0}, 0.6, 0.2));
  }
  for (uint32_t k = 0; k < car_lamps; k++) {
    const V3 c = cars[k % cars.size()];
    if (k % 2 == 0)
      l.push_back(Spot(c + V3{(k / 2) % 2 ? 0.6 : -0.6, 0.7, 2.3}, 10, {3, 3, 2.6}, {0, -0.2, 1}, 0.9, 0.75));
    else
      l.push_back(Spot(c + V3{(k / 2) % 2 ? 0.6 : -0.6, 0.8, -2.3}, 4, {1.5, 0.1, 0.1}, {0, -0.1, -1}, 0.7, 0.3));
  }
  while (l.size() + (head ? 1 : 0) < n)
    l.push_back(Point({-14 + 28 * u(rng), 0.5 + 6 * u(rng), 300 * u(rng)}, 3 + 5 * u(rng), {1, 0.8, 0.6}));
  if (head) l.push_back(Spot({0.6, 1.2, 0.5}, 40, {4, 4, 3.5}, {0, -0.1, 1}, 0.95, 0.85));
  View v;
  if (!MakeView(g, 3440, 1440, {0, 2, -4}, {0, 1.5, 30}, boxes, true, &v)) {
    std::printf("  timing: no view\n");
    return;
  }
  const RayManyLights c = Config();
  const uint32_t groups = ((v.w + kRtManyGroup - 1) / kRtManyGroup) * ((v.h + kRtManyGroup - 1) / kRtManyGroup);
  for (int skipped = 1; skipped >= 0; skipped--) {
    const RtManyLights* ml = skipped ? g.ml_skip.get() : g.ml_all.get();
    double best = 1e9, trace = 0, shade = 0;
    Lit r;
    for (uint32_t frame = 0; frame < 40; frame++) {
      if (!Pass(g, v, c, l, frame, true, &r, false, ml)) return;
      if (frame > 8 && r.gpu_ms < best) best = r.gpu_ms, trace = r.part_ms[0], shade = r.part_ms[1];
    }
    size_t entries = r.bins.index.size(), tiles = size_t(r.bins.tiles[0]) * r.bins.tiles[1];
    // (The shading's time with the pass that adds its light.)
    std::printf("  timing, 3440 x 1440, %u lights (%zu on screen, %.1f a tile, %u dropped), %s: %.2f ms GPU (trace "
                "%.2f, shading and adding %.2f), %.2f M rays",
                n, r.bins.lights.size(), double(entries) / tiles, r.bins.dropped,
                skipped ? "the groups nothing lights skipped" : "every group shaded", best, trace, shade,
                r.rays / 1e6);
    if (skipped) std::printf(", %u of %u groups nothing lit (%u of them with no light kept)", r.unlit, groups, r.bare);
    std::printf("\n");
  }
}

}  // namespace

// ---- ray-traced ambient occlusion (gfx/rt_occlusion.h) over the same views

// The occlusion of view v traced `frames` times (each its own frame), the
// mean per pixel (each pixel the half-size output's texel it falls in);
// the rays traced and blocked of the last, its GPU time.
bool Occlusion(Gpu& g, View& v, float radius, uint32_t frames, std::vector<double>* mean, uint32_t* rays,
               uint32_t* blocked, double* ms) {
  id<MTLTexture> out = g.ao->NewOut(v.w, v.h);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  v.scene->EncodeFrame(cb, v.f);
  if (!out || !g.Commit(cb)) return false;
  mean->assign(size_t(v.w) * v.h, 0.0);
  const uint32_t ow = uint32_t(out.width), oh = uint32_t(out.height);
  if (ow != (v.w + 1) / 2 || oh != (v.h + 1) / 2) return false;
  std::vector<uint8_t> px(size_t(ow) * oh);
  id<MTLBuffer> read = [g.dev newBufferWithLength:px.size() options:MTLResourceStorageModeShared];
  for (uint32_t fr = 0; fr < frames; fr++) {
    std::memset(counts.contents, 0, 16);
    RtOcclusionParams op{};
    op.frame = fr;
    op.stride = RtLightStride(kRtOcclusionBlock);
    op.radius = radius;
    cb = [g.queue commandBuffer];
    g.ao->Encode(cb, v.f, v.depth, v.normal, v.p, op, out, counts, 0);
    if (!g.Commit(cb)) return false;
    const double t = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    *ms = fr == 0 ? t : std::min(*ms, t);  // (the least: the GPU's clock ramps over the first)
    cb = [g.queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromTexture:out
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(ow, oh, 1)
                        toBuffer:read
               destinationOffset:0
          destinationBytesPerRow:ow
        destinationBytesPerImage:px.size()];
    [bl endEncoding];
    if (!g.Commit(cb)) return false;
    const uint8_t* b = static_cast<const uint8_t*>(read.contents);
    for (uint32_t j = 0; j < v.h; j++)
      for (uint32_t i = 0; i < v.w; i++) (*mean)[size_t(j) * v.w + i] += b[size_t(j / 2) * ow + i / 2] / 255.0 / frames;
  }
  const uint32_t* n = static_cast<const uint32_t*>(counts.contents);
  *rays = n[0], *blocked = n[1];
  return true;
}

// A box on the ground: open ground (no wall within the rays' length) and
// the sky exactly open; the ground at the foot of the box's walls darker
// (its mean over frames well below open); the box's top open; rays traced
// at a quarter of the pixels, some blocked.
int OcclusionChecks(Gpu& g) {
  int failures = 0;
  const Box box{{-2, 0, 8}, {2, 4, 12}};
  const float radius = 3.0f;
  for (bool reversed : {false, true}) {
    View v;
    if (!MakeView(g, 320, 200, {0, 6, -6}, {0, 0, 10}, {box}, reversed, &v))
      return Check(false, "occlusion: the view made");
    std::vector<double> m;
    uint32_t rays = 0, blocked = 0;
    double ms = 0;
    if (!Occlusion(g, v, radius, 16, &m, &rays, &blocked, &ms)) return Check(false, "occlusion: traced");
    auto off_box = [&](V3 p) {  // the distance from p to the box, on the ground plane
      const double dx = std::max({box.lo.x - p.x, 0.0, p.x - box.hi.x});
      const double dz = std::max({box.lo.z - p.z, 0.0, p.z - box.hi.z});
      return std::sqrt(dx * dx + dz * dz);
    };
    size_t open_n = 0, open_bad = 0, foot_n = 0, sky_bad = 0, top_n = 0, top_bad = 0;
    double foot = 0, worst_open = 1;
    for (uint32_t j = 1; j + 1 < v.h; j++)
      for (uint32_t i = 1; i + 1 < v.w; i++) {
        const size_t k = size_t(j) * v.w + i;
        if (!v.hit[k]) {
          sky_bad += m[k] < 0.999;
          continue;
        }
        const V3 p = v.points[k];
        if (std::fabs(p.y) < 1e-3) {
          const double d = off_box(p);
          if (d > radius + 0.5) open_n++, open_bad += m[k] < 0.98, worst_open = std::min(worst_open, m[k]);
          if (d > 0.02 && d < 0.3) foot_n++, foot += m[k];
        } else if (std::fabs(p.y - box.hi.y) < 1e-3 && p.x > box.lo.x + 0.3 && p.x < box.hi.x - 0.3 &&
                   p.z > box.lo.z + 0.3 && p.z < box.hi.z - 0.3) {
          top_n++, top_bad += m[k] < 0.98;
        }
      }
    foot = foot_n ? foot / foot_n : 1;
    char what[200];
    std::snprintf(what, sizeof what,
                  "occlusion (%s depth): open ground and sky open (%zu of %zu ground under 0.98, worst %.3f; sky %zu)",
                  reversed ? "reversed" : "float", open_bad, open_n, worst_open, sky_bad);
    failures += Check(open_n > 1000 && open_bad * 1000 <= open_n && sky_bad == 0, what);
    std::snprintf(what, sizeof what, "occlusion (%s depth): the foot of the walls darker (mean %.3f over %zu pixels)",
                  reversed ? "reversed" : "float", foot, foot_n);
    failures += Check(foot_n > 50 && foot < 0.8, what);
    std::snprintf(what, sizeof what, "occlusion (%s depth): the box's top open (%zu of %zu under 0.98)",
                  reversed ? "reversed" : "float", top_bad, top_n);
    failures += Check(top_n > 50 && top_bad * 100 <= top_n, what);
    const uint32_t samples =
        ((v.w + kRtOcclusionBlock - 1) / kRtOcclusionBlock) * ((v.h + kRtOcclusionBlock - 1) / kRtOcclusionBlock);
    std::snprintf(what, sizeof what, "occlusion (%s depth): %u rays for %u samples, %u blocked",
                  reversed ? "reversed" : "float", rays, samples, blocked);
    failures += Check(rays > 0 && rays <= samples * kRtOcclusionRays && blocked > 0 && blocked < rays, what);
  }
  {
    View v;
    if (MakeView(g, 3440, 1440, {0, 6, -6}, {0, 0, 10}, {box}, true, &v)) {
      std::vector<double> m;
      uint32_t rays = 0, blocked = 0;
      double ms = 0;
      if (Occlusion(g, v, radius, 20, &m, &rays, &blocked, &ms))
        std::printf("occlusion at 3440 x 1440: %.2f ms GPU, the least of 20 frames (%u rays)\n", ms, rays);
    }
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
    if (!g.Init()) return Check(false, "the passes made");
    failures += GpuChecks(g);
    failures += OcclusionChecks(g);
    Timing(g, 100);
    Timing(g, 380);
    Timing(g, 12, false);
    return failures ? 1 : 0;
  }
}
