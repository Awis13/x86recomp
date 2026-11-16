// Ray-traced shadow masks of local lights (gfx/rt_shadow.h: RtLocalLight,
// RtLocalLightBox, RtLocalLightBounds, RtLightParamsFor, RtLightMaskCache;
// RtShadowMasker::Encode with RtLightParams). On the CPU: a light read from
// a draw's registers (its radius's component, the cone only with all its
// registers); the bounds on screen (they hold every pixel the light
// reaches, smaller than the buffer; the whole buffer with the camera inside
// the light's box; the box clipped at the camera's plane beside it; empty
// off screen, wholly behind the camera, without a radius); a spot light's
// box (its cone's part of the sphere, either sign of the scale, none for a
// zero scale or a cone of no angle) and bounds; the parameters (relative to
// the camera, refused without a radius or pixels); the backend's choices
// over a frame's draws (one mask per light, the same light's second draw
// reusing it, another light traced, no entry for a light all lit, the
// least recently used taken over past the cache, traced again after a
// depth write and in the next frame, one snapshot for every light of a
// depth, a light past the frame's budget without a mask all lit; still
// lights over their budget across frames on their own earlier masks, the
// traced set turning round; moving lamps traced every frame ahead of
// them). On a GPU with hardware ray tracing: a ground plane and a box with a point light above, a
// synthetic depth buffer from a known camera (stored reversed in a depth
// texture, and not reversed in a float one): every judged ground pixel
// against the box's analytic shadow from the light; pixels beyond the
// radius lit and counted untraced, those within the lamp's gap too; an
// occluder above the light casting nothing; a spot light's cone (lit and
// untraced outside it); a lamp's housing around the light blocking every
// ray without a gap and none with one; two lights traced from one
// snapshot; only the bounds' pixels written (the rest keep what the mask
// held).
//
//   rt_light_shadow_test        (GPU part skipped, exit 77, without a GPU that traces rays)

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
double Len(V3 a) { return std::sqrt(Dot(a, a)); }

using Mat = float[4][4];

// D3D's left-handed camera at `eye` looking at `at` (y up), its
// perspective (depth 0..1 at near..far), as world to clip rows.
void Camera(V3 eye, V3 at, double fov_y, double aspect, double n, double f, double vp[16]) {
  const V3 fw = Unit(at - eye), rt = Unit(Cross({0, 1, 0}, fw)), up = Cross(fw, rt);
  const double view[16] = {rt.x, up.x, fw.x, 0, rt.y, up.y, fw.y, 0, rt.z, up.z, fw.z, 0,
                           -Dot(rt, eye), -Dot(up, eye), -Dot(fw, eye), 1};
  const double ys = 1.0 / std::tan(fov_y / 2), xs = ys / aspect;
  double proj[16] = {};
  proj[0] = xs;
  proj[5] = ys;
  proj[10] = f / (f - n);
  proj[11] = 1;
  proj[14] = -n * f / (f - n);
  Mul4(view, proj, vp);
}

// The identity world matrix and world * vp as float rows.
void Identity(Mat m) {
  std::memset(m, 0, sizeof(Mat));
  m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
}
void Rows(const double* vp, Mat out) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) out[r][k] = float(vp[r * 4 + k]);
}

// A frame whose camera is `vp` (its exact inverse given).
bool FrameOf(const double* vp, RtFrame* f) {
  double inv[16];
  if (!Invert4(vp, inv)) return false;
  for (int k = 0; k < 16; k++) f->inv_view_proj[k] = float(inv[k]);
  std::memcpy(f->inv_view_proj_exact, inv, sizeof inv);
  return true;
}

// Pixel (i, j) of a w x h buffer where world point p lands (D3D9's centres).
bool Project(const double* vp, V3 p, uint32_t w, uint32_t h, double* i, double* j) {
  double c[4];
  for (int k = 0; k < 4; k++) c[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
  if (!(c[3] > 0)) return false;
  *i = (c[0] / c[3] + 1) * 0.5 * w;
  *j = (1 - c[1] / c[3]) * 0.5 * h;
  return true;
}

// An axis-aligned box.
struct Box {
  V3 lo, hi;
};
// Whether the segment a -> b crosses box `b` grown by `grow` (negative:
// shrunk).
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

// ---- CPU

int CpuChecks() {
  int failures = 0;
  {
    // A light from a draw's registers.
    float c[224][4] = {};
    c[78][0] = 1, c[78][1] = 2, c[78][2] = 3;
    c[80][0] = 0.25f, c[80][1] = 9;
    c[79][1] = -1;
    c[81][0] = 0.5f, c[82][0] = 4;
    RayShadow s;
    s.local = true;
    s.position_register = 78;
    s.radius_register = 80;
    s.radius_component = 0;
    s.cone = true;
    s.direction_register = 79;
    s.cone_offset_register = 81;
    s.cone_scale_register = 82;
    s.lamp_gap = 0.5f;
    RtLocalLight l = RtLocalLightFor(s, c);
    bool ok = l.position[0] == 1 && l.position[2] == 3 && l.inv_sq_radius == 0.25f && l.cone &&
              l.direction[1] == -1 && l.cone_offset == 0.5f && l.cone_scale == 4 && l.lamp_gap == 0.5f &&
              RtLocalLightRadius(l) == 2.0;
    s.radius_component = 1;
    ok = ok && RtLocalLightFor(s, c).inv_sq_radius == 9;
    s.cone_scale_register = 224;  // a cone register out of range: no cone
    ok = ok && !RtLocalLightFor(s, c).cone;
    s.cone = false;
    s.cone_scale_register = 82;
    ok = ok && !RtLocalLightFor(s, c).cone;
    const RtLocalLight none = RtLocalLightFor(s, nullptr);
    ok = ok && none.inv_sq_radius == 0 && RtLocalLightRadius(none) == 0 && none.lamp_gap == 0.5f;
    RtLocalLight neg = l;
    neg.inv_sq_radius = -1;
    ok = ok && RtLocalLightRadius(neg) == 0;
    RtLocalLight m = l;
    ok = ok && RtSameLocalLight(l, m);
    m.cone_scale = 4.5f;
    ok = ok && !RtSameLocalLight(l, m);
    m = l;
    m.position[1] += 1e-4f;
    ok = ok && !RtSameLocalLight(l, m);
    failures += Check(ok, "a light from a draw's registers (the radius's component; a cone only with all its "
                          "registers; none without constants); the same light bit for bit");
  }
  {
    // The bounds on screen.
    constexpr uint32_t w = 320, h = 240;
    const V3 eye{0, 14, -14};
    double vp[16];
    Camera(eye, {0, 0, 2}, 1.0, double(w) / h, 0.5, 3000.0, vp);
    RtFrame f;
    bool ok = FrameOf(vp, &f);
    RtLocalLight l;
    const V3 lp{2, 3, 4};
    l.position[0] = float(lp.x), l.position[1] = float(lp.y), l.position[2] = float(lp.z);
    l.inv_sq_radius = 1.0f / 9.0f;  // radius 3
    const float jitter[2] = {0.4f, -0.3f};
    const RtRect r = RtLocalLightBounds(f, w, h, jitter, l);
    // Every point the light reaches (on a grid over its sphere, solid, and
    // in its cone where it has one) that lands on screen within `q`.
    auto holds = [&](const RtLocalLight& li, const RtRect& q, const float* jit) {
      const V3 c{li.position[0], li.position[1], li.position[2]};
      const double rad = RtLocalLightRadius(li);
      for (int s = 1; s <= 4; s++)
        for (int a = 0; a <= 24; a++)
          for (int b = 0; b <= 48; b++) {
            const double th = M_PI * a / 24, ph = 2 * M_PI * b / 48;
            const V3 u{std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)};
            if (li.cone && !((Dot(u, {li.direction[0], li.direction[1], li.direction[2]}) - li.cone_offset) *
                                 li.cone_scale >
                             0))
              continue;
            const V3 p = c + u * (rad * s / 4.0 * 0.999);
            double i, j;
            if (!Project(vp, p, w, h, &i, &j)) continue;  // behind the camera: not seen
            if (jit) i += jit[0], j += jit[1];
            const bool on = i >= 0 && i < w && j >= 0 && j < h;
            if (on && (i < double(q.x0) || i >= double(q.x1) || j < double(q.y0) || j >= double(q.y1)))
              return false;
          }
      return true;
    };
    ok = ok && holds(l, r, jitter) && !r.Empty() && r.Area() < uint64_t(w) * h / 2 && r.x1 <= w && r.y1 <= h;
    // The camera within the light's box: the whole buffer. Beside it (the
    // box spans the camera's plane, the camera outside it, the light to
    // its right): the box clipped there, the screen's right part.
    RtLocalLight near = l;
    near.position[0] = float(eye.x + 1), near.position[1] = float(eye.y), near.position[2] = float(eye.z);
    const RtRect rn = RtLocalLightBounds(f, w, h, nullptr, near);
    RtLocalLight beside = l;
    beside.position[0] = float(eye.x + 4), beside.position[1] = float(eye.y), beside.position[2] = float(eye.z);
    const RtRect rb = RtLocalLightBounds(f, w, h, nullptr, beside);
    ok = ok && rn.x0 == 0 && rn.y0 == 0 && rn.x1 == w && rn.y1 == h && holds(beside, rb, nullptr) &&
         rb.x0 > w / 2 && rb.x1 == w;
    std::printf("  beside the camera: [%u, %u) x [%u, %u)\n", rb.x0, rb.x1, rb.y0, rb.y1);
    // Off screen (far to the side), wholly behind the camera, no radius:
    // nothing.
    RtLocalLight side = l, behind = l, dark = l;
    side.position[0] = 400;
    behind.position[1] = 20, behind.position[2] = -40;
    dark.inv_sq_radius = 0;
    ok = ok && RtLocalLightBounds(f, w, h, nullptr, side).Empty() &&
         RtLocalLightBounds(f, w, h, nullptr, behind).Empty() && RtLocalLightBounds(f, w, h, nullptr, dark).Empty();
    // A singular camera: the whole buffer.
    RtFrame singular;
    const RtRect rs = RtLocalLightBounds(singular, w, h, nullptr, l);
    ok = ok && rs.x1 == w && rs.y1 == h && rs.x0 == 0;
    std::printf("  bounds: [%u, %u) x [%u, %u) of %u x %u\n", r.x0, r.x1, r.y0, r.y1, w, h);
    failures += Check(ok, "a light's bounds on screen: every pixel of its sphere (jittered), less than the buffer; "
                          "the whole buffer with the camera in its box or a singular camera; beside the camera the "
                          "box clipped at its plane (every visible pixel, the right part of the screen); nothing off "
                          "screen, behind the camera or without a radius");
    // A spot light: the box around its cone, within the sphere's, every
    // point of the cone inside; its bounds within the point light's.
    RtLocalLight spot = l;
    spot.cone = true;
    spot.direction[0] = 0, spot.direction[1] = -2, spot.direction[2] = 0;  // as the register holds it
    spot.cone_offset = float(2 * std::cos(0.5));  // half angle 0.5 rad
    spot.cone_scale = 3;
    double slo[3], shi[3], plo[3], phi[3];
    ok = RtLocalLightBox(spot, slo, shi) && RtLocalLightBox(l, plo, phi);
    const double sx = 3 * std::sin(0.5), sy = 3;  // the cone's extent sideways and down
    ok = ok && std::fabs(slo[0] - (2 - sx)) < 1e-4 && std::fabs(shi[0] - (2 + sx)) < 1e-4 &&
         std::fabs(slo[1] - (3 - sy)) < 1e-4 && std::fabs(shi[1] - 3) < 1e-6 && std::fabs(slo[2] - (4 - sx)) < 1e-4 &&
         std::fabs(plo[1] - 0) < 1e-6 && std::fabs(phi[1] - 6) < 1e-6;
    const RtRect rsp = RtLocalLightBounds(f, w, h, jitter, spot);
    ok = ok && holds(spot, rsp, jitter) && rsp.Area() < r.Area() && rsp.x0 >= r.x0 && rsp.x1 <= r.x1 &&
         rsp.y0 >= r.y0 && rsp.y1 <= r.y1;
    // A negative scale: the cone the other way (up); none with a zero scale
    // or an offset past the direction's length; a wide cone the sphere's box.
    RtLocalLight up = spot, flat = spot, past = spot, wide = spot;
    up.cone_scale = -3;
    up.cone_offset = -spot.cone_offset;
    flat.cone_scale = 0;
    past.cone_offset = 2.5f;
    wide.cone_offset = -0.5f;
    double ulo[3], uhi[3], wlo[3], whi[3];
    ok = ok && RtLocalLightBox(up, ulo, uhi) && std::fabs(ulo[1] - 3) < 1e-6 && std::fabs(uhi[1] - 6) < 1e-4 &&
         !RtLocalLightBox(flat, ulo, uhi) && !RtLocalLightBox(past, ulo, uhi) && RtLocalLightBox(wide, wlo, whi) &&
         wlo[1] == plo[1] && whi[1] == phi[1] && RtLocalLightBounds(f, w, h, nullptr, flat).Empty();
    std::printf("  spot: [%u, %u) x [%u, %u)\n", rsp.x0, rsp.x1, rsp.y0, rsp.y1);
    failures += Check(ok, "a spot light's box: its cone's part of the sphere (the direction as given, a negative "
                          "scale turning it round), none for a zero scale or a cone of no angle, a wide one the "
                          "sphere's; its bounds within the point light's, every pixel of the cone inside");
    // The parameters: relative to the camera, the cone and gap, the pixels.
    RtShadowParams p;
    RtLightParams q;
    l.cone = true;
    l.direction[1] = -2;
    l.cone_offset = 0.3f;
    l.cone_scale = 6;
    l.lamp_gap = 0.4f;
    ok = RtLightParamsFor(f, w, h, true, true, nullptr, l, r, &p, &q);
    ok = ok && q.local == 1 && q.cone == 1 && std::fabs(q.position[0] + p.eye[0] - 2) < 1e-4f &&
         std::fabs(q.position[1] + p.eye[1] - 3) < 1e-4f && std::fabs(q.position[2] + p.eye[2] - 4) < 1e-4f &&
         std::fabs(p.eye[1] - 14) < 1e-3f && q.position[3] == 1.0f / 9.0f && q.direction[1] == -2 &&
         q.direction[3] == 0.3f && q.cone_scale == 6 && q.lamp_gap == 0.4f && q.origin[0] == r.x0 &&
         q.origin[1] == r.y0 && q.extent[0] == r.x1 - r.x0 && q.extent[1] == r.y1 - r.y0 && p.size[0] == w;
    ok = ok && !RtLightParamsFor(f, w, h, true, true, nullptr, dark, r, &p, &q) &&
         !RtLightParamsFor(f, w, h, true, true, nullptr, l, RtRect{}, &p, &q) &&
         !RtLightParamsFor(f, w, h, true, true, nullptr, l, RtRect{0, 0, w + 1, h}, &p, &q);
    failures += Check(ok, "a light's parameters: its position relative to the camera, its radius, cone and gap, "
                          "its pixels; none without a radius or pixels");
  }
  {
    // The backend's choices over a frame's draws (metal_device.mm's
    // RtLightShadowFor): the entry, its memo, the snapshot.
    int depth = 0;
    RtShadowWant w;
    w.depth = &depth;
    w.epoch = 5;
    w.depth_camera = w.scene_camera = 2;
    w.main_size = true;
    w.light[2] = 1;
    RtLightMaskCache cache;
    RtSnapshotMemo snap;
    uint64_t serial = 10;
    int traces = 0, snapshots = 0, reused = 0, whites = 0, budgeted = 0;
    uint32_t budget = 16;
    auto used = [&] {
      int n = 0;
      for (const auto& e : cache.entry) n += e.used;
      return n;
    };
    auto draw = [&](float x, const RtShadowWant& want) {
      RtLocalLight l;
      l.position[0] = x;
      l.inv_sq_radius = 0.01f;
      int e = RtLightMaskFind(&cache, l);
      if (e >= 0) {
        switch (RtShadowNext(&cache.entry[e].memo, want)) {
          case RtShadowStep::Reuse: reused++; return e;
          case RtShadowStep::White: whites++; return e;
          case RtShadowStep::Trace: break;
        }
      } else if (RtShadowWhyWhite(want) != RtShadowWhite::None) {
        whites++;
        return e;
      }
      switch (RtLightMaskChoose(&cache, e, budget)) {
        case RtLightStep::Trace: break;
        case RtLightStep::Older:
          cache.entry[e].memo.white = false;
          return e;
        case RtLightStep::Budget:
          if (e < 0) {
            e = RtLightMaskTake(&cache, l);
            RtShadowNext(&cache.entry[e].memo, want);
            RtLightMaskChoose(&cache, e, budget);
          }
          cache.entry[e].memo.white = true;
          budgeted++;
          return e;
      }
      if (e < 0) {
        e = RtLightMaskTake(&cache, l);
        RtShadowNext(&cache.entry[e].memo, want);
        cache.entry[e].wanted = cache.frame;
      }
      traces++;
      RtLightMaskTraced(&cache, e, 100);
      if (!RtSnapshotReuse(&snap, want, serial)) snapshots++;
      return e;
    };
    // A light's two draws, another light's, the first again.
    const int a = draw(1, w), a2 = draw(1, w), b = draw(2, w), a3 = draw(1, w);
    bool ok = a == a2 && a == a3 && b != a && traces == 2 && reused == 2 && snapshots == 1;
    // A light all lit (another camera's depth) takes no entry.
    RtShadowWant other = w;
    other.depth_camera = 9;
    ok = ok && draw(5, other) == -1 && whites == 1 && used() == 2;
    // 16 more lights: the least recently used (light 1, then 2) taken over.
    budget = 100;
    for (int k = 0; k < 16; k++) draw(10.0f + k, w);
    ok = ok && traces == 18 && snapshots == 1 && used() == 16;
    draw(2, w);  // taken over: traced again
    ok = ok && traces == 19;
    // The depth written: traced again, a new snapshot.
    RtShadowWant x = w;
    x.epoch = 6;
    draw(2, x);
    ok = ok && traces == 20 && snapshots == 2;
    // The next frame: everything traced again, one snapshot.
    cache = {};
    snap = {};
    serial++;
    draw(2, x);
    draw(3, x);
    draw(2, x);
    ok = ok && traces == 22 && snapshots == 3;
    // A budget of two lights a frame: a third all lit (no mask of its own
    // yet), its entry taken (first in line next frame); the first two still
    // reused.
    cache = {};
    snap = {};
    serial++;
    budget = 2;
    draw(1, x);
    draw(2, x);
    const int third = draw(3, x);
    draw(3, x);
    draw(1, x);
    ok = ok && traces == 24 && budgeted == 1 && third >= 0 && used() == 3 && reused == 5 &&
         !cache.entry[third].mask && cache.entry[third].wanted == cache.frame;
    failures += Check(ok, "a frame's lights: one mask per light, its second draw reusing it, another light traced, "
                          "none taken for a light all lit, the least recently used taken over past the cache; traced "
                          "again after a depth write and in the next frame; one snapshot for every light of the same "
                          "depth; a light past the frame's budget without a mask all lit, its entry kept");
  }
  // Lights over the frames as RtLightShadowFor asks for them, a budget of 8:
  // each light's first draw of a frame (in `order`) needs a trace; per
  // frame, which were traced, which drew on an older mask of their own (and
  // its age), which were all lit, and whether an entry was taken.
  struct Outcome {
    std::vector<int> traced, older, lit;
    uint64_t oldest = 0;
    int taken = 0;
  };
  auto run = [](int frames, int lights, uint32_t budget, auto light_at) {
    int depth = 0;
    RtShadowWant w;
    w.depth = &depth;
    w.epoch = 5;
    w.depth_camera = w.scene_camera = 2;
    w.main_size = true;
    w.light[2] = 1;
    RtLightMaskCache cache;
    std::vector<Outcome> out(size_t(frames) + 1);
    for (int frame = 1; frame <= frames; frame++) {
      Outcome& o = out[size_t(frame)];
      for (int k = 0; k < lights; k++) {
        const RtLocalLight l = light_at(frame, k);
        int e = RtLightMaskFind(&cache, l);
        if (e >= 0 && RtShadowNext(&cache.entry[e].memo, w) != RtShadowStep::Trace) {
          o.lit.push_back(-1 - k);  // (each light's first draw of a frame needs a trace)
          continue;
        }
        const RtLightStep step = RtLightMaskChoose(&cache, e, budget);
        if (e < 0) {
          e = RtLightMaskTake(&cache, l);
          RtShadowNext(&cache.entry[e].memo, w);
          RtLightMaskChoose(&cache, e, budget);
          o.taken++;
        }
        if (step == RtLightStep::Trace) {
          RtLightMaskTraced(&cache, e, 100 + uint64_t(k));
          o.traced.push_back(k);
        } else if (step == RtLightStep::Older) {
          o.older.push_back(k);
          o.oldest = std::max(o.oldest, cache.frame - cache.entry[e].traced);
        } else {
          o.lit.push_back(k);
        }
      }
      RtLightMaskEndFrame(&cache, budget);
    }
    return out;
  };
  // Every light traced within each two frames from `from` on.
  auto turns = [](const std::vector<Outcome>& out, int lights, int from) {
    for (size_t f = size_t(from); f + 1 < out.size(); f++) {
      std::vector<bool> every(static_cast<size_t>(lights));
      for (size_t g : {f, f + 1})
        for (int k : out[g].traced) every[size_t(k)] = true;
      for (bool b : every)
        if (!b) return false;
    }
    return true;
  };
  // 12 lights that stay where they are, drawn in the same order every
  // frame: the first two frames in draw order (a light taken that frame
  // holds no reserved trace), then the four without a mask, then the
  // oldest: none all lit after its first trace, the traced set turning
  // round.
  {
    const auto out = run(6, 12, 8, [](int, int k) {
      RtLocalLight l;
      l.position[0] = 100.0f * float(k + 1);
      l.inv_sq_radius = 0.01f;
      return l;
    });
    const std::vector<int> want12 = {0, 1, 2, 3, 4, 5, 6, 7}, want3 = {4, 5, 6, 7, 8, 9, 10, 11},
                           want4 = {0, 1, 2, 3, 8, 9, 10, 11};
    bool ok = out[1].traced == want12 && out[2].traced == want12 && out[3].traced == want3 && out[4].traced == want4;
    for (int f = 1; f <= 6; f++) {
      ok = ok && out[size_t(f)].traced.size() == 8 && out[size_t(f)].lit.size() == (f <= 2 ? 4u : 0u) &&
           out[size_t(f)].oldest <= 2 && out[size_t(f)].taken == (f == 1 ? 12 : 0);
    }
    ok = ok && turns(out, 12, 3);
    failures += Check(ok, "12 lights, 8 traced a frame: none all lit after its first trace (an earlier frame's mask "
                          "of its own, at most 2 frames old), the traced set turning round so every light is traced "
                          "within two frames");
  }
  // 8 lamps of moving vehicles (a new position every frame) drawn before 8
  // street lights, a budget of 4 still lights a frame: each lamp keeps its
  // entry and, from the frame it is first seen moved, is traced every frame
  // (never an earlier frame's mask); the street lights turn round through
  // their budget, none all lit after its first trace.
  {
    const auto out = run(10, 16, 4, [](int frame, int k) {
      RtLocalLight l;
      l.inv_sq_radius = 0.01f;  // radius 10
      l.position[0] = 100.0f * float(k + 1) + (k < 8 ? 0.3f * float(frame) : 0.0f);
      l.position[1] = k < 8 ? 0.1f * float(frame) : 0.0f;
      if (k < 8) {
        l.cone = true;
        l.direction[0] = 1;
        l.cone_offset = 0.5f;
        l.cone_scale = 2;
      }
      return l;
    });
    bool ok = true;
    std::vector<bool> once(16);
    for (int f = 1; f <= 10; f++) {
      const Outcome& o = out[size_t(f)];
      int lamps = 0, still = 0;
      for (int k : o.traced) (k < 8 ? lamps : still)++;
      ok = ok && o.taken == (f == 1 ? 16 : 0) && (f == 1 || (lamps == 8 && still == 4));
      for (int k : o.older) ok = ok && k >= 8;
      for (int k : o.lit) ok = ok && k >= 0 && !once[size_t(k)];
      for (int k : o.traced) once[size_t(k)] = true;
    }
    // (Frame 1: the first four drawn, all lamps, as still lights: none
    // moved yet.)
    const std::vector<int> want1 = {0, 1, 2, 3};
    ok = ok && out[1].traced == want1;
    std::vector<Outcome> still(out.size());
    for (size_t f = 0; f < out.size(); f++)
      for (int k : out[f].traced)
        if (k >= 8) still[f].traced.push_back(k - 8);
    ok = ok && turns(still, 8, 3);
    failures += Check(ok, "8 moving lamps before 8 street lights, 4 still ones a frame: every lamp traced every frame "
                          "once seen moved (never an earlier frame's mask), the street lights turning round within two "
                          "frames, none all lit after its first trace");
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

constexpr uint32_t kW = 640, kH = 480;
constexpr uint8_t kUntouched = 0x40;  // the masks' texels before a trace

// A mesh of triangles (FLOAT3 positions, a triangle list).
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

const Box kBox{{-1, 0, -1}, {1, 2, 1}};  // on the ground, below the light

struct Light {
  V3 pos;
  double radius;
  bool cone = false;
  V3 dir;
  double offset = 0, scale = 0;
  double gap = 0;
};

struct Case {
  const char* what;
  bool reversed;           // depth stored reversed in a depth texture (else not, in a float one)
  Light light;
  bool ceiling = false;    // a quad above the light (y = 7)
  bool housing = false;    // a box around the light (its half side 0.3)
  bool two = false;        // a second light traced from the same snapshot into another mask
  uint32_t cap = kRtLightsCap;  // the lights traced at once: at most this many a point
  bool second_given = false;    // the second light `second` (else one off to the side)
  Light second;
};

struct Result {
  bool ran = false;
  int judged = 0, shadowed = 0, lit = 0, wrong = 0;
  int out_judged = 0, out_wrong = 0;   // ground pixels clearly beyond the light's reach
  int box_pixels = 0;
  int untouched_wrong = 0, written = 0;  // texels outside the bounds changed; inside them written
  int missed_by_bounds = 0;              // pixels reached by the light outside the bounds
  uint32_t rays = 0, hits = 0, outside = 0, in_gap = 0;
  uint32_t expect_in_rect = 0;           // pixels in the bounds, not sky
  RtRect rect;
  bool second_ok = true;
  uint32_t instances = 0;
  // The same light(s) traced at once (EncodeLights) at every pixel and at
  // one of every 2 x 2: texels compared with the lights' own masks (inside
  // their bounds; lit outside), those that differ, distances wrong, rays.
  int set_compared[2] = {}, set_wrong[2] = {}, set_dist_wrong[2] = {};
  uint32_t set_rays[2] = {}, set_capped[2] = {};
};

// Whether the ground point p is reached by `l` (with a margin: 1 clearly
// reached, -1 clearly not, 0 too close to tell).
int Reach(const Light& l, V3 p, double margin) {
  const V3 to = l.pos - p;
  const double d = Len(to);
  int r = d < l.radius - margin ? 1 : d > l.radius + margin ? -1 : 0;
  if (l.cone && r != -1) {
    // (dot(-l, dir) - offset) scale against the angle's margin.
    const double c = Dot(to * (-1.0 / d), l.dir), ang = std::acos(std::clamp(c, -1.0, 1.0));
    const double edge = std::acos(std::clamp(l.offset, -1.0, 1.0)), da = margin / d;
    const int cr = ang < edge - da ? 1 : ang > edge + da ? -1 : 0;
    r = cr == -1 ? -1 : std::min(r, cr);
  }
  return r;
}

// Whether the lights traced at once leave light `l` untraced at point p
// on surface `kind` (1 ground, 2 the box): the surface facing away from it
// (in shadow), or the light too faint there (falloff times cone below
// 1/64: lit); there the light's own mask may differ.
bool Untraced(const Light& l, int kind, V3 p) {
  if (!kind) return false;
  const V3 to = l.pos - p;
  const double d2 = Dot(to, to), f = 1.0 - d2 / (l.radius * l.radius);
  if (f <= 0.0) return false;
  double cw = 1.0;
  if (l.cone) cw = std::min(1.0, (Dot(to * (-1.0 / std::sqrt(d2)), l.dir) - l.offset) * l.scale);
  if (f * f * cw < 1.0 / 64.0 + 1e-3) return true;
  V3 n{0, 1, 0};
  if (kind == 2) {  // the box's face: the axis the point is nearest its side on
    const V3 c = (kBox.lo + kBox.hi) * 0.5, h = (kBox.hi - kBox.lo) * 0.5, r = p - c;
    const double ex = std::fabs(std::fabs(r.x) - h.x), ey = std::fabs(std::fabs(r.y) - h.y),
                 ez = std::fabs(std::fabs(r.z) - h.z);
    n = ex <= ey && ex <= ez ? V3{r.x > 0 ? 1.0 : -1.0, 0, 0}
        : ey <= ez           ? V3{0, r.y > 0 ? 1.0 : -1.0, 0}
                             : V3{0, 0, r.z > 0 ? 1.0 : -1.0};
  }
  return Dot(n, to) < 0.02 * std::sqrt(d2);
}

Result Run(Gpu& g, const Case& c) {
  Result res;
  std::unique_ptr<RtScene> s = MakeRtScene(g.dev, {});
  if (!s) return res;
  g.arena.used = 0;
  s->SetAllocator(ArenaAlloc, &g.arena);
  const V3 eye{0, 9, -9};
  double vp[16], inv[16];
  Camera(eye, {0, 0, 1}, 1.0, double(kW) / kH, 0.5, 3000.0, vp);
  Invert4(vp, inv);
  Mat w, mw;
  Identity(w);
  Rows(vp, mw);
  Mesh ground = QuadMesh(0, 60, 1), box = BoxMesh(kBox, 2), ceiling = QuadMesh(7, 6, 3);
  const Box housing{c.light.pos - V3{0.3, 0.3, 0.3}, c.light.pos + V3{0.3, 0.3, 0.3}};
  Mesh hm = BoxMesh(housing, 4);
  s->BeginFrame(1, 0);
  s->Add(ground.key, ground.g, w, mw, 1, 4);
  s->Add(box.key, box.g, w, mw, 1, 4);
  if (c.ceiling) s->Add(ceiling.key, ceiling.g, w, mw, 1, 4);
  if (c.housing) s->Add(hm.key, hm.g, w, mw, 1, 4);
  const RtFrame* f = s->Snapshot();
  res.instances = f ? f->instances : 0;

  // The depth buffer: per pixel the camera's ray, its first hit on the
  // ground or the box (the scene the camera sees: the ceiling and housing
  // stand only in the rays' way).
  std::vector<float> depth(size_t(kW) * kH);
  std::vector<V3> points(depth.size());
  std::vector<int> surface(depth.size(), 0);  // 0 sky, 1 ground, 2 box
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
      const size_t at = size_t(j) * kW + i;
      double t = 2, z = 1;
      if (d.y < 0) {
        const double tg = -pa.y / d.y;
        if (tg > 0 && tg < 1) t = tg, surface[at] = 1;
      }
      const double tb = RayBox(pa, d, kBox);
      if (tb > 0 && tb < t) t = tb, surface[at] = 2;
      if (surface[at]) {
        const V3 p = pa + d * t;
        double cc[4];
        for (int k = 0; k < 4; k++) cc[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
        z = cc[2] / cc[3];
        points[at] = p;
      }
      depth[at] = float(c.reversed ? 1.0 - z : z);
    }

  id<MTLCommandBuffer> cb = [g.queue commandBuffer];
  id<MTLTexture> dt;
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
  // The light as a draw's registers give it.
  auto local = [](const Light& li) {
    RtLocalLight l;
    l.position[0] = float(li.pos.x), l.position[1] = float(li.pos.y), l.position[2] = float(li.pos.z);
    l.inv_sq_radius = float(1.0 / (li.radius * li.radius));
    l.cone = li.cone;
    l.direction[0] = float(li.dir.x), l.direction[1] = float(li.dir.y), l.direction[2] = float(li.dir.z);
    l.cone_offset = float(li.offset);
    l.cone_scale = float(li.scale);
    l.lamp_gap = float(li.gap);
    return l;
  };
  const RtLocalLight l = local(c.light);
  // The second light: off to the side, no box between it and most of the
  // ground near it.
  Light other = c.light;
  other.pos = V3{-6, 4, 6};
  other.radius = 6;
  other.cone = false;
  if (c.second_given) other = c.second;
  const RtLocalLight l2 = local(other);
  if (!f || !f->tlas) return res;
  res.rect = RtLocalLightBounds(*f, kW, kH, nullptr, l);
  const RtRect rect2 = RtLocalLightBounds(*f, kW, kH, nullptr, l2);
  RtShadowParams p, p2;
  RtLightParams lp, lp2;
  if (!RtLightParamsFor(*f, kW, kH, c.reversed, c.reversed, nullptr, l, res.rect, &p, &lp)) return res;
  if (c.two && !RtLightParamsFor(*f, kW, kH, c.reversed, c.reversed, nullptr, l2, rect2, &p2, &lp2)) return res;
  // Masks holding kUntouched before the trace.
  auto mask_tex = [&] {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                  width:kW
                                                                                 height:kH
                                                                               mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> t = [g.dev newTextureWithDescriptor:td];
    const std::vector<uint8_t> fill(size_t(kW) * kH, kUntouched);
    [t replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:fill.data() bytesPerRow:kW];
    return t;
  };
  id<MTLTexture> mask = mask_tex(), mask2 = c.two ? mask_tex() : nil;
  s->EncodeFrame(cb, f);
  id<MTLBuffer> counts = [g.dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
  std::memset(counts.contents, 0, 32);
  g.masker->Encode(cb, f, dt, mask, p, counts, 0, nil, &lp);
  if (c.two) g.masker->Encode(cb, f, dt, mask2, p2, counts, 16, nil, &lp2);
  // The light (and the second) traced at once, at steps 1 and 2.
  double set_eye[3];
  RtLightGpu lg[2];
  const bool set_ok =
      RtLightsEye(*f, set_eye) && RtLightGpuFor(l, set_eye, &lg[0]) && RtLightGpuFor(l2, set_eye, &lg[1]);
  id<MTLTexture> set_masks[2] = {}, set_dist[2] = {};
  id<MTLBuffer> set_counts = [g.dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
  std::memset(set_counts.contents, 0, 32);
  for (uint32_t k = 0; k < 2 && set_ok; k++) {
    RtLightsParams sp{};
    sp.count = c.two ? 2 : 1;
    sp.step = k + 1;
    sp.cap = c.cap;
    // (At step 2 the samples turning, as in a jittered scene: frame 1.)
    sp.stride = k ? RtLightStride(sp.step) : 0;
    sp.frame = 1;
    sp.low[0] = (kW + sp.step - 1) / sp.step;
    sp.low[1] = (kH + sp.step - 1) / sp.step;
    MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                  width:sp.low[0]
                                                                                 height:sp.low[1]
                                                                              mipmapped:NO];
    md.textureType = MTLTextureType2DArray;
    md.arrayLength = kRtLightSlots;
    md.storageMode = MTLStorageModeShared;
    md.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    set_masks[k] = [g.dev newTextureWithDescriptor:md];
    MTLTextureDescriptor* dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                  width:sp.low[0]
                                                                                 height:sp.low[1]
                                                                              mipmapped:NO];
    dd.storageMode = MTLStorageModeShared;
    dd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    set_dist[k] = [g.dev newTextureWithDescriptor:dd];
    g.masker->EncodeLights(cb, f, dt, set_masks[k], set_dist[k], p, sp, lg, set_counts, k * 16);
  }
  if (!g.Commit(cb)) return res;
  const RtFrame* end = s->EndFrame(true);
  id<MTLCommandBuffer> cb2 = [g.queue commandBuffer];
  s->EncodeFrame(cb2, end);
  if (!g.Commit(cb2)) return res;
  res.ran = true;
  const auto* n = static_cast<const uint32_t*>(counts.contents);
  res.rays = n[0], res.hits = n[1], res.outside = n[2], res.in_gap = n[3];

  std::vector<uint8_t> m(size_t(kW) * kH), m2(size_t(kW) * kH);
  [mask getBytes:m.data() bytesPerRow:kW fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];
  if (c.two) [mask2 getBytes:m2.data() bytesPerRow:kW fromRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0];

  // The lights traced at once: every texel's layer as the light's own
  // mask at its pixel (one of every step x step, its top-left), lit outside
  // its bounds; its distance the point's from the camera (0: the sky).
  for (uint32_t k = 0; k < 2 && set_ok; k++) {
    const uint32_t step = k + 1, lw = (kW + step - 1) / step, lh = (kH + step - 1) / step;
    std::vector<uint8_t> layer(size_t(lw) * lh);
    std::vector<float> dist(size_t(lw) * lh);
    [set_dist[k] getBytes:dist.data() bytesPerRow:lw * 4 fromRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:0];
    res.set_rays[k] = static_cast<const uint32_t*>(set_counts.contents)[k * 4];
    res.set_capped[k] = static_cast<const uint32_t*>(set_counts.contents)[k * 4 + 2];
    for (uint32_t li = 0; li < (c.two ? 2u : 1u); li++) {
      [set_masks[k] getBytes:layer.data()
                 bytesPerRow:lw
               bytesPerImage:size_t(lw) * lh
                  fromRegion:MTLRegionMake2D(0, 0, lw, lh)
                 mipmapLevel:0
                       slice:li];
      const std::vector<uint8_t>& own = li ? m2 : m;
      const RtRect& rr = li ? rect2 : res.rect;
      for (uint32_t j = 0; j < lh; j++)
        for (uint32_t i = 0; i < lw; i++) {
          uint32_t pi, pj;
          RtLightSample(i, j, 1, k ? RtLightStride(step) : 0, step, &pi, &pj);
          pi = std::min(pi, kW - 1), pj = std::min(pj, kH - 1);
          const size_t at = size_t(pj) * kW + pi;
          const bool inr = pi >= rr.x0 && pi < rr.x1 && pj >= rr.y0 && pj < rr.y1;
          const uint8_t want = inr ? own[at] : 255;
          res.set_compared[k]++;
          if (layer[size_t(j) * lw + i] != want && !Untraced(li ? other : c.light, surface[at], points[at]))
            res.set_wrong[k]++;
          if (li) continue;
          const float dv = dist[size_t(j) * lw + i];
          const double dw = surface[at] ? Len(points[at] - eye) : 0.0;
          if (std::fabs(dv - dw) > 1e-3 * std::max(dw, 1.0)) res.set_dist_wrong[k]++;
        }
    }
  }

  // Judge: the bounds, then each ground pixel.
  const RtRect& r = res.rect;
  auto in_rect = [](const RtRect& q, uint32_t i, uint32_t j) { return i >= q.x0 && i < q.x1 && j >= q.y0 && j < q.y1; };
  const std::vector<Box> blockers = c.housing && c.light.gap <= 0 ? std::vector<Box>{kBox, housing}
                                                                  : std::vector<Box>{kBox};
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      const size_t at = size_t(j) * kW + i;
      const bool inr = in_rect(r, i, j);
      if (!inr) {
        if (m[at] != kUntouched) res.untouched_wrong++;
      } else {
        if (m[at] == 0 || m[at] == 255) res.written++;
        if (surface[at]) res.expect_in_rect++;
      }
      if (!surface[at]) continue;
      const V3 pw = points[at];
      if (!inr && Reach(c.light, pw, 0.0) == 1) res.missed_by_bounds++;
      if (surface[at] == 2) {
        res.box_pixels++;
        continue;
      }
      double foot = 0;
      if (i + 1 < kW && surface[at + 1]) foot = std::max(foot, Len(points[at + 1] - pw));
      if (j + 1 < kH && surface[at + kW]) foot = std::max(foot, Len(points[at + kW] - pw));
      if (foot == 0) continue;
      const double margin = 3.0 * foot + 0.02;
      const int reach = Reach(c.light, pw, margin);
      if (reach == 0) continue;
      if (reach < 0) {
        if (!inr) continue;  // (untouched there: the light adds nothing)
        res.out_judged++;
        if (m[at] != 255) res.out_wrong++;
        continue;
      }
      // The segment to the light (less the gap): through the blockers?
      const V3 to = c.light.pos - pw;
      const V3 end = pw + to * (1.0 - c.light.gap / Len(to));
      const V3 start = pw + V3{0, 1e-3, 0};
      bool sure_dark = false, sure_lit = true;
      for (const Box& b : blockers) {
        sure_dark = sure_dark || SegmentHits(start, end, b, -margin);
        sure_lit = sure_lit && !SegmentHits(start, end, b, margin);
      }
      if (!sure_dark && !sure_lit) continue;
      res.judged++;
      (sure_dark ? res.shadowed : res.lit)++;
      if ((m[at] > 127) == sure_dark) res.wrong++;
    }
  if (c.two) {
    // The second light: nothing between it and the ground within its reach
    // outside the box's shadow from it; lit everywhere it reaches but
    // there, untouched outside its bounds.
    int lit = 0;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++) {
        const size_t at = size_t(j) * kW + i;
        if (!in_rect(rect2, i, j)) {
          if (m2[at] != kUntouched) res.second_ok = false;
          continue;
        }
        if (surface[at] != 1) continue;
        const V3 pw = points[at];
        if (Len(pw - other.pos) > other.radius - 0.2) continue;
        if (SegmentHits(pw + V3{0, 1e-3, 0}, other.pos, kBox, 0.3)) continue;
        lit++;
        if (m2[at] != 255) res.second_ok = false;
      }
    res.second_ok = res.second_ok && lit > 500;
    std::printf("  second light: %d ground pixels lit, bounds [%u, %u) x [%u, %u)\n", lit, rect2.x0, rect2.x1,
                rect2.y0, rect2.y1);
  }
  std::printf("  %s: the lights at once: %d and %d texels compared, %d and %d unlike their own masks, %d and %d "
              "distances wrong, %u and %u rays\n",
              c.what, res.set_compared[0], res.set_compared[1], res.set_wrong[0], res.set_wrong[1],
              res.set_dist_wrong[0], res.set_dist_wrong[1], res.set_rays[0], res.set_rays[1]);
  std::printf("  %s: bounds [%u, %u) x [%u, %u), %d ground pixels judged (%d in shadow, %d lit), %d wrong; %d "
              "beyond its reach (%d not lit); %d box pixels; %u rays, %u blocked, %u untraced outside; %d texels "
              "outside the bounds changed\n",
              c.what, r.x0, r.x1, r.y0, r.y1, res.judged, res.shadowed, res.lit, res.wrong, res.out_judged,
              res.out_wrong, res.box_pixels, res.rays, res.hits, res.outside, res.untouched_wrong);
  return res;
}

// The light volumes' upsample of a pass's layer at pixel (x, y) whose
// point is `d` from the camera (r_rt_light_mask's, on the CPU): the texels
// of its block and the three beside it toward it, by a tent around the
// pixels they traced and the nearness of their distances.
double Upsample(const std::vector<uint8_t>& m, const std::vector<float>& dist, uint32_t lw, uint32_t lh,
                uint32_t step, uint32_t frame, uint32_t stride, uint32_t x, uint32_t y, double d) {
  const int s = int(step);
  const int ox = std::min(int(x) / s, int(lw) - 1), oy = std::min(int(y) / s, int(lh) - 1);
  const int dx = x >= ox * s + 0.5 * (s - 1) ? 1 : -1, dy = y >= oy * s + 0.5 * (s - 1) ? 1 : -1;
  double sum = 0, wsum = 0, first = 0;
  bool same = true;
  for (int k = 0; k < 4; k++) {
    const int tx = std::clamp(ox + (k & 1) * dx, 0, int(lw) - 1), ty = std::clamp(oy + (k >> 1) * dy, 0, int(lh) - 1);
    const double v = m[size_t(ty) * lw + tx] / 255.0;
    if (k == 0) first = v;
    same = same && v == first;
    uint32_t px, py;
    RtLightSample(uint32_t(tx), uint32_t(ty), frame, stride, step, &px, &py);
    const double t = std::clamp(1.0 - std::fabs(double(px) - x) / s, 0.0, 1.0) *
                         std::clamp(1.0 - std::fabs(double(py) - y) / s, 0.0, 1.0) +
                     1e-3;
    const double dk = dist[size_t(ty) * lw + tx];
    double near = 0;
    if (dk > 0) {
      const double rel = (dk - d) / d;
      near = 1.0 / (1e-4 + rel * rel * 400.0);
    }
    sum += t * near * v;
    wsum += t * near;
  }
  return same ? first : wsum > 0 ? sum / wsum : first;
}

// A thin caster: a rod 4 cm thick across the ground under a light, its
// shadow a line about one pixel tall along the rows. Traced at every pixel
// (the answer), then at one of every 3 x 3 over 9 frames with the samples
// turning (and on the fixed grid for comparison), each frame upsampled as
// the light volumes do: the shadow's sum over the frames close to the
// answer's, and in no frame gone.
int ThinChecks(Gpu& g) {
  std::unique_ptr<RtScene> sc = MakeRtScene(g.dev, {});
  if (!sc) return Check(false, "thin caster: a scene");
  g.arena.used = 0;
  sc->SetAllocator(ArenaAlloc, &g.arena);
  const V3 eye{0, 9, -9};
  double vp[16], inv[16];
  Camera(eye, {0, 0, 1}, 1.0, double(kW) / kH, 0.5, 3000.0, vp);
  Invert4(vp, inv);
  Mat w, mw;
  Identity(w);
  Rows(vp, mw);
  const Box rod{{-2.5, 0.98, 1.98}, {2.5, 1.02, 2.02}};
  Mesh ground = QuadMesh(0, 60, 1), rm = BoxMesh(rod, 2);
  sc->BeginFrame(1, 0);
  sc->Add(ground.key, ground.g, w, mw, 1, 4);
  sc->Add(rm.key, rm.g, w, mw, 1, 4);
  const RtFrame* f = sc->Snapshot();
  if (!f || !f->tlas) return Check(false, "thin caster: a TLAS");
  std::vector<float> depth(size_t(kW) * kH);
  std::vector<uint8_t> on_rod(depth.size(), 0);  // the rod's own pixels
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
      double t = 2, z = 1;
      if (d.y < 0 && -pa.y / d.y > 0 && -pa.y / d.y < 1) t = -pa.y / d.y;
      const double tb = RayBox(pa, d, rod);
      if (tb > 0 && tb < t) t = tb, on_rod[size_t(j) * kW + i] = 1;
      if (t < 2) {
        const V3 p = pa + d * t;
        double cc[4];
        for (int k = 0; k < 4; k++) cc[k] = p.x * vp[k] + p.y * vp[4 + k] + p.z * vp[8 + k] + vp[12 + k];
        z = cc[2] / cc[3];
      }
      depth[size_t(j) * kW + i] = float(z);
    }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                width:kW
                                                                               height:kH
                                                                            mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> dt = [g.dev newTextureWithDescriptor:td];
  [dt replaceRegion:MTLRegionMake2D(0, 0, kW, kH) mipmapLevel:0 withBytes:depth.data() bytesPerRow:kW * 4];
  RtLocalLight l;
  l.position[0] = 0, l.position[1] = 3.5f, l.position[2] = 0;
  l.inv_sq_radius = 1.0f / 64.0f;
  const float up[3] = {0, 0, 1};
  RtShadowParams p;
  double e[3];
  RtLightGpu lg;
  if (!RtShadowParamsFor(*f, kW, kH, false, false, nullptr, up, &p) || !RtLightsEye(*f, e) ||
      !RtLightGpuFor(l, e, &lg))
    return Check(false, "thin caster: the parameters");
  // One pass: its layer and distances (step, frame, stride).
  auto pass = [&](uint32_t step, uint32_t frame, uint32_t stride, std::vector<uint8_t>* m, std::vector<float>* dd) {
    RtLightsParams sp{};
    sp.count = 1;
    sp.step = step;
    sp.cap = kRtLightsCap;
    sp.frame = frame;
    sp.stride = stride;
    sp.low[0] = (kW + step - 1) / step;
    sp.low[1] = (kH + step - 1) / step;
    MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                  width:sp.low[0]
                                                                                 height:sp.low[1]
                                                                              mipmapped:NO];
    md.textureType = MTLTextureType2DArray;
    md.arrayLength = kRtLightSlots;
    md.storageMode = MTLStorageModeShared;
    md.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> masks = [g.dev newTextureWithDescriptor:md];
    MTLTextureDescriptor* ddsc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                    width:sp.low[0]
                                                                                   height:sp.low[1]
                                                                                mipmapped:NO];
    ddsc.storageMode = MTLStorageModeShared;
    ddsc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> dist = [g.dev newTextureWithDescriptor:ddsc];
    id<MTLBuffer> counts = [g.dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    std::memset(counts.contents, 0, 16);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    sc->EncodeFrame(cb, f);
    g.masker->EncodeLights(cb, f, dt, masks, dist, p, sp, &lg, counts, 0);
    if (!g.Commit(cb)) return false;
    m->resize(size_t(sp.low[0]) * sp.low[1]);
    dd->resize(m->size());
    [masks getBytes:m->data()
        bytesPerRow:sp.low[0]
      bytesPerImage:m->size()
         fromRegion:MTLRegionMake2D(0, 0, sp.low[0], sp.low[1])
        mipmapLevel:0
              slice:0];
    [dist getBytes:dd->data() bytesPerRow:sp.low[0] * 4 fromRegion:MTLRegionMake2D(0, 0, sp.low[0], sp.low[1]) mipmapLevel:0];
    return true;
  };
  std::vector<uint8_t> ref;
  std::vector<float> ref_dist;
  if (!pass(1, 0, 0, &ref, &ref_dist)) return Check(false, "thin caster: the full-resolution pass");
  // The rod's shadow on the ground: the answer's dark pixels there, grown
  // by 4 (the rod's own pixels, dark where they face away, left out).
  auto dark = [&](size_t at) { return !on_rod[at] && ref[at] < 128; };
  std::vector<uint8_t> region(ref.size(), 0);
  double answer = 0;
  int dark_rows_lo = int(kH), dark_rows_hi = -1;
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++)
      if (dark(size_t(j) * kW + i)) dark_rows_lo = std::min(dark_rows_lo, int(j)), dark_rows_hi = std::max(dark_rows_hi, int(j));
  for (uint32_t j = 0; j < kH; j++)
    for (uint32_t i = 0; i < kW; i++) {
      bool near = false;
      for (int dy = -4; dy <= 4 && !near; dy++)
        for (int dx = -4; dx <= 4 && !near; dx++) {
          const int x = int(i) + dx, y = int(j) + dy;
          near = x >= 0 && y >= 0 && x < int(kW) && y < int(kH) && dark(size_t(y) * kW + x);
        }
      if (!near || on_rod[size_t(j) * kW + i]) continue;
      region[size_t(j) * kW + i] = 1;
      answer += 1.0 - ref[size_t(j) * kW + i] / 255.0;
    }
  const uint32_t step = 3, lw = (kW + step - 1) / step, lh = (kH + step - 1) / step, stride = RtLightStride(step);
  auto shadow = [&](const std::vector<uint8_t>& m, const std::vector<float>& dd, uint32_t frame, uint32_t st) {
    double sum = 0;
    for (uint32_t j = 0; j < kH; j++)
      for (uint32_t i = 0; i < kW; i++) {
        if (!region[size_t(j) * kW + i]) continue;
        const double d = ref_dist[size_t(j) * kW + i];
        if (d <= 0) continue;
        sum += 1.0 - Upsample(m, dd, lw, lh, step, frame, st, i, j, d);
      }
    return sum;
  };
  double turning = 0, least = 1e30, fixed = 0;
  bool ran = true;
  for (uint32_t fr = 0; fr < step * step && ran; fr++) {
    std::vector<uint8_t> m;
    std::vector<float> dd;
    ran = pass(step, fr, stride, &m, &dd);
    const double sum = ran ? shadow(m, dd, fr, stride) : 0;
    turning += sum / (step * step);
    least = std::min(least, sum);
  }
  std::vector<uint8_t> fm;
  std::vector<float> fd;
  ran = ran && pass(step, 0, 0, &fm, &fd);
  if (ran) fixed = shadow(fm, fd, 0, 0);
  std::printf("  thin caster: shadow rows %d-%d, its sum at every pixel %.1f; at one of every 3 x 3 turning over 9 "
              "frames %.1f on average, %.1f in the least; on the fixed grid %.1f\n",
              dark_rows_lo, dark_rows_hi, answer, turning, least, fixed);
  return Check(ran && answer > 100 && dark_rows_hi - dark_rows_lo <= 4 && std::fabs(turning / answer - 1.0) < 0.3 &&
                   least > 0.25 * answer,
               "a thin caster's shadow (a line about a pixel tall): at one of every 3 x 3 pixels, the pixel turning "
               "frame to frame, its average over 9 frames within 30% of every pixel's, and in no frame gone");
}

int GpuChecks(Gpu& g) {
  int failures = 0;
  Light point;
  point.pos = {0.3, 3.5, 0.2};
  point.radius = 6;
  Light spot = point;
  spot.cone = true;
  spot.dir = {0, -1, 0};
  spot.offset = std::cos(40.0 * M_PI / 180.0);
  spot.scale = 5;
  Light gapped = point;
  gapped.gap = 0.6;  // past the housing's corners (0.52)
  Light wide_gap = point;
  wide_gap.gap = 2.0;  // past the box's top (1.5 below the light)
  Light small;  // away from the box, a small part of the screen
  small.pos = {3, 1, 6};
  small.radius = 2.5;
  const Case cases[] = {
      {"point light, reversed depth texture", true, point},
      {"point light, float depth", false, point},
      {"point light, a ceiling above it", true, point, true},
      {"spot light", true, spot},
      {"point light in a housing, no gap", true, point, false, true},
      {"point light in a housing, a gap", true, gapped, false, true},
      {"two lights, one snapshot", true, point, false, false, true},
      {"a small light", true, small},
      {"a point light, a gap past the box's top", true, wide_gap},
  };
  Result r[9];
  for (int k = 0; k < 9; k++) r[k] = Run(g, cases[k]);
  bool ran = true;
  for (const Result& x : r) ran = ran && x.ran;
  failures += Check(ran, "every case traced");
  if (!ran) return failures;
  auto good = [](const Result& x) {
    return x.wrong == 0 && x.shadowed > 300 && x.lit > 1000 && x.out_wrong == 0 && x.out_judged > 500;
  };
  failures += Check(good(r[0]) && good(r[1]) && std::abs(r[0].shadowed - r[1].shadowed) * 20 < r[0].shadowed,
                    "a point light: every judged ground pixel as the box's shadow from it predicts (in shadow and "
                    "lit), lit beyond its radius; reversed depth texture and float depth alike");
  // Every pixel of the bounds not the sky: traced, beyond the light or
  // within its lamp's gap; the blocked ones at least those judged in
  // shadow.
  bool counts = true;
  for (const Result& x : r)
    counts = counts && x.rays + x.outside + x.in_gap == x.expect_in_rect && x.hits >= uint32_t(x.shadowed);
  counts = counts && r[0].outside > 1000 && r[0].in_gap == 0 && r[8].in_gap > 1000 && r[8].rays < r[0].rays &&
           r[8].wrong == 0;
  failures += Check(counts, "a ray for each pixel of the bounds the light reaches, the rest counted untraced (beyond "
                            "it, or within its lamp's gap: the box's top with a 2 m gap); the blocked rays counted");
  failures += Check(good(r[2]) && r[2].shadowed == r[0].shadowed && r[2].lit == r[0].lit &&
                        r[2].instances == r[0].instances + 1,
                    "an occluder above the light casts nothing (rays end at the light)");
  failures += Check(r[3].wrong == 0 && r[3].out_wrong == 0 && r[3].out_judged > 1000 && r[3].shadowed > 300 &&
                        r[3].rays < r[0].rays && r[3].rect.Area() < r[0].rect.Area(),
                    "a spot light: its cone cuts (lit and untraced outside it, its bounds the cone's), the box's "
                    "shadow inside it");
  failures += Check(r[4].wrong == 0 && r[4].lit == 0 && r[4].shadowed > 3000,
                    "a housing around the light blocks every ray without a gap");
  failures += Check(good(r[5]) && r[5].shadowed == r[0].shadowed && r[5].lit == r[0].lit,
                    "the lamp's gap lets the rays past the housing: as without it");
  bool bounds = true;
  for (const Result& x : r)
    bounds = bounds && x.untouched_wrong == 0 && x.missed_by_bounds == 0 && x.written == int(x.rect.Area());
  const Result& sm = r[7];
  bounds = bounds && sm.rect.Area() * 4 < uint64_t(kW) * kH && sm.out_judged > 1000 && sm.out_wrong == 0 &&
           sm.lit > 1000 && sm.wrong == 0 && sm.rays + sm.outside == sm.expect_in_rect;
  failures += Check(bounds, "only the bounds' pixels dispatched (every texel there written, none outside), every "
                            "pixel the light reaches inside them; a small light's a small part of the buffer");
  failures += Check(good(r[6]) && r[6].second_ok && r[6].shadowed == r[0].shadowed,
                    "two lights traced from one snapshot into their own masks, each as alone");
  bool set = true;
  for (const Result& x : r)
    for (int k = 0; k < 2; k++)
      set = set && x.set_compared[k] > 1000 && x.set_wrong[k] == 0 && x.set_dist_wrong[k] == 0;
  // (Fewer rays than the light's own mask: none where the surface faces
  // away from the light, nor where it is too faint to matter.)
  set = set && r[0].set_rays[0] <= r[0].rays && r[0].set_rays[0] * 10 > r[0].rays * 6 &&
        r[6].set_rays[0] > r[6].set_rays[0] - r[0].set_rays[0] && r[6].set_rays[0] > r[0].set_rays[0] &&
        r[0].set_rays[1] * 3 < r[0].set_rays[0] && r[0].set_rays[1] * 5 > r[0].set_rays[0];
  failures += Check(set, "the lights traced at once (one pass, a layer each): every texel as the light's own mask "
                         "at the pixel it traced (lit beyond its bounds; dark untraced where the surface faces "
                         "away, lit untraced where too faint), at every pixel and at one of every 2 x 2 (a quarter "
                         "of the rays; the pixel turning within its block); the points' distances; two lights' "
                         "layers each its own");
  // The cap: of two lights reaching a point, only the stronger traced.
  {
    Light near = point, far = point;
    far.pos = point.pos + V3{0.0, 0.0, 0.5};
    Case cc{"two lights, a cap of one", true, near, false, false, true};
    cc.cap = 1;
    cc.second_given = true;
    cc.second = far;
    const Result x = Run(g, cc);
    failures += Check(x.ran && x.set_rays[0] > 1000 && x.set_rays[0] <= x.rays && x.set_capped[0] > 1000,
                      "a cap of one: one ray a point however many lights reach it, the points past the cap counted");
  }
  return failures;
}

}  // namespace

int main() {
  @autoreleasepool {
    int failures = CpuChecks();
    Gpu g;
    if (!g.dev || !RtSupported(g.dev)) {
      std::printf("skipped: %s\n", g.dev ? "the GPU does not trace rays in hardware (Apple9)" : "no Metal device");
      return failures ? 1 : 77;
    }
    if (!g.Init()) return Check(false, "shadow kernel made");
    failures += GpuChecks(g);
    failures += ThinChecks(g);
    return failures ? 1 : 0;
  }
}
