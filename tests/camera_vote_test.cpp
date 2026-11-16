// The frame's camera from the scene's draws (gfx/camera_vote.h). A camera
// (world to clip) and draws of it: two static meshes (their world matrices
// translations, one 4 km from the origin), and before them a skinned mesh
// whose world register holds a bone's matrix (its world-view-projection
// that of the mesh's own world): the first draw's world^-1 * wvp, as the
// camera was taken before, is no camera; the vote takes the camera the
// static meshes agree on and counts the skinned draws rejected; two draws
// of one world matrix confirm nothing; Settle takes the likeliest
// candidate, unconfirmed, where no two agreed; Reset starts over.
//
//   camera_vote_test

#include <cmath>
#include <cstdio>
#include <cstring>

#include "runtime/gfx/camera_vote.h"
#include "runtime/gfx/matrix4.h"

namespace {

using recomp::gfx::CameraVote;

int failures = 0;
void Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  failures += ok ? 0 : 1;
}

using Mat = float[4][4];

// A left-handed camera at `eye` looking along +z turned by `yaw`, its
// perspective (D3D's depth 0..1), world to clip rows.
void Camera(const double eye[3], double yaw, double vp[16]) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double rt[3] = {c, 0, -s}, up[3] = {0, 1, 0}, fw[3] = {s, 0, c};
  auto dot = [&](const double* a) { return a[0] * eye[0] + a[1] * eye[1] + a[2] * eye[2]; };
  const double view[16] = {rt[0], up[0], fw[0], 0, rt[1], up[1], fw[1], 0, rt[2], up[2], fw[2], 0,
                           -dot(rt), -dot(up), -dot(fw), 1};
  const double n = 0.3, f = 3000, ys = 1.0 / std::tan(0.45);
  double proj[16] = {};
  proj[0] = ys / (16.0 / 9.0);
  proj[5] = ys;
  proj[10] = f / (f - n);
  proj[11] = 1;
  proj[14] = -n * f / (f - n);
  recomp::gfx::Mul4(view, proj, vp);
}

// Rows of a rotation about y by `a` then a translation.
void World(double a, double x, double y, double z, double w[16]) {
  const double c = std::cos(a), s = std::sin(a);
  const double m[16] = {c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1};
  std::memcpy(w, m, sizeof m);
}

void ToFloat(const double* d, Mat out) {
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) out[r][k] = float(d[r * 4 + k]);
}

// A draw's registers: its world register holding `reg`, its wvp `world` * vp
// (as the game computes it, rounded to float).
struct Draw {
  Mat w, wvp;
};
Draw MakeDraw(const double* reg, const double* world, const double* vp) {
  Draw d;
  double m[16];
  recomp::gfx::Mul4(world, vp, m);
  ToFloat(reg, d.w);
  ToFloat(m, d.wvp);
  return d;
}

// The camera as it was taken before: the first draw's world^-1 * wvp.
void FirstDraw(const Draw& d, double out[16]) {
  double w[16], m[16], iw[16];
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < 4; k++) w[r * 4 + k] = d.w[r][k], m[r * 4 + k] = d.wvp[r][k];
  recomp::gfx::Invert4(w, iw);
  recomp::gfx::Mul4(iw, m, out);
}

}  // namespace

int main() {
  for (const double base : {0.0, 4000.0}) {
    const double eye[3] = {base + 10, 2, base - 5};
    double vp[16];
    Camera(eye, 0.7, vp);
    // A skinned mesh: its world (where it is) and two bones' matrices (what
    // its shaders hold in the world register).
    double ped[16], bone1[16], bone2[16], b1[16], b2[16];
    World(0.3, base + 12, 0, base + 3, ped);
    World(1.1, 0.2, 0.9, 0.1, b1);
    World(-0.4, 0.1, 1.4, -0.2, b2);
    recomp::gfx::Mul4(b1, ped, bone1);
    recomp::gfx::Mul4(b2, ped, bone2);
    double wall[16], car[16];
    World(0.0, base + 20, 0, base + 30, wall);
    World(2.0, base + 15, 0, base + 18, car);
    const Draw d_ped1 = MakeDraw(bone1, ped, vp), d_ped2 = MakeDraw(bone2, ped, vp);
    const Draw d_wall = MakeDraw(wall, wall, vp), d_car = MakeDraw(car, car, vp);

    double first[16];
    FirstDraw(d_ped1, first);
    const double before = CameraVote::Difference(vp, first);
    char what[240];
    std::snprintf(what, sizeof what,
                  "%.0f from the origin: a skinned draw first, its world register a bone's: the first draw's "
                  "world^-1 * wvp off the camera by %.2f of its largest entry",
                  base, before);
    Check(before > 0.01, what);

    CameraVote v;
    v.Reset();
    const bool a = v.Offer(d_ped1.w, d_ped1.wvp), b = v.Offer(d_ped2.w, d_ped2.wvp),
               c = v.Offer(d_wall.w, d_wall.wvp), d = v.Offer(d_car.w, d_car.wvp);
    const double after = CameraVote::Difference(vp, v.camera());
    std::snprintf(what, sizeof what,
                  "%.0f from the origin: the vote takes the camera the static meshes agree on (%.1e off), the two "
                  "skinned draws rejected (%llu)",
                  base, after, (unsigned long long)v.rejected());
    Check(!a && !b && !c && d && v.state() == CameraVote::State::Confirmed && after < 1e-5 && v.rejected() == 2,
          what);
    Check(!v.Offer(d_ped1.w, d_ped1.wvp) && CameraVote::Difference(vp, v.camera()) < 1e-5,
          "once taken, later draws change nothing");

    CameraVote same;
    same.Reset();
    const bool s1 = same.Offer(d_ped1.w, d_ped1.wvp), s2 = same.Offer(d_ped1.w, d_ped1.wvp);
    Check(!s1 && !s2 && same.state() == CameraVote::State::None,
          "two draws with the same world matrix confirm nothing (a skinned mesh's parts on one bone)");
    // Settle: the candidate most draws gave.
    same.Offer(d_wall.w, d_wall.wvp);
    Check(same.Settle() == CameraVote::State::Unconfirmed && CameraVote::Difference(first, same.camera()) < 1e-9,
          "Settle without two agreeing: the likeliest candidate, unconfirmed");
    same.Reset();
    Check(same.state() == CameraVote::State::None && same.Settle() == CameraVote::State::None,
          "Reset: none, and none to settle on");
  }
  return failures ? 1 : 0;
}
