// The size render targets and depth buffers are made at
// (runtime/gfx/target_size.h): RECOMP_RT_BOOST's sizes and its cap, the
// scene's buffers at the internal resolution, and a port's scope of
// targets left at the size the game asks for.

#include <cstdio>
#include <thread>

#include "runtime/gfx/target_size.h"

namespace {

using namespace recomp::gfx;

int g_failures = 0;

// What TargetSize answers for w x h: the size made, whether that is
// another one than asked for, and whether it is a scene buffer.
struct Made {
  int w, h;
  bool resized, scene;
};

Made Make(const TargetSizing& s, int w, int h) {
  Made m{w, h, false, false};
  m.resized = TargetSize(s, w, h, &m.w, &m.h, &m.scene);
  return m;
}

void Expect(const TargetSizing& s, int w, int h, Made want, const char* what) {
  const Made got = Make(s, w, h);
  if (got.w == want.w && got.h == want.h && got.resized == want.resized && got.scene == want.scene)
    return;
  std::printf("FAIL: %s: %dx%d made %dx%d resized %d scene %d, want %dx%d resized %d scene %d\n",
              what, w, h, got.w, got.h, got.resized, got.scene, want.w, want.h, want.resized,
              want.scene);
  g_failures++;
}

void Check(bool ok, const char* what) {
  if (ok) return;
  std::printf("FAIL: %s\n", what);
  g_failures++;
}


// The listed sizes: the last two share one side with the back buffer the
// tests give most cases, 1280x720, and that size itself is listed too.
constexpr RtBoost kBoosts[] = {{256, 256, 2},  {1024, 512, 3}, {8192, 4096, 4}, {640, 360, 2},
                               {1280, 720, 2}, {1280, 360, 2}, {640, 720, 3}};

// No upscaler: a 1280x720 back buffer, nothing at an internal resolution.
TargetSizing Plain() {
  TargetSizing s;
  s.bb_w = 1280;
  s.bb_h = 720;
  s.boost = kBoosts;
  return s;
}

void ListedSizes() {
  const TargetSizing plain = Plain();
  Expect(plain, 256, 256, {512, 512, true, false}, "a listed size is made k times larger");
  Expect(plain, 1024, 512, {3072, 1536, true, false}, "each listed size by its own factor");
  Expect(plain, 512, 256, {512, 256, false, false}, "a size that is not listed stays");
  Expect(plain, 256, 128, {256, 128, false, false}, "a size with one side of a listed one stays");
  Expect(plain, 640, 360, {1280, 720, true, false}, "a listed half of the back buffer");
  // Without the scene out parameter.
  int rw = 256, rh = 256;
  Check(TargetSize(plain, 256, 256, &rw, &rh) && rw == 512 && rh == 512, "scene may be null");
}

// The back buffer's own size is never boosted; a size with only its width
// or only its height is.
void BackBufferSize() {
  const TargetSizing plain = Plain();
  Expect(plain, 1280, 720, {1280, 720, false, false}, "the back buffer's size, listed, stays");
  Expect(plain, 1280, 360, {2560, 720, true, false}, "a listed size of the back buffer's width");
  Expect(plain, 640, 720, {1920, 2160, true, false}, "a listed size of the back buffer's height");
}

// The cap, per side.
void Cap() {
  Expect(Plain(), 8192, 4096, {16384, 16384, true, false}, "a boosted side stops at 16384");
  const RtBoost wide[] = {{5000, 100, 4}, {100, 5000, 4}, {4096, 4096, 4}, {4097, 16, 4}};
  TargetSizing capped;
  capped.boost = wide;
  Expect(capped, 5000, 100, {16384, 400, true, false}, "the cap holds the width, not the height");
  Expect(capped, 100, 5000, {400, 16384, true, false}, "the cap holds the height as the width");
  Expect(capped, 4096, 4096, {16384, 16384, true, false}, "exactly 16384 is reached");
  Expect(capped, 4097, 16, {16384, 64, true, false}, "one over the cap is cut to it");
}

// A port's scope: the same sizes, exact.
void Scope() {
  const TargetSizing plain = Plain();
  Check(!ExactTargets(), "no scope open at the start");
  BeginExactTargets();
  Check(ExactTargets(), "a scope is open after Begin");
  Expect(plain, 256, 256, {256, 256, false, false}, "a listed size in a scope stays as asked");
  Expect(plain, 1024, 512, {1024, 512, false, false}, "any listed size in a scope stays as asked");
  Expect(plain, 8192, 4096, {8192, 4096, false, false}, "a capped size in a scope stays as asked");
  EndExactTargets();
  Check(!ExactTargets(), "no scope open after End");
  Expect(plain, 256, 256, {512, 512, true, false}, "boosted again once the scope has ended");
}

// Another thread creating targets during a scope is not in it.
void ScopeIsItsThreadsAlone() {
  const TargetSizing plain = Plain();
  const ExactTargetScope scope;
  Made other{};
  bool other_exact = true;
  std::thread([&] {
    other_exact = ExactTargets();
    other = Make(plain, 256, 256);
  }).join();
  Check(!other_exact, "a scope is its thread's alone");
  Check(other.w == 512 && other.h == 512 && other.resized,
        "another thread's target is boosted during a scope");
  Expect(plain, 256, 256, {256, 256, false, false}, "and this thread's still is not");
}

// Scopes nest; an End too many opens nothing and breaks no later scope.
void NestedAndUnbalancedScopes() {
  const TargetSizing plain = Plain();
  {
    ExactTargetScope outer;
    {
      ExactTargetScope inner;
      Expect(plain, 256, 256, {256, 256, false, false}, "exact in a nested scope");
    }
    Expect(plain, 256, 256, {256, 256, false, false}, "still exact after the inner scope");
  }
  Expect(plain, 256, 256, {512, 512, true, false}, "boosted after the outer scope");
  EndExactTargets();
  Expect(plain, 256, 256, {512, 512, true, false}, "an End without a Begin changes nothing");
  BeginExactTargets();
  Expect(plain, 256, 256, {256, 256, false, false}, "a scope after such an End still holds");
  EndExactTargets();
  Expect(plain, 256, 256, {512, 512, true, false}, "and ends");
}

// The scene at half the back buffer's resolution: its buffers, and their
// halves and quarters, at the internal resolution, in a scope or not.
void SceneBuffers() {
  TargetSizing half = Plain();
  half.internal = true;
  half.render_scale = 0.5f;
  for (const bool scope : {false, true}) {
    if (scope) BeginExactTargets();
    Expect(half, 1280, 720, {640, 360, true, true},
           scope ? "scene buffer, in a scope" : "scene buffer");
    Expect(half, 640, 360, {320, 180, true, true},
           scope ? "listed half, in a scope: the scene's" : "listed half: the scene's");
    Expect(half, 320, 180, {160, 90, true, true}, scope ? "quarter, in a scope" : "quarter");
    Expect(half, 160, 90, {160, 90, false, false},
           scope ? "an eighth is no scene buffer, in a scope" : "an eighth is no scene buffer");
    if (scope) EndExactTargets();
  }
  Expect(half, 256, 256, {512, 512, true, false}, "boost beside the internal resolution");
  // Rounded, and never less than a pixel.
  TargetSizing odd;
  odd.internal = true;
  odd.bb_w = 1366;
  odd.bb_h = 3;
  odd.render_scale = 0.25f;
  Expect(odd, 1366, 3, {342, 1, true, true}, "the internal size is rounded");
  Expect(odd, 341, 0, {85, 1, true, true}, "and at least one pixel");
}

// TAA: scene buffers at their own size, known as the scene's.
void Taa() {
  TargetSizing taa = Plain();
  taa.internal = taa.taa = true;
  taa.render_scale = 0.5f;
  Expect(taa, 1280, 720, {1280, 720, false, true}, "TAA keeps the scene buffer's size");
  Expect(taa, 640, 360, {640, 360, false, true}, "TAA: a listed half is the scene's, not boosted");
  Expect(taa, 256, 256, {512, 512, true, false}, "boost with TAA");
}

// Before there is a back buffer: nothing is a scene buffer, and every
// listed size is boosted.
void NoBackBufferYet() {
  TargetSizing early;
  early.internal = true;
  early.render_scale = 0.5f;
  early.boost = kBoosts;
  Expect(early, 1280, 720, {2560, 1440, true, false}, "no back buffer yet: a listed size grows");
  Expect(early, 0, 0, {0, 0, false, false}, "no back buffer yet: 0x0 is not its size");
}

}  // namespace

int main() {
  ListedSizes();
  BackBufferSize();
  Cap();
  Scope();
  ScopeIsItsThreadsAlone();
  NestedAndUnbalancedScopes();
  SceneBuffers();
  Taa();
  NoBackBufferYet();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("target_size_test: ok\n");
  return 0;
}
