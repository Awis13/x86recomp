#include "runtime/gfx/target_size.h"

#include <algorithm>
#include <cmath>

namespace recomp::gfx {

namespace {
thread_local int t_exact_targets = 0;  // BeginExactTargets scopes open
}

void BeginExactTargets() { t_exact_targets++; }

void EndExactTargets() {
  if (t_exact_targets > 0) t_exact_targets--;
}

bool ExactTargets() { return t_exact_targets > 0; }

bool TargetSize(const TargetSizing& s, int w, int h, int* rw, int* rh, bool* scene) {
  if (scene) *scene = false;
  if (s.internal && s.bb_w > 0)
    for (int k = 0; k < 3; k++)
      if (w == s.bb_w >> k && h == s.bb_h >> k) {
        // TAA: the scene's buffers (jittered) at their own size; the scene
        // goes to the back buffer itself, as with no upscaler.
        if (s.taa) {
          if (scene) *scene = true;
          return false;
        }
        *rw = std::max(1, int(std::lround(w * s.render_scale)));
        *rh = std::max(1, int(std::lround(h * s.render_scale)));
        if (scene) *scene = true;
        return true;
      }
  // RECOMP_RT_BOOST: targets of these sizes made larger (shadow maps,
  // reflections), the game none the wiser; not those a port wants exact.
  if (ExactTargets()) return false;
  for (const RtBoost& b : s.boost)
    if (w == b.w && h == b.h && (s.bb_w <= 0 || w != s.bb_w || h != s.bb_h)) {
      *rw = std::min(kMaxBoostedSide, w * b.k);
      *rh = std::min(kMaxBoostedSide, h * b.k);
      return true;
    }
  return false;
}

}  // namespace recomp::gfx
