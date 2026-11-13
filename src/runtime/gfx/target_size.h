#pragma once
// The size a game's render target or depth buffer is really made at, where
// a backend makes some of them at another size than the game asked for
// (the Metal backend: the scene's buffers at the internal resolution, and
// RECOMP_RT_BOOST's larger ones), and a port's say over it.

#include <span>

namespace recomp::gfx {

// ---- for ports
//
// Render targets and depth buffers created on the calling thread between
// BeginExactTargets and EndExactTargets are left out of RECOMP_RT_BOOST:
// made at exactly the size the game asks for, whatever sizes are listed.
// For a target a game computes with rather than shows: a simulation that
// reads its own last result and offsets what it draws by half a texel of
// the size it asked for is wrong at any other size, where a picture (a
// shadow map, a reflection) only gets finer. The internal resolution of
// the scene's buffers (RECOMP_RENDER_SCALE) is not affected.
//
// A port calls Begin from a hook before the game creates such targets and
// End from one after (or holds an ExactTargetScope around a call into the
// game that creates them). A scope and not a count of targets to come: a
// creation the game skips, or one the graphics API refuses before it
// reaches the device, would leave a count standing for whatever target
// comes next.
//
// Per thread: only what the calling thread creates is affected, and a
// graphics API's creation calls reach the device on the thread the game
// calls them on, so a scope needs no lock and other threads loading
// resources meanwhile are not touched. Scopes nest; an End without a Begin
// does nothing. Any time, also before a device exists; backends that never
// resize targets ignore it.
void BeginExactTargets();
void EndExactTargets();
// Whether the calling thread is inside such a scope.
bool ExactTargets();

struct ExactTargetScope {
  ExactTargetScope() { BeginExactTargets(); }
  ~ExactTargetScope() { EndExactTargets(); }
  ExactTargetScope(const ExactTargetScope&) = delete;
  ExactTargetScope& operator=(const ExactTargetScope&) = delete;
};

// ---- for backends

// RECOMP_RT_BOOST=WxH:k: targets of w x h made k times larger each way.
struct RtBoost {
  int w, h, k;
};

// The longest side a boosted target is given.
inline constexpr int kMaxBoostedSide = 16384;

struct TargetSizing {
  // The scene is drawn at an internal resolution: its buffers (the back
  // buffer's size, its halves and quarters) are the scene's.
  bool internal = false;
  // TAA: the scene's buffers are jittered and stay at their own size,
  // not made at render_scale.
  bool taa = false;
  int bb_w = 0, bb_h = 0;  // the back buffer; 0 before there is one
  float render_scale = 1.0f;
  std::span<const RtBoost> boost;
};

// The size a render target or depth buffer the game asks for as w x h is
// made at, for one created now on this thread: true with it in *rw x *rh
// if it is another size than w x h, false (*rw, *rh untouched) if not.
// *scene: it is one of the scene's buffers.
bool TargetSize(const TargetSizing& s, int w, int h, int* rw, int* rh, bool* scene = nullptr);

}  // namespace recomp::gfx
