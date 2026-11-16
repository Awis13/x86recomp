#pragma once
// The frame's local lights, as a game hands them over (gfx::SetRtLights,
// device.h), for effects that light the scene by ray tracing: plain data,
// in world space, one list a presented frame.
//
// Threads: the list is given on the thread that records the frame (the one
// making the Direct3D calls) and read where that frame's work is encoded,
// which may be frames behind. RtLightFrames keeps the two apart: Give
// copies the list (on the recording thread) into a list of its own, which
// the recorded work hands to the encoding side in order (Publish), so that
// what reads it there (Current) sees the list of the frame it encodes,
// whatever frames the recording thread has gone on to; EndFrame, recorded
// at each presented frame's end, lets it go. A frame with no Give has an
// empty list; with several, each one replaces the one before for the work
// recorded after it (the last call wins). Lists are reused: as many exist
// as frames are in flight. RtLightsGive and RtLightsPresent are what a
// device records for SetRtLights and Present (its own recording given as
// `rec`), so that the wiring is tested without a GPU.

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace recomp::gfx {

// At most this many lights a frame are kept (the rest of a longer list is
// dropped).
inline constexpr size_t kRtLightsMax = 1024;

enum class RtLightKind : uint32_t {
  Point = 0,  // all around
  Spot = 1,   // in a cone (direction, cone_offset, cone_scale)
};

// RtLight::flags.
inline constexpr uint32_t kRtLightCastsShadow = 1u << 0;  // the game shadows it (a hint)
inline constexpr uint32_t kRtLightInterior = 1u << 1;     // lights what is indoors
inline constexpr uint32_t kRtLightExterior = 1u << 2;     // lights what is outdoors
inline constexpr uint32_t kRtLightVehicleLamp = 1u << 3;  // a vehicle's lamp (moves with it)
// How the light lights a pixel, for consumers that light a G-buffer with it
// (gfx/rt_many_lights.h): its model, an index into the game's light models
// (device.h, RayManyLights::models: 0-3), and its pixel class (RayManyLights::
// classes: 0 every pixel, 1-3 the pixels of that class).
inline constexpr uint32_t kRtLightModelShift = 8, kRtLightModelMask = 3u << kRtLightModelShift;
inline constexpr uint32_t kRtLightClassShift = 10, kRtLightClassMask = 3u << kRtLightClassShift;
inline constexpr uint32_t RtLightModel(uint32_t flags) { return (flags & kRtLightModelMask) >> kRtLightModelShift; }
inline constexpr uint32_t RtLightClass(uint32_t flags) { return (flags & kRtLightClassMask) >> kRtLightClassShift; }

struct RtLight {
  float position[3];  // world space
  float radius;       // > 0: it lights nothing farther
  float color[3];     // linear RGB, the intensity included
  RtLightKind kind;
  // Spot: the way it shines (unit), and its cone: at a point p, with D the
  // unit vector from the light to p, saturate((dot(D, direction) -
  // cone_offset) * cone_scale). Point: unused (zero).
  float direction[3];
  float cone_offset, cone_scale;
  uint32_t flags;  // kRtLight*, the model and class set
  // The same light's key from frame to frame, as the game names it (0:
  // none): for what a consumer keeps across frames.
  uint64_t key;
  // The space the light is in, as the game numbers it (an interior and a
  // room of it, say; 0 0 outside): only compared.
  uint32_t space[2];
};

// Whether a consumer can use `l`: every number finite, the radius above 0,
// no colour channel below 0 (a light that darkens is none: dropped), a
// known kind.
inline bool RtLightValid(const RtLight& l) {
  const float v[] = {l.position[0], l.position[1],  l.position[2],  l.radius,       l.color[0],    l.color[1],
                     l.color[2],    l.direction[0], l.direction[1], l.direction[2], l.cone_offset, l.cone_scale};
  for (float x : v)
    if (!std::isfinite(x)) return false;
  if (l.color[0] < 0 || l.color[1] < 0 || l.color[2] < 0) return false;
  return l.radius > 0 && (l.kind == RtLightKind::Point || l.kind == RtLightKind::Spot);
}

// What one Give took: lights given, kept (valid, within kRtLightsMax), and
// those kept by kind.
struct RtLightCounts {
  size_t given = 0, kept = 0, point = 0, spot = 0;
};

class RtLightFrames {
 public:
  struct List {
    std::vector<RtLight> lights;
  };

  // Recording thread: a copy of the valid ones of `lights` (the first
  // kRtLightsMax of them), for Publish; `counts` (optional) says what was
  // taken.
  List* Give(const RtLight* lights, size_t count, RtLightCounts* counts = nullptr) {
    List* l = nullptr;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (!free_.empty()) {
        l = free_.back();
        free_.pop_back();
      } else {
        all_.push_back(std::make_unique<List>());
        l = all_.back().get();
        l->lights.reserve(kRtLightsMax);
      }
    }
    l->lights.clear();
    RtLightCounts n;
    n.given = lights ? count : 0;
    for (size_t i = 0; i < n.given && l->lights.size() < kRtLightsMax; i++) {
      if (!RtLightValid(lights[i])) continue;
      l->lights.push_back(lights[i]);
      (lights[i].kind == RtLightKind::Spot ? n.spot : n.point)++;
    }
    n.kept = l->lights.size();
    if (counts) *counts = n;
    return l;
  }

  // Encoding thread, in the recorded order: `l` (Give's) is the frame's
  // list from here on.
  void Publish(List* l) {
    Release(current_);
    current_ = l;
  }

  // Encoding thread: the presented frame is over; the next one has no list
  // until a Publish.
  void EndFrame() {
    Release(current_);
    current_ = nullptr;
  }

  // Encoding thread: the frame's lights as of the work encoded now. The
  // span holds until the next Publish or EndFrame runs (the list then goes
  // back to be reused): what needs them later copies them before.
  std::span<const RtLight> Current() const {
    if (!current_) return {};
    return {current_->lights.data(), current_->lights.size()};
  }

  // Lists made so far (each reused once its frame is over).
  size_t Lists() const {
    std::lock_guard<std::mutex> g(mu_);
    return all_.size();
  }

 private:
  void Release(List* l) {
    if (!l) return;
    std::lock_guard<std::mutex> g(mu_);
    free_.push_back(l);
  }

  mutable std::mutex mu_;  // all_, free_
  std::vector<std::unique_ptr<List>> all_;
  std::vector<List*> free_;
  List* current_ = nullptr;  // encoding thread
};

// What a device records for SetRtLights: with ray tracing on (`on`), the
// lights given (Give) and their Publish recorded through `rec` (a callable
// taking a closure the encoding side runs in order, or one that runs it at
// once where nothing is deferred); false, with nothing recorded, when off.
template <typename Rec>
bool RtLightsGive(RtLightFrames& frames, bool on, const RtLight* lights, size_t count, Rec&& rec,
                  RtLightCounts* counts = nullptr) {
  if (!on) return false;
  RtLightFrames::List* list = frames.Give(lights, count, counts);
  RtLightFrames* f = &frames;
  rec([f, list] { f->Publish(list); });
  return true;
}

// What a device records at a presented frame's end: the frame's list let go
// (EndFrame) when ray tracing is on; nothing when off (no list was given).
template <typename Rec>
void RtLightsPresent(RtLightFrames& frames, bool on, Rec&& rec) {
  if (!on) return;
  RtLightFrames* f = &frames;
  rec([f] { f->EndFrame(); });
}

// The device that takes the lights (NoteRtLightDevice, device.cpp): Note(d,
// true) makes `d` it; Note(d, false) unmakes it only while it is `d` (a
// device going after another took over changes nothing).
template <typename T>
class RtLightTaker {
 public:
  void Note(T* d, bool on) {
    if (on) {
      d_.store(d, std::memory_order_release);
      return;
    }
    T* expected = d;
    d_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
  }
  T* Get() const { return d_.load(std::memory_order_acquire); }

 private:
  std::atomic<T*> d_{nullptr};
};

}  // namespace recomp::gfx
