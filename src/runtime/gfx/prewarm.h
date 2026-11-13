// Shaders and pipelines compiled before a game runs (macOS, iOS and tvOS), for a
// launcher to do while its window is open: the pipelines earlier runs
// compiled (their recipes, metal_cache.h) built again from the translated
// shaders those runs kept, into the pipeline archive and the OS's cache of
// compiled shaders, so that the game finds them instead of compiling them
// while it draws. Needed after an OS update (both are then empty for the
// new OS build) or when the archive has lost shards; no game is needed.
//
// Plain C++: a launcher includes it and links recomp_gfx. An iOS/tvOS launcher
// must wait for the call to return before gameplay: use drain_on_close so
// its archive writer is released first.

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>

namespace recomp::gfx {

// Whether the pipeline archive for this OS build and GPU lacks pipelines
// that earlier runs' recipes list (and that their kept shaders can make).
// Cheap: it reads the cache files and compiles nothing. `cache_dir`: the
// runtime's cache folder (empty: the user's, the one the game uses).
bool PrewarmNeeded(const std::string& cache_dir = {});
enum class PrewarmCacheState { Unavailable, Empty, Ready, Needed };
// Read-only inspection for a manual launcher's button and explanatory note.
PrewarmCacheState InspectPrewarmCache(const std::string& cache_dir = {});

enum class PrewarmResult {
  Done,         // all built
  Cancelled,    // `cancel` was set: stopped (what was built is kept)
  NothingToDo,  // nothing missing
  Busy,         // another process (the game) is saving to the archive
  Unavailable,  // no Metal device, or unsupported platform
  Incomplete,   // attempted, but some pipelines could not be built or saved
};

// Called from the threads doing the work (one call at a time): `done` of
// `total` (shaders compiled, then pipelines built), `phase` "shaders" or
// "pipelines".
using PrewarmProgress = std::function<void(size_t done, size_t total, const char* phase)>;

// A fixed pipeline total across cancel/resume, with shader function work
// counted separately. During a run, ready includes successful builds;
// Complete is sent after saving and reports only pipelines on disk.
struct PrewarmStatus {
  enum class Phase { Shaders, Pipelines, Saving, Complete };
  Phase phase = Phase::Shaders;
  size_t functions_done = 0, functions_total = 0;
  size_t pipelines_ready = 0, pipelines_total = 0;
};
using PrewarmStatusProgress = std::function<void(const PrewarmStatus&)>;

// Builds what PrewarmNeeded says is missing, with a Metal device of its
// own, on a few threads; returns when done or, once `cancel` is set,
// within about the time of one compile (the archive's last save then left
// to finish in the background, or to the next run). With drain_on_close,
// archive saving is also finished before returning; call off the UI thread.
PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir = {});

// Work and handoff limits: saved after save_every new pipelines; at most
// max_unsaved waiting to be saved. Tests can override the platform defaults.
struct PrewarmArchiveLimits {
  unsigned save_every = 64, max_unsaved = 1024;
  unsigned workers = 0;  // 0: automatic, 1: lower mobile memory pressure
  bool drain_on_close = false;  // same-process handoff, including cancellation
};
PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir, const PrewarmArchiveLimits& limits,
                            const PrewarmStatusProgress& status = {});

}  // namespace recomp::gfx
