// Shaders and pipelines compiled before a game runs (macOS), for a
// launcher to do while its window is open: the pipelines earlier runs
// compiled (their recipes, metal_cache.h) built again from the translated
// shaders those runs kept, into the pipeline archive and the OS's cache of
// compiled shaders, so that the game finds them instead of compiling them
// while it draws. Needed after an OS update (both are then empty for the
// new OS build) or when the archive has lost shards; no game is needed.
//
// Plain C++: a launcher includes it and links recomp_gfx. On iOS and tvOS
// (where the launcher runs in the game's process) PrewarmNeeded is false
// and PrewarmCaches returns at once.

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

enum class PrewarmResult {
  Done,         // all built
  Cancelled,    // `cancel` was set: stopped (what was built is kept)
  NothingToDo,  // nothing missing
  Unavailable,  // no Metal device, or not macOS
};

// Called from the threads doing the work (one call at a time): `done` of
// `total` (shaders compiled, then pipelines built), `phase` "shaders" or
// "pipelines".
using PrewarmProgress = std::function<void(size_t done, size_t total, const char* phase)>;

// Builds what PrewarmNeeded says is missing, with a Metal device of its
// own, on a few threads; returns when done or, once `cancel` is set,
// within about the time of one compile (the archive's last save then left
// to finish in the background, or to the next run).
PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir = {});

}  // namespace recomp::gfx
