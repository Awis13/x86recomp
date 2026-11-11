// Render pipelines kept compiled between runs (Metal binary archives), for
// the Metal backend (metal_device.mm); Objective-C++ only.
//
// Metal's own cache of compiled pipelines is per app and is not ours to
// rely on: an OS update empties it, and it is not always there when a run
// starts. Here every pipeline compiled in a run is also added to an archive
// in the user's caches, and later runs look there first: a pipeline found
// in an archive is made in well under a millisecond, where compiling it
// took tens.
//
// Files: in `dir`, per scope (the app, the OS build, the GPU: compiled code
// is only good for the OS's compiler and the GPU it was made with), a
// sidecar <scope>.scope with the scope's text and shards <scope>-<n>.bin,
// each grown until it passes the shard cap, then a new one. A scope's files
// are named from a hash of its text, so another OS build or GPU starts its
// own set. Files of a scope with the same app and GPU but another OS build
// (or format) are this app's and of no further use: removed when the
// archive is opened. Others' (another app, another GPU) are left alone.
//
// Threads: Build may be called from any thread. Loading, adding and saving
// run on one serial queue of the archive's own (they never run at the same
// time as each other: Metal has crashed saving an archive while another
// thread compiled into it); the archives looked in are the shards loaded
// when the archive was opened, never the one being added to. Loading is
// the queue's first work: pipelines built before it is done are compiled
// as if there were no archive (and added).
//
// Failures cost only time: a shard that does not load is removed; a save
// that fails removes the shard and starts it again with the pipelines not
// saved yet (a second failure stops adding for the run); shards that are
// all misses (outdated by a driver update the OS build did not show) are
// removed. In each case pipelines are compiled as before.

#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace recomp::gfx {

class PipelineArchive : public std::enable_shared_from_this<PipelineArchive> {
 public:
  // Changes when what is kept changes (part of the scope).
  static constexpr uint32_t kFormat = 1;
  // A shard is not grown past this (checked after each save): Metal has
  // failed saving archives of about 110 MB on iOS.
  static constexpr uint64_t kShardCap = 64ull << 20;

  struct Options {
    std::string dir;    // made if missing
    std::string scope;  // lines of key=value; DefaultScope's by default
    uint64_t shard_cap = kShardCap;
    // The queue's: its target queue (iOS and tvOS: the device's AheadQueue,
    // so archive work and the compiles ahead take turns), or nil: a queue
    // of its own at utility priority.
    dispatch_queue_t target = nil;
    // Saved after this many new pipelines, or this long after the last one
    // added (a save writes the whole shard).
    unsigned save_every = 128;
    double save_idle_s = 5.0;
  };

  struct Stats {
    uint64_t hits = 0, misses = 0, miss_ns = 0;  // Build's lookups (since the last TakeStats)
    uint64_t added = 0, pending = 0;             // added (in all), waiting to be added
    double save_ms = -1;                         // the last save (-1: none yet)
    uint64_t save_bytes = 0;
    size_t shards = 0;  // loaded and looked in
  };

  // A pipeline found by Build in no shard after this many lookups with not
  // one found: the shards are outdated.
  static constexpr uint64_t kStaleMisses = 64;

  // Loading starts at once, on the archive's queue.
  static std::shared_ptr<PipelineArchive> Open(id<MTLDevice> dev, Options o);

  // This run's scope: format, app (bundle id, else the process name), OS
  // build, GPU (name and architecture).
  static std::string DefaultScope(id<MTLDevice> dev);

  // Any thread. The pipeline of `pd`: from a shard if it is there, else
  // compiled (and, once per `key` in a run, a copy of `pd` queued to be
  // added). `key` names the pipeline the same in every run (its shaders'
  // translations and the rest of its key, not the run's shader ids).
  // `pd`'s binaryArchives is left nil.
  id<MTLRenderPipelineState> Build(MTLRenderPipelineDescriptor* pd, uint64_t key, NSError** error);

  // Tests: loading done; everything queued added, then saved.
  void WaitLoaded();
  void Flush();

  // Nothing more queued to be added (what is queued added in the first
  // half of `cap_s`, the rest dropped), what was added saved: waited for
  // at most `cap_s` (a save not done by then finishes
  // on the queue, or not at all if the process exits first: its
  // temporary file is then removed by a later run).
  void Close(double cap_s);

  Stats TakeStats();

  PipelineArchive(id<MTLDevice> dev, Options o);
  ~PipelineArchive();

 private:
  // All on the queue.
  void Load();
  void Add(MTLRenderPipelineDescriptor* pd);
  void Save();
  void SaveLater();
  void Failed(const char* what, NSError* e);
  void DropStale();
  void StartShard(int n, bool readd);
  std::string ShardPath(int n) const;

  void Log(const char* kind, const std::string& msg);

  id<MTLDevice> dev_;
  Options o_;
  std::string hex_;  // the scope's hash, naming its files
  dispatch_queue_t q_;

  // Looked in by Build (any thread).
  std::mutex mu_;
  NSArray<id<MTLBinaryArchive>>* shards_ = nil;
  std::vector<std::string> shard_paths_;  // the loaded shards' files
  bool loaded_ = false;
  std::condition_variable loaded_cv_;
  std::unordered_set<uint64_t> keys_;  // queued to be added this run

  std::atomic<bool> closing_{false};
  std::atomic<uint64_t> add_until_{0};  // closing: queued adds dropped after this (steady ns)
  std::atomic<bool> stale_{false};
  std::atomic<uint64_t> hits_{0}, misses_{0}, miss_ns_{0};
  std::atomic<uint64_t> found_{0}, missed_{0};  // in all, for the staleness test
  std::atomic<uint64_t> added_{0}, pending_{0};
  std::atomic<double> save_ms_{-1};
  std::atomic<uint64_t> save_bytes_{0};

  // The queue's.
  id<MTLBinaryArchive> writable_ = nil;  // added to; nil: adding stopped
  int writable_n_ = 0;
  // The pipelines added to writable_ since it was last saved, to add them
  // again if the save fails (only those: holding all of a run's would keep
  // every shader's library alive).
  std::vector<MTLRenderPipelineDescriptor*> unsaved_;
  uint64_t add_gen_ = 0;
  int failures_ = 0;
  // Archives whose save failed: never released (Metal has crashed
  // releasing such an archive).
  std::vector<id<MTLBinaryArchive>> kept_;
  std::unordered_set<std::string> logged_;
};

}  // namespace recomp::gfx
