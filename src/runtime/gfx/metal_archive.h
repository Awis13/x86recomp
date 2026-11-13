// Render pipelines kept compiled between runs (Metal binary archives), for
// the Metal backend (metal_device.mm); Objective-C++ only.
//
// Metal's own cache of compiled pipelines is per app and is not ours to
// rely on: an OS update empties it, and it is not always there when a run
// starts. Here the pipelines compiled in a run are also kept in an archive
// in the user's caches, and later runs look there first: a pipeline found
// in an archive is made in well under a millisecond, where compiling it
// took tens.
//
// Files: in `dir`, per scope (the app, the OS build, the GPU, the shader
// translation: compiled code is only good for the OS's compiler and the GPU
// it was made with), a sidecar <scope>.scope with the scope's text, and
// shards <scope>-<n>.bin, each with the keys of its pipelines in
// <scope>-<n>.keys. A scope's files are named from a hash of its text, so
// another OS build or GPU starts its own set. Files of a scope with the
// same app and GPU but another OS build (or format, or translation) are
// this app's and of no further use: removed when the archive is opened.
// Others' (another app, another GPU) are left alone. The shards used
// longest ago go when a scope has more than max_shards or max_bytes of
// them (a shard is marked used, its keys file touched, when one of its
// pipelines is found).
//
// What Metal allows: an archive saved once can be saved again only if
// nothing was added to it since (adding to it, or to the same archive
// opened from its file, and saving again fails with "cannot pack ...
// missing 'vertex' stage in pipeline no. N": seen with a few hundred real
// pipelines, at the second save of 64). So shards loaded from files are
// only looked in, never added to; the pipelines new in a run are kept in a
// list, and each save makes a new archive, adds the whole list (cheap: the
// pipelines are compiled already, a fraction of a ms each) and saves it
// once, as this run's own shard. Past max_pipelines (or the byte cap) that
// shard is done and the next pipelines start another, so a save re-adds
// and writes at most that many.
//
// Threads: Build may be called from any thread. Loading and saving run on
// one serial queue of the archive's own (iOS and tvOS: on the device's
// AheadQueue, as its target), never at the same time (a save when the app
// goes to the background, on iOS and tvOS, runs elsewhere so as not to
// wait behind the compiles ahead, but takes the same lock). Loading is its
// first work: pipelines built
// before it is done are compiled as if there were no archive, and saved
// only if the loaded shards do not have them. Two processes with the same
// scope: the one that opens it second only looks things up.
//
// Failures cost only time: a shard that does not load is removed; a save
// that fails leaves the files as they were and is tried again at the next
// save (two in a row stop saving for the run); shards whose own pipelines
// are all missed (outdated by a driver update the OS build did not show)
// are removed. In each case pipelines are compiled as before.

#pragma once

#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace recomp::gfx {

class PipelineArchive : public std::enable_shared_from_this<PipelineArchive> {
 public:
  // Changes when what is kept changes (part of the scope).
  static constexpr uint32_t kFormat = 2;
  // A shard is not grown past this (checked after each save): Metal has
  // failed saving archives of about 110 MB on iOS, which uses half of it.
  static constexpr uint64_t kShardCap = 64ull << 20;

  struct Options {
    std::string dir;    // made if missing
    std::string scope;  // lines of key=value; DefaultScope's by default
    uint64_t shard_cap = kShardCap;
    // A shard's pipelines at most (each save of the run's shard adds them
    // all again and writes them all).
    unsigned max_pipelines = 384;
    // A scope's shards in all: the oldest removed past either.
    uint64_t max_bytes = 4 * kShardCap;
    unsigned max_shards = 32;
    // The queue's: its target queue (iOS and tvOS: the device's AheadQueue,
    // so saves and the compiles ahead take turns), or nil: a queue of its
    // own at utility priority.
    dispatch_queue_t target = nil;
    // Saved after this many new pipelines, or this long after the last
    // one (a save writes the run's whole shard).
    unsigned save_every = 128;
    double save_idle_s = 5.0;
    // New pipelines kept to be saved at most (more are not kept: saves
    // that keep failing must not hold every shader's library).
    unsigned max_unsaved = 1024;
  };

  struct Stats {
    uint64_t hits = 0, misses = 0, miss_ns = 0;  // Build's lookups (since the last TakeStats)
    uint64_t saved = 0;    // new pipelines saved (in all)
    uint64_t pending = 0;  // new pipelines not saved yet
    double save_ms = -1;   // the last save (-1: none yet), and adding to it
    double add_ms = 0;
    uint64_t save_bytes = 0;
    uint64_t saves = 0;  // done (in all)
    size_t shards = 0;  // loaded and looked in
  };

  // Shards whose own pipelines (by their keys) have been looked for this
  // many times (fewer for small shards, at least 8) and never found are
  // outdated.
  static constexpr uint64_t kStaleMisses = 64;

  // Loading starts at once, on the archive's queue.
  static std::shared_ptr<PipelineArchive> Open(id<MTLDevice> dev, Options o);

  // This run's scope: format, app (bundle id, else the process name), OS
  // build, GPU (name and architecture).
  static std::string DefaultScope(id<MTLDevice> dev);

  // The keys of the pipelines saved in `dir` for `scope` (the keys files of
  // its shards that are there), without opening it: cheap.
  static std::unordered_set<uint64_t> SavedKeys(const std::string& dir, const std::string& scope);

  // Any thread. The pipeline of `pd`: from a shard if it is there (*found
  // true), else compiled (and, once per `key` in a run, a copy of `pd`
  // kept to be saved). `key` names the pipeline the same in every run (its
  // shaders' translations and the rest of its key, not the run's shader
  // ids). `pd`'s binaryArchives is left nil.
  id<MTLRenderPipelineState> Build(MTLRenderPipelineDescriptor* pd, uint64_t key, NSError** error,
                                   bool* found = nullptr);

  // No more new pipelines kept, saves not started any more but for
  // Close's (the device calls it first thing when it goes).
  void BeginClose();
  // BeginClose, then what is not saved yet saved: waited for at most
  // `cap_s`, or until `cancel` is set (a save not done by then finishes on the queue, or not at all
  // if the process exits first: its temporary file is then removed by a
  // later run).
  void Close(double cap_s, const std::atomic<bool>* cancel = nullptr);
  // A launcher sharing the game's process must finish queued work and
  // release the writer lease before the game's own archive opens.
  void CloseAndWait();

  Stats TakeStats();

  // New pipelines kept and not saved yet (Build keeps no more than
  // max_unsaved: a caller that must not lose any waits below it, asking
  // for a save meanwhile).
  uint64_t Pending() const { return pending_.load(); }
  void SaveSoon();  // a save queued, unless one is

  // Tests: loading done; a save now (waited for); whether this process
  // saves (not when another has the scope); the queue; a shard's file.
  void WaitLoaded();
  void Flush();
  bool Saves();
  dispatch_queue_t QueueForTests() const { return q_; }
  std::string ShardPath(int n) const;

  ~PipelineArchive();

 private:
  PipelineArchive(id<MTLDevice> dev, Options o);

  struct Entry {
    MTLRenderPipelineDescriptor* pd;
    uint64_t key;
    bool looked;  // looked up in the loaded shards (missed); else built before they were
  };

  // On the queue (Save also from a background save), under save_mu_ (but
  // Evict and Remove: their callers'). `final`: Close's save (else none
  // once closing has begun).
  void Load();
  void Save(bool final = false);
  bool SaveShard();
  void Evict(int keep);
  void DropStale();
  void Unlock();
  void Remove(int n);
  void Used(int n);  // any thread

  void SaveLater();
  void Log(const char* kind, const std::string& msg);

  id<MTLDevice> dev_;
  Options o_;
  std::string hex_;  // the scope's hash, naming its files
  dispatch_queue_t q_;

  // Build's (any thread), under mu_.
  std::mutex mu_;
  NSArray<id<MTLBinaryArchive>>* shards_ = nil;
  std::vector<int> loaded_n_;                 // the loaded shards' numbers
  std::unordered_map<uint64_t, int> listed_;  // the loaded shards' keys: their shard
  std::unordered_set<int> used_;              // shards marked used this run
  bool loaded_ = false;
  std::condition_variable loaded_cv_;
  std::unordered_set<uint64_t> keys_;  // kept this run
  std::vector<Entry> run_;             // this run's shard: its pipelines
  size_t saved_upto_ = 0;              // run_[0, saved_upto_) are in its file

  std::atomic<bool> closing_{false}, closed_{false};
  std::atomic<bool> stale_{false};
  std::atomic<bool> writer_{false};
  std::atomic<bool> timer_{false};
  std::atomic<bool> save_queued_{false};
  std::atomic<bool> stopped_{false};  // saving stopped (failures): nothing more kept
  std::atomic<uint64_t> last_new_ns_{0};
  std::atomic<uint64_t> hits_{0}, misses_{0}, miss_ns_{0};
  std::atomic<uint64_t> found_{0}, listed_missed_{0};  // in all, for the staleness test
  std::atomic<uint64_t> saved_{0}, pending_{0};
  std::atomic<double> save_ms_{-1}, add_ms_{0};
  std::atomic<uint64_t> save_bytes_{0}, saves_{0};

  // The queue's, under save_mu_.
  std::mutex save_mu_;
  int run_n_ = 0;  // this run's shard's number
  int failures_ = 0;  // saves failed in a row
  int lock_fd_ = -1;  // the sidecar, locked: this process saves
  std::unordered_set<std::string> logged_;
#if TARGET_OS_IOS || TARGET_OS_TV
  id background_observer_ = nil;
#endif
};

}  // namespace recomp::gfx
