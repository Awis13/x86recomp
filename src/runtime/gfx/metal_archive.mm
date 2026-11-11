// Render pipelines kept compiled between runs: see metal_archive.h.

#include "runtime/gfx/metal_archive.h"

#import <Foundation/Foundation.h>

#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace recomp::gfx {

namespace {

uint64_t Fnv64(const void* p, size_t n) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 0x100000001B3ull;
  return h;
}

uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

// The value of `key` in a scope's lines ("" if none).
std::string Value(const std::string& scope, const char* key) {
  const std::string k = std::string(key) + "=";
  for (size_t p = 0; p < scope.size();) {
    size_t e = scope.find('\n', p);
    if (e == std::string::npos) e = scope.size();
    if (scope.compare(p, k.size(), k) == 0) return scope.substr(p + k.size(), e - p - k.size());
    p = e + 1;
  }
  return {};
}

NSURL* Url(const std::string& path) { return [NSURL fileURLWithPath:@(path.c_str())]; }

uint64_t FileSize(const std::string& path) {
  struct stat st {};
  return stat(path.c_str(), &st) == 0 ? uint64_t(st.st_size) : 0;
}

id<MTLBinaryArchive> OpenArchive(id<MTLDevice> dev, const std::string& path, NSError** e) {
  MTLBinaryArchiveDescriptor* d = [MTLBinaryArchiveDescriptor new];
  d.url = path.empty() ? nil : Url(path);
  return [dev newBinaryArchiveWithDescriptor:d error:e];
}

const char* Text(NSError* e) { return e ? e.localizedDescription.UTF8String : "no reason given"; }

}  // namespace

std::string PipelineArchive::DefaultScope(id<MTLDevice> dev) {
  std::string s = "format=" + std::to_string(kFormat) + "\n";
  NSString* app = NSBundle.mainBundle.bundleIdentifier ?: NSProcessInfo.processInfo.processName;
  s += std::string("app=") + (app ? app.UTF8String : "") + "\n";
  // The OS build (e.g. 25G45): it changes with every update, the
  // compiler's included. operatingSystemVersionString if it is not to be
  // had.
  char build[64] = {};
  size_t len = sizeof build - 1;
  if (sysctlbyname("kern.osversion", build, &len, nullptr, 0) == 0 && build[0])
    s += std::string("os=") + build + "\n";
  else
    s += std::string("os=") + NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String + "\n";
  // The GPU by its name and architecture, not its registryID: that is an
  // IORegistry entry's id, documented only as the same for every process,
  // not as the same after a restart, and compiled code depends on the
  // GPU's model, not on which entry it is.
  std::string gpu = dev.name.UTF8String;
  if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) gpu += std::string(" ") + dev.architecture.name.UTF8String;
  s += "gpu=" + gpu + "\n";
  return s;
}

std::shared_ptr<PipelineArchive> PipelineArchive::Open(id<MTLDevice> dev, Options o) {
  auto a = std::make_shared<PipelineArchive>(dev, std::move(o));
  dispatch_async(a->q_, ^{
    a->Load();
  });
  return a;
}

PipelineArchive::PipelineArchive(id<MTLDevice> dev, Options o) : dev_(dev), o_(std::move(o)) {
  if (o_.scope.empty()) o_.scope = DefaultScope(dev);
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)Fnv64(o_.scope.data(), o_.scope.size()));
  hex_ = hex;
  q_ = o_.target ? dispatch_queue_create_with_target("recomp.gfx.archive", DISPATCH_QUEUE_SERIAL, o_.target)
                 : dispatch_queue_create("recomp.gfx.archive",
                                         dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                                                                 QOS_CLASS_UTILITY, 0));
}

PipelineArchive::~PipelineArchive() = default;

std::string PipelineArchive::ShardPath(int n) const { return o_.dir + "/" + hex_ + "-" + std::to_string(n) + ".bin"; }

void PipelineArchive::Log(const char* kind, const std::string& msg) {
  if (!logged_.insert(kind).second) return;
  std::fprintf(stderr, "gfx: pipeline archive: %s\n", msg.c_str());
}

// ---- loading (the queue's first work)

void PipelineArchive::Load() {
  @autoreleasepool {
    NSFileManager* fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:@(o_.dir.c_str()) withIntermediateDirectories:YES attributes:nil error:nil];
    NSArray<NSString*>* names = [fm contentsOfDirectoryAtPath:@(o_.dir.c_str()) error:nil] ?: @[];
    const std::string app = Value(o_.scope, "app"), gpu = Value(o_.scope, "gpu");
    std::map<int, std::string> mine;  // shard number -> path
    std::vector<std::string> gone_scopes;
    for (NSString* ns : names) {
      const std::string name = ns.UTF8String, path = o_.dir + "/" + name;
      // Only names of the forms above (16 hex digits first) are ours.
      if (name.size() < 17) continue;
      const std::string hex = name.substr(0, 16);
      if (hex.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
      const std::string rest = name.substr(16);
      // A save's temporary file (<shard>.<pid>.tmp) left by a run that
      // exited while saving (one of a run still saving is younger).
      if (rest.size() > 4 && rest.compare(rest.size() - 4, 4, ".tmp") == 0) {
        NSDate* m = [fm attributesOfItemAtPath:@(path.c_str()) error:nil].fileModificationDate;
        if (m && -m.timeIntervalSinceNow > 600) unlink(path.c_str());
        continue;
      }
      if (hex == hex_) {
        // -<n>.bin
        const std::string digits = rest.size() > 5 ? rest.substr(1, rest.size() - 5) : std::string();
        if (rest[0] == '-' && rest.compare(rest.size() - 4, 4, ".bin") == 0 && !digits.empty() &&
            digits.size() < 7 && digits.find_first_not_of("0123456789") == std::string::npos)
          mine[std::atoi(digits.c_str())] = path;
        continue;
      }
      // Another scope's sidecar: this app's on this GPU, of another OS
      // build or format, is of no further use.
      if (rest == ".scope") {
        NSString* text = [NSString stringWithContentsOfFile:@(path.c_str()) encoding:NSUTF8StringEncoding error:nil];
        const std::string other = text ? text.UTF8String : "";
        if (!other.empty() && Value(other, "app") == app && Value(other, "gpu") == gpu) gone_scopes.push_back(hex);
      }
    }
    for (const std::string& hex : gone_scopes) {
      size_t removed = 0;
      for (NSString* ns : names) {
        const std::string name = ns.UTF8String;
        if (name.compare(0, 16, hex) == 0 && unlink((o_.dir + "/" + name).c_str()) == 0) removed++;
      }
      std::fprintf(stderr, "gfx: pipeline archive: %zu files of an earlier OS or GPU driver removed\n", removed);
    }
    // This scope's sidecar (what its files are for).
    const std::string sidecar = o_.dir + "/" + hex_ + ".scope";
    if (access(sidecar.c_str(), F_OK) != 0)
      [@(o_.scope.c_str()) writeToFile:@(sidecar.c_str()) atomically:YES encoding:NSUTF8StringEncoding error:nil];

    const uint64_t t0 = NowNs();
    NSMutableArray<id<MTLBinaryArchive>>* loaded = [NSMutableArray array];
    std::vector<std::string> paths;
    int last = -1;
    uint64_t bytes = 0;
    for (const auto& [n, path] : mine) {
      NSError* e = nil;
      id<MTLBinaryArchive> a = OpenArchive(dev_, path, &e);
      if (!a) {
        std::fprintf(stderr, "gfx: pipeline archive: %s does not load (%s): removed\n", path.c_str(), Text(e));
        unlink(path.c_str());
        continue;
      }
      [loaded addObject:a];
      paths.push_back(path);
      last = n;
      bytes += FileSize(path);
    }
    if (loaded.count)
      std::fprintf(stderr, "gfx: pipeline archive: %zu shards (%.1f MB) loaded in %.0f ms\n", size_t(loaded.count),
                   double(bytes) / 1048576.0, double(NowNs() - t0) / 1e6);
    {
      std::lock_guard<std::mutex> l(mu_);
      shards_ = loaded.count ? [loaded copy] : nil;
      shard_paths_ = paths;
      loaded_ = true;
    }
    loaded_cv_.notify_all();
    // Added to: the last shard while it is under the cap (opened again: the
    // one Build looks in is never changed), else a new one.
    if (last >= 0 && FileSize(mine[last]) < o_.shard_cap) {
      NSError* e = nil;
      writable_ = OpenArchive(dev_, mine[last], &e);
      writable_n_ = last;
    }
    if (!writable_) StartShard(last + 1, false);
  }
}

// A new, empty shard `n` to add to; `readd`: the pipelines not saved yet
// added to it again (else dropped).
void PipelineArchive::StartShard(int n, bool readd) {
  std::vector<MTLRenderPipelineDescriptor*> again;
  again.swap(unsaved_);
  NSError* e = nil;
  writable_ = OpenArchive(dev_, std::string(), &e);
  writable_n_ = n;
  if (!writable_) {
    Log("new", std::string("cannot make an archive (") + Text(e) + "): pipelines not kept");
    return;
  }
  if (!readd) return;
  for (MTLRenderPipelineDescriptor* pd : again)
    if ([writable_ addRenderPipelineFunctionsWithDescriptor:pd error:nil]) unsaved_.push_back(pd);
}

void PipelineArchive::WaitLoaded() {
  std::unique_lock<std::mutex> l(mu_);
  loaded_cv_.wait(l, [this] { return loaded_; });
}

// ---- lookups and adds

id<MTLRenderPipelineState> PipelineArchive::Build(MTLRenderPipelineDescriptor* pd, uint64_t key, NSError** error) {
  NSArray<id<MTLBinaryArchive>>* shards;
  {
    std::lock_guard<std::mutex> l(mu_);
    shards = shards_;
  }
  if (shards) {
    pd.binaryArchives = shards;
    const uint64_t t0 = NowNs();
    NSError* e = nil;
    id<MTLRenderPipelineState> p = [dev_ newRenderPipelineStateWithDescriptor:pd
                                                                      options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                   reflection:nil
                                                                        error:&e];
    pd.binaryArchives = nil;
    if (p) {
      hits_.fetch_add(1, std::memory_order_relaxed);
      found_.fetch_add(1, std::memory_order_relaxed);
      return p;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    miss_ns_.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    const uint64_t missed = missed_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (missed >= kStaleMisses && !found_.load(std::memory_order_relaxed) && !stale_.exchange(true)) {
      {
        std::lock_guard<std::mutex> l(mu_);
        shards_ = nil;
      }
      auto self = shared_from_this();
      dispatch_async(q_, ^{
        self->DropStale();
      });
    }
  }
  id<MTLRenderPipelineState> p = [dev_ newRenderPipelineStateWithDescriptor:pd error:error];
  if (!p || closing_.load()) return p;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!keys_.insert(key).second) return p;
  }
  // A copy: the caller may change or reuse its descriptor.
  MTLRenderPipelineDescriptor* copy = [pd copy];
  copy.binaryArchives = nil;
  pending_.fetch_add(1, std::memory_order_relaxed);
  auto self = shared_from_this();
  dispatch_async(q_, ^{
    self->Add(copy);
    self->pending_.fetch_sub(1, std::memory_order_relaxed);
  });
  return p;
}

void PipelineArchive::Add(MTLRenderPipelineDescriptor* pd) {
  if (!writable_ || (closing_.load() && NowNs() > add_until_.load())) return;
  @autoreleasepool {
    NSError* e = nil;
    if (![writable_ addRenderPipelineFunctionsWithDescriptor:pd error:&e]) {
      Log("add", std::string("adding a pipeline: ") + Text(e) + " (logged once)");
      return;
    }
    unsaved_.push_back(pd);
    added_.fetch_add(1, std::memory_order_relaxed);
    if (unsaved_.size() >= o_.save_every)
      Save();
    else
      SaveLater();
  }
}

// A save once no pipeline has been added for save_idle_s.
void PipelineArchive::SaveLater() {
  const uint64_t gen = ++add_gen_;
  auto self = shared_from_this();
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, int64_t(o_.save_idle_s * 1e9)), q_, ^{
    if (self->closing_.load() || gen != self->add_gen_) return;
    self->Save();
  });
}

// ---- saving: the shard written to a temporary file in the same folder,
// then put in its place (a run that exits while saving leaves the shard as
// it was).

void PipelineArchive::Save() {
  if (!writable_ || unsaved_.empty()) return;
  @autoreleasepool {
    const std::string path = ShardPath(writable_n_);
    const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
    const uint64_t t0 = NowNs();
    NSError* e = nil;
    if (![writable_ serializeToURL:Url(tmp) error:&e]) {
      unlink(tmp.c_str());
      return Failed("saving", e);
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
      const int err = errno;
      unlink(tmp.c_str());
      return Failed("saving", [NSError errorWithDomain:NSPOSIXErrorDomain code:err userInfo:nil]);
    }
    const uint64_t bytes = FileSize(path);
    save_ms_.store(double(NowNs() - t0) / 1e6, std::memory_order_relaxed);
    save_bytes_.store(bytes, std::memory_order_relaxed);
    unsaved_.clear();
    if (bytes >= o_.shard_cap) {
      StartShard(writable_n_ + 1, false);
      return;
    }
    // The code collected for the shard held in memory until now: the
    // shard opened from its file instead (mapped, not the app's memory).
    if (id<MTLBinaryArchive> a = OpenArchive(dev_, path, nil)) writable_ = a;
  }
}

// Saving the shard failed: the archive kept (never released), the shard's
// file removed (what this run saved in it before goes too: a later run
// adds it again), and the shard started again with the pipelines not saved
// yet; a second failure in a run stops adding.
void PipelineArchive::Failed(const char* what, NSError* e) {
  kept_.push_back(writable_);
  writable_ = nil;
  const std::string path = ShardPath(writable_n_);
  unlink(path.c_str());
  if (++failures_ >= 2) {
    std::fprintf(stderr, "gfx: pipeline archive: %s %s failed again (%s): no more pipelines kept this run\n", what,
                 path.c_str(), Text(e));
    unsaved_.clear();
    return;
  }
  std::fprintf(stderr, "gfx: pipeline archive: %s %s failed (%s): removed, started again\n", what, path.c_str(),
               Text(e));
  StartShard(writable_n_, true);
}

// Every lookup a miss: the loaded shards (no longer looked in) removed, and
// the one added to started again (with what is not saved yet) if it was one
// of them.
void PipelineArchive::DropStale() {
  std::vector<std::string> paths;
  {
    std::lock_guard<std::mutex> l(mu_);
    paths.swap(shard_paths_);
  }
  std::fprintf(stderr, "gfx: pipeline archive: %llu lookups, none found: %zu outdated shards removed\n",
               (unsigned long long)missed_.load(), paths.size());
  bool writing_one = false;
  for (const std::string& p : paths) {
    unlink(p.c_str());
    writing_one = writing_one || p == ShardPath(writable_n_);
  }
  if (writing_one && writable_) StartShard(0, true);
}

// ---- closing

void PipelineArchive::Flush() {
  dispatch_sync(q_, ^{
    Save();
  });
}

void PipelineArchive::Close(double cap_s) {
  const uint64_t t0 = NowNs();
  // Pipelines still queued: added in the first half of the time, the rest
  // dropped (a later run adds them).
  add_until_ = t0 + uint64_t(cap_s * 0.5e9);
  if (closing_.exchange(true)) return;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  auto self = shared_from_this();
  dispatch_async(q_, ^{
    self->Save();
    dispatch_semaphore_signal(done);
  });
  if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, int64_t(cap_s * 1e9))) != 0)
    std::fprintf(stderr, "gfx: pipeline archive: not saved within %.1f s, left to finish\n", cap_s);
  else if (NowNs() - t0 > 500000000ull)
    std::fprintf(stderr, "gfx: pipeline archive: saved in %.1f s at exit\n", double(NowNs() - t0) / 1e9);
}

PipelineArchive::Stats PipelineArchive::TakeStats() {
  Stats s;
  s.hits = hits_.exchange(0);
  s.misses = misses_.exchange(0);
  s.miss_ns = miss_ns_.exchange(0);
  s.added = added_.load(std::memory_order_relaxed);
  s.pending = pending_.load(std::memory_order_relaxed);
  s.save_ms = save_ms_.load(std::memory_order_relaxed);
  s.save_bytes = save_bytes_.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> l(mu_);
  s.shards = shards_ ? size_t(shards_.count) : 0;
  return s;
}

}  // namespace recomp::gfx
