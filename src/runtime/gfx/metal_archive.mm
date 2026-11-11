// Render pipelines kept compiled between runs: see metal_archive.h.

#include "runtime/gfx/metal_archive.h"

#import <Foundation/Foundation.h>
#if TARGET_OS_IOS || TARGET_OS_TV
#import <UIKit/UIKit.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
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

const char* Text(NSError* e) { return e ? e.localizedDescription.UTF8String : "no reason given"; }

std::string KeysPath(const std::string& shard) { return shard.substr(0, shard.size() - 4) + ".keys"; }

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

std::unordered_set<uint64_t> PipelineArchive::SavedKeys(const std::string& dir, const std::string& scope) {
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)Fnv64(scope.data(), scope.size()));
  std::unordered_set<uint64_t> keys;
  for (NSString* ns in [NSFileManager.defaultManager contentsOfDirectoryAtPath:@(dir.c_str()) error:nil]) {
    const std::string name = ns.UTF8String;
    if (name.size() < 23 || name.compare(0, 17, std::string(hex) + "-") != 0 ||
        name.compare(name.size() - 5, 5, ".keys") != 0)
      continue;
    const std::string shard = dir + "/" + name.substr(0, name.size() - 5) + ".bin";
    if (access(shard.c_str(), F_OK) != 0) continue;  // a shard removed, its keys left
    if (FILE* f = std::fopen((dir + "/" + name).c_str(), "rb")) {
      uint64_t k;
      while (std::fread(&k, 8, 1, f) == 1) keys.insert(k);
      std::fclose(f);
    }
  }
  return keys;
}

std::shared_ptr<PipelineArchive> PipelineArchive::Open(id<MTLDevice> dev, Options o) {
  std::shared_ptr<PipelineArchive> a(new PipelineArchive(dev, std::move(o)));
  dispatch_async(a->q_, ^{
    a->Load();
  });
#if TARGET_OS_IOS || TARGET_OS_TV
  // Sent to the background (where it may be ended without notice): what is
  // not saved yet saved, in a background task.
  std::weak_ptr<PipelineArchive> weak = a;
  a->background_observer_ = [NSNotificationCenter.defaultCenter
      addObserverForName:UIApplicationDidEnterBackgroundNotification
                  object:nil
                   queue:nil
              usingBlock:^(NSNotification*) {
                std::shared_ptr<PipelineArchive> self = weak.lock();
                if (!self || self->closing_.load() || !self->pending_.load()) return;
                UIApplication* app = UIApplication.sharedApplication;
                __block UIBackgroundTaskIdentifier task = UIBackgroundTaskInvalid;
                task = [app beginBackgroundTaskWithName:@"pipeline archive"
                                      expirationHandler:^{
                                        [app endBackgroundTask:task];
                                        task = UIBackgroundTaskInvalid;
                                      }];
                // Not behind the compiles ahead on AheadQueue (the lock keeps
                // it apart from the queue's saves).
                dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                  self->Save();
                  dispatch_async(dispatch_get_main_queue(), ^{
                    if (task != UIBackgroundTaskInvalid) [app endBackgroundTask:task];
                    task = UIBackgroundTaskInvalid;
                  });
                });
              }];
#endif
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

PipelineArchive::~PipelineArchive() {
#if TARGET_OS_IOS || TARGET_OS_TV
  if (background_observer_) [NSNotificationCenter.defaultCenter removeObserver:background_observer_];
#endif
  if (lock_fd_ >= 0) close(lock_fd_);
}

std::string PipelineArchive::ShardPath(int n) const { return o_.dir + "/" + hex_ + "-" + std::to_string(n) + ".bin"; }

void PipelineArchive::Log(const char* kind, const std::string& msg) {
  if (!logged_.insert(kind).second) return;
  std::fprintf(stderr, "gfx: pipeline archive: %s\n", msg.c_str());
}

// Shard `n`'s files removed.
void PipelineArchive::Remove(int n) {
  const std::string p = ShardPath(n);
  unlink(p.c_str());
  unlink(KeysPath(p).c_str());
}

// ---- loading (the queue's first work)

void PipelineArchive::Load() {
  std::lock_guard<std::mutex> save_lock(save_mu_);
  @autoreleasepool {
    NSFileManager* fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:@(o_.dir.c_str()) withIntermediateDirectories:YES attributes:nil error:nil];
    // The scope's sidecar, locked while this process saves: a second
    // process with the same scope only looks things up.
    const std::string sidecar = o_.dir + "/" + hex_ + ".scope";
    lock_fd_ = open(sidecar.c_str(), O_RDWR | O_CREAT | O_EXLOCK | O_NONBLOCK | O_CLOEXEC, 0644);
    if (lock_fd_ >= 0) {
      writer_ = true;
      if (FileSize(sidecar) == 0 && write(lock_fd_, o_.scope.data(), o_.scope.size()) < 0)
        Log("scope", std::string("cannot write ") + sidecar);
    } else if (errno == EWOULDBLOCK || errno == EAGAIN) {
      std::fprintf(stderr, "gfx: pipeline archive: in use by another process: only looked in\n");
    } else {
      std::fprintf(stderr, "gfx: pipeline archive: cannot open %s (%s): only looked in\n", sidecar.c_str(),
                   std::strerror(errno));
    }
    const bool writer = writer_.load();

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
      // A save's temporary file (<file>.<pid>.tmp) left by a run that
      // exited while saving (one of a run still saving is younger).
      if (rest.size() > 4 && rest.compare(rest.size() - 4, 4, ".tmp") == 0) {
        NSDate* m = [fm attributesOfItemAtPath:@(path.c_str()) error:nil].fileModificationDate;
        if (writer && m && -m.timeIntervalSinceNow > 600) unlink(path.c_str());
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
      // build, format or translation, is of no further use.
      if (rest == ".scope" && writer) {
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
      std::fprintf(stderr, "gfx: pipeline archive: %zu files of an earlier OS, GPU driver or translation removed\n",
                   removed);
    }
    run_n_ = mine.empty() ? 0 : mine.rbegin()->first + 1;
    // Room for this run's shard.
    if (writer) {
      Evict(-1);
      for (auto it = mine.begin(); it != mine.end();)
        it = access(it->second.c_str(), F_OK) == 0 ? std::next(it) : mine.erase(it);
    }

    const uint64_t t0 = NowNs();
    NSMutableArray<id<MTLBinaryArchive>>* loaded = [NSMutableArray array];
    std::vector<int> numbers;
    std::unordered_map<uint64_t, int> listed;
    uint64_t bytes = 0;
    for (const auto& [n, path] : mine) {
      MTLBinaryArchiveDescriptor* d = [MTLBinaryArchiveDescriptor new];
      d.url = Url(path);
      NSError* e = nil;
      id<MTLBinaryArchive> a = [dev_ newBinaryArchiveWithDescriptor:d error:&e];
      if (!a) {
        std::fprintf(stderr, "gfx: pipeline archive: %s does not load (%s)%s\n", path.c_str(), Text(e),
                     writer ? ": removed" : "");
        if (writer) Remove(n);
        continue;
      }
      [loaded addObject:a];
      numbers.push_back(n);
      bytes += FileSize(path);
      if (FILE* f = std::fopen(KeysPath(path).c_str(), "rb")) {
        uint64_t k;
        while (std::fread(&k, 8, 1, f) == 1) listed.emplace(k, n);
        std::fclose(f);
      }
    }
    if (loaded.count)
      std::fprintf(stderr, "gfx: pipeline archive: %zu shards (%.1f MB, %zu pipelines) loaded in %.0f ms\n",
                   size_t(loaded.count), double(bytes) / 1048576.0, listed.size(), double(NowNs() - t0) / 1e6);
    {
      std::lock_guard<std::mutex> l(mu_);
      shards_ = loaded.count ? [loaded copy] : nil;
      loaded_n_ = numbers;
      listed_ = std::move(listed);
      loaded_ = true;
    }
    loaded_cv_.notify_all();
  }
}

// The shards used longest ago (by their keys file's time: touched when
// one of their pipelines is found, written when they are saved) removed
// while the scope has more than max_bytes or max_shards (counting one more,
// the run's shard to come, if `keep` is -1; else never `keep`, the run's
// shard).
void PipelineArchive::Evict(int keep) {
  NSArray<NSString*>* names = [NSFileManager.defaultManager contentsOfDirectoryAtPath:@(o_.dir.c_str())
                                                                                error:nil] ?: @[];
  struct Shard {
    int n;
    uint64_t bytes;
    double used;
  };
  std::vector<Shard> shards;
  for (NSString* ns : names) {
    const std::string name = ns.UTF8String;
    if (name.size() < 22 || name.compare(0, 17, hex_ + "-") != 0 || name.compare(name.size() - 4, 4, ".bin") != 0)
      continue;
    const std::string digits = name.substr(17, name.size() - 21);
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) continue;
    const std::string path = o_.dir + "/" + name;
    struct stat st {};
    if (stat(KeysPath(path).c_str(), &st) != 0 && stat(path.c_str(), &st) != 0) continue;
    shards.push_back({std::atoi(digits.c_str()), FileSize(path),
                      double(st.st_mtimespec.tv_sec) + double(st.st_mtimespec.tv_nsec) / 1e9});
  }
  std::sort(shards.begin(), shards.end(),
            [](const Shard& a, const Shard& b) { return a.used != b.used ? a.used < b.used : a.n < b.n; });
  uint64_t total = 0;
  for (const Shard& sh : shards) total += sh.bytes;
  size_t count = shards.size() + (keep < 0 ? 1 : 0);
  for (const Shard& sh : shards) {
    if (total <= o_.max_bytes && count <= o_.max_shards) break;
    if (sh.n == keep) continue;
    std::fprintf(stderr, "gfx: pipeline archive: shard %d (%.1f MB), used longest ago, removed\n", sh.n,
                 double(sh.bytes) / 1048576.0);
    Remove(sh.n);
    total -= sh.bytes;
    count--;
  }
}

// Shard `n` marked used (its keys file touched), once a run.
void PipelineArchive::Used(int n) {
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!used_.insert(n).second) return;
  }
  if (writer_.load()) utimes(KeysPath(ShardPath(n)).c_str(), nullptr);
}

void PipelineArchive::WaitLoaded() {
  std::unique_lock<std::mutex> l(mu_);
  loaded_cv_.wait(l, [this] { return loaded_; });
}

bool PipelineArchive::Saves() {
  WaitLoaded();
  return writer_.load();
}

// ---- lookups

id<MTLRenderPipelineState> PipelineArchive::Build(MTLRenderPipelineDescriptor* pd, uint64_t key, NSError** error,
                                                  bool* found) {
  if (found) *found = false;
  NSArray<id<MTLBinaryArchive>>* shards;
  bool loaded;
  {
    std::lock_guard<std::mutex> l(mu_);
    shards = shards_;
    loaded = loaded_;
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
      if (found) *found = true;
      int n = -1;
      {
        std::lock_guard<std::mutex> l(mu_);
        auto it = listed_.find(key);
        if (it != listed_.end()) n = it->second;
      }
      if (n >= 0) Used(n);
      return p;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    miss_ns_.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    // Outdated shards? Only a miss of a pipeline they hold counts (not one
    // never saved: built before they loaded, or not saved by its run).
    uint64_t threshold = 0;
    bool listed;
    {
      std::lock_guard<std::mutex> l(mu_);
      listed = listed_.count(key) != 0;
      threshold = std::min<uint64_t>(kStaleMisses, std::max<uint64_t>(8, listed_.size() / 2));
    }
    if (listed) {
      const uint64_t n = listed_missed_.fetch_add(1, std::memory_order_relaxed) + 1;
      if (n >= threshold && !found_.load(std::memory_order_relaxed) && !stale_.exchange(true)) {
        {
          std::lock_guard<std::mutex> l(mu_);
          shards_ = nil;
          listed_.clear();
        }
        auto self = shared_from_this();
        dispatch_async(q_, ^{
          self->DropStale();
        });
      }
    }
  }
  id<MTLRenderPipelineState> p = [dev_ newRenderPipelineStateWithDescriptor:pd error:error];
  // Kept to be saved: not when closing or saving has stopped, nor by a
  // process that only looks.
  if (!p || closing_.load() || stopped_.load() || (loaded && !writer_.load())) return p;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (pending_.load() >= o_.max_unsaved || !keys_.insert(key).second) return p;
    // A copy: the caller may change or reuse its descriptor.
    MTLRenderPipelineDescriptor* copy = [pd copy];
    copy.binaryArchives = nil;
    run_.push_back({copy, key, shards != nil});
    pending_.store(run_.size() - saved_upto_);
  }
  last_new_ns_ = NowNs();
  // One save queued at a time (a save takes a while; pipelines keep coming).
  if (pending_.load() >= o_.save_every) {
    if (!save_queued_.exchange(true)) {
      auto self = shared_from_this();
      dispatch_async(q_, ^{
        self->Save();
      });
    }
  } else {
    SaveLater();
  }
  return p;
}

// A save once no new pipeline has come for save_idle_s (one timer at a
// time, set again while they keep coming).
void PipelineArchive::SaveLater() {
  if (timer_.exchange(true)) return;
  auto self = shared_from_this();
  const uint64_t idle = uint64_t(o_.save_idle_s * 1e9);
  const uint64_t since = NowNs() - last_new_ns_.load();
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, int64_t(since < idle ? idle - since : 0)), q_, ^{
    self->timer_ = false;
    if (self->closing_.load()) return;
    if (NowNs() - self->last_new_ns_.load() + 1000000 < idle)
      self->SaveLater();
    else
      self->Save();
  });
}

// ---- saving: a new archive with this run's shard's pipelines, saved once
// to a temporary file in the same folder, then put in place of the shard
// (a run that exits while saving leaves the files as they were), then its
// keys likewise.

void PipelineArchive::Save(bool final) {
  std::lock_guard<std::mutex> save_lock(save_mu_);
  save_queued_ = false;
  if (!writer_.load() || (closing_.load() && !final)) return;
  // Close's: every shard's worth (the others, one: the rest soon after).
  while (SaveShard() && final) {
  }
}

// One save of the run's shard; true if it filled up and more pipelines
// are waiting for the next.
bool PipelineArchive::SaveShard() {
  if (stopped_.load()) return false;
  @autoreleasepool {
    std::vector<Entry> list;
    {
      std::lock_guard<std::mutex> l(mu_);
      if (run_.size() == saved_upto_) return false;
      // Built before the shards loaded and in them after all: not saved again.
      std::vector<Entry> keep;
      for (size_t i = 0; i < run_.size(); i++)
        if (i < saved_upto_ || run_[i].looked || !listed_.count(run_[i].key)) keep.push_back(run_[i]);
      run_.swap(keep);
      // The shard's: max_pipelines at most (the rest start the next).
      list.assign(run_.begin(), run_.begin() + ptrdiff_t(std::min<size_t>(run_.size(), o_.max_pipelines)));
      pending_.store(run_.size() - saved_upto_);
    }
    if (list.size() == saved_upto_) return false;
    const std::string path = ShardPath(run_n_);
    const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
    const uint64_t t0 = NowNs();
    NSError* e = nil;
    id<MTLBinaryArchive> a = [dev_ newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
    if (!a) {
      Log("new", std::string("cannot make an archive (") + Text(e) + "): pipelines not saved");
      failures_ = 2;
    }
    std::vector<uint64_t> keys;  // of the pipelines added
    for (size_t i = 0; a && i < list.size(); i++) {
      if ([a addRenderPipelineFunctionsWithDescriptor:list[i].pd error:&e])
        keys.push_back(list[i].key);
      else
        Log("add", std::string("adding a pipeline: ") + Text(e) + " (logged once)");
    }
    const uint64_t t1 = NowNs();
    bool ok = a != nil;
    struct stat st {};
    if (ok) {
      unlink(tmp.c_str());
      ok = [a serializeToURL:Url(tmp) error:&e];
      // Only a file put in place (something else in the temporary file's
      // way, e.g. a folder, is reported saved).
      if (ok && (stat(tmp.c_str(), &st) != 0 || !S_ISREG(st.st_mode))) {
        e = [NSError errorWithDomain:NSPOSIXErrorDomain code:EEXIST userInfo:nil];
        ok = false;
      }
      if (ok && rename(tmp.c_str(), path.c_str()) != 0) {
        e = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:nil];
        ok = false;
      }
    }
    if (!ok) {
      if (a) {
        if (stat(tmp.c_str(), &st) == 0 && S_ISREG(st.st_mode)) unlink(tmp.c_str());
        // Never released: Metal has crashed releasing an archive whose save
        // failed. The files are as they were; the next save tries again.
        (void)CFBridgingRetain(a);
        failures_++;
      }
      if (failures_ >= 2) {
        std::fprintf(stderr, "gfx: pipeline archive: saving %s failed again (%s): not saved any more this run\n",
                     path.c_str(), Text(e));
        // Nothing more kept (the pipelines held their shaders' libraries).
        stopped_ = true;
        std::lock_guard<std::mutex> l(mu_);
        run_.clear();
        saved_upto_ = 0;
        pending_.store(0);
      } else {
        std::fprintf(stderr, "gfx: pipeline archive: saving %s failed (%s): tried again at the next save\n",
                     path.c_str(), Text(e));
        // Soon, also if no new pipeline comes to start one.
        if (!closing_.load()) SaveLater();
      }
      return false;
    }
    failures_ = 0;
    // Its keys (for the staleness test and to mark it used), those added.
    const std::string keys_path = KeysPath(path), keys_tmp = keys_path + "." + std::to_string(getpid()) + ".tmp";
    if (FILE* f = std::fopen(keys_tmp.c_str(), "wb")) {
      if (!keys.empty()) std::fwrite(keys.data(), 8, keys.size(), f);
      const bool written = std::fclose(f) == 0;
      if (!written || rename(keys_tmp.c_str(), keys_path.c_str()) != 0) unlink(keys_tmp.c_str());
    }
    const uint64_t bytes = FileSize(path);
    save_ms_.store(double(NowNs() - t0) / 1e6, std::memory_order_relaxed);
    add_ms_.store(double(t1 - t0) / 1e6, std::memory_order_relaxed);
    save_bytes_.store(bytes, std::memory_order_relaxed);
    saves_.fetch_add(1, std::memory_order_relaxed);
    const bool done = list.size() >= o_.max_pipelines || bytes >= o_.shard_cap;
    bool more;
    {
      std::lock_guard<std::mutex> l(mu_);
      saved_.fetch_add(list.size() - saved_upto_, std::memory_order_relaxed);
      saved_upto_ = list.size();
      // Full: the shard is done (its pipelines no longer held), the next
      // pipelines start another.
      if (done) {
        run_.erase(run_.begin(), run_.begin() + ptrdiff_t(saved_upto_));
        saved_upto_ = 0;
      }
      pending_.store(run_.size() - saved_upto_);
      more = run_.size() > saved_upto_;
    }
    if (!done) return false;
    Evict(run_n_);
    run_n_++;
    // What came meanwhile: saved soon in the next shard.
    if (more && !closing_.load()) SaveLater();
    return more;
  }
}

// The loaded shards' pipelines all missed: their files removed (they are
// no longer looked in). This run's shard is another file.
void PipelineArchive::DropStale() {
  std::lock_guard<std::mutex> save_lock(save_mu_);
  std::vector<int> numbers;
  {
    std::lock_guard<std::mutex> l(mu_);
    numbers.swap(loaded_n_);
  }
  std::fprintf(stderr, "gfx: pipeline archive: %llu of their pipelines looked for, none found: %zu outdated shards%s\n",
               (unsigned long long)listed_missed_.load(), numbers.size(), writer_.load() ? " removed" : "");
  if (!writer_.load()) return;
  for (int n : numbers) Remove(n);
}

// ---- closing

void PipelineArchive::Unlock() {
  std::lock_guard<std::mutex> save_lock(save_mu_);
  if (lock_fd_ >= 0) close(lock_fd_);
  lock_fd_ = -1;
}

void PipelineArchive::BeginClose() { closing_ = true; }

void PipelineArchive::SaveSoon() {
  if (closing_.load() || save_queued_.exchange(true)) return;
  auto self = shared_from_this();
  dispatch_async(q_, ^{
    self->Save();
  });
}

void PipelineArchive::Flush() {
  dispatch_sync(q_, ^{
    Save(closing_.load());
  });
}

void PipelineArchive::Close(double cap_s, const std::atomic<bool>* cancel) {
  BeginClose();
  if (closed_.exchange(true)) return;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  auto self = shared_from_this();
  const uint64_t t0 = NowNs();
  dispatch_async(q_, ^{
    self->Save(true);
    self->Unlock();
    dispatch_semaphore_signal(done);
  });
  // In slices of 50 ms (one at least), to see `cancel`.
  bool saved = false;
  do
    saved = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 50000000)) == 0;
  while (!saved && NowNs() - t0 < uint64_t(cap_s * 1e9) && !(cancel && cancel->load()));
  if (!saved)
    std::fprintf(stderr, "gfx: pipeline archive: not saved within %.1f s, left to finish\n", double(NowNs() - t0) / 1e9);
  else if (NowNs() - t0 > 500000000ull)
    std::fprintf(stderr, "gfx: pipeline archive: saved in %.1f s at exit\n", double(NowNs() - t0) / 1e9);
}

PipelineArchive::Stats PipelineArchive::TakeStats() {
  Stats s;
  s.hits = hits_.exchange(0);
  s.misses = misses_.exchange(0);
  s.miss_ns = miss_ns_.exchange(0);
  s.saved = saved_.load(std::memory_order_relaxed);
  s.pending = pending_.load(std::memory_order_relaxed);
  s.save_ms = save_ms_.load(std::memory_order_relaxed);
  s.add_ms = add_ms_.load(std::memory_order_relaxed);
  s.save_bytes = save_bytes_.load(std::memory_order_relaxed);
  s.saves = saves_.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> l(mu_);
  s.shards = shards_ ? size_t(shards_.count) : 0;
  return s;
}

}  // namespace recomp::gfx
