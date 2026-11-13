// Shaders and pipelines compiled before a game runs: see prewarm.h.

#include "runtime/gfx/prewarm.h"

#include <TargetConditionals.h>

#if TARGET_OS_OSX || TARGET_OS_TV || TARGET_OS_IOS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/gfx/metal_archive.h"
#include "runtime/gfx/metal_cache.h"

namespace recomp::gfx {

namespace {

// A function to compile: a translated shader's, with the alpha test or
// without it (`full` false: its lean variant).
struct FunctionSpec {
  uint64_t hash;
  bool vertex, full;
};

uint64_t FunctionKey(uint64_t hash, bool full) { return hash * 2 + (full ? 0 : 1); }

// The cache files' content: the translated shaders' MSL and the recipes
// (each once), of this translation generation (metal_cache.h).
struct Sources {
  std::unordered_map<uint64_t, std::string> msl;  // Shader::hash -> MSL
  std::vector<PipelineRecipe> recipes;
};

// A file's size and time, to tell whether it changed.
std::string Stamp(const std::string& path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) return "-";
  return std::to_string(st.st_size) + "/" + std::to_string(st.st_mtimespec.tv_sec) + "." +
         std::to_string(st.st_mtimespec.tv_nsec);
}

// Read once for PrewarmNeeded and PrewarmCaches (again only if the files
// changed in between).
std::shared_ptr<const Sources> LoadSources(const std::string& dir) {
  static std::mutex mu;
  static std::string last_stamp;
#if TARGET_OS_TV || TARGET_OS_IOS
  // Do not retain the source cache during gameplay on memory-limited TVs.
  static std::weak_ptr<const Sources> last;
#else
  static std::shared_ptr<const Sources> last;
#endif
  const std::string msl_path = MslCacheFile(dir), recipe_path = RecipeFile(dir);
  const std::string stamp = dir + "|" + Stamp(msl_path) + "|" + Stamp(recipe_path);
  std::lock_guard<std::mutex> l(mu);
#if TARGET_OS_TV || TARGET_OS_IOS
  auto cached = last.lock();
#else
  auto cached = last;
#endif
  if (cached && stamp == last_stamp) return cached;
  auto src = std::make_shared<Sources>();
  if (FILE* f = std::fopen(msl_path.c_str(), "rb")) {
    uint64_t h;
    uint32_t n;
    while (std::fread(&h, 8, 1, f) == 1 && std::fread(&n, 4, 1, f) == 1 && n < (16u << 20)) {
      std::string text(n, '\0');
      if (std::fread(text.data(), 1, n, f) != n) break;
      src->msl[h] = std::move(text);  // the last of a hash wins, as in the game
    }
    std::fclose(f);
  }
  std::unordered_set<uint64_t> seen;
  if (FILE* f = std::fopen(recipe_path.c_str(), "rb")) {
    PipelineRecipe r;
    while (std::fread(&r, sizeof r, 1, f) == 1)
      if (seen.insert(ArchiveKey(r)).second) src->recipes.push_back(r);
    std::fclose(f);
  }
  last_stamp = stamp;
  last = src;
  return src;
}

// What is missing: the recipes whose pipelines the archive does not have
// (and whose shaders were kept), and the functions they need.
struct Plan {
  std::shared_ptr<const Sources> src;
  std::vector<PipelineRecipe> recipes;
  std::vector<FunctionSpec> functions;
  size_t cached = 0;
};

Plan MakePlan(const std::string& dir, id<MTLDevice> dev) {
  Plan p;
  p.src = LoadSources(dir);
  const auto& msl = p.src->msl;
  const std::unordered_set<uint64_t> saved = PipelineArchive::SavedKeys(ArchiveFolder(dir), ArchiveScope(dev));
  std::unordered_set<uint64_t> fns;
  for (const PipelineRecipe& r : p.src->recipes) {
    if (saved.count(ArchiveKey(r))) { p.cached++; continue; }
    auto vs = msl.find(r.vs), ps = msl.find(r.ps);
    if (vs == msl.end() || ps == msl.end()) continue;
    const bool lean = r.key.lean & kLeanPs;
    if (lean) {
      // A lean variant needs the alpha test's macro (MetalLeanMacros).
      std::string text = ps->second;
      if (!MetalLeanMacros(&text, false)) continue;
    }
    p.recipes.push_back(r);
    if (fns.insert(FunctionKey(r.vs, true)).second) p.functions.push_back({r.vs, true, true});
    if (fns.insert(FunctionKey(r.ps, !lean)).second) p.functions.push_back({r.ps, false, !lean});
  }
  return p;
}

// fn(i) for i in [0, n) on `threads` threads, stopping when `cancel` is set.
template <typename F>
void Parallel(size_t n, unsigned threads, const std::atomic<bool>& cancel, F fn) {
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; t++)
    pool.emplace_back([&] {
      for (size_t i; !cancel.load() && (i = next++) < n;) {
        @autoreleasepool {
          fn(i);
        }
      }
    });
  for (std::thread& t : pool) t.join();
}

}  // namespace

bool PrewarmNeeded(const std::string& cache_dir) {
  return InspectPrewarmCache(cache_dir) == PrewarmCacheState::Needed;
}

PrewarmCacheState InspectPrewarmCache(const std::string& cache_dir) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return PrewarmCacheState::Unavailable;
    Plan plan = MakePlan(cache_dir.empty() ? MetalCacheDir() : cache_dir, dev);
    if (plan.src->msl.empty() || plan.src->recipes.empty()) return PrewarmCacheState::Empty;
    return plan.recipes.empty() ? PrewarmCacheState::Ready : PrewarmCacheState::Needed;
  }
}

PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir) {
  PrewarmArchiveLimits limits;
#if TARGET_OS_TV || TARGET_OS_IOS
  limits.workers = 1;
  limits.max_unsaved = 64;
  limits.drain_on_close = true;
#endif
  return PrewarmCaches(progress, cancel, cache_dir, limits);
}

PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir, const PrewarmArchiveLimits& limits,
                            const PrewarmStatusProgress& status) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return PrewarmResult::Unavailable;
    const std::string dir = cache_dir.empty() ? MetalCacheDir() : cache_dir;
    RemoveOldGenerations(dir);
    Plan plan = MakePlan(dir, dev);
    if (plan.recipes.empty()) return PrewarmResult::NothingToDo;

    PipelineArchive::Options o;
    o.dir = ArchiveFolder(dir);
    o.scope = ArchiveScope(dev);
    o.save_every = limits.save_every;
    o.max_unsaved = limits.max_unsaved;
    std::shared_ptr<PipelineArchive> archive = PipelineArchive::Open(dev, o);
    // Another process (the game) saves to it: nothing built here would be
    // kept.
    if (!archive->Saves()) {
      archive->Close(0.1);
      std::fprintf(stderr, "gfx: prewarm: the pipeline archive is in use: not now\n");
      return PrewarmResult::Busy;
    }

    const size_t total = plan.functions.size() + plan.recipes.size();
    std::atomic<size_t> done{0};
    std::mutex report_mu;
    PrewarmStatus current;
    current.functions_total = plan.functions.size();
    current.pipelines_ready = plan.cached;
    current.pipelines_total = plan.cached + plan.recipes.size();
    // In order: the count taken under the lock the call is made under.
    auto report = [&](const char* phase, bool succeeded = true) {
      std::lock_guard<std::mutex> l(report_mu);
      const size_t n = ++done;
      if (std::strcmp(phase, "shaders") == 0) current.functions_done++;
      else {
        current.phase = PrewarmStatus::Phase::Pipelines;
        if (succeeded) current.pipelines_ready++;
      }
      if (status) status(current);
      if (progress) progress(n, total, phase);
    };
    if (status) status(current);
    if (progress) progress(0, total, "shaders");
    // Half the cores, at most 4: the launcher's window stays responsive.
    const unsigned threads = std::clamp<unsigned>(limits.workers ? limits.workers :
        unsigned(NSProcessInfo.processInfo.activeProcessorCount / 2), 1, 4);
    std::fprintf(stderr, "gfx: prewarm: %zu shaders, %zu pipelines, %u threads\n", plan.functions.size(),
                 plan.recipes.size(), threads);

    // The functions, compiled as the game compiles them (CompileShader,
    // Lean): the same MSL, the same macros, so that the archive matches.
    std::mutex fn_mu;
    std::unordered_map<uint64_t, id<MTLFunction>> fns;
    Parallel(plan.functions.size(), threads, cancel, [&](size_t i) {
      const FunctionSpec& f = plan.functions[i];
      std::string text = plan.src->msl.at(f.hash);
      const bool has_lean = MetalLeanMacros(&text, f.vertex);
      NSError* err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:@(text.c_str())
                                             options:has_lean ? LeanMslOptions(f.full) : nil
                                               error:&err];
      id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
      if (!fn)
        std::fprintf(stderr, "gfx: prewarm: shader %016llx (%s, %s) failed: %s\n",
                     (unsigned long long)f.hash, f.vertex ? "vertex" : "pixel", f.full ? "full" : "lean",
                     err ? err.localizedDescription.UTF8String : "main0 function unavailable");
      {
        std::lock_guard<std::mutex> l(fn_mu);
        fns[FunctionKey(f.hash, f.full)] = fn;
      }
      report("shaders");
    });

    // The pipelines: looked up in the archive, else compiled (into the OS's
    // cache) and kept in it.
    Parallel(plan.recipes.size(), threads, cancel, [&](size_t i) {
      const PipelineRecipe& r = plan.recipes[i];
      id<MTLFunction> vs = nil, ps = nil;
      {
        std::lock_guard<std::mutex> l(fn_mu);
        vs = fns[FunctionKey(r.vs, true)];
        ps = fns[FunctionKey(r.ps, !(r.key.lean & kLeanPs))];
      }
      bool built = false;
      if (vs && ps) {
        // Room in the archive's list (it keeps no more): else this one
        // would not be saved, and would be needed again next time. A save
        // asked for every 0.5 s meanwhile (after a failed one, none starts
        // by itself until a new pipeline comes; two failures in a row stop
        // saving and empty the list).
        for (int waited = 0; archive->Pending() + threads >= o.max_unsaved && !cancel.load(); waited++) {
          if (waited % 50 == 0) archive->SaveSoon();
          usleep(10000);
        }
        NSError* err = nil;
        built = archive->Build(DescribePipeline(r.key, vs, ps), ArchiveKey(r), &err) != nil;
        if (!built)
          std::fprintf(stderr, "gfx: prewarm: pipeline %016llx (vs %016llx, ps %016llx) failed: %s\n",
                       (unsigned long long)ArchiveKey(r), (unsigned long long)r.vs, (unsigned long long)r.ps,
                       err ? err.localizedDescription.UTF8String : "no reason given");
      }
      report("pipelines", built);
    });

    const bool cancelled = cancel.load();
    current.phase = PrewarmStatus::Phase::Saving;
    if (status) status(current);
    // What was built saved (cut short by `cancel`: then left to finish).
    if (limits.drain_on_close) archive->CloseAndWait();
    else archive->Close(cancelled ? 0.1 : 30.0, &cancel);
    const size_t remaining = MakePlan(dir, dev).recipes.size();
    current.phase = PrewarmStatus::Phase::Complete;
    current.pipelines_ready = current.pipelines_total - std::min(remaining, current.pipelines_total);
    if (status) status(current);
    std::fprintf(stderr, "gfx: prewarm: %s, %zu of %zu tasks attempted; %zu of %zu pipelines saved\n",
                 cancelled ? "cancelled" : remaining ? "incomplete" : "done", done.load(), total,
                 current.pipelines_ready, current.pipelines_total);
    return cancelled ? PrewarmResult::Cancelled : remaining ? PrewarmResult::Incomplete : PrewarmResult::Done;
  }
}

}  // namespace recomp::gfx

#else  // Other unsupported Apple platforms

namespace recomp::gfx {

bool PrewarmNeeded(const std::string&) { return false; }
PrewarmCacheState InspectPrewarmCache(const std::string&) { return PrewarmCacheState::Unavailable; }

PrewarmResult PrewarmCaches(const PrewarmProgress&, const std::atomic<bool>&, const std::string&) {
  return PrewarmResult::Unavailable;
}

PrewarmResult PrewarmCaches(const PrewarmProgress&, const std::atomic<bool>&, const std::string&,
                            const PrewarmArchiveLimits&, const PrewarmStatusProgress&) {
  return PrewarmResult::Unavailable;
}

}  // namespace recomp::gfx

#endif
