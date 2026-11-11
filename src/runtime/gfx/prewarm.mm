// Shaders and pipelines compiled before a game runs: see prewarm.h.

#include "runtime/gfx/prewarm.h"

#include <TargetConditionals.h>

#if TARGET_OS_OSX

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
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

// What is missing: the recipes whose pipelines the archive does not have
// (and whose shaders were kept), and the functions they need.
struct Plan {
  std::unordered_map<uint64_t, std::string> msl;  // Shader::hash -> MSL
  std::vector<PipelineRecipe> recipes;
  std::vector<FunctionSpec> functions;
};

Plan MakePlan(const std::string& dir, id<MTLDevice> dev) {
  Plan p;
  if (FILE* f = std::fopen(MslCacheFile(dir).c_str(), "rb")) {
    uint64_t h;
    uint32_t n;
    while (std::fread(&h, 8, 1, f) == 1 && std::fread(&n, 4, 1, f) == 1 && n < (16u << 20)) {
      std::string text(n, '\0');
      if (std::fread(text.data(), 1, n, f) != n) break;
      p.msl[h] = std::move(text);  // the last of a hash wins, as in the game
    }
    std::fclose(f);
  }
  const std::unordered_set<uint64_t> saved = PipelineArchive::SavedKeys(ArchiveFolder(dir), ArchiveScope(dev));
  std::unordered_set<uint64_t> seen, fns;
  if (FILE* f = std::fopen(RecipeFile(dir).c_str(), "rb")) {
    PipelineRecipe r;
    while (std::fread(&r, sizeof r, 1, f) == 1) {
      const uint64_t key = ArchiveKey(r);
      if (saved.count(key) || !seen.insert(key).second) continue;
      auto vs = p.msl.find(r.vs), ps = p.msl.find(r.ps);
      if (vs == p.msl.end() || ps == p.msl.end()) continue;
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
    std::fclose(f);
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
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    return !MakePlan(cache_dir.empty() ? MetalCacheDir() : cache_dir, dev).recipes.empty();
  }
}

PrewarmResult PrewarmCaches(const PrewarmProgress& progress, const std::atomic<bool>& cancel,
                            const std::string& cache_dir) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return PrewarmResult::Unavailable;
    const std::string dir = cache_dir.empty() ? MetalCacheDir() : cache_dir;
    Plan plan = MakePlan(dir, dev);
    if (plan.recipes.empty()) return PrewarmResult::NothingToDo;

    PipelineArchive::Options o;
    o.dir = ArchiveFolder(dir);
    o.scope = ArchiveScope(dev);
    o.save_every = 64;
    std::shared_ptr<PipelineArchive> archive = PipelineArchive::Open(dev, o);
    archive->WaitLoaded();

    const size_t total = plan.functions.size() + plan.recipes.size();
    std::atomic<size_t> done{0};
    std::mutex report_mu;
    // In order: the count taken under the lock the call is made under.
    auto report = [&](const char* phase) {
      std::lock_guard<std::mutex> l(report_mu);
      const size_t n = ++done;
      if (progress) progress(n, total, phase);
    };
    if (progress) progress(0, total, "shaders");
    // Half the cores, at most 4: the launcher's window stays responsive.
    const unsigned threads = std::clamp<unsigned>(unsigned(NSProcessInfo.processInfo.activeProcessorCount / 2), 1, 4);
    std::fprintf(stderr, "gfx: prewarm: %zu shaders, %zu pipelines, %u threads\n", plan.functions.size(),
                 plan.recipes.size(), threads);

    // The functions, compiled as the game compiles them (CompileShader,
    // Lean): the same MSL, the same macros, so that the archive matches.
    std::mutex fn_mu;
    std::unordered_map<uint64_t, id<MTLFunction>> fns;
    Parallel(plan.functions.size(), threads, cancel, [&](size_t i) {
      const FunctionSpec& f = plan.functions[i];
      std::string text = plan.msl.at(f.hash);
      const bool has_lean = MetalLeanMacros(&text, f.vertex);
      NSError* err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:@(text.c_str())
                                             options:has_lean ? LeanMslOptions(f.full) : nil
                                               error:&err];
      id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
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
      if (vs && ps) {
        NSError* err = nil;
        archive->Build(DescribePipeline(r.key, vs, ps), ArchiveKey(r), &err);
      }
      report("pipelines");
    });

    const bool cancelled = cancel.load();
    // What was built saved (cut short by `cancel`: then left to finish).
    archive->Close(cancelled ? 0.1 : 30.0, &cancel);
    std::fprintf(stderr, "gfx: prewarm: %s, %zu of %zu done\n", cancelled ? "cancelled" : "done", done.load(),
                 total);
    return cancelled ? PrewarmResult::Cancelled : PrewarmResult::Done;
  }
}

}  // namespace recomp::gfx

#else  // iOS, tvOS: the launcher runs in the game's process

namespace recomp::gfx {

bool PrewarmNeeded(const std::string&) { return false; }

PrewarmResult PrewarmCaches(const PrewarmProgress&, const std::atomic<bool>&, const std::string&) {
  return PrewarmResult::Unavailable;
}

}  // namespace recomp::gfx

#endif
