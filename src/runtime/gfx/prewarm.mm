// Shaders and pipelines compiled before a game runs: see prewarm.h.

#include "runtime/gfx/prewarm.h"

#include <TargetConditionals.h>

#if TARGET_OS_OSX

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
  static std::shared_ptr<const Sources> last;
  const std::string msl_path = MslCacheFile(dir), recipe_path = RecipeFile(dir);
  const std::string stamp = dir + "|" + Stamp(msl_path) + "|" + Stamp(recipe_path);
  std::lock_guard<std::mutex> l(mu);
  if (last && stamp == last_stamp) return last;
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
};

Plan MakePlan(const std::string& dir, id<MTLDevice> dev) {
  Plan p;
  p.src = LoadSources(dir);
  const auto& msl = p.src->msl;
  const std::unordered_set<uint64_t> saved = PipelineArchive::SavedKeys(ArchiveFolder(dir), ArchiveScope(dev));
  std::unordered_set<uint64_t> fns;
  for (const PipelineRecipe& r : p.src->recipes) {
    if (saved.count(ArchiveKey(r))) continue;
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
    RemoveOldGenerations(dir);
    Plan plan = MakePlan(dir, dev);
    if (plan.recipes.empty()) return PrewarmResult::NothingToDo;

    PipelineArchive::Options o;
    o.dir = ArchiveFolder(dir);
    o.scope = ArchiveScope(dev);
    o.save_every = 64;
    o.max_unsaved = 1024;
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
      std::string text = plan.src->msl.at(f.hash);
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
        // Room in the archive's list (it keeps no more): else this one
        // would not be saved, and would be needed again next time.
        while (archive->Pending() + threads >= o.max_unsaved && !cancel.load()) usleep(10000);
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
