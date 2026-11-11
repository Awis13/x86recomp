// PrewarmCaches (gfx/prewarm.h) on a cache folder of its own: an MSL
// cache and pipeline recipes as a game's runs leave them (a lean recipe
// among them, and one whose shader was not kept), then:
// - PrewarmNeeded says so; PrewarmCaches builds them all, its progress
//   reaching the total; afterwards nothing is needed, and the pipeline
//   archive finds every pipeline built as the game builds it;
// - with more of them, setting `cancel` stops it soon, with work left;
// - other translation generations' files are not used (the old ones
//   removed once unused for 30 days), nor keys files without their shard;
// - with the archive saved to by another process, it does nothing.
//
//   metal_prewarm_test        (skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "runtime/gfx/metal_archive.h"
#include "runtime/gfx/metal_cache.h"
#include "runtime/gfx/prewarm.h"

namespace {

using namespace recomp::gfx;

int failures = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

constexpr uint64_t kVs = 1, kPs = 100;  // shaders' hashes: the vertex one, pixel ones from kPs

std::string VertexMsl() {
  return R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V main0(uint id [[vertex_id]]) {
  V o;
  o.pos = float4(float(id & 1), float(id >> 1), 0.0, 1.0);
  return o;
}
)";
}

// Pixel shader i; the first has the alpha test's macro (a lean variant).
std::string PixelMsl(int i) {
  return std::string(R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
fragment float4 main0(V in [[stage_in]]) {
  float4 c = sin(in.pos.xyxy * )") +
         std::to_string(i + 1) + ".0);\n" + (i == 0 ? "  if (R_ALPHA_TEST && c.a < 0.5) discard_fragment();\n" : "") +
         "  return c;\n}\n";
}

PipelineRecipe Recipe(uint64_t ps, bool lean) {
  PipelineRecipe r;
  std::memset(&r, 0, sizeof r);
  r.vs = kVs;
  r.ps = ps;
  r.key.color[0] = uint16_t(MTLPixelFormatBGRA8Unorm);
  r.key.src = r.key.src_a = uint8_t(MTLBlendFactorOne);
  r.key.dst = r.key.dst_a = uint8_t(MTLBlendFactorZero);
  r.key.op = r.key.op_a = uint8_t(MTLBlendOperationAdd);
  r.key.mask[0] = 0xF;
  r.key.lean = lean ? kLeanPs : 0;
  return r;
}

// A cache folder with `n` pixel shaders: a recipe for each (the first also
// lean), and one for a pixel shader not kept.
std::string MakeCache(int n, std::vector<PipelineRecipe>* recipes) {
  char tmpl[1024];
  std::snprintf(tmpl, sizeof tmpl, "%s/metal_prewarm_test.XXXXXX", NSTemporaryDirectory().UTF8String);
  if (!mkdtemp(tmpl)) return {};
  const std::string dir = tmpl;
  FILE* f = std::fopen(MslCacheFile(dir).c_str(), "wb");
  auto put = [f](uint64_t h, const std::string& text) {
    const uint32_t len = uint32_t(text.size());
    std::fwrite(&h, 8, 1, f);
    std::fwrite(&len, 4, 1, f);
    std::fwrite(text.data(), 1, len, f);
  };
  put(kVs, VertexMsl());
  for (int i = 0; i < n; i++) put(kPs + uint64_t(i), PixelMsl(i));
  std::fclose(f);
  recipes->clear();
  for (int i = 0; i < n; i++) recipes->push_back(Recipe(kPs + uint64_t(i), false));
  recipes->push_back(Recipe(kPs, true));
  f = std::fopen(RecipeFile(dir).c_str(), "wb");
  for (const PipelineRecipe& r : *recipes) std::fwrite(&r, sizeof r, 1, f);
  const PipelineRecipe lost = Recipe(9999, false);  // its shader not kept: skipped
  std::fwrite(&lost, sizeof lost, 1, f);
  std::fclose(f);
  return dir;
}

// A cache file of another translation generation: the unversioned one of
// earlier runtimes (old: unused for 40 days) or a newer one. Shaders and
// recipes the current generation must not build.
void OtherGeneration(const std::string& path, bool msl, bool old) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (msl) {
    const std::string text = PixelMsl(50);
    const uint64_t h = 7777;
    const uint32_t len = uint32_t(text.size());
    std::fwrite(&h, 8, 1, f);
    std::fwrite(&len, 4, 1, f);
    std::fwrite(text.data(), 1, len, f);
  } else {
    const PipelineRecipe r = Recipe(7777, false);
    std::fwrite(&r, sizeof r, 1, f);
  }
  std::fclose(f);
  if (old) {
    timeval t[2];
    gettimeofday(&t[0], nullptr);
    t[0].tv_sec -= 40 * 86400;
    t[1] = t[0];
    utimes(path.c_str(), t);
  }
}

bool Exists(const std::string& p) { return access(p.c_str(), F_OK) == 0; }

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "metal_prewarm_test: no Metal device, skipped\n");
      return 77;
    }

    // ---- all of it.
    {
      std::vector<PipelineRecipe> recipes;
      const std::string dir = MakeCache(3, &recipes);
      // Other generations' files, and keys of every recipe in a keys file
      // whose shard is gone: none of it counts.
      OtherGeneration(dir + "/metal-msl-v1.bin", true, true);
      OtherGeneration(dir + "/metal-pipelines-v1.bin", false, true);
      OtherGeneration(dir + "/metal-msl-v9-0000000000000000.bin", true, false);
      {
        const std::string scope = ArchiveScope(dev);
        char name[64];
        std::snprintf(name, sizeof name, "/%016llx-7.keys", (unsigned long long)Fnv64(scope.data(), scope.size()));
        [NSFileManager.defaultManager createDirectoryAtPath:@(ArchiveFolder(dir).c_str())
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
        FILE* f = std::fopen((ArchiveFolder(dir) + name).c_str(), "wb");
        for (const PipelineRecipe& rc : recipes) {
          const uint64_t k = ArchiveKey(rc);
          std::fwrite(&k, 8, 1, f);
        }
        std::fclose(f);
      }
      Check(PrewarmNeeded(dir), "needed: nothing in the archive (a keys file without its shard does not count)");
      {
        // The game saving to the archive: nothing done meanwhile.
        PipelineArchive::Options o;
        o.dir = ArchiveFolder(dir);
        o.scope = ArchiveScope(dev);
        auto game = PipelineArchive::Open(dev, o);
        Check(game->Saves(), "the game's archive saves");
        std::atomic<bool> no{false};
        Check(PrewarmCaches(nullptr, no, dir) == PrewarmResult::Busy, "busy while another process saves");
        game->Close(2.0);
      }
      size_t last_done = 0, last_total = 0, calls = 0;
      bool pipelines = false;
      std::atomic<bool> cancel{false};
      const PrewarmResult r = PrewarmCaches(
          [&](size_t done, size_t total, const char* phase) {
            last_done = done;
            last_total = total;
            calls++;
            pipelines = pipelines || std::string(phase) == "pipelines";
          },
          cancel, dir);
      Check(r == PrewarmResult::Done, "done");
      // 1 vertex + 3 pixel + 1 lean function, 4 pipelines (none of other
      // generations).
      Check(last_total == 9 && last_done == 9 && calls == 10 && pipelines, "progress reaches the total");
      Check(!Exists(dir + "/metal-msl-v1.bin") && !Exists(dir + "/metal-pipelines-v1.bin") &&
                Exists(dir + "/metal-msl-v9-0000000000000000.bin"),
            "other generations' files removed once unused for 30 days, not before");
      Check(!PrewarmNeeded(dir), "nothing needed afterwards (the recipe without its shader not counted)");
      Check(PrewarmCaches(nullptr, cancel, dir) == PrewarmResult::NothingToDo, "a second run has nothing to do");

      // The archive has them, built as the game builds them.
      PipelineArchive::Options o;
      o.dir = ArchiveFolder(dir);
      o.scope = ArchiveScope(dev);
      auto a = PipelineArchive::Open(dev, o);
      a->WaitLoaded();
      size_t found_all = 0;
      for (const PipelineRecipe& rc : recipes) {
        const bool lean = rc.key.lean & kLeanPs;
        auto fn = [&](uint64_t h, bool vertex, bool full) {
          std::string text = h == kVs ? VertexMsl() : PixelMsl(int(h - kPs));
          const bool has_lean = MetalLeanMacros(&text, vertex);
          id<MTLLibrary> lib = [dev newLibraryWithSource:@(text.c_str())
                                                 options:has_lean ? LeanMslOptions(full) : nil
                                                   error:nil];
          return [lib newFunctionWithName:@"main0"];
        };
        NSError* e = nil;
        bool found = false;
        a->Build(DescribePipeline(rc.key, fn(rc.vs, true, true), fn(rc.ps, false, !lean)), ArchiveKey(rc), &e, &found);
        found_all += found;
      }
      Check(found_all == recipes.size(), "the archive finds every pipeline");
      a->Close(2.0);
      [NSFileManager.defaultManager removeItemAtPath:@(dir.c_str()) error:nil];
    }

    // ---- cancelled.
    {
      std::vector<PipelineRecipe> recipes;
      const std::string dir = MakeCache(60, &recipes);
      std::atomic<bool> cancel{false};
      size_t last_done = 0, last_total = 0;
      auto cancelled_at = std::chrono::steady_clock::now();
      const PrewarmResult r = PrewarmCaches(
          [&](size_t done, size_t total, const char*) {
            last_done = done;
            last_total = total;
            if (done == 3 && !cancel.load()) {
              cancelled_at = std::chrono::steady_clock::now();
              cancel = true;
            }
          },
          cancel, dir);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - cancelled_at).count();
      std::printf("metal_prewarm_test: cancelled after %zu of %zu, returned in %.0f ms\n", last_done, last_total,
                  s * 1e3);
      Check(r == PrewarmResult::Cancelled, "cancelled");
      Check(last_done < last_total && s < 0.5, "stops soon after cancel, with work left");
      Check(PrewarmNeeded(dir), "still needed after a cancel");
      [NSFileManager.defaultManager removeItemAtPath:@(dir.c_str()) error:nil];
    }
  }
  if (failures) return 1;
  std::printf("metal_prewarm_test: ok\n");
  return 0;
}
