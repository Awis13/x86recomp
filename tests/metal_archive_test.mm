// Pipelines kept compiled between runs on the Metal backend
// (gfx/metal_archive.h), in a folder of its own: pipelines built through
// one archive, saved, are found by the next archive opened on the folder
// (as by a later run), also those added to a shard loaded from a file; a
// damaged shard does not load, is removed, and pipelines are still built;
// another scope (another OS build) finds nothing, starts its own files and
// removes the earlier build's, but not those of another GPU; past the
// shard cap a new shard is started, and all shards are looked in; shards
// in which every lookup misses are removed.
//
//   metal_archive_test        (skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "runtime/gfx/metal_archive.h"

namespace {

using recomp::gfx::PipelineArchive;

int failures = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

constexpr const char* kMsl = R"(
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V v(uint id [[vertex_id]]) {
  V o;
  o.pos = float4(float(id & 1), float(id >> 1), 0.0, 1.0);
  return o;
}
fragment float4 f0() { return float4(1.0, 0.0, 0.0, 1.0); }
fragment float4 f1(V in [[stage_in]]) { return float4(in.pos.xy * 0.001, 0.5, 1.0); }
fragment float4 f2(V in [[stage_in]]) { return sin(in.pos.xxyy); }
fragment float4 f3(V in [[stage_in]]) { return cos(in.pos.yyxx) * 0.5; }
)";

std::string Scope(const char* os, const char* gpu) {
  return std::string("format=1\napp=metal_archive_test\nos=") + os + "\ngpu=" + gpu + "\n";
}

std::vector<std::string> Files(const std::string& dir, const char* suffix) {
  std::vector<std::string> out;
  for (NSString* n in [NSFileManager.defaultManager contentsOfDirectoryAtPath:@(dir.c_str()) error:nil]) {
    const std::string s = n.UTF8String;
    if (s.size() >= std::strlen(suffix) && s.compare(s.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0)
      out.push_back(s);
  }
  return out;
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "metal_archive_test: no Metal device, skipped\n");
      return 77;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kMsl) options:nil error:&err];
    if (!lib) {
      std::fprintf(stderr, "metal_archive_test: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> vs = [lib newFunctionWithName:@"v"];
    id<MTLFunction> fs[4] = {[lib newFunctionWithName:@"f0"], [lib newFunctionWithName:@"f1"],
                             [lib newFunctionWithName:@"f2"], [lib newFunctionWithName:@"f3"]};
    auto describe = [&](int i, MTLPixelFormat fmt = MTLPixelFormatBGRA8Unorm, MTLColorWriteMask mask = MTLColorWriteMaskAll) {
      MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
      pd.vertexFunction = vs;
      pd.fragmentFunction = fs[i];
      pd.colorAttachments[0].pixelFormat = fmt;
      pd.colorAttachments[0].writeMask = mask;
      return pd;
    };

    char tmpl[1024];
    std::snprintf(tmpl, sizeof tmpl, "%s/metal_archive_test.XXXXXX", NSTemporaryDirectory().UTF8String);
    if (!mkdtemp(tmpl)) return 1;
    const std::string dir = tmpl;
    auto open = [&](const std::string& scope, uint64_t cap = PipelineArchive::kShardCap, unsigned every = 128) {
      PipelineArchive::Options o;
      o.dir = dir;
      o.scope = scope;
      o.shard_cap = cap;
      o.save_every = every;
      auto a = PipelineArchive::Open(dev, o);
      a->WaitLoaded();
      return a;
    };
    // Builds pipelines 0..n-1 through `a`: all made, and how many found.
    auto build = [&](PipelineArchive& a, int n, int first = 0) {
      a.TakeStats();
      bool made = true;
      for (int i = first; i < first + n; i++) {
        NSError* e = nil;
        made = made && a.Build(describe(i), uint64_t(i), &e) != nil;
      }
      Check(made, "every pipeline is made");
      return a.TakeStats();
    };
    const std::string a_scope = Scope("A1", "gpu0");

    // A first run: nothing to find; three pipelines added and saved.
    {
      auto a = open(a_scope);
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0 && s.misses == 0, "an empty folder: nothing looked in");
      a->Flush();
      s = a->TakeStats();
      Check(s.added == 3 && s.pending == 0 && s.save_bytes > 0, "three pipelines added and saved");
      a->Close(2.0);
      Check(Files(dir, ".bin").size() == 1 && Files(dir, ".scope").size() == 1, "one shard and its sidecar");
    }
    // A later run: all three found; a fourth added to the loaded shard.
    {
      auto a = open(a_scope);
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 1 && s.hits == 3 && s.misses == 0, "the three found in the shard of the run before");
      s = build(*a, 1, 3);
      Check(s.hits == 0 && s.misses == 1, "a new pipeline is not found");
      a->Close(2.0);
      Check(a->TakeStats().added == 1, "the new pipeline added at closing");
    }
    {
      auto a = open(a_scope);
      PipelineArchive::Stats s = build(*a, 4);
      Check(s.shards == 1 && s.hits == 4, "the earlier three and the one added to the loaded shard found");
      a->Close(2.0);
    }

    // A damaged shard: cut to half its size.
    {
      const std::string shard = dir + "/" + Files(dir, ".bin")[0];
      NSData* d = [NSData dataWithContentsOfFile:@(shard.c_str())];
      [[d subdataWithRange:NSMakeRange(0, d.length / 2)] writeToFile:@(shard.c_str()) atomically:NO];
      auto a = open(a_scope);
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0, "a cut shard does not load");
      a->Close(2.0);
      // Saved again from the pipelines of this run.
      auto b = open(a_scope);
      s = build(*b, 3);
      Check(s.shards == 1 && s.hits == 3, "the shard written again after the damaged one was removed");
      b->Close(2.0);
    }
    // Garbage in place of a shard.
    {
      const std::string shard = dir + "/" + Files(dir, ".bin")[0];
      std::string junk(4096, '\0');
      for (size_t i = 0; i < junk.size(); i++) junk[i] = char((i * 2654435761u) >> 13);
      [[NSData dataWithBytes:junk.data() length:junk.size()] writeToFile:@(shard.c_str()) atomically:NO];
      auto a = open(a_scope);
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0, "garbage does not load, pipelines still made");
      a->Close(2.0);
    }

    // Another OS build: nothing found, its own files; the earlier build's
    // removed. Another GPU's are left alone.
    {
      const std::vector<std::string> before = Files(dir, ".bin");
      auto a = open(Scope("B2", "gpu0"));
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0, "another OS build finds nothing");
      a->Close(2.0);
      const std::vector<std::string> after = Files(dir, ".bin");
      Check(after.size() == 1 && after[0] != before[0], "another OS build: its own shard, the earlier one removed");
      auto c = open(Scope("B2", "gpu1"));
      build(*c, 1);
      c->Close(2.0);
      Check(Files(dir, ".bin").size() == 2 && Files(dir, ".scope").size() == 2, "another GPU's files left alone");
      auto b = open(Scope("B2", "gpu0"));
      s = build(*b, 3);
      Check(s.shards == 1 && s.hits == 3, "the OS build's files still found after another GPU's run");
      b->Close(2.0);
    }

    // The shard cap: each save past it, the next pipeline in a new shard;
    // a later run looks in all of them.
    {
      const std::string scope = Scope("C3", "gpu0");
      auto a = open(scope, 1, 1);
      build(*a, 3);
      a->Close(2.0);
      auto b = open(scope, 1, 1);
      PipelineArchive::Stats s = build(*b, 3);
      Check(s.shards == 3 && s.hits == 3, "three shards past the cap, all looked in");
      b->Close(2.0);
    }

    // Outdated shards: every lookup a miss (pipelines never added), removed.
    {
      const std::string scope = Scope("D4", "gpu0");
      auto a = open(scope);
      build(*a, 3);
      a->Close(2.0);
      const size_t files = Files(dir, ".bin").size();
      auto b = open(scope);
      const MTLPixelFormat fmts[] = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float, MTLPixelFormatRG16Float,
                                     MTLPixelFormatRGBA32Float, MTLPixelFormatR16Float};
      uint64_t misses = 0, hits = 0;
      for (int i = 0; i < int(PipelineArchive::kStaleMisses); i++) {
        NSError* e = nil;
        Check(b->Build(describe(i % 4, fmts[(i / 4) % 5], MTLColorWriteMask(1 + (i / 20) % 15)),
                       uint64_t(1000 + i), &e) != nil,
              "a pipeline missed is made");
        const PipelineArchive::Stats s = b->TakeStats();
        misses += s.misses;
        hits += s.hits;
      }
      b->Flush();
      Check(hits == 0 && misses == PipelineArchive::kStaleMisses, "pipelines never added are all missed");
      Check(b->TakeStats().shards == 0, "outdated shards no longer looked in");
      b->Close(2.0);
      Check(Files(dir, ".bin").size() == files, "outdated shard removed, this run's saved in its place");
      auto c = open(scope);
      const PipelineArchive::Stats s = build(*c, 3);
      Check(s.hits == 0, "the outdated pipelines gone");
      c->Close(2.0);
    }

    [NSFileManager.defaultManager removeItemAtPath:@(dir.c_str()) error:nil];
  }
  if (failures) return 1;
  std::printf("metal_archive_test: ok\n");
  return 0;
}
