// Pipelines kept compiled between runs on the Metal backend
// (gfx/metal_archive.h), in a folder of its own, each "run" an archive
// opened on the folder (as by a later process):
// - pipelines saved by one run are found by the next; a run's new ones go
//   to a shard of its own, saved again whole as it grows (Metal fails a
//   second save of an archive added to since: with 80 pipelines saved
//   every 16 here, as with hundreds of real ones);
// - a damaged shard does not load, is removed, pipelines are still built;
// - another scope (another OS build) finds nothing, starts its own files
//   and removes the earlier build's, but not another GPU's;
// - the shard cap starts new shards, all looked in; past max_shards the
//   oldest go;
// - a save that fails leaves the files as they were and is tried again;
//   two in a row stop saving;
// - shards are outdated only when their own pipelines are missed and none
//   found, not for misses of pipelines they never had;
// - lookups while saving, from several threads; Close's time cap;
//   pipelines built before the shards loaded and found in them are not
//   saved again (no file grows without new pipelines);
// - a second process with the same scope only looks things up.
//
//   metal_archive_test        (skipped, exit 77, without a Metal device)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
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

constexpr int kFns = 80;  // fragment functions, one pipeline each

std::string Msl() {
  std::string s = R"(
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V v(uint id [[vertex_id]]) {
  V o;
  o.pos = float4(float(id & 1), float(id >> 1), 0.0, 1.0);
  return o;
}
)";
  for (int i = 0; i < kFns; i++)
    s += "fragment float4 f" + std::to_string(i) + "(V in [[stage_in]]) { return sin(in.pos.xyxy * " +
         std::to_string(i + 1) + ".0) * " + std::to_string(0.5 + i * 0.01) + "; }\n";
  return s;
}

std::string Scope(const char* os, const char* gpu) {
  return std::string("format=2\napp=metal_archive_test\nos=") + os + "\ngpu=" + gpu + "\n";
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

uint64_t FileBytes(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0 ? uint64_t(st.st_size) : 0;
}

double Seconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(Msl().c_str()) options:nil error:&err];
    if (!lib) {
      std::fprintf(stderr, "metal_archive_test: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> vs = [lib newFunctionWithName:@"v"];
    std::vector<id<MTLFunction>> fs;
    for (int i = 0; i < kFns; i++) fs.push_back([lib newFunctionWithName:@(("f" + std::to_string(i)).c_str())]);
    // Pipeline i: fragment function i % kFns, colour format by i / kFns.
    auto describe = [&](int i) {
      static const MTLPixelFormat fmts[] = {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float,
                                            MTLPixelFormatRGBA8Unorm};
      MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
      pd.vertexFunction = vs;
      pd.fragmentFunction = fs[size_t(i % kFns)];
      pd.colorAttachments[0].pixelFormat = fmts[(i / kFns) % 3];
      return pd;
    };

    char tmpl[1024];
    std::snprintf(tmpl, sizeof tmpl, "%s/metal_archive_test.XXXXXX", NSTemporaryDirectory().UTF8String);
    if (!mkdtemp(tmpl)) return 1;
    const std::string dir = tmpl;
    auto options = [&](const std::string& scope) {
      PipelineArchive::Options o;
      o.dir = dir;
      o.scope = scope;
      return o;
    };
    auto open = [&](PipelineArchive::Options o) {
      auto a = PipelineArchive::Open(dev, o);
      a->WaitLoaded();
      return a;
    };
    // Builds pipelines [first, first + n) (key i for pipeline i): all made;
    // the lookups' counts.
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

    // ---- runs, a shard each, and the save Metal fails when done twice
    // to one archive.
    const std::string a_scope = Scope("A1", "gpu0");
    {
      auto a = open(options(a_scope));
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0 && s.misses == 0, "an empty folder: nothing looked in");
      a->Flush();
      s = a->TakeStats();
      Check(s.saved == 3 && s.pending == 0 && s.save_bytes > 0, "three pipelines saved");
      a->Close(2.0);
      Check(Files(dir, ".bin").size() == 1 && Files(dir, ".keys").size() == 1 && Files(dir, ".scope").size() == 1,
            "a shard, its keys and the scope's sidecar");
    }
    {
      auto a = open(options(a_scope));
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 1 && s.hits == 3 && s.misses == 0, "the three found in the shard of the run before");
      s = build(*a, 1, 3);
      Check(s.hits == 0 && s.misses == 1, "a new pipeline is not found");
      a->Flush();
      // A lookup in the loaded shard after this run's save.
      bool found = false;
      NSError* e = nil;
      Check(a->Build(describe(0), 0, &e, &found) != nil && found, "the loaded shard looked in after a save");
      a->Close(2.0);
      Check(a->TakeStats().saved == 1, "the new pipeline saved");
      Check(Files(dir, ".bin").size() == 2, "the run's new pipeline in a shard of its own");
    }
    {
      // 80 new pipelines saved every 16, then all looked up in a later run
      // (a second save of one archive added to since fails in Metal).
      PipelineArchive::Options o = options(a_scope);
      o.save_every = 16;
      auto a = open(o);
      a->TakeStats();
      for (int i = 0; i < 80; i++) {
        NSError* e = nil;
        a->Build(describe(kFns + i), uint64_t(kFns + i), &e);
        if (i % 16 == 15) a->Flush();
      }
      PipelineArchive::Stats s = a->TakeStats();
      Check(s.saved == 80 && s.pending == 0, "80 pipelines saved in five saves of the run's shard");
      a->Close(2.0);
      auto b = open(options(a_scope));
      s = build(*b, 4);
      PipelineArchive::Stats t = build(*b, 80, kFns);
      Check(s.hits == 4 && t.hits == 80 && b->TakeStats().shards == 3, "all 84 found in three shards");
      b->Close(2.0);
    }

    // ---- damaged shards.
    const std::string d_scope = Scope("D1", "gpu0");
    {
      auto a = open(options(d_scope));
      build(*a, 3);
      a->Close(2.0);
      const std::string shard = a->ShardPath(0);
      NSData* d = [NSData dataWithContentsOfFile:@(shard.c_str())];
      [[d subdataWithRange:NSMakeRange(0, d.length / 2)] writeToFile:@(shard.c_str()) atomically:NO];
      auto b = open(options(d_scope));
      PipelineArchive::Stats s = build(*b, 3);
      Check(s.shards == 0 && s.hits == 0 && FileBytes(shard) == 0, "a cut shard does not load and is removed");
      b->Close(2.0);
      auto c = open(options(d_scope));
      s = build(*c, 3);
      Check(s.shards == 1 && s.hits == 3, "the pipelines saved again after the damaged shard went");
      c->Close(2.0);
      const std::string again = c->ShardPath(1);
      std::string junk(4096, '\0');
      for (size_t i = 0; i < junk.size(); i++) junk[i] = char((i * 2654435761u) >> 13);
      [[NSData dataWithBytes:junk.data() length:junk.size()] writeToFile:@(again.c_str()) atomically:NO];
      auto g = open(options(d_scope));
      s = build(*g, 3);
      Check(s.shards == 0 && s.hits == 0, "garbage does not load, pipelines still made");
      g->Close(2.0);
    }

    // ---- scopes: another OS build removes the earlier one's files, not
    // another GPU's.
    {
      auto a = open(options(Scope("B2", "gpu0")));
      PipelineArchive::Stats s = build(*a, 3);
      Check(s.shards == 0 && s.hits == 0, "another OS build finds nothing");
      a->Close(2.0);
      Check(Files(dir, ".scope").size() == 1 && Files(dir, ".bin").size() == 1,
            "another OS build: its own files, the earlier builds' removed");
      auto c = open(options(Scope("B2", "gpu1")));
      build(*c, 1);
      c->Close(2.0);
      Check(Files(dir, ".bin").size() == 2 && Files(dir, ".scope").size() == 2, "another GPU's files left alone");
      auto b = open(options(Scope("B2", "gpu0")));
      s = build(*b, 3);
      Check(s.shards == 1 && s.hits == 3, "the OS build's files still found after another GPU's run");
      b->Close(2.0);
    }

    // ---- the shard cap and max_shards.
    {
      PipelineArchive::Options o = options(Scope("C3", "gpu0"));
      o.shard_cap = 1;
      o.save_every = 1;
      auto a = open(o);
      for (int i = 0; i < 3; i++) {
        build(*a, 1, i);
        a->Flush();
      }
      a->Close(2.0);
      auto b = open(o);
      PipelineArchive::Stats s = build(*b, 3);
      Check(s.shards == 3 && s.hits == 3, "three shards past the cap, all looked in");
      b->Close(2.0);
      o.max_shards = 2;
      auto c = open(o);  // room for one more: the two oldest go
      s = build(*c, 3);
      Check(s.shards == 1 && s.hits == 1, "past max_shards the oldest shards are removed");
      c->Close(2.0);
    }

    // ---- saves that fail.
    {
      const std::string scope = Scope("F4", "gpu0");
      auto a = open(options(scope));
      // The folder made read-only: no temporary file can be made (Metal
      // replaces a file or folder in its way).
      auto block = [&] { chmod(dir.c_str(), 0555); };
      auto unblock = [&] { chmod(dir.c_str(), 0755); };
      block();
      build(*a, 3);
      a->Flush();
      PipelineArchive::Stats s = a->TakeStats();
      Check(s.save_ms < 0 && s.pending == 3 && FileBytes(a->ShardPath(0)) == 0, "a failed save writes nothing");
      unblock();
      a->Flush();
      s = a->TakeStats();
      Check(s.saved == 3 && s.pending == 0 && FileBytes(a->ShardPath(0)) > 0, "the next save tries again");
      const uint64_t good = FileBytes(a->ShardPath(0));
      block();
      build(*a, 1, 3);
      a->Flush();  // fails
      a->Flush();  // tried again: the second failure in a row
      build(*a, 1, 4);
      a->Flush();  // not tried any more
      unblock();
      build(*a, 1, 5);
      a->Flush();
      s = a->TakeStats();
      Check(s.saved == 3 && s.pending == 3 && FileBytes(a->ShardPath(0)) == good,
            "two failures in a row stop saving, the shard saved before kept");
      a->Close(2.0);
      a.reset();  // the archives whose save failed are never released: no crash here
      auto b = open(options(scope));
      s = build(*b, 6);
      Check(s.hits == 3 && s.misses == 3, "the pipelines saved before the failures found");
      b->Close(2.0);
    }

    // ---- outdated shards: only their own pipelines' misses count. Each
    // case its own GPU (a scope of its own, left alone by the others): a
    // shard of pipelines 0..19 (20 keys listed: 10 misses make it stale).
    {
      auto shard_of_20 = [&](const char* gpu) {
        auto a = open(options(Scope("S5", gpu)));
        build(*a, 20);
        a->Close(2.0);
        return a->ShardPath(0);
      };
      // Listed keys, other pipelines (never saved under them): misses.
      auto listed_misses = [&](PipelineArchive& a, int n, int first_key) {
        for (int i = 0; i < n; i++) {
          NSError* e = nil;
          a.Build(describe(kFns + 40 + i), uint64_t(first_key + i), &e);
        }
      };
      shard_of_20("s1");
      auto b = open(options(Scope("S5", "s1")));
      PipelineArchive::Stats s = build(*b, 70, 2 * kFns);
      b->Flush();
      Check(s.misses == 70 && b->TakeStats().shards == 1, "misses of pipelines never saved do not make a shard stale");
      b->Close(2.0);

      shard_of_20("s2");
      auto c = open(options(Scope("S5", "s2")));
      build(*c, 1);
      c->TakeStats();
      listed_misses(*c, 19, 1);
      c->Flush();
      s = c->TakeStats();
      Check(s.misses == 19 && s.shards == 1, "with one found, listed misses keep the shard");
      c->Close(2.0);

      const std::string old = shard_of_20("s3");
      auto d = open(options(Scope("S5", "s3")));
      d->TakeStats();
      listed_misses(*d, 12, 0);
      d->Flush();
      Check(d->TakeStats().shards == 0, "listed misses and nothing found: no longer looked in");
      d->Close(2.0);
      Check(FileBytes(old) == 0 && FileBytes(d->ShardPath(1)) > 0, "the outdated shard removed, the run's own saved");
    }

    // ---- lookups from several threads while saving.
    {
      const std::string scope = Scope("T6", "gpu0");
      auto a = open(options(scope));
      build(*a, 10);
      a->Close(2.0);
      PipelineArchive::Options o = options(scope);
      o.save_every = 4;
      auto b = open(o);
      std::vector<std::thread> threads;
      std::atomic<int> made{0}, found{0};
      for (int t = 0; t < 4; t++)
        threads.emplace_back([&, t] {
          @autoreleasepool {
            for (int i = 0; i < 30; i++) {
              const int p = i < 10 ? i : kFns + t * 20 + i;
              NSError* e = nil;
              bool f = false;
              if (b->Build(describe(p), uint64_t(p), &e, &f)) made++;
              if (f) found++;
            }
          }
        });
      for (int i = 0; i < 20; i++) b->Flush();
      for (auto& t : threads) t.join();
      b->Close(2.0);
      Check(made == 120 && found == 40, "every lookup made, the loaded ones found, while saving");
      auto c = open(options(scope));
      PipelineArchive::Stats s = build(*c, 80, kFns + 10);
      Check(s.hits == 80, "the pipelines new on four threads all saved");
      c->Close(2.0);
    }

    // ---- Close: its time cap; nothing kept after it began.
    {
      auto a = open(options(Scope("K7", "gpu0")));
      build(*a, 2);
      dispatch_async(a->QueueForTests(), ^{
        usleep(1500 * 1000);
      });
      const auto t0 = std::chrono::steady_clock::now();
      a->Close(0.2);
      Check(Seconds(t0) < 1.0, "Close returns at its cap");
      build(*a, 1, 2);
      Check(a->TakeStats().pending <= 2, "nothing kept after closing began");
      dispatch_sync(a->QueueForTests(), ^{
      });
      Check(a->TakeStats().saved == 2 && FileBytes(a->ShardPath(0)) > 0, "the last save finished after the cap");
    }

    // ---- built before the shards loaded and in them: not saved again;
    // a run that only finds what it looks for writes nothing.
    {
      const std::string scope = Scope("Q8", "gpu0");
      auto a = open(options(scope));
      build(*a, 5);
      a->Close(2.0);
      const uint64_t size = FileBytes(a->ShardPath(0));
      dispatch_queue_t gate = dispatch_queue_create("test.gate", DISPATCH_QUEUE_SERIAL);
      dispatch_suspend(gate);  // loading waits
      PipelineArchive::Options o = options(scope);
      o.target = gate;
      auto b = PipelineArchive::Open(dev, o);
      PipelineArchive::Stats s = build(*b, 5);
      Check(s.shards == 0 && s.misses == 0, "built before loading: nothing looked in");
      dispatch_resume(gate);
      b->WaitLoaded();
      b->Close(2.0);
      Check(Files(dir, ".bin").size() >= 1 && FileBytes(b->ShardPath(1)) == 0 && FileBytes(b->ShardPath(0)) == size,
            "pipelines the loaded shard has not saved again");
      auto c = open(options(scope));
      s = build(*c, 5);
      c->Close(2.0);
      Check(s.hits == 5 && FileBytes(c->ShardPath(1)) == 0 && FileBytes(c->ShardPath(0)) == size,
            "a run with only hits writes nothing");
    }

    // ---- two processes, one scope: the second only looks.
    {
      const std::string scope = Scope("P9", "gpu0");
      auto a = open(options(scope));
      auto b = open(options(scope));
      Check(a->Saves() && !b->Saves(), "the scope's second user does not save");
      build(*b, 2);
      Check(b->TakeStats().pending == 0, "nothing kept by the second");
      build(*a, 2);
      a->Close(2.0);
      b->Close(2.0);
      auto c = open(options(scope));
      Check(c->Saves(), "the scope free again once the first closed");
      Check(build(*c, 2).hits == 2, "the first's pipelines saved");
      c->Close(2.0);
    }

    [NSFileManager.defaultManager removeItemAtPath:@(dir.c_str()) error:nil];
  }
  if (failures) return 1;
  std::printf("metal_archive_test: ok\n");
  return 0;
}
