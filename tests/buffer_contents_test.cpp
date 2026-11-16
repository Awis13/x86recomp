// Versions of Direct3D buffer contents (runtime/hle/buffer_contents.h):
// identities and versions never 0 and never repeated; a writing lock and
// each unlock after one (nested too) give new contents, a read-only one
// (no WriteLock) and its unlock none; untracked buffers stay at 0; locks
// in three frames running mark a buffer rewritten, for as long as the
// next frame, and a gap starts the count over.
//
//   buffer_contents_test

#include <cstdio>
#include <set>
#include <thread>
#include <vector>

#include "runtime/hle/buffer_contents.h"

namespace {

using recomp::d3d9::BufferContents;

int Check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;
  {
    BufferContents a, b;
    a.Track();
    b.Track();
    std::set<uint64_t> seen = {a.id, a.Version(), b.id, b.Version()};
    failures += Check(seen.size() == 4 && !seen.count(0), "identities and versions: never 0, all apart");
  }
  {
    BufferContents u;
    u.WriteLock(1);
    u.Unlock();
    failures += Check(u.id == 0 && u.Version() == 0 && !u.Rewritten(1), "an untracked buffer stays at 0");
  }
  {
    BufferContents c;
    c.Track();
    const uint64_t v0 = c.Version();
    c.WriteLock(1);
    const uint64_t v1 = c.Version();  // what a draw between lock and unlock sees
    c.Unlock();
    const uint64_t v2 = c.Version();
    failures += Check(v1 != v0 && v2 != v1 && v2 != v0, "a writing lock, then its unlock: new contents each");
    c.Unlock();  // after a read-only lock (no WriteLock)
    failures += Check(c.Version() == v2, "an unlock after a read-only lock: the same contents");
    c.WriteLock(1);
    c.WriteLock(1);
    const uint64_t n0 = c.Version();
    c.Unlock();
    const uint64_t n1 = c.Version();
    c.Unlock();
    const uint64_t n2 = c.Version();
    c.Unlock();
    failures += Check(n1 != n0 && n2 != n1 && c.Version() == n2, "nested writing locks: each unlock new contents, no more");
  }
  {
    BufferContents r;
    r.Track();
    for (uint64_t f : {10, 10, 11, 12}) {
      r.WriteLock(f);
      r.Unlock();
    }
    const bool on = r.Rewritten(12) && r.Rewritten(13) && !r.Rewritten(14);
    BufferContents gap;
    gap.Track();
    for (uint64_t f : {10, 11, 13, 14}) gap.WriteLock(f);
    BufferContents two;
    two.Track();
    for (uint64_t f : {10, 11}) two.WriteLock(f);
    failures += Check(on && !gap.Rewritten(14) && !two.Rewritten(11),
                      "locked in three frames running: rewritten until the frame after; a gap or two frames: not");
  }
  {
    // Versions from several threads at once stay apart.
    std::vector<uint64_t> got[4];
    std::vector<std::thread> ts;
    for (auto& g : got)
      ts.emplace_back([&g] {
        for (int i = 0; i < 10000; i++) g.push_back(recomp::d3d9::NewBufferVersion());
      });
    for (std::thread& t : ts) t.join();
    std::set<uint64_t> all;
    for (const auto& g : got) all.insert(g.begin(), g.end());
    failures += Check(all.size() == 40000, "versions made on several threads all apart");
  }
  return failures ? 1 : 0;
}
