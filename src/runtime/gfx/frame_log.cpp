// The frame log (frame_log.h).

#include "runtime/gfx/frame_log.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace recomp::gfx {
namespace {

// A few KB to tens of KB a second (a frame's "q" line alone is 1 to 3 KB):
// 1 MB holds many seconds of a disk that does not answer.
constexpr size_t kBuffer = 1 << 20;

// Lines are formatted into `buf` under `mu` (no I/O under it: its holders
// are the render, encoding and main threads). The flusher swaps `buf` for
// `spare` under `mu` and writes `spare` to the file under `io` alone, so a
// write that takes long (the file is on a USB disk) blocks no writer.
struct Log {
  std::FILE* f = nullptr;
  std::chrono::steady_clock::time_point t0;
  std::mutex mu;  // buf, used, dropped
  char* buf = nullptr;
  size_t used = 0;
  uint64_t dropped = 0;  // lines that found the buffer full since the last swap
  std::mutex io;  // spare and the file; taken before mu
  char* spare = nullptr;
};

// The open log, for FrameLogFlush: it must not open one at exit.
std::atomic<Log*> g_log{nullptr};

void Flush(Log* l) {
  std::lock_guard<std::mutex> io(l->io);
  size_t n = 0;
  {
    std::lock_guard<std::mutex> lk(l->mu);
    if (!l->used) return;
    std::swap(l->buf, l->spare);
    n = l->used;
    l->used = 0;
    if (l->dropped) {
      const int m = std::snprintf(l->buf, kBuffer, "# %llu lines dropped here: the buffer was full\n",
                                  (unsigned long long)l->dropped);
      if (m > 0) l->used = size_t(m);
      l->dropped = 0;
    }
  }
  std::fwrite(l->spare, 1, n, l->f);
  std::fflush(l->f);
}

Log* Open() {
  const char* path = std::getenv("RECOMP_FRAME_LOG");
  if (!path || !*path) return nullptr;
  std::FILE* f = std::fopen(path, "w");
  if (!f) {
    std::fprintf(stderr, "gfx: frame log: cannot write %s\n", path);
    return nullptr;
  }
  // (Never freed: the flusher and late writers outlive every destructor.)
  Log* l = new Log;
  l->f = f;
  l->buf = static_cast<char*>(std::malloc(kBuffer));
  l->spare = static_cast<char*>(std::malloc(kBuffer));
  if (!l->buf || !l->spare) {
    std::fprintf(stderr, "gfx: frame log: no memory for its buffers\n");
    std::free(l->buf);
    std::free(l->spare);
    delete l;
    std::fclose(f);
    return nullptr;
  }
  // Each flush is one write of the whole buffer: no second buffer in stdio.
  std::setvbuf(f, nullptr, _IONBF, 0);
  l->t0 = std::chrono::steady_clock::now();
  const double began =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  const int n = std::snprintf(
      l->buf, kBuffer,
      "# frame log v2\n"
      "# began at unix time %.3f; t_s: seconds since then (one steady clock for every line); a line's first "
      "token is its kind, kinds a reader does not know are to be skipped\n"
      "# p serial t_s interval_ms draws passes ops (a frame presented, the render thread)\n"
      "# g serial gpu_busy_ms gpu_start_s gpu_end_s (the GPU finished it, all its parts' busy time; the host's "
      "clock)\n"
      "# n pass_id name (a GPU pass name's id, before its first use)\n"
      "# q serial pass_id:ms ... (the frame's GPU time, every pass in the order it ran)\n"
      "# c serial slot_wait_ms record_ms rt_note_ms snapshot_ms end_ms (the render thread's CPU of the frame)\n"
      "# r serial tlas_instances blas_builds captured_instances lights_on_screen forward_ray_draws\n"
      "# s display_s | sg display_s (a rendered | generated frame shown; the host's clock, 0: not shown, dropped "
      "or the window hidden)\n"
      "# w t_s x y z heading in_car speed_mps cam_x cam_y cam_z cam_heading cam_pitch fov hour minute weather "
      "wanted interior\n"
      "# i t_s lx ly rx ry lt rt buttons_hex keys_hex\n"
      "# m t_s main_frame_ms wait_render_ms\n"
      "# e t_s text\n",
      began);
  l->used = n > 0 ? size_t(n) : 0;
  g_log.store(l, std::memory_order_release);
  std::atexit(FrameLogFlush);
  // To the file four times a second whether or not lines keep coming: a
  // process that is killed (the lab ends its runs with SIGTERM) or leaves
  // through _Exit runs no hook, and loses a quarter second at most.
  std::thread([l] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      Flush(l);
    }
  }).detach();
  return l;
}

Log* TheLog() {
  static Log* const log = Open();
  return log;
}

}  // namespace

bool FrameLogOn() { return TheLog() != nullptr; }

double FrameLogSeconds() {
  Log* l = TheLog();
  return l ? std::chrono::duration<double>(std::chrono::steady_clock::now() - l->t0).count() : 0.0;
}

void FrameLogLine(const char* fmt, ...) {
  Log* l = TheLog();
  if (!l) return;
  va_list ap;
  va_start(ap, fmt);
  {
    std::lock_guard<std::mutex> lk(l->mu);
    // Formatted where it stays: a line may be several KB (a frame's passes).
    char* at = l->buf + l->used;
    const size_t room = kBuffer - l->used;
    const int n = std::vsnprintf(at, room, fmt, ap);
    if (n >= 0 && size_t(n) < room) {
      // (A newline in its text would start a line of no kind.)
      for (char* p = at; (p = static_cast<char*>(std::memchr(p, '\n', size_t(at + n - p)))) != nullptr;) *p = ' ';
      at[n] = '\n';
      l->used += size_t(n) + 1;
    } else {
      l->dropped++;
    }
  }
  va_end(ap);
}

void FrameLogFlush() {
  if (Log* l = g_log.load(std::memory_order_acquire)) Flush(l);
}

}  // namespace recomp::gfx
