// OnceJobs (runtime/gfx/shader_jobs.h): a key's job runs once, whoever
// asks first runs it, those who ask while it runs wait for it, a value
// asked for after it ran comes at once, lanes run jobs ahead without
// running any twice, and after Close queued lane work does nothing.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "runtime/gfx/shader_jobs.h"

namespace {

using recomp::gfx::OnceJobs;
using Jobs = OnceJobs<int, std::string>;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    g_failures++;
  }
}

// A lane whose tasks wait until the test runs them.
struct ManualLane {
  std::vector<std::function<void()>> tasks;
  Jobs::Lane Get() {
    return [this](std::function<void()> t) { tasks.push_back(std::move(t)); };
  }
  void RunAll() {
    auto t = std::move(tasks);
    tasks.clear();
    for (auto& f : t) f();
  }
};

// A lane of its own thread, running tasks in order (a serial queue).
struct ThreadLane {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::function<void()>> q;
  bool stop = false;
  std::thread th{[this] {
    for (;;) {
      std::function<void()> t;
      {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [this] { return stop || !q.empty(); });
        if (q.empty()) return;
        t = std::move(q.front());
        q.pop_front();
      }
      t();
    }
  }};
  Jobs::Lane Get() {
    return [this](std::function<void()> t) {
      {
        std::lock_guard<std::mutex> l(mu);
        q.push_back(std::move(t));
      }
      cv.notify_one();
    };
  }
  // Waits for every task queued so far (the device's drain).
  void Drain() {
    {
      std::lock_guard<std::mutex> l(mu);
      stop = true;
    }
    cv.notify_one();
    th.join();
  }
};

// Many threads asking for one key at once: one run, one value for all.
void SameKeyAtOnce() {
  Jobs jobs;
  std::atomic<int> runs{0};
  jobs.Add(1, [&] {
    runs++;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return std::string("one");
  });
  constexpr int kThreads = 16;
  std::atomic<int> ready{0};
  std::vector<std::string*> got(kThreads);
  std::vector<Jobs::How> how(kThreads);
  std::vector<std::thread> ts;
  for (int i = 0; i < kThreads; i++)
    ts.emplace_back([&, i] {
      ready++;
      while (ready.load() < kThreads) std::this_thread::yield();
      got[i] = jobs.Get(1, &how[i]);
    });
  for (auto& t : ts) t.join();
  Check(runs == 1, "one key asked for by 16 threads at once runs once");
  int ran = 0;
  for (int i = 0; i < kThreads; i++) {
    Check(got[i] && *got[i] == "one", "every thread gets the value");
    Check(got[i] == got[0], "every thread gets the same value");
    if (how[i] == Jobs::How::RanHere) ran++;
  }
  Check(ran == 1, "one thread ran it, the others waited or found it");
  Check(jobs.Pending() == 0, "nothing pending once it ran");
}

// Asked for again after it ran: at once, not run again.
void AfterDone() {
  Jobs jobs;
  int runs = 0;
  jobs.Add(7, [&] {
    runs++;
    return std::string("seven");
  });
  Check(jobs.Pending() == 1, "a registered job is pending");
  Jobs::How how;
  std::string* v = jobs.Get(7, &how);
  Check(v && *v == "seven" && how == Jobs::How::RanHere, "the first ask runs it");
  const auto t0 = std::chrono::steady_clock::now();
  v = jobs.Get(7, &how);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Check(v && *v == "seven" && how == Jobs::How::Ready, "the second finds it ready");
  Check(ms < 5, "and at once");
  Check(runs == 1, "run once");
  Check(!jobs.Add(7, [] { return std::string("again"); }), "a key's job is added once");
  Check(jobs.Get(8) == nullptr, "a key without a job has no value");
  jobs.Remove(7);
  Check(!jobs.Contains(7) && jobs.Get(7) == nullptr, "removed, the key is gone");
}

// Queued on a lane, asked for before the lane gets to it: run by the one
// who asks; the lane's task then does nothing.
void AskedBeforeTheLane() {
  Jobs jobs;
  ManualLane lane;
  int runs = 0;
  jobs.Add(1, [&] {
    runs++;
    return std::string("a");
  });
  jobs.Prefetch(1, lane.Get());
  jobs.Prefetch(1, lane.Get());
  Check(lane.tasks.size() == 1, "queued on a lane once");
  Jobs::How how;
  std::string* v = jobs.Get(1, &how);
  Check(v && *v == "a" && how == Jobs::How::RanHere, "asked for while queued: run here");
  lane.RunAll();
  Check(runs == 1, "the lane's task does not run it again");
  Check(jobs.TakeFinished().empty(), "nor reports it finished");
}

// Run on a lane while another thread asks: that one waits for it.
void JoinedOnALane() {
  Jobs jobs;
  ThreadLane lane;
  std::atomic<int> runs{0};
  std::atomic<bool> started{false};
  jobs.Add(3, [&] {
    runs++;
    started = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return std::string("three");
  });
  jobs.Add(4, [&] {
    runs++;
    return std::string("four");
  });
  jobs.Prefetch(3, lane.Get());
  while (!started) std::this_thread::yield();
  Jobs::How how;
  std::string* v = jobs.Get(3, &how);
  Check(v && *v == "three", "the value of a job a lane ran");
  Check(how == Jobs::How::Joined, "asked for while a lane runs it: waited for");
  jobs.Prefetch(4, lane.Get());
  lane.Drain();
  Check(runs == 2, "each run once");
  std::vector<int> done = jobs.TakeFinished();
  Check(done.size() == 2, "both reported finished by the lane");
  v = jobs.Get(4, &how);
  Check(v && *v == "four" && how == Jobs::How::Ready, "a job a lane finished is ready");
  Check(jobs.TakeFinished().empty(), "reported once");
}

// Removed while a lane runs it: the run finishes, its value is dropped.
void RemovedWhileRunning() {
  Jobs jobs;
  ThreadLane lane;
  std::atomic<bool> started{false}, go{false};
  jobs.Add(5, [&] {
    started = true;
    while (!go) std::this_thread::yield();
    return std::string("five");
  });
  jobs.Prefetch(5, lane.Get());
  while (!started) std::this_thread::yield();
  jobs.Remove(5);
  go = true;
  lane.Drain();
  Check(!jobs.Contains(5), "removed stays removed");
  Check(jobs.TakeFinished().empty(), "a removed job is not reported finished");
  Check(jobs.Pending() == 0, "nor pending");
}

// Closing with work queued: the lane's tasks do nothing, asking still
// works.
void CloseWithQueuedWork() {
  Jobs jobs;
  ManualLane lane;
  std::atomic<int> runs{0};
  for (int k = 0; k < 10; k++)
    jobs.Add(k, [&, k] {
      runs++;
      return std::to_string(k);
    });
  for (int k = 0; k < 10; k++) jobs.Prefetch(k, lane.Get());
  jobs.Close();
  lane.RunAll();
  Check(runs == 0, "queued lane work does nothing after Close");
  Check(jobs.Pending() == 10, "so all of it is still pending");
  jobs.Prefetch(0, lane.Get());
  Check(lane.tasks.empty(), "nothing queued after Close");
  std::string* v = jobs.Get(9);
  Check(v && *v == "9" && runs == 1, "a job asked for after Close runs here");
}

// Many keys on several lanes while one thread asks for them in order:
// every job runs exactly once.
void LanesAndAsker() {
  Jobs jobs;
  constexpr int kKeys = 400;
  std::vector<std::atomic<int>> runs(kKeys);
  for (int k = 0; k < kKeys; k++)
    jobs.Add(k, [&, k] {
      runs[k]++;
      return std::to_string(k * 3);
    });
  ThreadLane lanes[3];
  for (int k = 0; k < kKeys; k++) jobs.Prefetch(k, lanes[k % 3].Get());
  bool all = true;
  for (int k = kKeys - 1; k >= 0; k--) {
    std::string* v = jobs.Get(k);
    all = all && v && *v == std::to_string(k * 3);
    jobs.Remove(k);
  }
  for (ThreadLane& l : lanes) l.Drain();
  Check(all, "every value right while lanes run them");
  bool once = true;
  for (int k = 0; k < kKeys; k++) once = once && runs[k] == 1;
  Check(once, "every job run exactly once");
  Check(jobs.Pending() == 0, "nothing left");
}

}  // namespace

int main() {
  SameKeyAtOnce();
  AfterDone();
  AskedBeforeTheLane();
  JoinedOnALane();
  RemovedWhileRunning();
  CloseWithQueuedWork();
  LanesAndAsker();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("shader_jobs: all passed\n");
  return 0;
}
