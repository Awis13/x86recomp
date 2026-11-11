// OnceJobs (runtime/gfx/shader_jobs.h): a key's job runs once, whoever
// asks first runs it, those who ask while it runs wait for it, a value
// asked for after it ran comes at once, lanes run jobs ahead without
// running any twice, a value outlives its key's removal, and after Close
// queued lane work does nothing. Waiters: the ids sharing a key's job.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <memory>
#include <thread>
#include <vector>

#include "runtime/gfx/shader_jobs.h"

namespace {

using recomp::gfx::OnceJobs;
using recomp::gfx::Waiters;
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
  jobs.Add(1, [&](bool) {
    runs++;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return std::string("one");
  });
  constexpr int kThreads = 16;
  std::atomic<int> ready{0};
  std::vector<std::shared_ptr<std::string>> got(kThreads);
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
  bool on_lane = true;
  jobs.Add(7, [&](bool l) {
    runs++;
    on_lane = l;
    return std::string("seven");
  });
  Check(jobs.Pending() == 1, "a registered job is pending");
  Jobs::How how;
  std::shared_ptr<std::string> v = jobs.Get(7, &how);
  Check(v && *v == "seven" && how == Jobs::How::RanHere, "the first ask runs it");
  Check(!on_lane, "told it runs for one who asked");
  const auto t0 = std::chrono::steady_clock::now();
  v = jobs.Get(7, &how);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Check(v && *v == "seven" && how == Jobs::How::Ready, "the second finds it ready");
  Check(ms < 5, "and at once");
  Check(runs == 1, "run once");
  Check(!jobs.Add(7, [](bool) { return std::string("again"); }), "a key's job is added once");
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
  jobs.Add(1, [&](bool) {
    runs++;
    return std::string("a");
  });
  jobs.Prefetch(1, lane.Get());
  jobs.Prefetch(1, lane.Get());
  Check(lane.tasks.size() == 1, "queued on a lane once");
  Jobs::How how;
  std::shared_ptr<std::string> v = jobs.Get(1, &how);
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
  jobs.Add(3, [&](bool) {
    runs++;
    started = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return std::string("three");
  });
  std::atomic<bool> four_on_lane{false};
  jobs.Add(4, [&](bool on_lane) {
    runs++;
    four_on_lane = on_lane;
    return std::string("four");
  });
  jobs.Prefetch(3, lane.Get());
  while (!started) std::this_thread::yield();
  Jobs::How how;
  std::shared_ptr<std::string> v = jobs.Get(3, &how);
  Check(v && *v == "three", "the value of a job a lane ran");
  Check(how == Jobs::How::Joined, "asked for while a lane runs it: waited for");
  jobs.Prefetch(4, lane.Get());
  lane.Drain();
  Check(runs == 2, "each run once");
  Check(four_on_lane, "a job a lane runs is told so");
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
  jobs.Add(5, [&](bool) {
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

// A value asked for while a lane ran it, its key removed before the lane
// was done: the value is still there for the one who waited. (Tried a few
// times: on a busy host the asker may not be waiting yet when the key goes.)
bool RemovedWhileWaitedOnce() {
  Jobs jobs;
  ThreadLane lane;
  std::atomic<bool> started{false}, go{false};
  jobs.Add(6, [&](bool) {
    started = true;
    while (!go) std::this_thread::yield();
    return std::string(4096, 'x');
  });
  jobs.Prefetch(6, lane.Get());
  while (!started) std::this_thread::yield();
  std::shared_ptr<std::string> got;
  Jobs::How how{};
  std::atomic<bool> asking{false};
  std::thread asker([&] {
    asking = true;
    got = jobs.Get(6, &how);
  });
  while (!asking) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // (Get waiting on the lane)
  jobs.Remove(6);
  go = true;
  asker.join();
  lane.Drain();
  Check(!jobs.Contains(6) && jobs.Pending() == 0, "a removed key stays removed");
  if (!got) return false;  // asked too late: not the case under test
  Check(got->size() == 4096 && (*got)[4095] == 'x', "the value outlives its key's removal");
  Check(how == Jobs::How::Joined, "waited for on the lane");
  return true;
}

void RemovedWhileWaited() {
  bool seen = false;
  for (int i = 0; i < 5 && !seen; i++) seen = RemovedWhileWaitedOnce();
  Check(seen, "a removal while one waited for the value happened");
}

// Shaders of one translation: one job for all, dropped with the last.
void SharedByWaiters() {
  Waiters<int, int> w;
  Check(w.Add(100, 1), "the first of a key makes its job");
  Check(!w.Add(100, 2), "the second waits for it");
  Check(w.Add(200, 3), "another key, another job");
  Check(!w.Add(100, 4), "a third waits too");
  Check(!w.Remove(100, 2), "one of three gone: the job stays");
  Check(!w.Remove(100, 9), "an id not waiting changes nothing");
  Check(w.Add(50, 5), "a key made after");
  const std::vector<int> order = w.KeysInOrder();
  Check(order == std::vector<int>({100, 200, 50}), "keys in the order their first id came");
  const std::vector<int> ids = w.Take(100);
  Check(ids == std::vector<int>({1, 4}), "a key's ids, in order, at its job's end");
  Check(!w.Has(100) && w.Take(100).empty(), "taken, the key is gone");
  Check(w.Remove(200, 3), "the last of a key gone: its job is dropped");
  Check(!w.Has(200), "and the key with it");
}

// Closing with work queued: the lane's tasks do nothing, asking still
// works.
void CloseWithQueuedWork() {
  Jobs jobs;
  ManualLane lane;
  std::atomic<int> runs{0};
  for (int k = 0; k < 10; k++)
    jobs.Add(k, [&, k](bool) {
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
  std::shared_ptr<std::string> v = jobs.Get(9);
  Check(v && *v == "9" && runs == 1, "a job asked for after Close runs here");
}

// Many keys on several lanes while one thread asks for them in order:
// every job runs exactly once.
void LanesAndAsker() {
  Jobs jobs;
  constexpr int kKeys = 400;
  std::vector<std::atomic<int>> runs(kKeys);
  for (int k = 0; k < kKeys; k++)
    jobs.Add(k, [&, k](bool) {
      runs[k]++;
      return std::to_string(k * 3);
    });
  ThreadLane lanes[3];
  for (int k = 0; k < kKeys; k++) jobs.Prefetch(k, lanes[k % 3].Get());
  bool all = true;
  for (int k = kKeys - 1; k >= 0; k--) {
    std::shared_ptr<std::string> v = jobs.Get(k);
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
  RemovedWhileWaited();
  CloseWithQueuedWork();
  LanesAndAsker();
  SharedByWaiters();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("shader_jobs: all passed\n");
  return 0;
}
