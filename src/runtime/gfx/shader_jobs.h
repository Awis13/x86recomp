#pragma once
// Work done once per key, by whichever thread needs it first: the thread
// that asks for a key's value runs its job if nobody has started it, waits
// for it if another thread is running it, and gets it at once if it has
// run. Background lanes may run jobs ahead (Prefetch); a job is never run
// twice, nor by two threads at once.
//
// For shaders compiled when first drawn (metal_device.mm): the game's
// thread asks for a shader mid-frame (Get), background lanes compile the
// others meanwhile; one job per translation, the shaders made from it its
// Waiters. The thread that waits for a job running on a lane
// lends that lane its priority: the runner holds the job's RunLock for as
// long as it runs, and the waiter blocks on it (on Apple's systems an
// os_unfair_lock, whose owner the kernel boosts; a lane at background
// priority would otherwise keep the game waiting behind everything else).
//
// Jobs must not throw. The lanes' tasks use the object: whoever owns it
// calls Close and then waits for every lane's tasks to finish (those
// still queued then do nothing) before destroying it.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <os/lock.h>
#endif

namespace recomp::gfx {

#if defined(__APPLE__)
class RunLock {
 public:
  void lock() { os_unfair_lock_lock(&l_); }
  void unlock() { os_unfair_lock_unlock(&l_); }

 private:
  os_unfair_lock l_ = OS_UNFAIR_LOCK_INIT;
};
#else
using RunLock = std::mutex;
#endif

template <class Key, class Value, class Hash = std::hash<Key>>
class OnceJobs {
 public:
  // Told whether a lane runs it (else a thread that asked for its value).
  using Job = std::function<Value(bool on_lane)>;
  // Runs a task later on some thread (a background lane).
  using Lane = std::function<void(std::function<void()>)>;

  // How Get came by the value.
  enum class How { Ready, Joined, RanHere };

  // `job` registered for `key`; false (and nothing changed) if the key
  // already has one.
  bool Add(const Key& key, Job job) {
    std::lock_guard<std::mutex> l(mu_);
    auto [it, fresh] = slots_.try_emplace(key);
    if (!fresh) return false;
    it->second = std::make_shared<Slot>();
    it->second->job = std::move(job);
    return true;
  }

  bool Contains(const Key& key) const {
    std::lock_guard<std::mutex> l(mu_);
    return slots_.count(key) != 0;
  }

  // Keys registered whose jobs have not finished.
  size_t Pending() const {
    std::lock_guard<std::mutex> l(mu_);
    return slots_.size() - done_;
  }

  // The value of key's job: at once if the job has run, once it has if it
  // is running on another thread, else run on this one. Null if the key
  // has no job. The value stays valid as long as it is held, the key
  // removed meanwhile or not (the caller may move from it).
  std::shared_ptr<Value> Get(const Key& key, How* how = nullptr) {
    std::unique_lock<std::mutex> l(mu_);
    auto it = slots_.find(key);
    if (it == slots_.end()) return nullptr;
    std::shared_ptr<Slot> s = it->second;
    How h = How::Ready;
    if (s->state == State::Running) {
      h = How::Joined;
      l.unlock();
      s->run.lock();  // until the runner is done with it
      s->run.unlock();
      l.lock();
    } else if (s->state != State::Done) {
      h = How::RanHere;
      RunLocked(s, l, false);
    }
    if (how) *how = h;
    return std::shared_ptr<Value>(s, &*s->value);  // (keeps the slot)
  }

  // Key's job queued on `lane`, unless it has been started or queued
  // already, or the key has none. Not after Close.
  void Prefetch(const Key& key, const Lane& lane) {
    std::shared_ptr<Slot> s;
    {
      std::lock_guard<std::mutex> l(mu_);
      if (closing_) return;
      auto it = slots_.find(key);
      if (it == slots_.end() || it->second->state != State::Idle) return;
      s = it->second;
      s->state = State::Queued;
    }
    lane([this, s, key] {
      std::unique_lock<std::mutex> l(mu_);
      // Asked for (run or running), removed, or closing: nothing to do.
      if (closing_ || s->state != State::Queued || s->forgotten) return;
      RunLocked(s, l, true);
      if (!s->forgotten) finished_.push_back(key);
    });
  }

  // The keys whose jobs lanes have finished since the last call (their
  // values wait for Get).
  std::vector<Key> TakeFinished() {
    std::lock_guard<std::mutex> l(mu_);
    std::vector<Key> keys;
    keys.swap(finished_);
    return keys;
  }

  // Key's job (or value) dropped: not run if it has not started; if a lane
  // is running it, its value is dropped when done.
  void Remove(const Key& key) {
    std::lock_guard<std::mutex> l(mu_);
    auto it = slots_.find(key);
    if (it != slots_.end()) Forget(key, it->second);
  }

  // From now on lanes' tasks do nothing; Get still runs jobs on the thread
  // that asks.
  void Close() {
    std::lock_guard<std::mutex> l(mu_);
    closing_ = true;
  }

 private:
  enum class State { Idle, Queued, Running, Done };
  struct Slot {
    State state = State::Idle;
    bool forgotten = false;
    Job job;
    std::optional<Value> value;
    RunLock run;  // held by the thread running the job
  };

  // The job of `s` run on this thread, `l` (mu_) released meanwhile and
  // held again when it returns.
  void RunLocked(const std::shared_ptr<Slot>& s, std::unique_lock<std::mutex>& l, bool on_lane) {
    s->state = State::Running;
    s->run.lock();  // free: only a waiter takes it, and only while Running
    Job job = std::move(s->job);
    l.unlock();
    Value v = job(on_lane);
    job = nullptr;  // what it captured, freed outside the lock
    l.lock();
    s->value = std::move(v);
    s->state = State::Done;
    if (!s->forgotten) done_++;
    s->run.unlock();  // Done first: a waiter then sees it
  }

  // Under mu_.
  void Forget(const Key& key, const std::shared_ptr<Slot>& s) {
    if (s->forgotten) return;
    s->forgotten = true;
    if (s->state == State::Done) done_--;
    slots_.erase(key);
    std::erase(finished_, key);
  }

  mutable std::mutex mu_;
  std::unordered_map<Key, std::shared_ptr<Slot>, Hash> slots_;
  std::vector<Key> finished_;
  size_t done_ = 0;  // slots_ whose jobs have finished
  bool closing_ = false;
};

// The ids waiting for each key's job: shaders made from one translation
// (the same source) share one compile. Not thread-safe (the device's under
// its lock).
template <class Key, class Id, class Hash = std::hash<Key>>
class Waiters {
 public:
  // `id` waits for key's job; true if it is the first (the job to make).
  bool Add(const Key& key, Id id) {
    std::vector<Id>& ids = by_key_[key];
    ids.push_back(id);
    return ids.size() == 1;
  }

  // `id` waits no more; true if it was the last (the job to drop).
  bool Remove(const Key& key, Id id) {
    auto it = by_key_.find(key);
    if (it == by_key_.end()) return false;
    auto at = std::find(it->second.begin(), it->second.end(), id);
    if (at == it->second.end()) return false;
    it->second.erase(at);
    if (!it->second.empty()) return false;
    by_key_.erase(it);
    return true;
  }

  bool Has(const Key& key) const { return by_key_.count(key) != 0; }

  // Key's ids, in the order they came, and the key forgotten.
  std::vector<Id> Take(const Key& key) {
    auto it = by_key_.find(key);
    if (it == by_key_.end()) return {};
    std::vector<Id> ids = std::move(it->second);
    by_key_.erase(it);
    return ids;
  }

  // The keys, in the order of the first id of each (ids grow as they are
  // made: the order the first shader of each was made in).
  std::vector<Key> KeysInOrder() const {
    std::vector<std::pair<Id, Key>> firsts;
    for (const auto& [key, ids] : by_key_) firsts.emplace_back(ids.front(), key);
    std::sort(firsts.begin(), firsts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<Key> keys;
    for (const auto& f : firsts) keys.push_back(f.second);
    return keys;
  }

 private:
  std::unordered_map<Key, std::vector<Id>, Hash> by_key_;
};

}  // namespace recomp::gfx
