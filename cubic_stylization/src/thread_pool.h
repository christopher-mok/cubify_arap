// Minimal persistent thread pool with a dynamically scheduled parallel_for.
//
// std::thread only (no OpenMP), so the server has no runtime dependencies on
// any platform — Apple clang ships without OpenMP, and MSVC's vcomp140.dll is
// not guaranteed to be present next to Blender.

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cubify {

class ThreadPool {
 public:
  using RangeFn = std::function<void(int begin, int end)>;

  // `threads` is the total worker count including the calling thread;
  // <= 0 means one per hardware thread.
  explicit ThreadPool(int threads = 0) {
    if (threads <= 0) threads = hardware_threads();
    size_ = threads;
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return size_; }

  static int hardware_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
  }

  // Calls fn(begin, end) over [0, n) in chunks of `grain`, on all threads.
  // fn must not throw.
  void parallel_for(int n, int grain, const RangeFn& fn) {
    if (n <= 0) return;
    grain = std::max(grain, 1);
    if (workers_.empty() || n <= grain) {
      fn(0, n);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(m_);
      job_ = &fn;
      n_ = n;
      grain_ = grain;
      next_.store(0);
      active_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    run_chunks(fn, n, grain);
    std::unique_lock<std::mutex> lock(m_);
    done_cv_.wait(lock, [this] { return active_ == 0; });
    job_ = nullptr;
  }

 private:
  void run_chunks(const RangeFn& fn, int n, int grain) {
    for (;;) {
      int b = next_.fetch_add(grain);
      if (b >= n) break;
      fn(b, std::min(n, b + grain));
    }
  }

  void worker_loop() {
    uint64_t seen = 0;
    for (;;) {
      const RangeFn* job;
      int n, grain;
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job = job_;
        n = n_;
        grain = grain_;
      }
      run_chunks(*job, n, grain);
      {
        std::lock_guard<std::mutex> lock(m_);
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  int size_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  const RangeFn* job_ = nullptr;
  int n_ = 0, grain_ = 1;
  std::atomic<int> next_{0};
  int active_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace cubify
