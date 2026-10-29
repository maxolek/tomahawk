// threads.h
// Lazy SMP helper threads.
//
// A search runs on the calling (UCI) thread, the "main" thread, plus Threads - 1
// helpers. Every helper searches the same root position with its own board,
// move generator, NNUE accumulator stack and Searcher (killers, history), and
// shares the transposition table with the main thread; that shared table is how
// helper work reaches the main search. The main thread's result is the one
// reported, and when it finishes it stops the helpers. At the start of every
// search each thread moves itself to its CPUs (Affinity option, affinity.h).

#ifndef THREADS_H
#define THREADS_H

#include "searcher.h"
#include "affinity.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class HelperThread {
public:
    HelperThread(NNUE& nnue, TranspositionTable& tt);
    ~HelperThread();
    HelperThread(const HelperThread&) = delete;
    HelperThread& operator=(const HelperThread&) = delete;

    // Start searching `root` in the background on `cpus` (see affinity.h), or on
    // `fallback` if the OS refuses them; the helper must be idle.
    void start(const Board& root, const SearchLimits& limits, const SearchParams& params,
               const std::vector<int>& cpus, const std::vector<int>& fallback);
    // Block until the current search (if any) has returned.
    void wait();
    // Nodes of the last search; valid after wait().
    uint64_t nodes() const { return searcher->nodes; }
    // The OS refused the placement of the last search; valid after wait().
    bool placement_failed() const { return pin_failed; }

private:
    void loop();
    void run_search();

    Board board;
    MoveGenerator movegen;
    AccumulatorStack acc;
    std::unique_ptr<Searcher> searcher;
    SearchLimits limits;
    std::vector<int> cpus;          // placement for the next search, applied by the helper itself
    std::vector<int> fallback_cpus; // used if the OS refuses `cpus`
    bool pin_failed = false;

    std::mutex mutex;
    std::condition_variable cv;
    bool searching = false;
    bool exiting = false;
    std::thread thread; // started last, after everything it uses exists
};

class SearchThreadPool {
public:
    // Shared by every thread of the running search (limits.stop_signal).
    std::atomic<bool> stop_signal{false};

    ~SearchThreadPool() { stop_and_wait(); }

    // Keep `count` helper threads (Threads - 1). Not while a search is running.
    void set_helpers(int count, NNUE& nnue, TranspositionTable& tt);
    int helper_count() const { return static_cast<int>(helpers.size()); }

    // Starts every helper on `root`; the caller clears the stop signal before
    // launching a search. Helper i runs on placement.threads[i + 1].
    void start(const Board& root, const SearchLimits& limits, const SearchParams& params,
               const affinity::Plan& placement);
    // Raises the stop signal and waits for every helper to return.
    void stop_and_wait();
    // Sum of the helpers' node counts for the last search.
    uint64_t helper_nodes() const;
    // Helpers whose placement the OS refused in the last search (after stop_and_wait()).
    int placement_failures() const;

private:
    std::vector<std::unique_ptr<HelperThread>> helpers;
};

#endif // SEARCH_THREADS_H
