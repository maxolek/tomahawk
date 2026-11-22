// threads.h
// Lazy SMP helper threads.
//
// A search runs on the calling (UCI) thread, the "main" thread, plus Threads - 1
// helpers. Every helper searches the same root position with its own board,
// move generator, NNUE accumulator stack and Searcher (killers, history), and
// shares the transposition table with the main thread; that shared table is how
// helper work reaches the main search. Each thread's final best move contributes
// its completed depth as vote weight; the move with the greatest total weight is
// reported. Equal weights prefer more votes, then the main move. The deepest
// thread voting for the selected move supplies its eval and PV. Search threads
// are pinned once when they start; changing Threads or Affinity recreates helpers.

#ifndef THREADS_H
#define THREADS_H

#include "searcher.h"
#include "affinity.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

class HelperThread {
public:
    HelperThread(NNUE& nnue, TranspositionTable& tt,
                 const std::vector<int>& cpus, const std::vector<int>& fallback_cpus);
    ~HelperThread();
    HelperThread(const HelperThread&) = delete;
    HelperThread& operator=(const HelperThread&) = delete;

    // Start searching `root` in the background; the helper must be idle.
    void start(const Board& root, const SearchLimits& limits, const SearchParams& params);
    // Block until the current search (if any) has returned.
    void wait();
    void wait_until_placed();
    SearchStats completed_stats() const;
    // Nodes of the last search; valid after wait().
    uint64_t nodes() const { return searcher->nodes; }
    // The OS refused this helper's configured placement.
    bool placement_failed() const { return pin_failed; }

private:
    void loop();
    void run_search();

    Board board;
    MoveGenerator movegen;
    AccumulatorStack acc;
    std::unique_ptr<Searcher> searcher;
    SearchLimits limits;
    std::vector<int> cpus;
    std::vector<int> fallback_cpus;
    bool pin_failed = false;

    mutable std::mutex mutex;
    std::condition_variable cv;
    bool searching = false;
    bool placement_ready = false;
    bool exiting = false;
    SearchStats last_search_stats;
    std::thread thread; // started last, after everything it uses exists
};

struct ThreadVoteResult {
    Move best_move = Move::NullMove();
    int eval = -MATE_SCORE;
    int depth = 0;
    PV best_line;
    int votes = 0;
    int total_votes = 0;
    int weight = 0;
    int total_weight = 0;
    struct Tally {
        Move move;
        int votes;
        int weight;
    };
    std::vector<Tally> tallies;
    std::vector<SearchStats> helper_stats;
};

class SearchThreadPool {
public:
    // Shared by every thread of the running search (limits.stop_signal).
    std::atomic<bool> stop_signal{false};

    ~SearchThreadPool() { stop_and_wait(); }

    // Keep `count` helper threads (Threads - 1), using the configured placement.
    void set_helpers(int count, NNUE& nnue, TranspositionTable& tt,
                     const affinity::Plan& placement);
    int helper_count() const { return static_cast<int>(helpers.size()); }

    // Starts every helper on `root`; the caller clears the stop signal before
    // launching a search.
    void start(const Board& root, const SearchLimits& limits, const SearchParams& params);
    // Raises the stop signal and waits for every helper to return.
    void stop_and_wait();
    // Collect helper results and vote with the main thread's completed result.
    ThreadVoteResult vote(const SearchResult& main_result, int main_depth) const;
    // Sum of the helpers' node counts for the last search.
    uint64_t helper_nodes() const;
    // Helpers whose placement the OS refused in the last search (after stop_and_wait()).
    int placement_failures() const;

private:
    std::vector<std::unique_ptr<HelperThread>> helpers;
};

#endif // SEARCH_THREADS_H
