// threads.cpp
// Lazy SMP helper threads (see threads.h).

#include <threads.h>

#include <algorithm>
#include <iterator>

namespace {

struct CandidateVote {
    Move move;
    int eval;
    int depth;
    PV best_line;
    bool main_thread;
};

struct VoteTally {
    Move move;
    int count = 0;
    int weight = 0; // sum of depths (instead of counts)
    bool includes_main = false;
    size_t representative = 0;
};

bool better_candidate(const CandidateVote& candidate, const CandidateVote& current) {
    if (candidate.depth != current.depth) return candidate.depth > current.depth;
    if (candidate.eval != current.eval) return candidate.eval > current.eval;
    return candidate.main_thread && !current.main_thread;
}

PV helper_pv(Move best_move, const std::vector<Move>& line) {
    PV pv;
    if (line.empty() || !Move::SameMove(line.front(), best_move)) {
        pv.moves[0] = best_move.Value();
        pv.length = 1;
        return pv;
    }

    const int count = std::min(static_cast<int>(line.size()), PV::MAX_LEN);
    for (int i = 0; i < count; ++i)
        pv.moves[i] = line[static_cast<size_t>(i)].Value();
    pv.length = count;
    return pv;
}

} // namespace

// ------------------------------------------------------------
// HelperThread
// ------------------------------------------------------------

HelperThread::HelperThread(NNUE& nnue, TranspositionTable& tt,
                           const std::vector<int>& assigned_cpus, const std::vector<int>& fallback_assignment)
    : movegen(board),
      searcher(std::make_unique<Searcher>(board, movegen, nnue, acc, tt)),
      cpus(assigned_cpus),
      fallback_cpus(fallback_assignment) {
    searcher->is_main = false;
    thread = std::thread(&HelperThread::loop, this);
}

HelperThread::~HelperThread() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !searching; });
        exiting = true;
    }
    cv.notify_all();
    thread.join();
}

void HelperThread::start(const Board& root, const SearchLimits& search_limits, const SearchParams& params) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        board = root;
        limits = search_limits;
        searcher->params = params;
        searching = true;
    }
    cv.notify_all();
}

void HelperThread::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !searching; });
}

void HelperThread::wait_until_placed() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return placement_ready; });
}

SearchStats HelperThread::completed_stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return last_search_stats;
}

void HelperThread::loop() {
    pin_failed = !affinity::apply_to_current_thread(cpus);
    if (pin_failed) affinity::apply_to_current_thread(fallback_cpus);

    std::unique_lock<std::mutex> lock(mutex);
    placement_ready = true;
    cv.notify_all();
    while (true) {
        cv.wait(lock, [&] { return searching || exiting; });
        if (exiting) return;

        lock.unlock();
        run_search();
        lock.lock();

        last_search_stats = g_stats;
        searching = false;
        cv.notify_all();
    }
}

void HelperThread::run_search() {
    resetSearchStats();
    const auto start_time = std::chrono::steady_clock::now();
    Move root_moves[MAX_MOVES];
    const int count = movegen.generateMoves(board, false);
    std::copy_n(movegen.moves, count, root_moves);
    searcher->nodes = 0;
    SearchResult result;
    if (count > 0)
        result = searcher->iterativeDeepening(root_moves, count, limits);
    g_stats.time_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count());
    g_stats.eval = result.eval;
    g_stats.move = result.bestMove;
    g_stats.principal_variation = result.best_line.to_vector();
    #if !defined(DEV) && !defined(STATS)
        g_stats.nodes = searcher->nodes;
    #endif
}

// ------------------------------------------------------------
// SearchThreadPool
// ------------------------------------------------------------

void SearchThreadPool::set_helpers(int count, NNUE& nnue, TranspositionTable& tt,
                                   const affinity::Plan& placement) {
    count = std::max(0, count);
    stop_and_wait();
    helpers.clear();
    for (int i = 0; i < count; ++i) {
        const size_t slot = static_cast<size_t>(i + 1);
        const std::vector<int> cpus = slot < placement.threads.size()
            ? placement.threads[slot] : std::vector<int>{};
        const std::vector<int> fallback = slot < placement.unpinned.size()
            ? placement.unpinned[slot] : std::vector<int>{};
        helpers.push_back(std::make_unique<HelperThread>(nnue, tt, cpus, fallback));
    }
    for (const auto& helper : helpers) helper->wait_until_placed();
}

void SearchThreadPool::start(const Board& root, const SearchLimits& limits, const SearchParams& params) {
    for (size_t i = 0; i < helpers.size(); ++i)
        helpers[i]->start(root, limits, params);
}

void SearchThreadPool::stop_and_wait() {
    stop_signal.store(true, std::memory_order_relaxed);
    for (auto& helper : helpers) helper->wait();
}

ThreadVoteResult SearchThreadPool::vote(const SearchResult& main_result, int main_depth) const {
    ThreadVoteResult result;
    result.helper_stats.reserve(helpers.size());
    for (const auto& helper : helpers)
        result.helper_stats.push_back(helper->completed_stats());

    std::vector<CandidateVote> candidates;
    candidates.reserve(result.helper_stats.size() + 1);
    candidates.push_back({
        main_result.bestMove, main_result.eval, main_depth, main_result.best_line, true
    });
    for (const SearchStats& stats : result.helper_stats) {
        candidates.push_back({
            stats.move, stats.eval, stats.max_completed_depth,
            helper_pv(stats.move, stats.principal_variation), false
        });
    }

    std::vector<VoteTally> tallies;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const CandidateVote& candidate = candidates[i];
        if (Move::SameMove(candidate.move, Move::NullMove())) continue;

        auto tally = std::find_if(tallies.begin(), tallies.end(), [&](const VoteTally& entry) {
            return Move::SameMove(entry.move, candidate.move);
        });
        if (tally == tallies.end()) {
            tallies.push_back({candidate.move, 0, 0, false, i});
            tally = std::prev(tallies.end());
        }

        ++tally->count;
        tally->weight += std::max(candidate.depth, 0);

        tally->includes_main = tally->includes_main || candidate.main_thread;
        if (better_candidate(candidate, candidates[tally->representative]))
            tally->representative = i;
    }

    size_t winner = tallies.size();
    for (size_t i = 0; i < tallies.size(); ++i) {
        const VoteTally& tally = tallies[i];
        result.total_votes += tally.count;
        result.total_weight += tally.weight;
        result.tallies.push_back({tally.move, tally.count, tally.weight});
        if (winner == tallies.size()) {
            winner = i;
            continue;
        }

        const VoteTally& current = tallies[winner];
        bool better_tally = false;
        if (tally.weight != current.weight) {
            better_tally = tally.weight > current.weight;
        } else if (tally.count != current.count) {
            better_tally = tally.count > current.count;
        } else if (tally.includes_main != current.includes_main) {
            better_tally = tally.includes_main;
        } else {
            better_tally = better_candidate(candidates[tally.representative],
                                            candidates[current.representative]);
        }
        if (better_tally) {
            winner = i;
        }
    }

    if (winner != tallies.size()) {
        const VoteTally& winning_tally = tallies[winner];
        const CandidateVote& representative = candidates[winning_tally.representative];
        result.best_move = representative.move;
        result.eval = representative.eval;
        result.depth = representative.depth;
        // Keep the winning representative thread's PV intact; never splice lines across threads.
        result.best_line = representative.best_line;
        result.votes = winning_tally.count;
        result.weight = winning_tally.weight;
    }
    return result;
}

uint64_t SearchThreadPool::helper_nodes() const {
    uint64_t total = 0;
    for (const auto& helper : helpers) total += helper->nodes();
    return total;
}

int SearchThreadPool::helper_seldepth() const {
    int maximum = 0;
    for (const auto& helper : helpers)
        maximum = std::max(maximum, helper->seldepth());
    return maximum;
}

int SearchThreadPool::placement_failures() const {
    int failures = 0;
    for (const auto& helper : helpers) failures += helper->placement_failed() ? 1 : 0;
    return failures;
}
