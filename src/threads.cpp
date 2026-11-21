// threads.cpp
// Lazy SMP helper threads (see threads.h).

#include <threads.h>

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

uint64_t SearchThreadPool::helper_nodes() const {
    uint64_t total = 0;
    for (const auto& helper : helpers) total += helper->nodes();
    return total;
}

int SearchThreadPool::placement_failures() const {
    int failures = 0;
    for (const auto& helper : helpers) failures += helper->placement_failed() ? 1 : 0;
    return failures;
}

std::vector<SearchStats> SearchThreadPool::completed_stats() const {
    std::vector<SearchStats> stats;
    stats.reserve(helpers.size());
    for (const auto& helper : helpers)
        stats.push_back(helper->completed_stats());
    return stats;
}
