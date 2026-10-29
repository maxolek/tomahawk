// threads.cpp
// Lazy SMP helper threads (see threads.h).

#include <threads.h>

// ------------------------------------------------------------
// HelperThread
// ------------------------------------------------------------

HelperThread::HelperThread(NNUE& nnue, TranspositionTable& tt)
    : movegen(board),
      searcher(std::make_unique<Searcher>(board, movegen, nnue, acc, tt)) {
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

void HelperThread::start(const Board& root, const SearchLimits& search_limits, const SearchParams& params,
                         const std::vector<int>& placement, const std::vector<int>& fallback) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        board = root;
        limits = search_limits;
        searcher->params = params;
        cpus = placement;
        fallback_cpus = fallback;
        searching = true;
    }
    cv.notify_all();
}

void HelperThread::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !searching; });
}

void HelperThread::loop() {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
        cv.wait(lock, [&] { return searching || exiting; });
        if (exiting) return;

        lock.unlock();
        run_search();
        lock.lock();

        searching = false;
        cv.notify_all();
    }
}

void HelperThread::run_search() {
    // A refused placement would leave the thread where it was, which may be a CPU of
    // another search thread (Linux threads inherit their creator's affinity).
    pin_failed = !affinity::apply_to_current_thread(cpus);
    if (pin_failed) affinity::apply_to_current_thread(fallback_cpus);
    Move root_moves[MAX_MOVES];
    const int count = movegen.generateMoves(board, false);
    std::copy_n(movegen.moves, count, root_moves);
    searcher->nodes = 0;
    if (count > 0)
        searcher->iterativeDeepening(root_moves, count, limits); // result intentionally unused
}

// ------------------------------------------------------------
// SearchThreadPool
// ------------------------------------------------------------

void SearchThreadPool::set_helpers(int count, NNUE& nnue, TranspositionTable& tt) {
    count = std::max(0, count);
    if (static_cast<int>(helpers.size()) > count) stop_and_wait(); // no helper may still be searching
    while (static_cast<int>(helpers.size()) > count) helpers.pop_back();
    while (static_cast<int>(helpers.size()) < count) helpers.push_back(std::make_unique<HelperThread>(nnue, tt));
}

void SearchThreadPool::start(const Board& root, const SearchLimits& limits, const SearchParams& params,
                             const affinity::Plan& placement) {
    static const std::vector<int> unpinned;
    auto slot = [&](const std::vector<std::vector<int>>& sets, size_t i) -> const std::vector<int>& {
        return i < sets.size() ? sets[i] : unpinned;
    };
    for (size_t i = 0; i < helpers.size(); ++i)
        helpers[i]->start(root, limits, params, slot(placement.threads, i + 1), slot(placement.unpinned, i + 1));
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
