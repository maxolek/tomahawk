#pragma once

#include <engine.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace {

uint64_t perft_nodes(Board& board, MoveGenerator& movegen, int depth) {
    if (depth == 0) return 1;

    Move moves[MAX_MOVES];
    const int count = movegen.generateMoves(board, false);
    std::copy_n(movegen.moves, count, moves);

    uint64_t nodes = 0;
    for (int i = 0; i < count; ++i) {
        board.MakeMove(moves[i]);
        nodes += perft_nodes(board, movegen, depth - 1);
        board.UnmakeMove(moves[i]);
    }
    return nodes;
}

struct PerftTask {
    size_t root_index;
    std::vector<Move> path;
    int remaining_depth;
};

struct PerftWorkerStats {
    uint64_t nodes = 0;
    uint64_t elapsed_ns = 0;
};

struct PerftResult {
    std::vector<Move> moves;
    std::vector<uint64_t> counts;
    std::vector<PerftWorkerStats> workers;
    uint64_t elapsed_ns = 0;
};

PerftResult parallel_perft(Board& root, MoveGenerator& root_movegen, int depth, int thread_count,
                           const affinity::Plan& placement) {
    const auto start = std::chrono::steady_clock::now();
    auto elapsed_ns = [&] {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    };

    const int move_count = root_movegen.generateMoves(root, false);
    PerftResult result;
    result.moves.assign(root_movegen.moves, root_movegen.moves + move_count);
    result.counts.assign(result.moves.size(), 0);
    if (result.moves.empty()) {
        result.elapsed_ns = elapsed_ns();
        return result;
    }

    const int worker_count = std::min(thread_count, static_cast<int>(result.moves.size()));
    if (worker_count <= 1) {
        result.workers.push_back({0, 0});
        for (size_t i = 0; i < result.moves.size(); ++i) {
            root.MakeMove(result.moves[i]);
            result.counts[i] = perft_nodes(root, root_movegen, depth - 1);
            root.UnmakeMove(result.moves[i]);
        }
        result.elapsed_ns = elapsed_ns();
        for (uint64_t nodes : result.counts)
            result.workers[0].nodes += nodes;
        result.workers[0].elapsed_ns = result.elapsed_ns;
        return result;
    }

    std::vector<PerftTask> tasks;
    tasks.reserve(result.moves.size());
    for (size_t i = 0; i < result.moves.size(); ++i)
        tasks.push_back({i, {result.moves[i]}, depth - 1});

    const size_t target_tasks = static_cast<size_t>(worker_count) * 2;
    while (tasks.size() < target_tasks && tasks.front().remaining_depth > 0) {
        std::vector<PerftTask> expanded;
        for (const PerftTask& task : tasks) {
            Board board(root);
            MoveGenerator movegen(board);
            for (Move move : task.path) board.MakeMove(move);

            Move children[MAX_MOVES];
            const int child_count = movegen.generateMoves(board, false);
            std::copy_n(movegen.moves, child_count, children);
            for (int i = 0; i < child_count; ++i) {
                PerftTask child = task;
                child.path.push_back(children[i]);
                --child.remaining_depth;
                expanded.push_back(std::move(child));
            }
        }
        tasks = std::move(expanded);
        if (tasks.empty()) {
            result.elapsed_ns = elapsed_ns();
            return result;
        }
    }

    std::atomic<size_t> next_task{0};
    std::vector<std::vector<uint64_t>> worker_counts(
        static_cast<size_t>(worker_count), std::vector<uint64_t>(result.moves.size(), 0));
    result.workers.resize(static_cast<size_t>(worker_count));
    std::vector<Timing> worker_timings(static_cast<size_t>(worker_count));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));

    for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers.emplace_back([&, worker_id] {
            const auto worker_start = std::chrono::steady_clock::now();
            Timing* previous_timing = g_thread_timing;
            g_thread_timing = &worker_timings[static_cast<size_t>(worker_id)];

            const size_t index = static_cast<size_t>(worker_id);
            if (index < placement.threads.size() &&
                !affinity::apply_to_current_thread(placement.threads[index]) &&
                index < placement.unpinned.size())
                affinity::apply_to_current_thread(placement.unpinned[index]);

            Board board(root);
            MoveGenerator movegen(board);
            size_t task_index = 0;
            while ((task_index = next_task.fetch_add(1, std::memory_order_relaxed)) < tasks.size()) {
                const PerftTask& task = tasks[task_index];
                for (Move move : task.path) board.MakeMove(move);
                const uint64_t nodes = task.remaining_depth == 0 ? 1 :
                    perft_nodes(board, movegen, task.remaining_depth);
                worker_counts[index][task.root_index] += nodes;
                result.workers[index].nodes += nodes;
                for (auto move = task.path.rbegin(); move != task.path.rend(); ++move)
                    board.UnmakeMove(*move);
            }
            result.workers[index].elapsed_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - worker_start).count());
            g_thread_timing = previous_timing;
        });
    }
    for (std::thread& worker : workers) worker.join();

    for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
        const Timing& timing = worker_timings[static_cast<size_t>(worker_id)];
        for (int timer = 0; timer < T_COUNT; ++timer) {
            g_timing.stats[timer].cycles += timing.stats[timer].cycles;
            g_timing.stats[timer].calls += timing.stats[timer].calls;
        }
        for (size_t i = 0; i < result.counts.size(); ++i)
            result.counts[i] += worker_counts[static_cast<size_t>(worker_id)][i];
    }
    result.elapsed_ns = elapsed_ns();
    return result;
}

uint64_t perft_total(const PerftResult& result) {
    uint64_t total = 0;
    for (uint64_t nodes : result.counts) total += nodes;
    return total;
}

void print_perft_info(int depth, uint64_t total, const PerftResult& result) {
    const double seconds = static_cast<double>(result.elapsed_ns) / 1'000'000'000.0;
    const double nps = seconds > 0.0 ? static_cast<double>(total) / seconds : 0.0;
    const double effective_bf = depth > 0
        ? std::pow(static_cast<double>(total), 1.0 / static_cast<double>(depth))
        : 1.0;

    std::ostringstream info;
    info << std::fixed << std::setprecision(2)
         << "info string perft depth " << depth
         << " threads " << result.workers.size()
         << " nodes " << total
         << " time_ms " << static_cast<double>(result.elapsed_ns) / 1'000'000.0
         << " nps " << nps
         << " effective_bf " << std::setprecision(3) << effective_bf;
    std::cout << info.str() << '\n';

    for (size_t i = 0; i < result.workers.size(); ++i) {
        const PerftWorkerStats& worker = result.workers[i];
        const double worker_seconds = static_cast<double>(worker.elapsed_ns) / 1'000'000'000.0;
        const double worker_nps = worker_seconds > 0.0
            ? static_cast<double>(worker.nodes) / worker_seconds : 0.0;
        std::ostringstream worker_info;
        worker_info << std::fixed << std::setprecision(2)
                    << "info string perft thread " << (i + 1)
                    << " nodes " << worker.nodes
                    << " time_ms " << static_cast<double>(worker.elapsed_ns) / 1'000'000.0
                    << " nps " << worker_nps;
        std::cout << worker_info.str() << '\n';
    }
}

PerftResult single_node_perft_result() {
    const auto start = std::chrono::steady_clock::now();
    PerftResult result;
    result.workers.push_back({1, 0});
    result.elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    result.workers[0].elapsed_ns = result.elapsed_ns;
    return result;
}

PerftResult engine_perft(Engine& engine, int depth) {
    return parallel_perft(engine.search_board, *engine.movegen, depth,
                          engine.searchThreads(), engine.affinity_plan);
}

struct TTBenchmarkSlot {
    std::atomic<uint64_t> check{0};
    std::atomic<uint64_t> data{0};
};

static_assert(sizeof(TTBenchmarkSlot) == 16, "TT benchmark slot layout changed");

struct TTBenchmarkSample {
    double milliseconds;
    uint64_t checksum;
};

TTBenchmarkSample run_tt_memory_benchmark(TTBenchmarkSlot* table, size_t count,
                                          size_t operations) {
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < operations; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t key = state * 0x2545F4914F6CDD1DULL;
        TTBenchmarkSlot& slot = table[static_cast<size_t>(key) & (count - 1)];
        const uint64_t data = slot.data.load(std::memory_order_relaxed);
        const uint64_t stored_key = slot.check.load(std::memory_order_relaxed) ^ data;
        if (stored_key != key) {
            const uint64_t new_data = key ^ 0xD6E8FEB86659FD93ULL;
            slot.data.store(new_data, std::memory_order_relaxed);
            slot.check.store(key ^ new_data, std::memory_order_relaxed);
        }
        checksum += data + stored_key;
    }

    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return {milliseconds, checksum};
}

const char* tt_benchmark_page_mode(mem::PageMode mode) {
    switch (mode) {
    case mem::PageMode::Large:
        return "large pages";
    case mem::PageMode::TransparentHugePageHint:
        return "transparent huge-page hint";
    case mem::PageMode::Standard:
    default:
        return "standard pages (large pages unavailable)";
    }
}

} // namespace

void Engine::ttPageBenchmark() {
    constexpr int memory_mb = 32;
    constexpr int rounds = 3;
    constexpr size_t bytes_per_mb = 1024 * 1024;
    const size_t requested_bytes = static_cast<size_t>(memory_mb) * bytes_per_mb;
    const size_t available_slots = requested_bytes / sizeof(TTBenchmarkSlot);
    size_t slot_count = 1;
    while (slot_count <= available_slots / 2)
        slot_count *= 2;
    const size_t operations = slot_count * 2;

    std::unique_ptr<TTBenchmarkSlot[]> standard_table(
        new (std::nothrow) TTBenchmarkSlot[slot_count]);
    if (!standard_table) {
        std::cerr << "info string ttbench: failed to allocate standard-page table\n";
        return;
    }

    mem::Allocation page_allocation = mem::alloc_large_pages(requested_bytes);
    if (!page_allocation.ptr ||
        page_allocation.allocated_bytes < slot_count * sizeof(TTBenchmarkSlot)) {
        mem::free_large_pages(page_allocation);
        std::cerr << "info string ttbench: failed to allocate page-aware table\n";
        return;
    }

    TTBenchmarkSlot* page_table = static_cast<TTBenchmarkSlot*>(page_allocation.ptr);
    for (size_t i = 0; i < slot_count; ++i)
        ::new (static_cast<void*>(page_table + i)) TTBenchmarkSlot{};

    const TTBenchmarkSample standard_warmup =
        run_tt_memory_benchmark(standard_table.get(), slot_count, operations);
    const TTBenchmarkSample page_warmup =
        run_tt_memory_benchmark(page_table, slot_count, operations);
    (void)standard_warmup;
    (void)page_warmup;

    std::cout << "info string ttbench start mb " << memory_mb
              << " slots " << slot_count
              << " probes_per_round " << operations
              << " rounds " << rounds
              << " page_mode " << tt_benchmark_page_mode(page_allocation.page_mode)
              << '\n';

    double standard_total_ms = 0.0;
    double page_total_ms = 0.0;
    uint64_t standard_checksum = 0;
    uint64_t page_checksum = 0;
    for (int round = 0; round < rounds; ++round) {
        TTBenchmarkSample standard_sample{};
        TTBenchmarkSample page_sample{};
        if ((round & 1) == 0) {
            standard_sample = run_tt_memory_benchmark(
                standard_table.get(), slot_count, operations);
            page_sample = run_tt_memory_benchmark(page_table, slot_count, operations);
        } else {
            page_sample = run_tt_memory_benchmark(page_table, slot_count, operations);
            standard_sample = run_tt_memory_benchmark(
                standard_table.get(), slot_count, operations);
        }

        standard_total_ms += standard_sample.milliseconds;
        page_total_ms += page_sample.milliseconds;
        standard_checksum ^= standard_sample.checksum;
        page_checksum ^= page_sample.checksum;
        std::ostringstream round_info;
        round_info << std::fixed << std::setprecision(2)
                   << "info string ttbench round " << (round + 1)
                   << " standard_ms " << standard_sample.milliseconds
                   << " page_aware_ms " << page_sample.milliseconds;
        std::cout << round_info.str() << '\n';
    }

    const double standard_average_ms = standard_total_ms / rounds;
    const double page_average_ms = page_total_ms / rounds;
    const double speedup = page_average_ms > 0.0
        ? standard_average_ms / page_average_ms : 0.0;
    std::ostringstream average_info;
    average_info << std::fixed << std::setprecision(2)
                 << "info string ttbench average standard_ms " << standard_average_ms
                 << " page_aware_ms " << page_average_ms;
    if (page_allocation.page_mode == mem::PageMode::Standard)
        average_info << " speedup n/a";
    else
        average_info << " speedup " << speedup << "x";
    average_info << " checksum_match "
                 << (standard_checksum == page_checksum ? "true" : "false");
    std::cout << average_info.str() << '\n';
    if (page_allocation.page_mode == mem::PageMode::Standard) {
        std::cout << "info string ttbench note large pages unavailable; both tables use standard pages\n";
    } else if (page_allocation.page_mode == mem::PageMode::TransparentHugePageHint) {
        std::cout << "info string ttbench note transparent huge-page hint requested; OS may ignore it\n";
    }

    for (size_t i = 0; i < slot_count; ++i)
        page_table[i].~TTBenchmarkSlot();
    mem::free_large_pages(page_allocation);
}

uint64_t Engine::perft(int depth) {
    if (depth < 0) return 0;
    if (depth == 0) return 1;
    const PerftResult perft_result = engine_perft(*this, depth);
    return perft_total(perft_result);
}

void Engine::perftPrint(int depth) {
    if (depth < 0) {
        std::cerr << "Perft depth must not be negative\n";
        return;
    }
    PerftResult perft_result;
    uint64_t total = 1;
    if (depth == 0) {
        perft_result = single_node_perft_result();
    } else {
        perft_result = engine_perft(*this, depth);
        total = perft_total(perft_result);
    }
    std::cout << "Nodes searched: " << total << '\n';
    print_perft_info(depth, total, perft_result);
}

void Engine::perftDivide(int depth) {
    if (depth < 0) {
        std::cerr << "Perft depth must not be negative\n";
        return;
    }
    if (depth == 0) {
        std::cout << "Total: 1\n";
        const PerftResult perft_result = single_node_perft_result();
        print_perft_info(depth, 1, perft_result);
        return;
    }

    const PerftResult perft_result = engine_perft(*this, depth);
    const uint64_t total = perft_total(perft_result);
    for (size_t i = 0; i < perft_result.moves.size(); ++i) {
        std::cout << perft_result.moves[i].uci() << ": " << perft_result.counts[i] << '\n';
    }

    std::cout << "Total: " << total << "\n";
    print_perft_info(depth, total, perft_result);

#if defined(DEV) || defined(TIMING)
    logTimingStats(game_board.getBoardFEN());
#endif
}

void Engine::SEETest(int capture_square) {
    const int count = movegen->generateMoves(search_board, true);

    for (int i = 0; i < count; ++i) {
        const Move move = movegen->moves[i];
        if (move.TargetSquare() == capture_square) {
            const int see = searcher->SEE(search_board, move);
            std::cout << "SEE (" << move.uci() << ") = " << see << std::endl;
        }
    }
}

void Engine::moveOrderingTest(int /*depth*/) {
    std::cout << "=== Move Ordering Test ===\n";

    const int move_count = movegen->generateMoves(search_board, false);
    Move* moves = movegen->moves;
    std::vector<std::pair<Move, int>> scored_moves;
    Move tt_move;
    std::vector<Move> previous_pv;

    for (int i = 0; i < move_count; ++i) {
        const int score = searcher->moveScore(moves[i], search_board, 0, tt_move, previous_pv);
        scored_moves.emplace_back(moves[i], score);
    }

    std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for (const auto& [move, score] : scored_moves)
        std::cout << move.uci() << ": " << score << "\n";
}
