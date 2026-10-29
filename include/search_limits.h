#pragma once
#include <chrono>
#include <atomic>
#include "helpers.h"

struct SearchLimits {
    // In-tree searches read the clock only once per TIME_POLL_INTERVAL calls.
    static constexpr uint32_t TIME_POLL_INTERVAL = 1024;

    std::chrono::steady_clock::time_point start_time;

    int time_limit_ms;
    int max_depth = MAX_DEPTH;
    int node_limit;
    
    bool stopped;
    uint32_t poll_count = 0;
    // Stop request shared by all threads of one search (optional). Each thread
    // searches with its own copy of the limits; the flag makes them all stop.
    std::atomic<bool>* stop_signal = nullptr;

    SearchLimits(int ms = 0, int depth = -1, int n = 0)
        : start_time(std::chrono::steady_clock::now()),
          time_limit_ms(ms),
          max_depth(depth < 0 ? MAX_DEPTH : depth),
          node_limit(n),
          stopped(false) {}

    inline bool out_of_time() {
        if (stopped) return true;
        if (stop_signal && stop_signal->load(std::memory_order_relaxed)) {
            stopped = true;
            return true;
        }
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= time_limit_ms && time_limit_ms > 0) {
            stopped = true;
            return true;
        }
        return false;
    }

    // Hot-path variant of out_of_time(): time and external stop requests are
    // checked only periodically.
    inline bool poll_out_of_time() {
        if (stopped) return true;
        if (++poll_count & (TIME_POLL_INTERVAL - 1)) return false;
        return out_of_time();
    }

    inline bool nodes_reached(int current_nodes) {
        return (node_limit > 0 && current_nodes >= node_limit);
    }

    inline bool depth_reached(int current_depth) {
        return (max_depth >= 0 && current_depth > max_depth);
    }

    inline bool should_stop(int current_depth, int current_nodes) {
        return out_of_time() 
                || depth_reached(current_depth) 
                || nodes_reached(current_nodes);
    }
};
