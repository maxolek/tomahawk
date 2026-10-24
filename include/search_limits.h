#pragma once
#include <chrono>

#include "helpers.h"

struct SearchLimits {
    std::chrono::steady_clock::time_point start_time;
    int node_limit;
    int time_limit_ms;
    int max_depth = MAX_DEPTH;
    bool stopped;

    SearchLimits(int n = 0, int ms = 0, int depth = -1)
        : start_time(std::chrono::steady_clock::now()),
          node_limit(n),
          time_limit_ms(ms),
          max_depth(depth < 0 ? MAX_DEPTH : depth),
          stopped(false) {}

    inline bool out_of_time() {
        if (stopped) return true;
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= time_limit_ms && time_limit_ms > 0) {
            stopped = true;
            return true;
        }
        return false;
    }

    inline bool nodes_reached(int current_nodes) {
        return (node_limit > 0 && current_nodes >= node_limit);
    }

    inline bool depth_reached(int current_depth) {
        return (max_depth >= 0 && current_depth > max_depth);
    }

    inline bool should_stop(int current_depth, int current_nodes) {
        return out_of_time() || depth_reached(current_depth) || nodes_reached(current_nodes);
    }
};
