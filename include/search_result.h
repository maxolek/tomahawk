#ifndef SEARCH_RESULT_H
#define SEARCH_RESULT_H

#include "move.h"
#include "helpers.h"
#include <vector>
#include <cstdint>

struct RootMove {
    Move move = Move::NullMove();
    int eval = -INF;
    int64_t time_ms = 0;
    uint64_t nodes = 0;
    bool exact = false; // true only if this came from a full-window (or fail-high full window re-)search
};

struct PV {
    // Fixed capacity, so building lines in the search never touches the heap.
    // Moves are kept as raw values so constructing a PV does not initialize storage.
    // Lines longer than MAX_LEN keep their first MAX_LEN moves.
    static constexpr int MAX_LEN = 128;
    int length = 0;
    ushort moves[MAX_LEN];

    PV() {}
    PV(const PV& other) : length(other.length) { std::memcpy(moves, other.moves, sizeof(ushort) * (size_t)length); }
    PV& operator=(const PV& other) {
        length = other.length;
        std::memcpy(moves, other.moves, sizeof(ushort) * (size_t)length);
        return *this;
    }

    void clear() { length = 0; }
    int size() const { return length; }
    Move operator[](int i) const { return Move(moves[i]); }

    inline void set(Move first, const PV& child) {
        const int n = std::min(child.length, MAX_LEN - 1);
        moves[0] = first.Value();
        std::memcpy(moves + 1, child.moves, sizeof(ushort) * (size_t)n);
        length = n + 1;
    }

    std::vector<Move> to_vector() const {
        std::vector<Move> out;
        out.reserve((size_t)length);
        for (int i = 0; i < length; ++i) out.push_back(Move(moves[i]));
        return out;
    }
};

struct SearchResult {
    Move bestMove = Move::NullMove();
    int eval = -MATE_SCORE;
    PV best_line;

    RootMove root_moves[MAX_MOVES];
    int root_count = 0;

    inline void setPV(Move first, const PV& child) {
        best_line.set(first, child);
    }
};

#endif
