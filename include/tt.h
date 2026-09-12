#pragma once
#include "helpers.h"
#include "move.h"
#include <vector>
#include <algorithm>
#include <cmath>

// -------------------------------
// Stats tracking
// -------------------------------
struct TTStats {
    size_t totalStores = 0;
    size_t totalHits = 0;
    size_t overwritten = 0;
};

// -------------------------------
// Bound types for TT entries
// -------------------------------
enum BoundType : uint8_t {
    EXACT,
    LOWERBOUND,
    UPPERBOUND
};

// -------------------------------
// Transposition Table Entry
// -------------------------------
// key      16 bit
// eval     16 bit
// depth    16 bit
// age      16 bit
// flag     2 bit
// bestMove 16 bit
struct TTEntry {
    U64 key = 0;           // Zobrist key
    int16_t eval = 0;      // Stored evaluation (centipawns)
    int16_t depth = 0;     // iterative_depth recorded
    uint16_t age = 0;      // Age counter
    BoundType flag = EXACT;
    uint16_t bestMove = 0; // Encoded move
};

// -------------------------------
// Transposition Table
// -------------------------------
class TranspositionTable {
public:
    TTStats stats;

    TranspositionTable(int mbSize = 512) {
        resize(mbSize);
    }

    // Resize table to given MB size
    void resize(int mbSize) {
        int bytes = mbSize * 1024 * 1024;
        entriesCount = 1ULL << static_cast<size_t>(
            std::log2((unsigned long)bytes / sizeof(TTEntry))
        );

        table.assign(entriesCount, TTEntry{});
        filledCount = 0;
        stats = TTStats{};
    }

    // Probe TT for a given key
    inline TTEntry* probe(U64 key) {
        #ifdef DEV
            ScopedTimer timer(T_TT_PROBE);
        #endif
        TTEntry* entry = &table[key & (entriesCount - 1)];

        if (entry->key == key && key != 0)
            stats.totalHits++;

        return entry;
    }

    // Store an entry
    inline void store(U64 key, int depth, [[maybe_unused]] int ply, int score,
                      BoundType flag, Move bestMove) {
        #ifdef DEV
            ScopedTimer timer(T_TT_STORE);
        #endif
        TTEntry* entry = probe(key);
        
        bool wasEmpty   = (entry->key == 0);
        bool overwritten = (!wasEmpty && entry->key != key);

        // Replace if new key or deeper search depth
        if (entry->key != key || depth >= entry->depth) {
            if (overwritten)
                stats.overwritten++;

            if (wasEmpty)
                filledCount++;   

            entry->key = key;
            entry->eval = static_cast<int16_t>(score);
            entry->depth = static_cast<int16_t>(depth);
            entry->flag = flag;
            entry->bestMove = bestMove.Value();
            entry->age++;

            //stats.totalStores++;
            #ifdef DEV
                STATS_TT_STORE(depth+ply, ply);
            #endif
        }
    }

    // Clear all entries
    void clear() {
        std::fill(table.begin(), table.end(), TTEntry{});
        stats = TTStats{};
        filledCount = 0;
    }

    // Fill metrics (O(1))
    size_t filled() const {
        return filledCount;
    }

    double fillRatio() const {
        return static_cast<double>(filledCount) / static_cast<double>(entriesCount);
    }

    size_t entriesCount = 0;
    size_t filledCount = 0;   
private:
    std::vector<TTEntry> table;
};
