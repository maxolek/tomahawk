#pragma once
#include "helpers.h"
#include "move.h"
#include <atomic>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

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
// Snapshot of one table slot, as returned by probe().
// key      64 bit
// eval     16 bit
// depth    16 bit
// age      16 bit
// flag      8 bit
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
// Shared by all search threads without locks. Each slot holds its payload in one
// 64-bit word and key ^ payload in another; both are accessed with relaxed atomic
// loads/stores. A slot read while another thread rewrites it therefore fails the
// key check and reads as a miss instead of mixing two positions' data.
class TranspositionTable {
public:
    TranspositionTable(int mbSize = 512) {
        resize(mbSize);
    }

    // Resize table to given MB size (not while a search is running)
    void resize(int mbSize) {
        int bytes = mbSize * 1024 * 1024;
        const size_t count = 1ULL << static_cast<size_t>(
            std::log2((unsigned long)bytes / sizeof(Slot))
        );

        if (table && count == entriesCount) {  // same size: just empty it
            clear();
            return;
        }
        table.reset(); // release the old table before allocating the new one
        entriesCount = count;
        table = std::make_unique<Slot[]>(entriesCount);
    }

    // Probe TT for a given key; the result's key equals `key` on a hit
    inline TTEntry probe(U64 key) const {
        #ifdef DEV
            ScopedTimer timer(T_TT_PROBE);
        #endif
        const Slot& slot = table[key & (entriesCount - 1)];
        const uint64_t data = slot.data.load(std::memory_order_relaxed);
        const uint64_t check = slot.check.load(std::memory_order_relaxed);

        TTEntry entry;
        entry.key = check ^ data;
        entry.eval = static_cast<int16_t>(data & 0xFFFF);
        entry.depth = static_cast<int16_t>((data >> 16) & 0xFFFF);
        entry.flag = static_cast<BoundType>((data >> 32) & 0xFF);
        entry.bestMove = static_cast<uint16_t>((data >> 40) & 0xFFFF);
        entry.age = static_cast<uint16_t>(slot.age.load(std::memory_order_relaxed));
        return entry;
    }

    // Start loading the entry for a position that is about to be probed
    // (an entry may straddle two cache lines).
    inline void prefetch(U64 key) const {
        const char* entry = reinterpret_cast<const char*>(&table[key & (entriesCount - 1)]);
        __builtin_prefetch(entry);
        __builtin_prefetch(entry + sizeof(Slot) - 1);
    }

    // Store an entry
    inline void store(U64 key, int depth, [[maybe_unused]] int ply, int score,
                      BoundType flag, Move bestMove) {
        #ifdef DEV
            ScopedTimer timer(T_TT_STORE);
        #endif
        Slot& slot = table[key & (entriesCount - 1)];
        const uint64_t old_data = slot.data.load(std::memory_order_relaxed);
        const uint64_t old_key = slot.check.load(std::memory_order_relaxed) ^ old_data;
        const int16_t old_depth = static_cast<int16_t>((old_data >> 16) & 0xFFFF);

        // Replace if new key or deeper search depth
        if (old_key != key || depth >= old_depth) {
            const uint64_t data =
                  static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(score)))
                | static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(depth))) << 16
                | static_cast<uint64_t>(flag) << 32
                | static_cast<uint64_t>(bestMove.Value()) << 40;
            slot.data.store(data, std::memory_order_relaxed);
            slot.check.store(key ^ data, std::memory_order_relaxed);
            slot.age.store(static_cast<uint16_t>(slot.age.load(std::memory_order_relaxed) + 1),
                           std::memory_order_relaxed);

            #ifdef DEV
                STATS_TT_STORE(depth+ply, ply);
            #endif
        }
    }

    // Clear all entries (not while a search is running)
    void clear() {
        for (size_t i = 0; i < entriesCount; ++i) {
            table[i].check.store(0, std::memory_order_relaxed);
            table[i].data.store(0, std::memory_order_relaxed);
            table[i].age.store(0, std::memory_order_relaxed);
        }
    }

    // Number of occupied slots (scans the table; for diagnostics)
    size_t filled() const {
        size_t count = 0;
        for (size_t i = 0; i < entriesCount; ++i)
            count += (table[i].check.load(std::memory_order_relaxed)
                      ^ table[i].data.load(std::memory_order_relaxed)) != 0;
        return count;
    }

    double fillRatio() const {
        return static_cast<double>(filled()) / static_cast<double>(entriesCount);
    }

    size_t entriesCount = 0;

private:
    struct Slot {
        std::atomic<uint64_t> check{0}; // key ^ data
        std::atomic<uint64_t> data{0};  // eval | depth << 16 | flag << 32 | bestMove << 40
        std::atomic<uint64_t> age{0};   // store counter
    };
    // same slot size (and so the same slot count per MB) as the former layout
    static_assert(sizeof(Slot) == 24, "TT slot layout changed");

    std::unique_ptr<Slot[]> table;
};