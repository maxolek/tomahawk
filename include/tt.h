#pragma once
#include "helpers.h"
#include "move.h"
#include "large_pages.h"
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>

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
// flag      8 bit
// bestMove 16 bit
struct TTEntry {
    U64 key = 0;           // Zobrist key
    int16_t eval = 0;      // Stored evaluation (centipawns)
    int16_t depth = 0;     // iterative_depth recorded
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
        if (!resize(mbSize))
            throw std::bad_alloc();
    }

    ~TranspositionTable() {
        release_storage();
    }

    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    // Resize table to given MB size (not while a search is running)
    bool resize(int mbSize) {
        constexpr std::size_t bytes_per_mb = 1024 * 1024;
        if (mbSize <= 0 ||
            static_cast<std::size_t>(mbSize) >
                (std::numeric_limits<std::size_t>::max)() / bytes_per_mb) {
            std::cerr << "info string Error: Invalid TT size\n";
            return false;
        }

        const std::size_t bytes = static_cast<std::size_t>(mbSize) * bytes_per_mb;
        if (table && bytes == requested_bytes_) {
            clear();
            return true;
        }

        const std::size_t available_slots = bytes / sizeof(Slot);
        std::size_t count = 1;
        while (count <= available_slots / 2)
            count *= 2;

        mem::Allocation new_allocation = mem::alloc_large_pages(bytes);
        if (!new_allocation.ptr || new_allocation.allocated_bytes < count * sizeof(Slot)) {
            mem::free_large_pages(new_allocation);
            std::cerr << "info string Error: Failed to allocate TT memory\n";
            return false;
        }

        Slot* new_table = static_cast<Slot*>(new_allocation.ptr);
        for (std::size_t i = 0; i < count; ++i)
            ::new (static_cast<void*>(new_table + i)) Slot{};

        release_storage();
        alloc_ = new_allocation;
        table = new_table;
        entriesCount = count;
        requested_bytes_ = bytes;
        overwrite_count_.store(0, std::memory_order_relaxed);

        const char* page_mode = "standard pages";
        if (alloc_.page_mode == mem::PageMode::Large)
            page_mode = "large pages";
        else if (alloc_.page_mode == mem::PageMode::TransparentHugePageHint)
            page_mode = "transparent huge-page hint";
        std::cout << "info string TT allocated " << (alloc_.allocated_bytes / (1024 * 1024))
                  << " MB (" << page_mode << ")\n";
        return true;
    }

    // Probe TT for a given key; the result's key equals `key` on a hit
    inline TTEntry probe(U64 key) const {
        #if defined(DEV) || defined(TIMING)
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
        #if defined(DEV) || defined(TIMING)
            ScopedTimer timer(T_TT_STORE);
        #endif
        Slot& slot = table[key & (entriesCount - 1)];
        const uint64_t old_data = slot.data.load(std::memory_order_relaxed);
        const uint64_t old_check = slot.check.load(std::memory_order_relaxed);
        const uint64_t old_key = old_check ^ old_data;
        const bool overwriting = old_key != key && (old_check != 0 || old_data != 0);
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

            #if defined(DEV) || defined(STATS)
                STATS_TT_STORE(depth+ply, ply);
            #endif
            if (overwriting)
                overwrite_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Clear all entries (not while a search is running)
    void clear() {
        for (size_t i = 0; i < entriesCount; ++i) {
            table[i].check.store(0, std::memory_order_relaxed);
            table[i].data.store(0, std::memory_order_relaxed);
        }
        overwrite_count_.store(0, std::memory_order_relaxed);
    }

    // Number of occupied slots (scans the table; for diagnostics)
    size_t filled() const {
        size_t count = 0;
        for (size_t i = 0; i < entriesCount; ++i)
            count += (table[i].check.load(std::memory_order_relaxed)
                      ^ table[i].data.load(std::memory_order_relaxed)) != 0;
        return count;
    }

    size_t size() const {
        return entriesCount;
    }

    size_t occupiedBytes(size_t occupied_entries) const {
        return occupied_entries * sizeof(Slot);
    }

    uint64_t overwrites() const {
        return overwrite_count_.load(std::memory_order_relaxed);
    }

    double fillRatio() const {
        return entriesCount == 0 ? 0.0
                                 : static_cast<double>(filled()) / static_cast<double>(entriesCount);
    }


private:
    struct Slot {
        std::atomic<uint64_t> check{0}; // key ^ data
        std::atomic<uint64_t> data{0};  // eval | depth << 16 | flag << 32 | bestMove << 40
    };

    void release_storage() {
        for (std::size_t i = 0; i < entriesCount; ++i)
            table[i].~Slot();
        table = nullptr;
        entriesCount = 0;
        requested_bytes_ = 0;
        mem::free_large_pages(alloc_);
    }

    // same slot size (and so the same slot count per MB) as the former layout
    static_assert(sizeof(Slot) == 16, "TT slot layout changed");

    mem::Allocation alloc_{};
    Slot* table = nullptr;
    size_t entriesCount = 0;
    std::size_t requested_bytes_ = 0;
    std::atomic<uint64_t> overwrite_count_{0};
};