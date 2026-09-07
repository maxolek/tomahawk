#pragma once
#include <vector>
#include <cstdint>
#include <string>

#ifdef DEBUG
    #include <unordered_set>
#endif

#include "board.h"
#include "move.h"
#include "stats.h"
#include "timer.h"
#include "simd.h"
#include "network.h"
#include "accumulator.h"
#include "utils.h"

// ============================================================
// Network
// ============================================================

class NNUE {
public:
    // Cached accumulators
    // dual perspective
    // during tracking stm=white and ntm=black always
    // flipped appropriately during eval for [stm,ntm] actual [us/them] concat
    //      changed to pointers for integration with DEBUG class
    Accumulator acc_stm;
    Accumulator acc_ntm;
  
    // ========== L0: 768xINPUT_BUCKETS → 512 ==========
    // Stored column-major: W0[feature][hidden]
    int16_t l0w[INPUT_SIZE][L1_SIZE];
    int16_t l0b[L1_SIZE];

    // ========== L1: 512 → 8x16 ==========
    // Dual-perspective: [stm_hidden, ntm_hidden]
    int8_t l1w[L2_SIZE * NUM_OUTPUT_BUCKETS][L1_SIZE]; // 2*L0_SIZE if not using pairwise multiply
    int32_t l1b[L2_SIZE * NUM_OUTPUT_BUCKETS];          // otherwise the concat is reduced back down to L0_SIZE

    // ========== L2: 16 → 8x32 ==========
    int8_t l2w[L3_SIZE * NUM_OUTPUT_BUCKETS][L2_SIZE];
    int32_t l2b[L3_SIZE * NUM_OUTPUT_BUCKETS];

    // ========== L3: 32 → NUM_OUTPUT_BUCKETS ==========
    int8_t l3w[NUM_OUTPUT_BUCKETS][L3_SIZE];
    int32_t l3b[NUM_OUTPUT_BUCKETS];

    // constructors 
    NNUE() {};
    NNUE(const fs::path& path) { load(path); };

    // Load quantised network
    bool load(const fs::path& path);

    // Compute final output from accumulators
    int evaluate(bool is_white_move, U64 occ); 
    int eval_simd(bool is_white_move, U64 occ);
    int full_eval(const Board& b);
    //int eval_smallnet(bool is_white_move, U64 occ)

    // Incremental updates for search
    void on_make_move(const Board& board, const Move& mv);
    void on_make_move_halfka(const Board& board, const Move& mv);
    void on_unmake_move(const Board& board, const Move& mv);
    void on_unmake_move_halfka(const Board& board, const Move& mv);

    // Build full accumulators from board
    void build_accumulators(const Board& b);
    void build_halfka_accumulators(const Board& b);

    void set_accumulators(Accumulator* stm, Accumulator* ntm);
    void reset_accumulators();

    // debugging
#ifdef DEBUG
    void debug_simd(const Board& b);
    void debug_acc(const Accumulator& acc, const std::string& name) const;
    void debug_acc_full(const Accumulator& acc, const std::string& name) const;
    void debug_evaluate(const Accumulator& us, const Accumulator& them) const;
    void debug_on_move(const std::string& name, const Move& mv, int color, int moved_piece,
                         int f_from, int f_to) const;
    void on_make_move_debug(const Board& before, const Move& mv);
    void on_unmake_move_debug(const Board& board, const Move& mv);
    int evaluate_debug(bool is_white_move) const;
    void debug_check_incr_vs_full_after_make(
        const Board& before,
        const Move& mv,
        bool is_king_move
    );
    void debug_check_incr_vs_full_after_unmake(
        const Board& board_with_move,
        const Move& mv,
        bool is_king_move
    );
    void debug_replay_feature_changes(const Board& before,
                                        const Move& mv,
                                        const Board& after);
    void debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after);
    bool check_active_features_consistency(const Accumulator& incr,
                                              const Accumulator& full,
                                              const char* name,
                                              bool abort_on_mismatch = true);

    void debug_check_features_after_move(const Board& b);
#endif

};


