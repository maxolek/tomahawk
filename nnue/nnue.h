#pragma once
#include <vector>
#include <cstdint>
#include <string>

#ifdef DEBUG
    #include <unordered_set>
#endif

#include "board.h"
#include "move.h"
#include "search_stats.h"
#include "timer.h"
#include "simd.h"
#include "network.h"
#include "accumulator.h"
#include "utils.h"

#include "network.h"
#include "accumulator.h"

// ============================================================
// Network
// ============================================================

class NNUE {
public:
    // Cached accumulators
    // dual perspective
    // during tracking stm=white and ntm=black always
    // flipped appropriately during eval for [stm,ntm] actual [us/them] concat
    //      
    // weights below are read-only after load() and shared by every search thread;
    // each thread keeps its incremental state in an AccumulatorStack
    // acc_state is the stack used by the overloads without a stack argument
    // (the main search thread and diagnostics); acc_stm()/acc_ntm() are its current ply

    AccumulatorStack acc_state;
    Accumulator& acc_stm() { return acc_state.white(); };
    Accumulator& acc_ntm() { return acc_state.black(); };

    // ----------- SMALL NET 1024x2 -----------

    alignas(VEC_ALIGN) int16_t l1w_small[NUM_OUTPUT_BUCKETS][2*hl_size];
    alignas(VEC_ALIGN) int16_t l1b_small[NUM_OUTPUT_BUCKETS];

    // ----------- BIG NET 1024-16-32 ---------------
  
    // ========== L0: 768xINPUT_BUCKETS → 512 ==========
    // Stored column-major: W0[feature][hidden]
    alignas(VEC_ALIGN) int16_t l0w[INPUT_SIZE][L1_SIZE];
    alignas(VEC_ALIGN) int16_t l0b[L1_SIZE];

    // ========== L1: 512 → 8x16 ==========
    // Dual-perspective: [stm_hidden, ntm_hidden]
    alignas(VEC_ALIGN) int8_t l1w[L2_SIZE * NUM_OUTPUT_BUCKETS][L1_SIZE]; // 2*L0_SIZE if not using pairwise multiply
    alignas(VEC_ALIGN) int32_t l1b[L2_SIZE * NUM_OUTPUT_BUCKETS];          // otherwise the concat is reduced back down to L0_SIZE

    // ========== L2: 16 → 8x32 ==========
    alignas(VEC_ALIGN) int8_t l2w[L3_SIZE * NUM_OUTPUT_BUCKETS][L2_SIZE];
    alignas(VEC_ALIGN) int32_t l2b[L3_SIZE * NUM_OUTPUT_BUCKETS];

    // ========== L3: 32 → NUM_OUTPUT_BUCKETS ==========
    alignas(VEC_ALIGN) int8_t l3w[NUM_OUTPUT_BUCKETS][L3_SIZE];
    alignas(VEC_ALIGN) int32_t l3b[NUM_OUTPUT_BUCKETS];

    // ---- inference layouts derived from the weights above at load time ----
    // L1 by input pair: [bucket][pair k][output o][2] = l1w[o][2k], l1w[o][2k+1],
    // so one broadcast input pair multiplies against all 16 outputs at once.
    alignas(64) int16_t l1w_pairs[NUM_OUTPUT_BUCKETS][L1_SIZE / 2][2 * L2_SIZE];
    // L2 by input: [bucket][input][output] sign-extended to 64-bit lanes.
    alignas(64) int64_t l2w_by_input[NUM_OUTPUT_BUCKETS][L2_SIZE][L3_SIZE];
    // L3 widened to 32 bits: [bucket][input].
    alignas(64) int32_t l3w_i32[NUM_OUTPUT_BUCKETS][L3_SIZE];
    void build_inference_layouts();


    // constructors 
    NNUE() {};
    NNUE(const fs::path& path) { load(path); };

    // Load quantised network
    bool load(const fs::path& path);
    bool loadSmall(const fs::path& path);

    // Compute final output from accumulators
    int evaluate(bool is_white_move, U64 occ) { return evaluate(acc_state, is_white_move, occ); }
    int evaluate(const AccumulatorStack& stack, bool is_white_move, U64 occ) const;
    int eval_simd(bool is_white_move, U64 occ) { return eval_simd(acc_state, is_white_move, occ); }
    int eval_simd(const AccumulatorStack& stack, bool is_white_move, U64 occ) const;
    int full_eval(const Board& b);
    // Direct SCReLU -> output forward passes, available for alternate heads.
    // Supply decoded output-major weights and matching quantization; the active
    // loader and multi-layer evaluation do not populate or select this head.
    int eval_screlu(bool is_white_move, U64 occ);
    int eval_screlu_simd(bool is_white_move, U64 occ);

    // Accumulator stack (search). Every push is undone by stack.pop().
    //   move_delta + push_delta: incremental update; the delta is computed from the
    //                   pre-move board (so the board can be updated in between)
    //   push_king_move: the mover's king changed input bucket or mirror; board in
    //                   its post-move state. Refreshes the mover's perspective and
    //                   updates the other one incrementally (castling refreshes both).
    struct AccumulatorDelta { int add[2]; int sub[2]; int cap[2]; }; // [white, black]; cap -1 = none
    void reset_stack(const Board& b) { reset_stack(acc_state, b); }
    void reset_stack(AccumulatorStack& stack, const Board& b) const;
    AccumulatorDelta move_delta(const Board& before, const Move& mv) const;
    void push_delta(AccumulatorStack& stack, const AccumulatorDelta& delta) const;
    void push_king_move(AccumulatorStack& stack, const Board& after, const Move& mv) const;


    // Incremental updates for search
    void on_make_move_halfka(const Board& board, const Move& mv);
    void on_unmake_move_halfka(const Board& board, const Move& mv);

    // Build full accumulators from board
    void build_halfka_accumulators(const Board& b) { build_halfka_accumulators(b, acc_state); };
    void build_halfka_accumulators(AccumulatorStack& stack, const Board& b) const;
    // rebuild a single perspective (0 = white, 1 = black) of the current ply
    void refresh_perspective(AccumulatorStack& stack, const Board& b, int perspective) const;

    // debugging
#ifdef DEBUG
    void debug_simd(const Board& b);
    void debug_acc_full(const Accumulator& acc, const std::string& name) const;
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


