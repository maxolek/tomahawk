#pragma once
#include <vector>
#include <cstdint>
#include <string>
#ifndef NDEBUG
#include <unordered_set>
#endif
#include "board.h"
#include "move.h"
#include "stats.h"
#include "timer.h"

// ============================================================
// Network Dimensions
// ============================================================

constexpr int INPUT_SIZE  = 768*64;   // Chess768 features 64*12 -- sq*piece*color (+ sq)
constexpr int HIDDEN_SIZE = 1024;   // Hidden dimension

// bucketing
constexpr int FILE_GROUP[8] = {0, 1, 2, 3, 3, 2, 1, 0};

// Quantisation factors used in training
constexpr int QA = 255;
constexpr int QB = 64;
constexpr int SCALE = 400;

// ============================================================
// Accumulator: holds hidden activations BEFORE SCReLU
// ============================================================

struct Accumulator {
    int32_t vals[HIDDEN_SIZE];   // pre-activation
    //std::unordered_set<int> active_features;

    void init_bias(const int16_t* bias) {
        for (int i = 0; i < HIDDEN_SIZE; i++)
            vals[i] = bias[i];
        //active_features.clear();
    }

    inline void add_feature(int feature_idx, int16_t (*W)[HIDDEN_SIZE]) {
        const int16_t* col = W[feature_idx];
        for (int i = 0; i < HIDDEN_SIZE; i++)
            vals[i] += col[i];
        //active_features.insert(feature_idx);
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[HIDDEN_SIZE]) {
        const int16_t* col = W[feature_idx];
        for (int i = 0; i < HIDDEN_SIZE; i++)
            vals[i] -= col[i];
        //active_features.erase(feature_idx);
    }

    /*
    inline void get_bucket_and_flip(int king_sq, int& bucket, int& flip) {
        int file = king_sq % 8;
        int rank_group = king_sq / 8;
        flip = (file > 3) ? 7 : 0;
        bucket = BUCKET_LAYOUT[rank_group * 4 + FILE_GROUP[file]];
    }

    inline int feature_index_stm(int sq, int piece, int color, int bucket) {
        int file = white_king_sq % 8;
        int flip = (file > 3) ? 7 : 0;
        int sq_m = sq ^ flip;
        return 768*bucket + (color==0 ? 0:384) + piece*64 + sq_m;
    }

    inline int feature_index_ntm(int sq, int piece, int color, int bucket) {
        int black_king_canon = black_king_sq ^ 56;
        int file = black_king_canon % 8;          // == black_king_sq % 8, ^56 doesn't touch file bits
        int flip = (file > 3) ? 7 : 0;
        int sq_m = (sq ^ 56) ^ flip;
        return 768*bucket + (color==0 ? 384:0) + piece*64 + sq_m;
    }
    */

    /*
    void dump_active_features(const char* name) const {
        std::cout << "[ACTIVE FEATURES] " << name << " count=" << active_features.size() << "\n";
        int count = 0;
        for (int f : active_features) {
            std::cout << f << " ";
            if (++count % 16 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }
    */
};

// ============================================================
// Network
// ============================================================

class NNUE {
public:
    NNUE() {};
    NNUE(const fs::path& path) { load(path); };

    // Load quantised network
    bool load(const fs::path& path);

    // Compute final output from accumulators
    int evaluate(bool is_white_move); 
    int eval_simd(bool is_white_move);
    int full_eval(const Board& b);

    // Incremental updates for search
    void on_make_move(const Board& board, const Move& mv);
    void on_make_move_halfka(const Board& board, const Move& mv);
    void on_unmake_move(const Board& board, const Move& mv);
    void on_unmake_move_halfka(const Board& board, const Move& mv);

    // Build full accumulators from board
    void build_accumulators(const Board& b);
    void build_halfka_accumulators(const Board& b);

    // ========== L0: 768 → 128 ==========
    // Stored column-major: W0[feature][hidden]
    int16_t l0w[INPUT_SIZE][HIDDEN_SIZE];
    int16_t l0b[HIDDEN_SIZE];

    // ========== L1: 2*128 → 1 ==========
    // Dual-perspective: [stm_hidden, ntm_hidden]
    int16_t l1w[2 * HIDDEN_SIZE];
    int16_t l1b;

    // Cached accumulators
    // dual perspective
    // during tracking stm=white and ntm=black always
    // flipped appropriately during eval for [stm,ntm] actual [us/them] concat
    Accumulator acc_stm;
    Accumulator acc_ntm;

    // ========================================================
    // Helpers
    // ========================================================

    inline int32_t screlu(int32_t x) const {
        int32_t y = std::clamp<int32_t>(x, 0, QA);
        return y * y;
    }

    // debugging
    void debug_simd(const Board& b);
    //void debug_acc(const Accumulator& acc, const std::string& name) const;
    void debug_acc_full(const Accumulator& acc, const std::string& name) const;
    //void debug_evaluate(const Accumulator& us, const Accumulator& them) const;
    //void debug_on_move(const std::string& name, const Move& mv, int color, int moved_piece,
    //                     int f_from, int f_to) const;
    //void on_make_move_debug(const Board& before, const Move& mv);
    //void on_unmake_move_debug(const Board& board, const Move& mv);
    //int evaluate_debug(bool is_white_move) const;
    void debug_check_incr_vs_full_after_make(const Board& before, const Move& mv, NNUE& nnue);
    void debug_check_incr_vs_full_after_unmake(const Board& board_with_move, const Move& mv, NNUE& nnue);
    void debug_replay_feature_changes(const Board& before,
                                        const Move& mv,
                                        const Board& after);
    void debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after);
/*
    bool check_active_features_consistency(const Accumulator& incr,
                                              const Accumulator& full,
                                              const char* name,
                                              bool abort_on_mismatch = true);
    void debug_check_features_after_move(const Board& b);
*/
    
};
