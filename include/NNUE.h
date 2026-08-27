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

// bucketing
constexpr int NUM_INPUT_BUCKETS = 10;
constexpr int NUM_OUTPUT_BUCKETS = 8;
constexpr int OUTPUT_BUCKET_DIVISOR = (32 + NUM_OUTPUT_BUCKETS - 1) / NUM_OUTPUT_BUCKETS;

constexpr int FILE_GROUP[8] = {0, 1, 2, 3, 3, 2, 1, 0};
constexpr int BUCKET_LAYOUT[32] = {
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9
};

constexpr int INPUT_SIZE  = 768 * NUM_INPUT_BUCKETS;   // (chessbucketgsmirrored) 10 horizontal king buckets
constexpr int HIDDEN_SIZE = 1024;   // Hidden dimension

// Quantisation factors used in training
constexpr int QA = 255;
constexpr int QB = 64;
constexpr int SCALE = 400;

// ============================================================
// external exposure funcs +++ helpera
// ============================================================

inline int other_color(int c) { return c ^ 1; }

inline int mirrored_flip(int king_sq) {
    return (king_sq % 8 > 3) ? 7 : 0;
}

inline int mirrored_bucket(int king_sq) {
    const int rank = king_sq / 8;
    const int file_group = FILE_GROUP[king_sq % 8];

    return BUCKET_LAYOUT[rank * 4 + file_group];
}

inline int output_bucket(U64 occ) {
    return (countBits(occ) - 2) / OUTPUT_BUCKET_DIVISOR; // -2 for kings
}

// ============================================================
// Accumulator: holds hidden activations BEFORE SCReLU
// ============================================================
struct Accumulator {
    int32_t vals[HIDDEN_SIZE];   // pre-activation <--> post-weight_transform
    //std::unordered_set<int> active_features;

    inline void init_bias(const int16_t* bias) {
        for (int i = 0; i < HIDDEN_SIZE; i += 8) {
            __m256i b32 = _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + i))
            );
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(vals + i), b32);
        }
        
        //for (int i = 0; i < HIDDEN_SIZE; i++)
        //    vals[i] = bias[i];
        //active_features.clear();
    }

    inline void add_feature(int feature_idx, int16_t (*W)[HIDDEN_SIZE]) {
        const int16_t* col = W[feature_idx];

        for (int i = 0; i < HIDDEN_SIZE; i += 8) {
            // load 8 int32 accumulator values
            __m256i acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(vals + i)
            );
            // load 8 int16 weights, widen to int32 inline
            __m256i w32 = _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col + i))
            );

            acc = _mm256_add_epi32(acc, w32);
            
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(vals + i), acc
            );
        }

        //for (int i = 0; i < HIDDEN_SIZE; i++)
        //    vals[i] += col[i];
        //active_features.insert(feature_idx);
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[HIDDEN_SIZE]) {
        const int16_t* col = W[feature_idx];
        
        for (int i = 0; i < HIDDEN_SIZE; i += 8) {
            __m256i acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(vals + i)
            );
            __m256i w32 = _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col + i))
            );

            acc = _mm256_sub_epi32(acc, w32);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(vals + i), acc
            );
        }
        
        //for (int i = 0; i < HIDDEN_SIZE; i++)
        //    vals[i] -= col[i];
        //active_features.erase(feature_idx);
    }

    // typical moves perform both of these
    // so combine to best utilize SIMD
    inline void add_sub_feature(int add_idx, int sub_idx, int16_t (*W)[HIDDEN_SIZE]) {
        const int16_t* add_col = W[add_idx];
        const int16_t* sub_col = W[sub_idx];

        for (int i = 0; i < HIDDEN_SIZE; i += 8) {
            __m256i acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(vals + i)
            );
            __m256i add_w = _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(add_col + i))
            );
            __m256i sub_w = _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(sub_col + i))
            );

            acc = _mm256_add_epi32(acc, add_w);
            acc = _mm256_sub_epi32(acc, sub_w);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
    }

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
    int evaluate(bool is_white_move, U64 occ); 
    int eval_simd(bool is_white_move, U64 occ);
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
    int16_t l1w[NUM_OUTPUT_BUCKETS][2 * HIDDEN_SIZE];
    int16_t l1b[NUM_OUTPUT_BUCKETS];

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
    
};
