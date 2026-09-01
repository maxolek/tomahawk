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
#include "simd.h"

// ============================================================
// Network Dimensions
// ============================================================

// bucketing
/*
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
constexpr int L1_SIZE = 1024;   // Hidden dimensions
constexpr int L2_SIZE = 16;
constexpr int L3_SIZE = 32;

// Quantisation factors used in training
constexpr int QA = 127; // L0 (feature-transformer) [must fit in screlu int32]
constexpr int QB = 64;  // L1 weights
constexpr int QC = 64;  // L2+3 weights
constexpr int SCALE = 400;
*/

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
    alignas(32) int16_t vals[L1_SIZE];   // pre-activation <--> post-weight_transform
    //std::unordered_set<int> active_features;

    inline void init_bias(const int16_t* bias) {
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), b);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] = bias[i];
#endif
        //active_features.clear();
    }

    inline void add_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            // load 8 int32 accumulator values
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            // load 8 int16 weights
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

            acc = _mm256_add_epi16(acc, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] += col[i];
#endif
        //active_features.insert(feature_idx);
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

            acc = _mm256_sub_epi32(acc, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] -= col[i];
#endif
        //active_features.erase(feature_idx);
    }

    // typical moves perform both of these
    // so combine to best utilize SIMD
    inline void add_sub_feature(int add_idx, int sub_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* add_col = W[add_idx];
        const int16_t* sub_col = W[sub_idx];

#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            __m256i add_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add_col + i));
            __m256i sub_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub_col + i));

            acc = _mm256_add_epi32(acc, add_w);
            acc = _mm256_sub_epi32(acc, sub_w);

            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else 
        add_feature(add_idx, W);
        remove_feature(sub_idx, W);
#endif
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

    // Cached accumulators
    // dual perspective
    // during tracking stm=white and ntm=black always
    // flipped appropriately during eval for [stm,ntm] actual [us/them] concat
    Accumulator acc_stm;
    Accumulator acc_ntm;

    // ========================================================
    // Helpers
    // ========================================================

    inline int32_t crelu(int32_t x, int32_t clamp_bound) const {
        return std::clamp<int32_t>(x, 0, clamp_bound);
    }

    template <typename T>
    inline T screlu(T x, T clamp_bound) const {
        T y = std::clamp<T>(x, 0, clamp_bound);
        return y * y;
    }

    // pairwise multiply WITHIN each accumulator
    // concat resulting 1/2-stm and 1/2-ntm accumulators
    inline void pairwise_mul(
        const int32_t* stm,
        const int32_t* ntm,
        int32_t* result
    ) {
        constexpr int HALF = L1_SIZE / 2;

        for (int i = 0; i < HALF; ++i) {
            result[i] = stm[i] * stm[i + HALF];
            result[i + HALF] = ntm[i] * ntm[i + HALF];
        }
    }

#ifdef _WIN32
    
#endif

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
