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
constexpr int L1_SIZE = 1024;   // Hidden dimensions
constexpr int L2_SIZE = 16;
constexpr int L3_SIZE = 32;

// Quantisation factors used in training
constexpr int QA = 127; // L0 (feature-transformer) [must fit in screlu int32]
constexpr int QB = 64;  // L1 weights
constexpr int QC = 64;  // L2+3 weights
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
    template <typename T>
    inline void pairwise_mul_simd(
        const T* stm,
        const T* ntm,
        int32_t* result
    ) {
        constexpr int HALF = L1_SIZE / 2;
        
        for (int i = 0; i < HALF; i += 16) {
            __m256i stm_a = _mm256_load_si256( reinterpret_cast<const __m256i*>(&stm[i]));
            __m256i stm_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&stm[i + HALF]));

            __m256i ntm_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ntm[i]));
            __m256i ntm_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&ntm[i + HALF]));

            __m256i stm_r = _mm256_mullo_epi16(stm_a, stm_b);
            __m256i ntm_r = _mm256_mullo_epi16(ntm_a, ntm_b);

            // Widen STM 16-bit products -> 32-bit
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(&result[i]),
                _mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(stm_r)
                )
            );
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(&result[i + 8]),
                _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(stm_r, 1)
                )
            );

            // Widen NTM 16-bit products -> 32-bit
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(&result[i + HALF]),
                _mm256_cvtepi16_epi32(
                    _mm256_castsi256_si128(ntm_r)
                )
            );
            _mm256_store_si256(
                reinterpret_cast<__m256i*>(&result[i + HALF + 8]),
                _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(ntm_r, 1)
                )
            );
        }
    }


    // horizontal sum int32
    // performs weight multplication for !! output layer !!
    inline int32_t hsum_epi32(__m256i v) {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        lo = _mm_add_epi32(lo, hi);

        __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
        lo = _mm_add_epi32(lo, shuf);

        shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(0, 1, 0, 1));
        lo = _mm_add_epi32(lo, shuf);

        return _mm_cvtsi128_si32(lo);
    }

    // horizontal sum of 4x int64
    inline int64_t hsum_epi64(__m256i v) {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i sum128 = _mm_add_epi64(lo, hi);
        __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
        __m128i sum64  = _mm_add_epi64(sum128, hi64);
        return _mm_cvtsi128_si64(sum64);
    }

    // dot product of int64 activations x int8 weights, exact (no overflow/truncation)
    // a[] must be non-negative (true for screlu output) and size must be a multiple of 4
    inline int64_t dot_i64_i8(const int64_t* a, const int8_t* w, int size) {
        __m256i acc = _mm256_setzero_si256();
        const __m256i mask_lo32 = _mm256_set1_epi64x(0xFFFFFFFFLL);
        const __m256i one       = _mm256_set1_epi64x(1);

        for (int i = 0; i < size; i += 4) {
            __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));

            // widen 4 int8 weights -> 4 int64 (sign-extended)
            __m128i w8   = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i)); // loads 8 bytes, only low 4 used
            __m256i w32  = _mm256_cvtepi8_epi32(w8);       // low 8 int8 -> 8 int32
            __m128i w32l = _mm256_castsi256_si128(w32);    // low 4 int32 == w[i..i+3]
            __m256i wv   = _mm256_cvtepi32_epi64(w32l);    // 4 int64, sign-extended

            // split a into 32-bit hi/lo halves (a >= 0 guaranteed, from screlu square)
            __m256i a_lo = _mm256_and_si256(av, mask_lo32);
            __m256i a_hi = _mm256_srli_epi64(av, 32);

            // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane as SIGNED.
            // a_lo's true value is unsigned (0..2^32-1); if its top bit is set,
            // the signed read is (a_lo - 2^32), so we carry a +1 into a_hi to
            // compensate: a = (a_hi + carry)*2^32 + a_lo_signed, exactly.
            __m256i carry    = _mm256_and_si256(_mm256_srli_epi64(av, 31), one);
            __m256i a_hi_adj = _mm256_add_epi64(a_hi, carry);

            __m256i lo_prod = _mm256_mul_epi32(a_lo, wv);      // a_lo_signed * w  (exact 64-bit)
            __m256i hi_prod = _mm256_mul_epi32(a_hi_adj, wv);  // (a_hi + carry) * w

            __m256i prod = _mm256_add_epi64(lo_prod, _mm256_slli_epi64(hi_prod, 32));
            acc = _mm256_add_epi64(acc, prod);
        }

        return hsum_epi64(acc);
    }

    // dot product of int32 activations x int8 weights, widened to int64 (no overflow)
    inline int64_t dot_i32_i8_widen(const int32_t* a, const int8_t* w, int size) {
        __m256i acc_lo = _mm256_setzero_si256();  // holds even-indexed lane products (0,2,4,6)
        __m256i acc_hi = _mm256_setzero_si256();  // holds odd-indexed lane products (1,3,5,7)

        for (int i = 0; i < size; i += 8) {
            __m256i av  = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
            __m128i w8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i));
            __m256i wv  = _mm256_cvtepi8_epi32(w8);

            // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane, signed,
            // and produces a true 64-bit product -- widening, no overflow.
            __m256i lo = _mm256_mul_epi32(av, wv);                                     // lanes 0,2,4,6
            __m256i hi = _mm256_mul_epi32(_mm256_srli_si256(av, 4), _mm256_srli_si256(wv, 4)); // lanes 1,3,5,7

            acc_lo = _mm256_add_epi64(acc_lo, lo);
            acc_hi = _mm256_add_epi64(acc_hi, hi);
        }

        return hsum_epi64(acc_lo) + hsum_epi64(acc_hi);
    }

    inline void activate_crelu(const int16_t* in, int16_t* out, int size, int QA) {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i qa   = _mm256_set1_epi16(QA);
        for (int i = 0; i < size; i += 16) {
            __m256i v       = _mm256_load_si256((const __m256i*)&in[i]);
            __m256i clipped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
            _mm256_store_si256((__m256i*)&out[i], clipped);
        }
    }

    // fold SCRELU into multiply-add
    inline void activate_screlu32(const int32_t* in, int32_t* out, int size, int QA) {
        // TYPE = int32 or int64
        // handled the same (int16 is different)
        const __m256i zero = _mm256_setzero_si256();
        const __m256i qa   = _mm256_set1_epi32(QA);

        for (int i = 0; i < size; i += 8) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));

            v = _mm256_max_epi32(v, zero);
            v = _mm256_min_epi32(v, qa);
            v = _mm256_mullo_epi32(v, v);

            _mm256_store_si256(reinterpret_cast<__m256i*>(out + i), v);
        }
    }

    inline void activate_screlu64(const int32_t* in, int64_t* out, int size, int32_t clamp_bound) {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i qa   = _mm256_set1_epi32(clamp_bound);

        for (int i = 0; i < size; i += 8) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
            v = _mm256_max_epi32(v, zero);
            v = _mm256_min_epi32(v, qa);          // clamp in 32-bit, values are small enough here

            // widening square: 32x32 -> 64, 4 lanes at a time, done twice for 8 int32 inputs
            __m256i lo = _mm256_mul_epi32(v, v);                                    // squares elements 0,2,4,6 -> 64-bit
            __m256i hi = _mm256_mul_epi32(_mm256_srli_si256(v, 4), _mm256_srli_si256(v, 4)); // elements 1,3,5,7

            // interleave lo/hi back into correct order for a contiguous int64 output
            _mm256_store_si256(reinterpret_cast<__m256i*>(out + i),     _mm256_unpacklo_epi64(lo, hi));
            _mm256_store_si256(reinterpret_cast<__m256i*>(out + i + 4), _mm256_unpackhi_epi64(lo, hi));
        }
    }
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
