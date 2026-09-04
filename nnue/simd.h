#ifndef SIMD_H
#define SIMD_H

#include "helpers.h"


// ============================================================
// Network Dimensions
// ============================================================

// stored here for circular inclusion reasons

// ------------ big net --------------

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

/* --------- SMALL NET -------------------

    the simd for this is also not as tightly optimized as the larger net
    _smallnet suffixes instead of _simd 

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
*/


// --------- Accumulator -------------

inline void init_bias_simd(const int16_t* bias, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), b);
    }
}

/*
inline void init_bias_smallnet(const int16_t* bias, int32_t* vals) {
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        __m256i b32 = _mm256_cvtepi16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + i))
        );
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(vals + i), b32);
    }
*/

inline void add_feature_simd(const int16_t* col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        // load 8 int32 accumulator values
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
        // load 8 int16 weights
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

        acc = _mm256_add_epi16(acc, w);
        _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
    }
}

/*
inline void add_feature_smallnet(const int16_t* col, int32_t* vals) {
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
}
*/

inline void remove_feature_simd(const int16_t* col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

        acc = _mm256_sub_epi16(acc, w);
        _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
    }
}

/*
inline void remove_feature_smallnet(const int16_t* col, int32_t* vals) {
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
}
*/

inline void add_sub_feature_simd(const int16_t* add_col, const int16_t* sub_col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
        __m256i add_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add_col + i));
        __m256i sub_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub_col + i));

        acc = _mm256_add_epi16(acc, add_w);
        acc = _mm256_sub_epi16(acc, sub_w);

        _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
    }
}

/*
inline void add_sub_feature_smallnet(const int16_t* add_col, const int16_t* sub_col, int16_t* vals) {
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
*/


// ---------- feature transformer -----------

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

// ----------- sum accumulation -------------

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

// -------------- weight transforms -------------

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

// ----------- activations -------------

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

        // unpacklo/hi only interleave within each 128-bit half:
        //   u_lo = [v0²,v1²,v4²,v5²]   u_hi = [v2²,v3²,v6²,v7²]
        __m256i u_lo = _mm256_unpacklo_epi64(lo, hi);
        __m256i u_hi = _mm256_unpackhi_epi64(lo, hi);

        // reassemble across the 128-bit boundary to get true sequential order
        __m256i out_lo = _mm256_permute2x128_si256(u_lo, u_hi, 0x20); // v0²,v1²,v2²,v3²
        __m256i out_hi = _mm256_permute2x128_si256(u_lo, u_hi, 0x31); // v4²,v5²,v6²,v7²
        
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + i),     out_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + i + 4), out_hi);
    }
}

#endif // SIMD_H