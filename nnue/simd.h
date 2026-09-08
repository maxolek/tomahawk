#ifndef SIMD_H
#define SIMD_H

#include "helpers.h"
#include "network.h"
#include "simd_defs.h"

#ifdef _WIN32

// --------- Accumulator -------------

inline void init_bias_simd(const int16_t* bias, int16_t* vals) {
    alignas(VEC_ALIGN) __m256i b[L1_SIZE];
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i b = vec_load(bias + i);
        vec_store(vals + i, b);
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
    alignas(VEC_ALIGN) __m256i acc[L1_SIZE];
    alignas(VEC_ALIGN) __m256i w[L1_SIZE];
    for (int i = 0; i < L1_SIZE; i += 16) {
        // load 16 int16 accumulator + weight values
        __m256i acc = vec_load(vals + i);
        __m256i w = vec_load(col + i);

        // forward pass weight layer
        acc = vec_add16(acc, w);
        vec_store(vals + i, acc);
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
    alignas(VEC_ALIGN) __m256i acc[L1_SIZE];
    alignas(VEC_ALIGN) __m256i w[L1_SIZE];
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i acc = vec_load(vals + i);
        __m256i w = vec_load(col + i);

        acc = vec_sub16(acc, w);
        vec_store(vals + i, acc);
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
    alignas(VEC_ALIGN) __m256i acc[L1_SIZE];
    alignas(VEC_ALIGN) __m256i add_w[L1_SIZE];
    alignas(VEC_ALIGN) __m256i sub_w[L1_SIZE];
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i acc = vec_load(vals + i);
        __m256i add_w = vec_load(add_col + i);
        __m256i sub_w = vec_load(sub_col + i);

        acc = vec_add16(acc, add_w);
        acc = vec_sub16(acc, sub_w);

        vec_store(vals + i, acc);
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

inline void pairwise_mul_simd(
    int16_t* stm,
    int16_t* ntm,
    int32_t* result
) {
    constexpr int HALF = L1_SIZE / 2;

    alignas(VEC_ALIGN) __m256i stm_a[L1_SIZE];
    alignas(VEC_ALIGN) __m256i stm_b[L1_SIZE];
    alignas(VEC_ALIGN) __m256i stm_r[L1_SIZE];
    alignas(VEC_ALIGN) __m256i ntm_a[L1_SIZE];
    alignas(VEC_ALIGN) __m256i ntm_b[L1_SIZE];
    alignas(VEC_ALIGN) __m256i ntm_r[L1_SIZE];
    
    for (int i = 0; i < HALF; i += 16) {
        __m256i stm_a = vec_load(stm[i]);
        __m256i stm_b = vec_load(stm[i + HALF]);

        __m256i ntm_a = vec_load(ntm[i]);
        __m256i ntm_b = vec_load(ntm[i + HALF]);

        __m256i stm_r = vec_mullo16(stm_a, stm_b);
        __m256i ntm_r = vec_mullo16(ntm_a, ntm_b);

        // Widen STM 16-bit products -> 32-bit
        vec_store(
            result[i],
            vec_convert_16_32<__m256i>(
                vec_cast_256_128<__m128i>(stm_r)
            )
        );
        vec_store(
            result[i + 8],
            vec_convert_16_32<__m256i>(
                vec_extract_256_128<__m128i>(stm_r, 1)
            )
        );

        // Widen NTM 16-bit products -> 32-bit
        vec_store(
            result[i + HALF],
            vec_convert_16_32<__m256i>(
                vec_cast_256_128<__m128i>(ntm_r)
            )
        );
        vec_store(
            result[i + HALF + 8],
            vec_convert_16_32<__m256i>(
                vec_extract_256_128<__m128i>(ntm_r, 1)
            )
        );
    }
}

// ----------- sum accumulation -------------

// horizontal sum int32
// performs weight multplication for !! output layer !!
inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = vec_cast_256_128<__m128i>(v);
    __m128i hi = vec_extract_256_128<__m128i>(v, 1);
    lo = vec_add32(lo, hi);

    __m128i shuf = vec_shuffle32<__m128i>(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = vec_add32(lo, shuf);

    shuf = vec_shuffle32<__m128i>(lo, _MM_SHUFFLE(0, 1, 0, 1));
    lo = vec_add32(lo, shuf);
    return vec_convert_si128_32<__m128i>(lo);
}

// horizontal sum of 4x int64
inline int64_t hsum_epi64(__m256i v) {
    __m128i lo = vec_cast_256_128<__m128i>(v);
    __m128i hi = vec_extract_256_128<__m128i>(v, 1);
    __m128i sum128 = vec_add64(lo, hi);
    __m128i hi64   = vec_unpackhi64(sum128, sum128);
    __m128i sum64  = vec_add64(sum128, hi64);
    return vec_convert_si128_64<__m128i>(sum64);
}

// -------------- weight transforms -------------

// dot product of int64 activations x int8 weights, exact (no overflow/truncation)
// a[] must be non-negative (true for screlu output) and size must be a multiple of 4
inline int64_t dot_i64_i8(const int64_t* a, const int8_t* w, int size) {
    __m256i acc = zeros256;
    const __m256i one       = vec_set_64<__m256i>(1);

    for (int i = 0; i < size; i += 4) {
        __m256i av = vec_load(reinterpret_cast<const __m256i*>(a + i));

        // widen 4 int8 weights -> 4 int64 (sign-extended)
        __m128i w8   = vec_load(reinterpret_cast<const __m128i*>(w + i)); // loads 8 bytes, only low 4 used
        __m256i w32  = vec_convert_8_32(w8);       // low 8 int8 -> 8 int32
        __m128i w32l = vec_cast_256_128(w32);    // low 4 int32 == w[i..i+3]
        __m256i wv   = vec_convert_32_64(w32l);    // 4 int64, sign-extended

        // split a into 32-bit hi/lo halves (a >= 0 guaranteed, from screlu square)
        __m256i a_lo = vec_lo_half(av);
        __m256i a_hi = vec_hi_half(av);

        // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane as SIGNED.
        // a_lo's true value is unsigned (0..2^32-1); if its top bit is set,
        // the signed read is (a_lo - 2^32), so we carry a +1 into a_hi to
        // compensate: a = (a_hi + carry)*2^32 + a_lo_signed, exactly.
        __m256i carry    = _mm256_and_si256(_mm256_srli_epi64(av, 31), one);
        __m256i a_hi_adj = vec_add64(a_hi, carry);

        __m256i lo_prod = vec_mul32(a_lo, wv);      // a_lo_signed * w  (exact 64-bit)
        __m256i hi_prod = vec_mul32(a_hi_adj, wv);  // (a_hi + carry) * w

        __m256i prod = vec_add64(lo_prod, vec_slli64(hi_prod, 32));
        acc = vec_add64(acc, prod);
    }

    return hsum_epi64(acc);
}

// dot product of int32 activations x int8 weights, widened to int64 (no overflow)
inline int64_t dot_i32_i8_widen(const int32_t* a, const int8_t* w, int size) {
    __m256i acc_lo = zeros256;  // holds even-indexed lane products (0,2,4,6)
    __m256i acc_hi = zeros256;  // holds odd-indexed lane products (1,3,5,7)

    for (int i = 0; i < size; i += 8) {
        __m256i av  = vec_load(a + i);
        __m128i w8  = vec_load(reinterpret_cast<const __m128i*>(w + i));
        __m256i wv  = vec_convert_8_32<__m256i>(w8);

        // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane, signed,
        // and produces a true 64-bit product -- widening, no overflow.
        __m256i lo = vec_mul32<__m256i>(av, wv);                                     // lanes 0,2,4,6
        __m256i hi = vec_mul32<__m256i>(_mm256_srli_si256(av, 4), _mm256_srli_si256(wv, 4)); // lanes 1,3,5,7

        acc_lo = vec_add64<__m256i>(acc_lo, lo);
        acc_hi = vec_add64<__m256i>(acc_hi, hi);
    }

    return hsum_epi64(acc_lo) + hsum_epi64(acc_hi);
}

// ----------- activations -------------

inline void activate_crelu(const int16_t* in, int16_t* out, int size, int QA) {
    const __m256i zero = zeros256;
    const __m256i qa   = vec_set_16<__m256i>(QA);
    for (int i = 0; i < size; i += 16) {
        __m256i v       = vec_load(in + i);
        __m256i clipped = vec_clamp16(v, zero, qa);
        vec_store(out + i, clipped);
    }
}

// fold SCRELU into multiply-add
inline void activate_screlu32(const int32_t* in, int32_t* out, int size, int QA) {
    // TYPE = int32 or int64
    // handled the same (int16 is different)
    const __m256i zero = zeros256;
    const __m256i qa   = vec_set_32<__m256i>(QA);

    for (int i = 0; i < size; i += 8) {
        __m256i v = vec_load(reinterpret_cast<const __m256i*>(in + i));
        v = vec_clamp32(v, zero, qa);
        v = vec_mullo32(v, v);

        vec_store(out + i, v);
    }
}

inline void activate_screlu64(const int32_t* in, int64_t* out, int size, int32_t clamp_bound) {
    const __m256i zero = zeros256;
    const __m256i qa   = vec_set_32<__m256i>(clamp_bound);

    for (int i = 0; i < size; i += 8) {
        __m256i v = vec_load(in + i);
        v = vec_clamp32(v, zero, qa);         // clamp in 32-bit, values are small enough here

        // widening square: 32x32 -> 64, 4 lanes at a time, done twice for 8 int32 inputs
        __m256i lo = vec_mul32(v, v);                                    // squares elements 0,2,4,6 -> 64-bit
        __m256i hi = vec_mul32(_mm256_srli_si256(v, 4), _mm256_srli_si256(v, 4)); // elements 1,3,5,7
        // unpacklo/hi only interleave within each 128-bit half:
        //   u_lo = [v0²,v1²,v4²,v5²]   u_hi = [v2²,v3²,v6²,v7²]
        __m256i u_lo = vec_unpacklo64(u_lo, u_hi);
        __m256i u_hi = vec_unpackhi64(u_lo, u_hi);

        // reassemble across the 128-bit boundary to get true sequential order
        __m256i out_lo = vec_permute2x128(u_lo, u_hi, 0x20); // v0²,v1²,v2²,v3²
        __m256i out_hi = vec_permute2x128(u_lo, u_hi, 0x31); // v4²,v5²,v6²,v7²
        
        vec_store(out + i, out_lo);
        vec_store(out + i + 4, out_hi);
    }
}

#endif // _WIN32

#endif // SIMD_H