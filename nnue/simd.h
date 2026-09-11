#ifndef SIMD_H
#define SIMD_H

#include "helpers.h"
#include "network.h"
#include "simd_defs.h"

// --------- Accumulator -------------

inline void init_bias_simd(const int16_t* bias, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        vec256_t b = vec_load<int16_t, vec256_t>(bias + i);
        vec_store<vec256_t, int16_t>(vals + i, b);
    }
}

inline void init_bias_simd(const int16_t* bias, int32_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 8) {
        const auto b = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(bias + i));
        vec_store<vec256_t, int32_t>(vals + i, b);
    }
}

inline void add_feature_simd(const int16_t* col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        // load 16 int16 accumulator + weight values
        vec256_t acc = vec_load<int16_t, vec256_t>(vals + i);
        vec256_t w = vec_load<int16_t, vec256_t>(col + i);

        // forward pass weight layer
        acc = vec_add16<vec256_t>(acc, w);
        vec_store<vec256_t, int16_t>(vals + i, acc);
    }
}

inline void add_feature_simd(const int16_t* col, int32_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 8) {
        auto acc = vec_load<int32_t, vec256_t>(vals + i);
        const auto w = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(col + i));
        vec_store<vec256_t, int32_t>(vals + i, vec_add32(acc, w));
    }
}

inline void remove_feature_simd(const int16_t* col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        vec256_t acc = vec_load<int16_t, vec256_t>(vals + i);
        vec256_t w = vec_load<int16_t, vec256_t>(col + i);

        acc = vec_sub16<vec256_t>(acc, w);
        vec_store<vec256_t, int16_t>(vals + i, acc);
    }
}

inline void remove_feature_simd(const int16_t* col, int32_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 8) {
        auto acc = vec_load<int32_t, vec256_t>(vals + i);
        const auto w = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(col + i));
        vec_store<vec256_t, int32_t>(vals + i, vec_sub32(acc, w));
    }
}

inline void add_sub_feature_simd(const int16_t* add_col, const int16_t* sub_col, int16_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        vec256_t acc = vec_load<int16_t, vec256_t>(vals + i);
        vec256_t add_w = vec_load<int16_t, vec256_t>(add_col + i);
        vec256_t sub_w = vec_load<int16_t, vec256_t>(sub_col + i);

        acc = vec_add16<vec256_t>(acc, add_w);
        acc = vec_sub16<vec256_t>(acc, sub_w);

        vec_store<vec256_t, int16_t>(vals + i, acc);
    }
}

inline void add_sub_feature_simd(const int16_t* add_col, const int16_t* sub_col, int32_t* vals) {
    for (int i = 0; i < L1_SIZE; i += 8) {
        auto acc = vec_load<int32_t, vec256_t>(vals + i);
        const auto add = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(add_col + i));
        const auto sub = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(sub_col + i));
        vec_store<vec256_t, int32_t>(vals + i, vec_sub32(vec_add32(acc, add), sub));
    }
}


// ---------- feature transformer -----------

inline void pairwise_mul_simd(
    int16_t* stm,
    int16_t* ntm,
    int32_t* result
) {
    constexpr int HALF = L1_SIZE / 2;
    
    for (int i = 0; i < HALF; i += 16) {
        vec256_t stm_a = vec_load<int16_t, vec256_t>(&stm[i]);
        vec256_t stm_b = vec_load<int16_t, vec256_t>(&stm[i + HALF]);

        vec256_t ntm_a = vec_load<int16_t, vec256_t>(&ntm[i]);
        vec256_t ntm_b = vec_load<int16_t, vec256_t>(&ntm[i + HALF]);

        vec256_t stm_r = vec_mullo16<vec256_t>(stm_a, stm_b);
        vec256_t ntm_r = vec_mullo16<vec256_t>(ntm_a, ntm_b);

        // Widen STM 16-bit products -> 32-bit
        vec_store<vec256_t, int32_t>(
            &result[i],
            vec_convert_16_32<vec128_t, vec256_t>(
                vec_cast_256_128<vec256_t, vec128_t>(stm_r)
            )
        );
        vec_store<vec256_t, int32_t>(
            &result[i + 8],
            vec_convert_16_32<vec128_t, vec256_t>(
                vec_extract_256_128<vec256_t, vec128_t>(stm_r, 1)
            )
        );

        // Widen NTM 16-bit products -> 32-bit
        vec_store<vec256_t, int32_t>(
            &result[i + HALF],
            vec_convert_16_32<vec128_t, vec256_t>(
                vec_cast_256_128<vec256_t, vec128_t>(ntm_r)
            )
        );
        vec_store<vec256_t, int32_t>(
            &result[i + HALF + 8],
            vec_convert_16_32<vec128_t, vec256_t>(
                vec_extract_256_128<vec256_t, vec128_t>(ntm_r, 1)
            )
        );
    }
}

// ----------- sum accumulation -------------

// horizontal sum int32
// performs weight multplication for !! output layer !!
inline int32_t hsum_epi32(vec256_t v) {
    vec128_t lo = vec_cast_256_128<vec256_t, vec128_t>(v);
    vec128_t hi = vec_extract_256_128<vec256_t, vec128_t>(v, 1);
    lo = vec_add32<vec128_t>(lo, hi);

    vec128_t shuf = vec_shuffle32<vec128_t>(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = vec_add32<vec128_t>(lo, shuf);

    shuf = vec_shuffle32<vec128_t>(lo, _MM_SHUFFLE(0, 1, 0, 1));
    lo = vec_add32<vec128_t>(lo, shuf);
    return vec_convert_si128_32<vec128_t, int32_t>(lo);
}

// horizontal sum of 4x int64
inline int64_t hsum_epi64(vec256_t v) {
    vec128_t lo = vec_cast_256_128<vec256_t, vec128_t>(v);
    vec128_t hi = vec_extract_256_128<vec256_t, vec128_t>(v, 1);
    vec128_t sum128 = vec_add64<vec128_t>(lo, hi);
    vec128_t hi64   = vec_unpackhi64<vec128_t>(sum128, sum128);
    vec128_t sum64  = vec_add64(sum128, hi64);
    return vec_convert_si128_64<vec128_t, int64_t>(sum64);
}

// Fused SCReLU and int16 output weights. Widen before squaring (255^2
// exceeds signed int16), then accumulate signed products in 64 bits.
// Inputs use the current accumulator width; arrays must have L1_SIZE entries.
inline int64_t dot_screlu_i16(const int16_t* values, const int16_t* weights, int16_t qa) {
    auto even = zeros256;
    auto odd = zeros256;
    const auto cap = vec_set_32<vec256_t>(qa);
    for (int i = 0; i < L1_SIZE; i += 8) {
        auto x = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(values + i));
        x = vec_clamp32(x, zeros256, cap);
        x = vec_mullo32(x, x);
        const auto w = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(weights + i));
        even = vec_add64(even, vec_mul32(x, w));
        odd = vec_add64(odd, vec_mul32(vec_shift_right64(x, 32), vec_shift_right64(w, 32)));
    }
    return hsum_epi64(vec_add64(even, odd));
}

// -------------- weight transforms -------------

// dot product of int64 activations x int8 weights, exact (no overflow/truncation)
// a[] must be non-negative (true for screlu output) and size must be a multiple of 4
inline int64_t dot_i64_i8(const int64_t* a, const int8_t* w, int size) {
    vec256_t acc = zeros256;
    const vec256_t one       = vec_set_64<vec256_t>(1);

    for (int i = 0; i < size; i += 4) {
        vec256_t av = vec_load<int64_t, vec256_t>(a + i);

        // widen 4 int8 weights -> 4 int64 (sign-extended)
        // Only four weights remain in the final iteration: avoid an 8-byte overread.
        int8_t weights[8] = {};
        std::memcpy(weights, w + i, 4);
        vec128_t w8   = vec_load<int8_t, vec128_t>(weights);
        vec256_t w32  = vec_convert_8_32<vec128_t, vec256_t>(w8);       // low 8 int8 -> 8 int32
        vec128_t w32l = vec_cast_256_128<vec256_t, vec128_t>(w32);    // low 4 int32 == w[i..i+3]
        vec256_t wv   = vec_convert_32_64<vec128_t, vec256_t>(w32l);    // 4 int64, sign-extended

        // split a into 32-bit hi/lo halves (a >= 0 guaranteed, from screlu square)
        vec256_t a_lo = vec_lo_half<vec256_t>(av);
        vec256_t a_hi = vec_hi_half<vec256_t>(av);

        // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane as SIGNED.
        // a_lo's true value is unsigned (0..2^32-1); if its top bit is set,
        // the signed read is (a_lo - 2^32), so we carry a +1 into a_hi to
        // compensate: a = (a_hi + carry)*2^32 + a_lo_signed, exactly.
        vec256_t carry    = vec_and<vec256_t>(
                                vec_shift_right64<vec256_t>(av, 31), 
                                one
                            );
        vec256_t a_hi_adj = vec_add64<vec256_t>(a_hi, carry);

        vec256_t lo_prod = vec_mul32<vec256_t>(a_lo, wv);      // a_lo_signed * w  (exact 64-bit)
        vec256_t hi_prod = vec_mul32<vec256_t>(a_hi_adj, wv);  // (a_hi + carry) * w

        vec256_t prod = vec_add64<vec256_t>(lo_prod, vec_shift_left64<vec256_t>(hi_prod, 32));
        acc = vec_add64<vec256_t>(acc, prod);
    }

    return hsum_epi64(acc);
}

// dot product of int32 activations x int8 weights, widened to int64 (no overflow)
inline int64_t dot_i32_i8_widen(const int32_t* a, const int8_t* w, int size) {
    vec256_t acc_lo = zeros256;  // holds even-indexed lane products (0,2,4,6)
    vec256_t acc_hi = zeros256;  // holds odd-indexed lane products (1,3,5,7)

    for (int i = 0; i < size; i += 8) {
        vec256_t av  = vec_load<int32_t, vec256_t>(a + i);
        vec128_t w8  = vec_load<int8_t, vec128_t>(w + i);
        vec256_t wv  = vec_convert_8_32<vec128_t, vec256_t>(w8);

        // _mm256_mul_epi32 reads the low 32 bits of each 64-bit lane, signed,
        // and produces a true 64-bit product -- widening, no overflow.
        vec256_t lo = vec_mul32<vec256_t>(av, wv);                                     // lanes 0,2,4,6
        vec256_t hi = vec_mul32<vec256_t>(vec_shift_right256<vec256_t>(av, 4), vec_shift_right256<vec256_t>(wv, 4)); // lanes 1,3,5,7

        acc_lo = vec_add64<vec256_t>(acc_lo, lo);
        acc_hi = vec_add64<vec256_t>(acc_hi, hi);
    }

    return hsum_epi64(acc_lo) + hsum_epi64(acc_hi);
}

// ----------- activations -------------

inline void activate_crelu(const int16_t* in, int16_t* out, int size, int16_t clamp_bound) {
    const vec256_t zero = zeros256;
    const vec256_t qa   = vec_set_16<vec256_t>(clamp_bound);
    for (int i = 0; i < size; i += 16) {
        vec256_t v       = vec_load<int16_t, vec256_t>(in + i);
        vec256_t clipped = vec_clamp16<vec256_t>(v, zero, qa);
        vec_store<vec256_t, int16_t>(out + i, clipped);
    }
}

// fold SCRELU into multiply-add
inline void activate_screlu32(const int32_t* in, int32_t* out, int size, int32_t clamp_bound) {
    // TYPE = int32 or int64
    // handled the same (int16 is different)
    const vec256_t zero = zeros256;
    const vec256_t qa   = vec_set_32<vec256_t>(clamp_bound);
    const vec256_t qa   = vec_set_32<vec256_t>(clamp_bound);

    for (int i = 0; i < size; i += 8) {
        vec256_t v = vec_load<int32_t, vec256_t>(in + i);
        v = vec_clamp32<vec256_t>(v, zero, qa);
        v = vec_mullo32<vec256_t>(v, v);

        vec_store<vec256_t, int32_t>(out + i, v);
    }
}

inline void activate_screlu64(const int32_t* in, int64_t* out, int size, int32_t clamp_bound) {
    const vec256_t zero = zeros256;
    const vec256_t qa   = vec_set_32<vec256_t>(clamp_bound);

    for (int i = 0; i < size; i += 8) {
        vec256_t v = vec_load<int32_t, vec256_t>(in + i);
        v = vec_clamp32<vec256_t>(v, zero, qa);         // clamp in 32-bit, values are small enough here

        // widening square: 32x32 -> 64, 4 lanes at a time, done twice for 8 int32 inputs
        vec256_t lo = vec_mul32<vec256_t>(v, v);                                    // squares elements 0,2,4,6 -> 64-bit
        vec256_t hi = vec_mul32<vec256_t>(vec_shift_right256<vec256_t>(v, 4), vec_shift_right256<vec256_t>(v, 4)); // elements 1,3,5,7
        // unpacklo/hi only interleave within each 128-bit half:
        //   u_lo = [v0²,v1²,v4²,v5²]   u_hi = [v2²,v3²,v6²,v7²]
        vec256_t u_lo = vec_unpacklo64<vec256_t>(lo, hi);
        vec256_t u_hi = vec_unpackhi64<vec256_t>(lo, hi);

        // reassemble across the 128-bit boundary to get true sequential order
        vec256_t out_lo = vec_permute2x128<vec256_t>(u_lo, u_hi, 0x20); // v0²,v1²,v2²,v3²
        vec256_t out_hi = vec_permute2x128<vec256_t>(u_lo, u_hi, 0x31); // v4²,v5²,v6²,v7²
        
        vec_store<vec256_t, int64_t>(out + i, out_lo);
        vec_store<vec256_t, int64_t>(out + i + 4, out_hi);
    }
}

#endif // SIMD_H
