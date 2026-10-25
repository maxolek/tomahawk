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

// copy-on-update kernels for the accumulator stack: dst = src + add - sub (- sub2)
// int16 arithmetic wraps, so the result equals any sequence of individual updates
inline void copy_add_sub_simd(const int16_t* src, int16_t* dst, 
                              const int16_t* add, const int16_t* sub) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        vec256_t s = vec_load<int16_t, vec256_t>(src + i);
        vec256_t a = vec_load<int16_t, vec256_t>(add + i);
        vec256_t b = vec_load<int16_t, vec256_t>(sub + i);

        s = vec_add16<vec256_t>(s, a);
        s = vec_sub16<vec256_t>(s, b);

        vec_store<vec256_t, int16_t>(dst + i, s);
    }
}

inline void copy_add_sub_sub_simd(const int16_t* src, int16_t* dst, 
                                  const int16_t* add, const int16_t* sub1, const int16_t* sub2) {
    for (int i = 0; i < L1_SIZE; i += 16) {
        vec256_t s = vec_load<int16_t, vec256_t>(src + i);
        vec256_t a = vec_load<int16_t, vec256_t>(add + i);
        vec256_t b = vec_load<int16_t, vec256_t>(sub1 + i);
        vec256_t c = vec_load<int16_t, vec256_t>(sub2 + i);

        s = vec_add16<vec256_t>(s, a);
        s = vec_sub16<vec256_t>(s, b);
        s = vec_sub16<vec256_t>(s, c);

        vec_store<vec256_t, int16_t>(dst + i, s);
    }
}

// dst = bias + sum of the given feature rows. works on register-resident tiles 
// so accumulator is written once instead of once per feature 
inline void refresh_simd(int16_t* dst, const int16_t* bias, const int16_t (*W)[L1_SIZE], 
                         const int* features, int count) {

    constexpr int TILE_REGS = 8; // 8x16 int16 = 128 int16 per tile
    constexpr int TILE_SIZE = TILE_REGS * 16;
    static_assert(L1_SIZE % TILE_SIZE == 0, "L1_SIZE must be a multiple of the tile size");

    for (int t = 0; t < L1_SIZE; t += TILE_SIZE) {
        vec256_t r[TILE_REGS];;

        for (int k = 0; k < TILE_REGS; k++) 
            r[k] = vec_load<int16_t, vec256_t>(bias + t + 16 * k);

        for (int f = 0; f < count; f++) {
            const int16_t* w = W[features[f]] + t;
            for (int k = 0; k < TILE_REGS; k++) 
                r[k] = vec_add16<vec256_t>(r[k], vec_load<int16_t, vec256_t>(w + 16 * k));
        }

        for (int k = 0; k < TILE_REGS; k++) 
            vec_store<vec256_t, int16_t>(dst + t + 16 * k, r[k]);
    }
}

// ---------- feature transformer -----------

// Preserve the pairwise products in int16 for the first dense layer.
inline void pairwise_mul_simd(const int16_t* stm, const int16_t* ntm, int16_t* result) {
    static_assert(QA > 0 && int64_t(QA) * QA <= INT16_MAX,
                  "Pairwise activations must fit signed int16");
    static_assert(L1_SIZE % 32 == 0, "Each perspective must contain whole SIMD vectors");
    
    constexpr int HALF = L1_SIZE / 2;
    for (int i = 0; i < HALF; i += 16) {
        const auto us = vec_mullo16(vec_load<int16_t, vec256_t>(stm + i),
                                   vec_load<int16_t, vec256_t>(stm + HALF + i));
        const auto them = vec_mullo16(vec_load<int16_t, vec256_t>(ntm + i),
                                     vec_load<int16_t, vec256_t>(ntm + HALF + i));
        vec_store<vec256_t, int16_t>(result + i, us);
        vec_store<vec256_t, int16_t>(result + HALF + i, them);
    }
}

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
                vec_extract_256_128<1>(stm_r)
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
                vec_extract_256_128<1>(ntm_r)
            )
        );
    }
}

// ----------- sum accumulation -------------

// horizontal sum int32
// performs weight multplication for !! output layer !!
inline int32_t hsum_epi32(vec256_t v) {
    vec128_t lo = vec_cast_256_128<vec256_t, vec128_t>(v);
    vec128_t hi = vec_extract_256_128<1>(v);
    lo = vec_add32<vec128_t>(lo, hi);

    vec128_t shuf = vec_shuffle32<_MM_SHUFFLE(1, 0, 3, 2)>(lo);
    lo = vec_add32<vec128_t>(lo, shuf);

    shuf = vec_shuffle32<_MM_SHUFFLE(0, 1, 0, 1)>(lo);
    lo = vec_add32<vec128_t>(lo, shuf);
    return vec_convert_si128_32<vec128_t, int32_t>(lo);
}

// horizontal sum of 4x int64
inline int64_t hsum_epi64(vec256_t v) {
    vec128_t lo = vec_cast_256_128<vec256_t, vec128_t>(v);
    vec128_t hi = vec_extract_256_128<1>(v);
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
    for (int i = 0; i < hl_size; i += 8) {
        auto x = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(values + i));
        // screlu
        x = vec_clamp32(x, zeros256, cap);
        x = vec_mullo32(x, x);
        // dot
        const auto w = vec_convert_16_32<vec128_t, vec256_t>(vec_load<int16_t, vec128_t>(weights + i));
        even = vec_add64(even, vec_mul32(x, w));
        odd = vec_add64(odd, vec_mul32(vec_shift_right64(x, 32), vec_shift_right64(w, 32)));
    }
    return hsum_epi64(vec_add64(even, odd));
}

// -------------- weight transforms -------------

// First dense layer only: values are in [0, QA^2], weights in [-128, 127].
// The absolute sum bound also protects every partial sum and horizontal add.
// Promote the result before adding the bias; later layers still require int64.
inline int32_t dot_pairwise_i16_i8(const int16_t* a, const int8_t* w) {
    static_assert(int64_t(L1_SIZE) * QA * QA * 128 <= INT32_MAX,
                  "First dense layer sum must fit signed int32");
    static_assert(L1_SIZE % 32 == 0, "Dot product requires whole SIMD vectors");
    auto acc0 = zeros256;
    auto acc1 = zeros256;
    for (int i = 0; i < L1_SIZE; i += 32) {
        acc0 = vec_madd_i16_i8(acc0, vec_load<int16_t, vec256_t>(a + i), w + i);
        acc1 = vec_madd_i16_i8(acc1, vec_load<int16_t, vec256_t>(a + i + 16), w + i + 16);
    }
    return hsum_epi32(vec_add32(acc0, acc1));
}

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
        vec256_t hi = vec_mul32<vec256_t>(vec_shift_right256<4>(av), vec_shift_right256<4>(wv)); // lanes 1,3,5,7

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
        vec256_t hi = vec_mul32<vec256_t>(vec_shift_right256<4>(v), vec_shift_right256<4>(v)); // elements 1,3,5,7
        // unpacklo/hi only interleave within each 128-bit half:
        //   u_lo = [v0²,v1²,v4²,v5²]   u_hi = [v2²,v3²,v6²,v7²]
        vec256_t u_lo = vec_unpacklo64<vec256_t>(lo, hi);
        vec256_t u_hi = vec_unpackhi64<vec256_t>(lo, hi);

        // reassemble across the 128-bit boundary to get true sequential order
        vec256_t out_lo = vec_permute2x128<0x20>(u_lo, u_hi); // v0²,v1²,v2²,v3²
        vec256_t out_hi = vec_permute2x128<0x31>(u_lo, u_hi); // v4²,v5²,v6²,v7²
        
        vec_store<vec256_t, int64_t>(out + i, out_lo);
        vec_store<vec256_t, int64_t>(out + i + 4, out_hi);
    }
}

// ============================================================
// AVX2 inference kernels
// ============================================================
// Integer-exact replacements for the dense layers above: they produce the same
// sums, only with fewer instructions. Other targets use the generic kernels.
#if defined(__AVX2__)

// first dense layer: CReLU of both accumulators, pairwise multiply, and the 1024 -> 16 affine product
// after CReLU most pairwise products are zero, so the products are grouped in in16 pairs
// and only non-zero pairs are multiplied, each against all 16 outputs at once
// p[i] = clamp(us[i], 0, QA) * clamp(us[i + HALF], 0, QA) 
// sums[o] = bias[o] + sum_i p[i] * w_pairs[o][i]
inline void l1_sparse_avx2(const int16_t* us, const int16_t* them, 
                            const int16_t (*w_pairs)[2 * L2_SIZE], int32_t* sums) {
    static_assert(L2_SIZE == 16,  "one input pair is multiplied into two 8-lane accumulators");
    static_assert(int64_t(QA) * QA < 32768, "pairwise products must be non-negative int16");
    static_assert(L1_SIZE % 256 == 0, "whole 64-pair mask words per perspective");

    constexpr int HALF = L1_SIZE / 2;
    constexpr int WORDS = L1_SIZE / 128;    // one mask bit per int16 pair
    alignas(32) int32_t act[L1_SIZE / 2];   // pairwise products, two int16 per enty
    uint64_t masks[WORDS];

    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);

    // within layer transforms
    // (crelu + pairmul) for both perspectives
    for (int perspective = 0; perspective < 2; perspective++) {
        const int16_t* acc = perspective ? them : us;

        for (int group = 0; group < HALF / 128; group++) {
            uint64_t mask = 0;

            for (int v = 0; v < 8; v++) {
                const int i = group * 128 + v * 16;

                // CReLU
                const __m256i a = _mm256_min_epi16(_mm256_max_epi16(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i)), zero), qa);
                const __m256i b = _mm256_min_epi16(_mm256_max_epi16(
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + HALF + i)), zero), qa);
                
                // pair-mul
                const __m256i p = _mm256_mullo_epi16(a, b);
                _mm256_store_si256(reinterpret_cast<__m256i*>(act + (perspective * HALF + i) / 2), p);

                const unsigned nz = static_cast<unsigned>(
                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(p, zero))));
                
                mask |= static_cast<uint64_t>(nz) << (8 * v);
            }

            masks[perspective * (HALF / 128) + group] = mask;
        }
    }

    // (sparse) affine transform
    // walk the non-zero pairs, alternating two accumulator pairs so the 
    // multiply-adds (single multi-cycle instruction on AVX-VNNI target)
    // do not form one long dependency chain
    __m256i a0 = zero, a1 = zero, b0 = zero, b1 = zero;
    for (int word = 0; word < WORDS; word++) {
        uint64_t mask = masks[word];
        const int32_t* x = act + 64 * word;
        const int16_t (*w) [2 * L2_SIZE] = w_pairs + 64 * word;

        while (mask) {
            int k = __builtin_ctzll(mask);
            mask &= mask-1;
            const __m256i xk = _mm256_set1_epi32(x[k]);
            const __m256i* wk = reinterpret_cast<const __m256i*>(w[k]);

            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(xk, _mm256_load_si256(wk)));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(xk, _mm256_load_si256(wk + 1)));
            
            if (!mask) break;

            k = __builtin_ctzll(mask);
            mask &= mask-1;
            const __m256i xk2 = _mm256_set1_epi32(x[k]);
            const __m256i* wk2 = reinterpret_cast<const __m256i*>(w[k]);

            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(xk2 , _mm256_load_si256(wk2)));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(xk2, _mm256_load_si256(wk2 + 1)));
        }
    }

    // add bias to all 16 outputs
    const __m256i s0 = _mm256_add_epi32(a0, b0);
    const __m256i s1 = _mm256_add_epi32(a1, b1);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums), s0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums + 8), s1);
}

// Second dense layer, output-parallel: out[o] = sum_i x[i] * w[i][o] with exact
// 64-bit products (each x[i] fits a signed 32-bit lane).
inline void l2_avx2(const int32_t* x, const int64_t (*w_by_input)[L3_SIZE], int64_t* out) {
    static_assert(L3_SIZE == 32, "eight 4-lane int64 accumulators");
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int i = 0; i < L2_SIZE; ++i) {
        const __m256i xi = _mm256_set1_epi32(x[i]);
        const __m256i* wi = reinterpret_cast<const __m256i*>(w_by_input[i]);
        a0 = _mm256_add_epi64(a0, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 0)));
        a1 = _mm256_add_epi64(a1, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 1)));
        a2 = _mm256_add_epi64(a2, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 2)));
        a3 = _mm256_add_epi64(a3, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 3)));
        a4 = _mm256_add_epi64(a4, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 4)));
        a5 = _mm256_add_epi64(a5, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 5)));
        a6 = _mm256_add_epi64(a6, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 6)));
        a7 = _mm256_add_epi64(a7, _mm256_mul_epi32(xi, _mm256_load_si256(wi + 7)));
    }
    __m256i* o = reinterpret_cast<__m256i*>(out);
    _mm256_storeu_si256(o + 0, a0); _mm256_storeu_si256(o + 1, a1);
    _mm256_storeu_si256(o + 2, a2); _mm256_storeu_si256(o + 3, a3);
    _mm256_storeu_si256(o + 4, a4); _mm256_storeu_si256(o + 5, a5);
    _mm256_storeu_si256(o + 6, a6); _mm256_storeu_si256(o + 7, a7);
}

// output layer: sum_i clamp(x[i], 0, cap)^2 * w[i], computed as c * (c * w)
// so every product is exact (c * w fits 32 bits, the result is accumulated in 64)
inline int64_t l3_screlu_avx2(const int32_t* x, const int32_t* w, const int32_t* bias, int32_t cap) {
    static_assert(L3_SIZE % 8 == 0, "whole 8-lane vectors");
    static_assert(int64_t(QA) * QB * QC * 128 <= INT32_MAX, "c * w must fit a signed 32-bit lane");

    const __m256i zero = _mm256_setzero_si256();
    const __m256i capv = _mm256_set1_epi32(cap);
    __m256i acc = zero;

    for (int i = 0; i < L3_SIZE; i += 8) {
        const __m256i c = _mm256_min_epi32(_mm256_max_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)), zero), capv);
        const __m256i cw = _mm256_mullo_epi32(c, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i)));

        acc = _mm256_add_epi64(acc, _mm256_mul_epi32(c, cw)); // c * (cw)
        acc = _mm256_add_epi64(acc, _mm256_mul_epi32(
            _mm256_srli_epi64(c, 32), _mm256_srli_epi64(cw, 32)));
    }

    // affine (output)
    return hsum_epi64(acc);
}

#endif // avx2

#endif // SIMD_H
