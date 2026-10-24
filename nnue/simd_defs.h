#ifndef SIMD_DEFS_H
#define SIMD_DEFS_H

#include "helpers.h"

static constexpr size_t VEC_ALIGN = 32;

// ============================================================
// Templates
// ============================================================
template <typename T> static inline T vec_set_16(int16_t x);
template <typename T> static inline T vec_set_32(int32_t x);
template <typename T> static inline T vec_set_64(int64_t x);
template <typename T> static inline void vec_set_zero(T &x);

template <typename T_in, typename T_out> static inline T_out vec_load(const T_in* x);
template <typename T_in, typename T_out> static inline void vec_store(T_out* to, T_in from);

template <typename T> static inline T vec_unpacklo64(T x, T y);
template <typename T> static inline T vec_unpackhi64(T x, T y);
template <typename T_out, typename T_in> static inline T_out vec_lo_half(T_in x);
template <typename T_out, typename T_in> static inline T_out vec_hi_half(T_in x);

template <typename T> static inline T vec_shift_left64(T x, int shift);
template <typename T> static inline T vec_shift_right64(T x, int shift);
template <typename T> static inline T vec_shift_right256(T x, int shift);

template <typename T> static inline T vec_add16(T x, T y);
template <typename T> static inline T vec_add32(T x, T y);
template <typename T> static inline T vec_add64(T x, T y);
template <typename T> static inline T vec_sub16(T x, T y);
template <typename T> static inline T vec_sub32(T x, T y);

template <typename T> static inline T vec_mul16(T x, T y);
template <typename T> static inline T vec_mul32(T x, T y);
template <typename T> static inline T vec_mullo16(T x, T y);
template <typename T> static inline T vec_mullo32(T x, T y);
template <typename T> static inline T vec_mulhi16(T x, T y);
template <typename T> static inline T vec_mulhi32(T x, T y);
// Accumulate 16 signed products into int32 lanes. Lane grouping is platform
// specific; callers must horizontally sum all lanes to obtain the dot product.
template <typename T> static inline T vec_madd_i16_i8(T acc, T values, const int8_t* weights);

template <typename T> static inline T vec_clamp16(T x, T min, T max);
template <typename T> static inline T vec_clamp32(T x, T min, T max);

template <typename T> static inline T vec_shuffle32(T x, int imm);
template <typename T> static inline T vec_permute2x128(T x, T y, int imm);

template <typename T_in, typename T_out> static inline T_out vec_cast_256_128(T_in x);
template <typename T_in, typename T_out> static inline T_out vec_extract_256_128(T_in x, int imm);

template <typename T_in, typename T_out> static inline T_out vec_convert_8_32(T_in x);
template <typename T_in, typename T_out> static inline T_out vec_convert_16_32(T_in x);
template <typename T_in, typename T_out> static inline T_out vec_convert_32_64(T_in x);
template <typename T_in, typename T_out> static inline T_out vec_convert_si128_32(T_in x);
template <typename T_in, typename T_out> static inline T_out vec_convert_si128_64(T_in x);

template <typename T> static inline T vec_and(T x, T y);

// ============================================================
// x86 (AVX2) specializations
// ============================================================
#ifdef _WIN32

// enabled through this inclusion
#include <immintrin.h>

// type defs
using vec256_t = __m256i;
using vec128_t = __m128i;

// consts
static const vec256_t zeros256 = _mm256_setzero_si256();
static const vec256_t ones32  = _mm256_set1_epi32(1);

// set
template <> inline vec256_t vec_set_16<vec256_t>(int16_t x) { return _mm256_set1_epi16(x); }
template <> inline vec256_t vec_set_32<vec256_t>(int32_t x) { return _mm256_set1_epi32(x); }
template <> inline vec256_t vec_set_64<vec256_t>(int64_t x) { return _mm256_set1_epi64x(x); }
template <> inline void vec_set_zero(vec256_t &x) { x = zeros256; }

// load
template <> inline vec128_t vec_load<int8_t, vec128_t>(const int8_t* x) { // different load function due to immediate registry widening in dot functions
    return _mm_loadl_epi64(reinterpret_cast<const vec128_t*>(x)); // load 8 bytes (64 bits) into low half of vec128_t
}
template <> inline vec128_t vec_load<int16_t, vec128_t>(const int16_t* x) {
    return _mm_loadu_si128(reinterpret_cast<const vec128_t*>(x));
}
template <> inline vec256_t vec_load<int16_t, vec256_t>(const int16_t* x) { 
    return _mm256_load_si256(reinterpret_cast<const vec256_t*>(x)); 
}
template <> inline vec256_t vec_load<int32_t, vec256_t>(const int32_t* x) { 
    return _mm256_load_si256(reinterpret_cast<const vec256_t*>(x)); 
}
template <> inline vec256_t vec_load<int64_t, vec256_t>(const int64_t* x) { 
    return _mm256_load_si256(reinterpret_cast<const vec256_t*>(x)); 
}
template <> inline vec256_t vec_load<vec256_t, vec256_t>(const vec256_t* x) { 
    return _mm256_load_si256(x); 
}

// store
template <> inline void vec_store<vec256_t, vec256_t>(vec256_t* to, vec256_t from) { 
    _mm256_store_si256(to, from); 
}
template <> inline void vec_store<vec256_t, int16_t>(int16_t* to, vec256_t from) { 
    _mm256_store_si256(reinterpret_cast<vec256_t*>(to), from); 
}
template <> inline void vec_store<vec256_t, int32_t>(int32_t* to, vec256_t from) { 
    _mm256_store_si256(reinterpret_cast<vec256_t*>(to), from); 
}
template <> inline void vec_store<vec256_t, int64_t>(int64_t* to, vec256_t from) { 
    _mm256_store_si256(reinterpret_cast<vec256_t*>(to), from); 
}

// unpack
template <> inline vec128_t vec_unpacklo64<vec128_t>(vec128_t x, vec128_t y) { 
    return _mm_unpacklo_epi64(x, y); 
}
template <> inline vec256_t vec_unpacklo64<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_unpacklo_epi64(x, y); 
}

template <> inline vec128_t vec_unpackhi64<vec128_t>(vec128_t x, vec128_t y) { 
    return _mm_unpackhi_epi64(x, y); 
}
template <> inline vec256_t vec_unpackhi64<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_unpackhi_epi64(x, y); 
}

// shift
template <> inline vec256_t vec_shift_left64<vec256_t>(vec256_t x, int shift) { 
    return _mm256_slli_epi64(x, shift); 
}
template <> inline vec256_t vec_shift_right64<vec256_t>(vec256_t x, int shift) { 
    return _mm256_srli_epi64(x, shift); 
}
template <> inline vec256_t vec_shift_right256<vec256_t>(vec256_t x, int shift) {
    return _mm256_srli_si256(x, shift);
}

// halves
template <> inline vec256_t vec_lo_half<vec256_t>(vec256_t x) { 
    const vec256_t mask_lo32 = vec_set_64<vec256_t>(0xFFFFFFFFLL);
    return _mm256_and_si256(x, mask_lo32); 
}
template <> inline vec256_t vec_hi_half<vec256_t>(vec256_t x) { 
    return _mm256_srli_epi64(x, 32); 
}

// add
template <> inline vec256_t vec_add16<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_add_epi16(x, y); 
}
template <> inline vec128_t vec_add32<vec128_t>(vec128_t x, vec128_t y) { 
    return _mm_add_epi32(x, y); 
}
template <> inline vec256_t vec_add32<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_add_epi32(x, y); 
}
template <> inline vec128_t vec_add64<vec128_t>(vec128_t x, vec128_t y) { 
    return _mm_add_epi64(x, y); 
}
template <> inline vec256_t vec_add64<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_add_epi64(x, y); 
}

// sub
template <> inline vec256_t vec_sub16<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_sub_epi16(x, y); 
}
template <> inline vec256_t vec_sub32<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_sub_epi32(x, y); 
}

// widening (multiply)
//  reads low 32 bits of each 64-bit lane
//  (even-indexed 32-bit lanes 0,2,4,6) and produces genuine 64-bit products
//  distinct from vec_mullo32 which truncates every lane back to 32 bits
//  (computes 4 64-bit products, not 8 32-bit like mullo)
template <> inline vec256_t vec_mul32<vec256_t>(vec256_t x, vec256_t y) {
    return _mm256_mul_epi32(x, y);
}

// multiply
template <> inline vec256_t vec_madd_i16_i8<vec256_t>(vec256_t acc, vec256_t values, const int8_t* weights) {
    const auto w = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(weights)));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(values, w));
}

template <> inline vec256_t vec_mullo16<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_mullo_epi16(x, y); 
}
template <> inline vec256_t vec_mullo32<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_mullo_epi32(x, y); 
}
template <> inline vec256_t vec_mulhi16<vec256_t>(vec256_t x, vec256_t y) { 
    return _mm256_mulhi_epi16(x, y); 
}

// shuffle 
template <> inline vec256_t vec_shuffle32<vec256_t>(vec256_t x, int imm) { 
    return _mm256_shuffle_epi32(x, imm); 
}
template <> inline vec128_t vec_shuffle32<vec128_t>(vec128_t x, int imm) { 
    return _mm_shuffle_epi32(x, imm); 
}

// permute
template <> inline vec256_t vec_permute2x128<vec256_t>(vec256_t x, vec256_t y, int imm) { 
    return _mm256_permute2x128_si256(x, y, imm); 
} 

// clamp
template <> inline vec256_t vec_clamp16<vec256_t>(vec256_t x, vec256_t min, vec256_t max) {
    return _mm256_min_epi16(_mm256_max_epi16(x, min), max);
}
template <> inline vec256_t vec_clamp32<vec256_t>(vec256_t x, vec256_t min, vec256_t max) {
    return _mm256_min_epi32(_mm256_max_epi32(x, min), max);
}

// convert
template <> inline vec256_t vec_convert_8_32<vec128_t, vec256_t>(vec128_t x) {
    return _mm256_cvtepi8_epi32(x);
}
template <> inline vec256_t vec_convert_16_32<vec128_t, vec256_t>(vec128_t x) {
    return _mm256_cvtepi16_epi32(x);
}
template <> inline vec256_t vec_convert_32_64<vec128_t, vec256_t>(vec128_t x) {
    return _mm256_cvtepi32_epi64(x);
}

template <> inline int32_t vec_convert_si128_32<vec128_t, int32_t>(vec128_t x) {
    return _mm_cvtsi128_si32(x);
}
template <> inline int64_t vec_convert_si128_64<vec128_t, int64_t>(vec128_t x) {
    return _mm_cvtsi128_si64(x);
}

template <> inline vec128_t vec_cast_256_128<vec256_t, vec128_t>(vec256_t x) {
    return _mm256_castsi256_si128(x);
}

template <> inline vec128_t vec_extract_256_128<vec256_t, vec128_t>(vec256_t x, int index) {
    return _mm256_extracti128_si256(x, index);
}

// utils 
template <> inline vec256_t vec_and<vec256_t>(vec256_t x, vec256_t y) {
    return _mm256_and_si256(x, y);
}


// ============================================================
// ARM (NEON) specializations
// ============================================================
# else

#include <arm_neon.h>

// Two NEON registers preserve the AVX2 lane layout.
struct alignas(VEC_ALIGN) vec256_t { int8x16_t lo; int8x16_t hi; };
using vec128_t = int8x16_t;

// override _MM_SHUFFLE to prevent different arch compile errors
#ifndef _MM_SHUFFLE
#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif // _MM_SHUFFLE

// consts
static const vec256_t zeros256 = { vdupq_n_s8(0), vdupq_n_s8(0) };
static const vec256_t ones32  = { 
    vreinterpretq_s8_s32(vdupq_n_s32(1)), 
    vreinterpretq_s8_s32(vdupq_n_s32(1)) 
};

// ---- internal helpers (not part of the public vec_* API) ----
namespace neon_detail {
    inline int32x4_t shuffle32(int32x4_t x, int imm) {
        // Table lookup allows runtime indices; vgetq_lane requires literals.
        uint8_t indices[16];
        for (int lane = 0; lane < 4; ++lane)
            for (int byte = 0; byte < 4; ++byte)
                indices[4 * lane + byte] = static_cast<uint8_t>(4 * ((imm >> (2 * lane)) & 3) + byte);
        return vreinterpretq_s32_s8(vqtbl1q_s8(vreinterpretq_s8_s32(x), vld1q_u8(indices)));
    }
}

// set
template <> inline vec256_t vec_set_16<vec256_t>(int16_t x) {
    int8x16_t v = vreinterpretq_s8_s16(vdupq_n_s16(x));
    return { v, v };
}
template <> inline vec256_t vec_set_32<vec256_t>(int32_t x) {
    int8x16_t v = vreinterpretq_s8_s32(vdupq_n_s32(x));
    return { v, v };
}
template <> inline vec256_t vec_set_64<vec256_t>(int64_t x) {
    int8x16_t v = vreinterpretq_s8_s64(vdupq_n_s64(x));
    return { v, v };
}
template <> inline void vec_set_zero(vec256_t &x) { x = zeros256; }
 
// load
template <> inline vec128_t vec_load<int8_t, vec128_t>(const int8_t* x) {
    // load 8 bytes into the low half, zero-fill the high half -- matches
    // _mm_loadl_epi64 (see simd_defs.h note on why int8 differs from the rest)
    return vcombine_s8(vld1_s8(x), vdup_n_s8(0));
}
template <> inline vec128_t vec_load<int16_t, vec128_t>(const int16_t* x) {
    return vreinterpretq_s8_s16(vld1q_s16(x));
}
template <> inline vec256_t vec_load<int16_t, vec256_t>(const int16_t* x) {
    return {
        vreinterpretq_s8_s16(vld1q_s16(x)),
        vreinterpretq_s8_s16(vld1q_s16(x + 8))
    };
}
template <> inline vec256_t vec_load<int32_t, vec256_t>(const int32_t* x) {
    return {
        vreinterpretq_s8_s32(vld1q_s32(x)),
        vreinterpretq_s8_s32(vld1q_s32(x + 4))
    };
}
template <> inline vec256_t vec_load<int64_t, vec256_t>(const int64_t* x) {
    return {
        vreinterpretq_s8_s64(vld1q_s64(x)),
        vreinterpretq_s8_s64(vld1q_s64(x + 2))
    };
}
template <> inline vec256_t vec_load<vec256_t, vec256_t>(const vec256_t* x) {
    const int8_t* p = reinterpret_cast<const int8_t*>(x);
    return { vld1q_s8(p), vld1q_s8(p + 16) };
}
 
// store
template <> inline void vec_store<vec256_t, vec256_t>(vec256_t* to, vec256_t from) {
    int8_t* p = reinterpret_cast<int8_t*>(to);
    vst1q_s8(p, from.lo);
    vst1q_s8(p + 16, from.hi);
}
template <> inline void vec_store<vec256_t, int16_t>(int16_t* to, vec256_t from) {
    vst1q_s16(to,     vreinterpretq_s16_s8(from.lo));
    vst1q_s16(to + 8, vreinterpretq_s16_s8(from.hi));
}
template <> inline void vec_store<vec256_t, int32_t>(int32_t* to, vec256_t from) {
    vst1q_s32(to,     vreinterpretq_s32_s8(from.lo));
    vst1q_s32(to + 4, vreinterpretq_s32_s8(from.hi));
}
template <> inline void vec_store<vec256_t, int64_t>(int64_t* to, vec256_t from) {
    vst1q_s64(to,     vreinterpretq_s64_s8(from.lo));
    vst1q_s64(to + 2, vreinterpretq_s64_s8(from.hi));
}
 
// unpack (x86 unpacklo/hi64 operate independently within each 128-bit half)
template <> inline vec128_t vec_unpacklo64<vec128_t>(vec128_t x, vec128_t y) {
    int64x2_t xi = vreinterpretq_s64_s8(x), yi = vreinterpretq_s64_s8(y);
    return vreinterpretq_s8_s64(vcombine_s64(vget_low_s64(xi), vget_low_s64(yi)));
}
template <> inline vec128_t vec_unpackhi64<vec128_t>(vec128_t x, vec128_t y) {
    int64x2_t xi = vreinterpretq_s64_s8(x), yi = vreinterpretq_s64_s8(y);
    return vreinterpretq_s8_s64(vcombine_s64(vget_high_s64(xi), vget_high_s64(yi)));
}
template <> inline vec256_t vec_unpacklo64<vec256_t>(vec256_t x, vec256_t y) {
    return { vec_unpacklo64<vec128_t>(x.lo, y.lo), vec_unpacklo64<vec128_t>(x.hi, y.hi) };
}
template <> inline vec256_t vec_unpackhi64<vec256_t>(vec256_t x, vec256_t y) {
    return { vec_unpackhi64<vec128_t>(x.lo, y.lo), vec_unpackhi64<vec128_t>(x.hi, y.hi) };
}
 
// shift (NEON immediates must be compile-time constants; use the
// variable-shift intrinsics since these wrappers take a runtime `shift`)
template <> inline vec256_t vec_shift_left64<vec256_t>(vec256_t x, int shift) {
    if (shift < 0 || shift >= 64) return zeros256;
    int64x2_t s = vdupq_n_s64(shift);
    return {
        vreinterpretq_s8_s64(vshlq_s64(vreinterpretq_s64_s8(x.lo), s)),
        vreinterpretq_s8_s64(vshlq_s64(vreinterpretq_s64_s8(x.hi), s))
    };
}
template <> inline vec256_t vec_shift_right64<vec256_t>(vec256_t x, int shift) {
    if (shift < 0 || shift >= 64) return zeros256;
    int64x2_t s = vdupq_n_s64(-shift);
    return {
        vreinterpretq_s8_u64(vshlq_u64(vreinterpretq_u64_s8(x.lo), s)),
        vreinterpretq_s8_u64(vshlq_u64(vreinterpretq_u64_s8(x.hi), s))
    };
}
 
// AVX2 byte shifts are independent within each 128-bit half.
template <> inline vec256_t vec_shift_right256<vec256_t>(vec256_t x, int shift) {
    if (shift < 0 || shift >= 16) return zeros256;
    static constexpr uint8_t bytes[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t indices = vaddq_u8(
        vld1q_u8(bytes),
        vdupq_n_u8(static_cast<uint8_t>(shift)));
    return { vqtbl1q_s8(x.lo, indices), vqtbl1q_s8(x.hi, indices) };
}

template <> inline vec256_t vec_lo_half<vec256_t>(vec256_t x) {
    uint64x2_t mask = vdupq_n_u64(0xFFFFFFFFULL);
    return {
        vreinterpretq_s8_u64(vandq_u64(vreinterpretq_u64_s8(x.lo), mask)),
        vreinterpretq_s8_u64(vandq_u64(vreinterpretq_u64_s8(x.hi), mask))
    };
}
template <> inline vec256_t vec_hi_half<vec256_t>(vec256_t x) {
    return vec_shift_right64<vec256_t>(x, 32);
}
 
// add
template <> inline vec256_t vec_add16<vec256_t>(vec256_t x, vec256_t y) {
    return {
        vreinterpretq_s8_s16(vaddq_s16(vreinterpretq_s16_s8(x.lo), vreinterpretq_s16_s8(y.lo))),
        vreinterpretq_s8_s16(vaddq_s16(vreinterpretq_s16_s8(x.hi), vreinterpretq_s16_s8(y.hi)))
    };
}
template <> inline vec128_t vec_add32<vec128_t>(vec128_t x, vec128_t y) {
    return vreinterpretq_s8_s32(vaddq_s32(vreinterpretq_s32_s8(x), vreinterpretq_s32_s8(y)));
}
template <> inline vec256_t vec_add32<vec256_t>(vec256_t x, vec256_t y) {
    return { vec_add32<vec128_t>(x.lo, y.lo), vec_add32<vec128_t>(x.hi, y.hi) };
}
template <> inline vec128_t vec_add64<vec128_t>(vec128_t x, vec128_t y) {
    return vreinterpretq_s8_s64(vaddq_s64(vreinterpretq_s64_s8(x), vreinterpretq_s64_s8(y)));
}
template <> inline vec256_t vec_add64<vec256_t>(vec256_t x, vec256_t y) {
    return { vec_add64<vec128_t>(x.lo, y.lo), vec_add64<vec128_t>(x.hi, y.hi) };
}
 
// sub
template <> inline vec256_t vec_sub16<vec256_t>(vec256_t x, vec256_t y) {
    return {
        vreinterpretq_s8_s16(vsubq_s16(vreinterpretq_s16_s8(x.lo), vreinterpretq_s16_s8(y.lo))),
        vreinterpretq_s8_s16(vsubq_s16(vreinterpretq_s16_s8(x.hi), vreinterpretq_s16_s8(y.hi)))
    };
}
template <> inline vec256_t vec_sub32<vec256_t>(vec256_t x, vec256_t y) {
    return {
        vreinterpretq_s8_s32(vsubq_s32(vreinterpretq_s32_s8(x.lo), vreinterpretq_s32_s8(y.lo))),
        vreinterpretq_s8_s32(vsubq_s32(vreinterpretq_s32_s8(x.hi), vreinterpretq_s32_s8(y.hi)))
    };
}
 
// Signed widening products of lanes 0, 2, 4 and 6, matching AVX2.
template <> inline vec256_t vec_mul32<vec256_t>(vec256_t x, vec256_t y) {
    auto half = [](int8x16_t a, int8x16_t b) {
        int32x4_t av = vreinterpretq_s32_s8(a);
        int32x4_t bv = vreinterpretq_s32_s8(b);
        return vreinterpretq_s8_s64(vmull_s32(
            vget_low_s32(vuzp1q_s32(av, av)),
            vget_low_s32(vuzp1q_s32(bv, bv))));
    };
    return { half(x.lo, y.lo), half(x.hi, y.hi) };
}

// multiply
template <> inline vec256_t vec_madd_i16_i8<vec256_t>(vec256_t acc, vec256_t values, const int8_t* weights) {
    const int8x16_t w = vld1q_s8(weights);
    auto half = [](int8x16_t sum, int8x16_t a, int16x8_t b) {
        const int16x8_t av = vreinterpretq_s16_s8(a);
        auto result = vmlal_s16(vreinterpretq_s32_s8(sum), vget_low_s16(av), vget_low_s16(b));
        return vreinterpretq_s8_s32(vmlal_high_s16(result, av, b));
    };
    return { half(acc.lo, values.lo, vmovl_s8(vget_low_s8(w))),
             half(acc.hi, values.hi, vmovl_high_s8(w)) };
}

template <> inline vec256_t vec_mullo16<vec256_t>(vec256_t x, vec256_t y) {
    return {
        vreinterpretq_s8_s16(vmulq_s16(vreinterpretq_s16_s8(x.lo), vreinterpretq_s16_s8(y.lo))),
        vreinterpretq_s8_s16(vmulq_s16(vreinterpretq_s16_s8(x.hi), vreinterpretq_s16_s8(y.hi)))
    };
}
template <> inline vec256_t vec_mullo32<vec256_t>(vec256_t x, vec256_t y) {
    return {
        vreinterpretq_s8_s32(vmulq_s32(vreinterpretq_s32_s8(x.lo), vreinterpretq_s32_s8(y.lo))),
        vreinterpretq_s8_s32(vmulq_s32(vreinterpretq_s32_s8(x.hi), vreinterpretq_s32_s8(y.hi)))
    };
}
template <> inline vec256_t vec_mulhi16<vec256_t>(vec256_t x, vec256_t y) {
    auto half = [](int16x8_t a, int16x8_t b) -> int16x8_t {
        int32x4_t lo = vmull_s16(vget_low_s16(a),  vget_low_s16(b));
        int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
        return vcombine_s16(vshrn_n_s32(lo, 16), vshrn_n_s32(hi, 16));
    };
    return {
        vreinterpretq_s8_s16(half(vreinterpretq_s16_s8(x.lo), vreinterpretq_s16_s8(y.lo))),
        vreinterpretq_s8_s16(half(vreinterpretq_s16_s8(x.hi), vreinterpretq_s16_s8(y.hi)))
    };
}
 
// clamp
template <> inline vec256_t vec_clamp16<vec256_t>(vec256_t x, vec256_t min, vec256_t max) {
    auto half = [](int8x16_t xv, int8x16_t mn, int8x16_t mx) -> int8x16_t {
        int16x8_t v = vminq_s16(vmaxq_s16(vreinterpretq_s16_s8(xv), vreinterpretq_s16_s8(mn)), vreinterpretq_s16_s8(mx));
        return vreinterpretq_s8_s16(v);
    };
    return { half(x.lo, min.lo, max.lo), half(x.hi, min.hi, max.hi) };
}
template <> inline vec256_t vec_clamp32<vec256_t>(vec256_t x, vec256_t min, vec256_t max) {
    auto half = [](int8x16_t xv, int8x16_t mn, int8x16_t mx) -> int8x16_t {
        int32x4_t v = vminq_s32(vmaxq_s32(vreinterpretq_s32_s8(xv), vreinterpretq_s32_s8(mn)), vreinterpretq_s32_s8(mx));
        return vreinterpretq_s8_s32(v);
    };
    return { half(x.lo, min.lo, max.lo), half(x.hi, min.hi, max.hi) };
}
 
// Fast paths for the two shuffles used by hsum_epi32.
template <> inline vec128_t vec_shuffle32<vec128_t>(vec128_t x, int imm) {
    // [src2,src3,src0,src1] == swap the two 64-bit halves
    if (imm == _MM_SHUFFLE(1, 0, 3, 2)) { 
        int64x2_t v = vreinterpretq_s64_s8(x);
        return vreinterpretq_s8_s64(vextq_s64(v, v, 1));
    } // [src1,src0,src1,src0] == swap adjacent pairs, duplicate low 64 into high 64
    else if (imm == _MM_SHUFFLE(0, 1, 0, 1)) { 
        int32x4_t t = vrev64q_s32(vreinterpretq_s32_s8(x));
        return vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(t), vget_low_s32(t)));
    }
    // generic fallback for any other immediate
    return vreinterpretq_s8_s32(neon_detail::shuffle32(vreinterpretq_s32_s8(x), imm));
}

template <> inline vec256_t vec_shuffle32<vec256_t>(vec256_t x, int imm) {
    return { vec_shuffle32<vec128_t>(x.lo, imm), vec_shuffle32<vec128_t>(x.hi, imm) };
}
 
// permute (only ever called with imm 0x20 / 0x31 in this codebase, via activate_screlu64
// both 0x20 and 0x31 are pure field selection, no computation
template <> inline vec256_t vec_permute2x128<vec256_t>(vec256_t x, vec256_t y, int imm) {
    if (imm == 0x20) return { x.lo, y.lo };
    if (imm == 0x31) return { x.hi, y.hi };
    
    // generic fallback for any other immediate
    auto select = [&](int sel) -> int8x16_t {
        if (sel & 0x8) return vdupq_n_s8(0);
        switch (sel & 0x3) {
            case 0:  return x.lo;
            case 1:  return x.hi;
            case 2:  return y.lo;
            default: return y.hi;
        }
    };
    return { select(imm & 0xF), select((imm >> 4) & 0xF) };
}
 
// convert
template <> inline vec256_t vec_convert_8_32<vec128_t, vec256_t>(vec128_t x) {
    // only the low 8 bytes are meaningful input (see vec_load<int8_t,...> note)
    int8x8_t low8 = vget_low_s8(x);
    int16x8_t w16 = vmovl_s8(low8);
    return {
        vreinterpretq_s8_s32(vmovl_s16(vget_low_s16(w16))),
        vreinterpretq_s8_s32(vmovl_s16(vget_high_s16(w16)))
    };
}
template <> inline vec256_t vec_convert_16_32<vec128_t, vec256_t>(vec128_t x) {
    int16x8_t v = vreinterpretq_s16_s8(x);
    return {
        vreinterpretq_s8_s32(vmovl_s16(vget_low_s16(v))),
        vreinterpretq_s8_s32(vmovl_s16(vget_high_s16(v)))
    };
}
template <> inline vec256_t vec_convert_32_64<vec128_t, vec256_t>(vec128_t x) {
    int32x4_t v = vreinterpretq_s32_s8(x);
    return {
        vreinterpretq_s8_s64(vmovl_s32(vget_low_s32(v))),
        vreinterpretq_s8_s64(vmovl_s32(vget_high_s32(v)))
    };
}
template <> inline int32_t vec_convert_si128_32<vec128_t, int32_t>(vec128_t x) {
    return vgetq_lane_s32(vreinterpretq_s32_s8(x), 0);
}
template <> inline int64_t vec_convert_si128_64<vec128_t, int64_t>(vec128_t x) {
    return vgetq_lane_s64(vreinterpretq_s64_s8(x), 0);
}
 
template <> inline vec128_t vec_cast_256_128<vec256_t, vec128_t>(vec256_t x) { return x.lo; }
template <> inline vec128_t vec_extract_256_128<vec256_t, vec128_t>(vec256_t x, int index) {
    return index == 0 ? x.lo : x.hi;
}
 
// and
template <> inline vec256_t vec_and<vec256_t>(vec256_t x, vec256_t y) {
    return { vandq_s8(x.lo, y.lo), vandq_s8(x.hi, y.hi) };
}



#endif // cpu arch's
#endif // SIMD_DEFS_H
