#ifndef SIMD_DEFS_H
#define SIMD_DEFS_H

#include "helpers.h"

#ifdef _WIN32

#include <immintrin.h>

// templates
template <typename T> static inline T vec_set_16(int x);
template <typename T> static inline T vec_set_32(int x);
template <typename T> static inline T vec_set_64(int x);
template <typename T> static inline void vec_set_zero(T &x);

template <typename T> static inline T vec_load(const int8_t *x);
template <typename T> static inline T vec_load(const int16_t *x);
template <typename T> static inline T vec_load(const int32_t *x);
template <typename T> static inline T vec_load(const T *x);

template <typename T> static inline void vec_store(T *to, T from);
template <typename T> static inline void vec_store(int16_t *to, T from);
template <typename T> static inline void vec_store(int32_t *to, T from);
template <typename T> static inline void vec_store(int64_t *to, T from);

template <typename T> static inline T vec_unpacklo64(T x, T y);
template <typename T> static inline T vec_unpackhi64(T x, T y);
template <typename T> static inline T vec_lo_half(T x);
template <typename T> static inline T vec_hi_half(T x);

template <typename T> static inline T vec_shift_left32(T x, int shift);
template <typename T> static inline T vec_shift_left64(T x, int shift);
template <typename T> static inline T vec_shift_right32(T x, int shift);
template <typename T> static inline T vec_shift_right64(T x, int shift);

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

template <typename T> static inline T vec_clamp16(T x, T min, T max);
template <typename T> static inline T vec_clamp32(T x, T min, T max);

template <typename T> static inline T vec_shuffle32(T x, int imm);
template <typename T> static inline T vec_permute2x128(T x, T y, int imm);

template <typename T> static inline T vec_cast_256_128(T x);
template <typename T> static inline T vec_extract_256_128(T x, int imm);

template <typename T> static inline T vec_convert_16_32(T x);
template <typename T> static inline int32_t vec_convert_si128_32(T x);

// consts
static constexpr size_t VEC_ALIGN = 32;
static const __m256i zeros256 = _mm256_setzero_si256();
static const __m256i ones32  = _mm256_set1_epi32(1);

// set
static inline __m256i vec_set_16(int x) { return _mm256_set1_epi16(x); }
template <> __m256i vec_set_16<__m256i>(int16_t x) { return _mm256_set1_epi16(x); }
template <> __m256i vec_set_32<__m256i>(int32_t x) { return _mm256_set1_epi32(x); }
template <> __m256i vec_set_64<__m256i>(int64_t x) { return _mm256_set1_epi64x(x); }
template <> void vec_set_zero(__m256i &x) { x = zero256; }

// load
template <> __m128i vec_load(const int8_t *x) { 
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x)); 
}
template <> __m256i vec_load(const int16_t *x) { 
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(x)); 
}
template <> __m256i vec_load(const int32_t *x) { 
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(x)); 
}
template <> __m256i vec_load(const __m256i *x) { 
    return _mm256_load_si256(x); 
}

// store
template <> void vec_store(__m256i *to, __m256i from) { 
    _mm256_store_si256(to, from); 
}
template <> void vec_store(int16_t *to, __m256i from) { 
    _mm256_store_si256(reinterpret_cast<__m256i*>(to), from); 
}
template <> void vec_store(int32_t *to, __m256i from) { 
    _mm256_store_si256(reinterpret_cast<__m256i*>(to), from); 
}
template <> void vec_store(int64_t *to, __m256i from) { 
    _mm256_store_si256(reinterpret_cast<__m256i*>(to), from); 
}

// unpack
template <> __m256i vec_unpacklo64<__m256i>(__m256i x, __m256i y) { 
    return _mm_unpacklo_epi64(x, y); 
}
template <> __m256i vec_unpackhi64<__m256i>(__m256i x, __m256i y) { 
    return _mm_unpackhi_epi64(x, y); 
}

// shift
template <> __m256i vec_shift_left32<__m256i>(__m256i x, int shift) { 
    return _mm256_slli_epi32(x, shift); 
}
template <> __m256i vec_shift_left64<__m256i>(__m256i x, int shift) { 
    return _mm256_slli_epi64(x, shift); 
}

template <> __m256i vec_shift_right32<__m256i>(__m256i x, int shift) { 
    return _mm256_srli_epi32(x, shift); 
}
template <> __m256i vec_shift_right64<__m256i>(__m256i x, int shift) { 
    return _mm256_srli_epi64(x, shift); 
}

template <> __m256i vec_lo_half<__m256i>(__m256i x) { 
    const __m256i mask_lo32 = vec_set_64<__m256i>(0xFFFFFFFFLL);
    return _mm256_and_si256(x, mask_lo32); 
}
template <> __m256i vec_hi_half<__m256i>(__m256i x) { 
    return _mm256_srli_epi64(x, 32); 
}

// add
template <> __m256i vec_add16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_add_epi16(x, y); 
}
template <> __m256i vec_add32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_add_epi32(x, y); 
}
template <> __m256i vec_add64<__m256i>(__m256i x, __m256i y) { 
    return _mm256_add_epi64(x, y); 
}

// sub
template <> __m256i vec_sub16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_sub_epi16(x, y); 
}
template <> __m256i vec_sub32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_sub_epi32(x, y); 
}

// multiply
template <> __m256i vec_mul16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mul_epi16(x, y); 
}
template <> __m256i vec_mul32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mul_epi32(x, y); 
}

template <> __m256i vec_mullo16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mullo_epi16(x, y); 
}
template <> __m256i vec_mullo32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mullo_epi32(x, y); 
}

template <> __m256i vec_mulhi16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mulhi_epi16(x, y); 
}
template <> __m256i vec_mulhi32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_mulhi_epi32(x, y); 
}

// shuffle 
template <> __m256i vec_shuffle32<__m256i>(__m256i x, int imm) { 
    return _mm256_shuffle_epi32(x, imm); 
}
template <> __m128i vec_shuffle32<__m128i>(__m128i x, int imm) { 
    return _mm_shuffle_epi32(x, imm); 
}

// permute
template <> __m256i vec_permute2x128<__m256i>(__m256i x, __m256i y, int imm) { 
    return _mm256_permute2x128_si256(x, y, imm); 
} 

// clamp
template <> __m256i vec_clamp16<__m256i>(__m256i x, __m256i min, __m256i max) {
    return _mm256_min_epi16(_mm256_max_epi16(x, min), max);
}
template <> __m256i vec_clamp32<__m256i>(__m256i x, __m256i min, __m256i max) {
    return _mm256_min_epi32(_mm256_max_epi32(x, min), max);
}

// convert
template <> __m256i vec_convert_16_32<__m256i>(__m128i x) {
    return _mm256_cvtepi16_epi32(x);
}

template <> int32_t vec_convert_si128_32<__m128i>(__m128i x) {
    return _mm_cvtsi128_si32(x);
}

template <> __m128i vec_cast_256_128<__m128i>(__m256i x) {
    return _mm256_castsi256_si128(x);
}

template <> __m128i vec_extract_256_128<__m128i>(__m256i x, int index) {
    return _mm256_extracti128_si256(x, index);
}


#endif // _WIN32

#endif // SIMD_DEFS_H