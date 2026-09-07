#ifndef SIMD_DEFS_H
#define SIMD_DEFS_H

#include "helpers.h"

static constexpr size_t VEC_ALIGN = 32;

#ifdef _WIN32

// consts
static const __m256i zeros256 = _mm256_setzero_si256();
static const __m256i ones32  = _mm256_set1_epi32(1);

// set
static inline __m256i vec_set_16(int x) { return _mm256_set1_epi16(x); }
template <> __m256i vec_set_16<__m256i>(int16_t x) { return _mm256_set1_epi16(x); }
template <> __m256i vec_set_32<__m256i>(int32_t x) { return _mm256_set1_epi32(x); }
template <> void vec_set_zero(__m256i &x) { x = zero256; }

// store/load
template <> __m256i vec_load(const __m256i *x) { r
    eturn _mm256_load_si256(x); 
}
template <> void vec_store(__m256i *to, __m256i from) { 
    _mm256_store_si256(to, from); 
}

// add/sub
template <> __m256i vec_add16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_add_epi16(x, y); 
}
template <> __m256i vec_add32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_add_epi32(x, y); 
}

template <> __m256i vec_sub16<__m256i>(__m256i x, __m256i y) { 
    return _mm256_sub_epi16(x, y); 
}
template <> __m256i vec_sub32<__m256i>(__m256i x, __m256i y) { 
    return _mm256_sub_epi32(x, y); 
}

// multiply
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

// clamp
template <> __m256i vec_clamp16<__m256i>(__m256i x, __m256i min, __m256i max) {
    return _mm256_min_epi16(_mm256_max_epi16(x, min), max);
}
template <> __m256i vec_clamp32<__m256i>(__m256i x, __m256i min, __m256i max) {
    return _mm256_min_epi32(_mm256_max_epi32(x, min), max);
}

#endif // _WIN32

#endif // SIMD_DEFS_H