#ifndef UTILS_H
#define UTILS_H

#include "helpers.h"
#include "network.h"

// ============================================================
// FEATURE INDEXING - INPUT BUCKETS
//
// ChessBucketsMirrored / 768x32hm
//
// Matches bullet trainer:
// ChessBucketsMirrored::new([usize; 32])
//
// expanded bucket mapping:
//     rank * 4 + [0,1,2,3,3,2,1,0][file]
// and:
//     flip = 7 if file > 3
//            0 otherwise
//
// The underlying 768 feature numbering is the existing
// Chess768 numbering used before
// ============================================================


inline int feature_index_stm_halfka(
    int sq,
    int piece,
    int color,
    int base,
    int flip
) {
    return base
         + (color == 0 ? 0 : 384)
         + piece * 64
         + (sq ^ flip);
}

inline int feature_index_ntm_halfka(
    int sq,
    int piece,
    int color,
    int base,
    int flip
) {
    return base
         + (color == 0 ? 384 : 0)
         + piece * 64
         + ((sq ^ 56) ^ flip);
}

// ========================================================
// Network Functions
// ========================================================

template <typename T>
inline T crelu(const T x, const T clamp_bound) {
    return std::clamp<T>(x, 0, clamp_bound);
}

template <typename T>
inline T screlu(const T x, const T clamp_bound) {
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


// ============================================================
// Helper functions
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

inline int stm_flip(int w_king_sq) {
    return mirrored_flip(w_king_sq);
    
}

inline int ntm_flip(int b_king_sq) {
    return mirrored_flip(b_king_sq ^ 56);
}

inline int stm_base(int w_king_sq) {
    return mirrored_bucket(w_king_sq) * 768;
}

inline int ntm_base(int b_king_sq) {
    return mirrored_bucket(b_king_sq ^ 56) * 768;
}


#endif // UTILS_H