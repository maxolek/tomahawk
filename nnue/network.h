#ifndef NETWORK_H
#define NETWORK_H


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


#endif // NETWORK_H