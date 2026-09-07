#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "network.h"
#include "simd.h"
#include "utils.h"

// ============================================================
// Accumulator: holds hidden activations BEFORE SCReLU
// ============================================================
struct Accumulator {
    alignas(32) int16_t vals[L1_SIZE];   // pre-activation <--> post-weight_transform
#ifdef DEBUG
    std::unordered_set<int> active_features;
#endif

    inline void init_bias(const int16_t* bias) {
        #ifdef _WIN32
            init_bias_simd(bias, vals);
        #else
            for (int i = 0; i < L1_SIZE; i++)
                vals[i] = bias[i];
        #endif
        #ifdef DEBUG
            active_features.clear();
        #endif
    }

    inline void add_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];

        #ifdef _WIN32
            add_feature_simd(col, vals);
        #else
            for (int i = 0; i < L1_SIZE; i++)
                vals[i] += col[i];
        #endif
        #ifdef DEBUG
            active_features.insert(feature_idx);
        #endif
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];

        #ifdef _WIN32
                remove_feature_simd(col, vals);
        #else
                for (int i = 0; i < L1_SIZE; i++)
                    vals[i] -= col[i];
        #endif
        #ifdef DEBUG
                active_features.erase(feature_idx);
        #endif
    }

    // typical moves perform both of these
    // so combine to best utilize SIMD
    inline void add_sub_feature(int add_idx, int sub_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* add_col = W[add_idx];
        const int16_t* sub_col = W[sub_idx];

        #ifdef _WIN32
                add_sub_feature_simd(add_col, sub_col, vals);
        #else 
                add_feature(add_idx, W);
                remove_feature(sub_idx, W);
        #endif
    }

#ifdef DEBUG
    void dump_active_features(const char* name) const {
        std::cout << "[ACTIVE FEATURES] " << name << " count=" << active_features.size() << "\n";
        int count = 0;
        for (int f : active_features) {
            std::cout << f << " ";
            if (++count % 16 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }
#endif
};

#endif // ACCUMULATOR_H