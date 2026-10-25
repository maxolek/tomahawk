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
        init_bias_simd(bias, vals);
        //for (int i = 0; i < L1_SIZE; i++)
            //vals[i] = bias[i];

        #ifdef DEBUG
            active_features.clear();
        #endif
    }

    inline void add_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];

        add_feature_simd(col, vals);
        //for (int i = 0; i < L1_SIZE; i++)
        //    vals[i] += col[i];

        #ifdef DEBUG
            active_features.insert(feature_idx);
        #endif
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];

        remove_feature_simd(col, vals);
        //for (int i = 0; i < L1_SIZE; i++)
            //vals[i] -= col[i];

        #ifdef DEBUG
                active_features.erase(feature_idx);
        #endif
    }

    // typical moves perform both of these
    // so combine to best utilize SIMD
    inline void add_sub_feature(int add_idx, int sub_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* add_col = W[add_idx];
        const int16_t* sub_col = W[sub_idx];

        add_sub_feature_simd(add_col, sub_col, vals);
        //add_feature(add_idx, W);
        //remove_feature(sub_idx, W);
    }

    // this = src + W[add] - W[sub1] (- W[sub2] when sub >= 0), in a signel pass
    inline void copy_update(const Accumulator& src, int add, int sub1, int sub2, const int16_t (*W)[L1_SIZE]) {
        if (sub2 >= 0)  copy_add_sub_sub_simd(src.vals, vals, W[add], W[sub1], W[sub2]);
        else            copy_add_sub_simd(src.vals, vals, W[add], W[sub1]);

        #ifdef DEBUG 
            active_features = src.active_features;
            active_features.insert(add);
            active_features.erase(sub1);
            if (sub2 >= 0) active_features.erase(sub2);
        #endif 
    }

    // this = bias + sum of W[features[i]] 
    inline void refresh(const int16_t* bias, const int16_t (*W)[L1_SIZE], const int* features, int count) {
        refresh_simd(vals, bias, W, features, count);

        #ifdef DEBUG
            active_features.clear();
            for (int f : features)
                active_features.insert(f);
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


// ============================================================
// Accumulator Stack
// ============================================================

// Initial accumulator-stack depth; deeper searches grow the stack on demand.
constexpr int ACC_STACK_RESERVE = 256;

// Incremental accumulator state of one search thread: a [white, black] pair per
// ply. Making a move writes the next ply from the current one and unmaking a move
// steps back. Network weights are shared read-only; each thread owns its stack.
struct AccumulatorStack {
    struct alignas(64) Pair { Accumulator acc[2]; };
    std::vector<Pair> entries = std::vector<Pair>(ACC_STACK_RESERVE);
    int ply = 0;

    Accumulator& white() { return entries[static_cast<size_t>(ply)].acc[0]; }
    Accumulator& black() { return entries[static_cast<size_t>(ply)].acc[1]; }
    const Accumulator& white() const { return entries[static_cast<size_t>(ply)].acc[0]; }
    const Accumulator& black() const { return entries[static_cast<size_t>(ply)].acc[1]; }

    // make sure ply + 1 exists (grows the stack for unusually deep lines)
    void reserve_next() {
        if (ply + 1 >= static_cast<int>(entries.size())) entries.resize(entries.size() * 2);
    }
    void pop() { --ply; }
};

#endif // ACCUMULATOR_H