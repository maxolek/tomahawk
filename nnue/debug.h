#ifndef DEBUG_H
#define DEBUG_H

#include <nnue.h>
#include <simd.h>

// child of accumulator class
struct DEBUG_ACCUMULATOR : public Accumulator {
    std::unordered_set<int> active_features;

    inline void init_bias(const int16_t* bias) {
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), b);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] = bias[i];
#endif
        active_features.clear();
    }

    inline void add_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            // load 8 int32 accumulator values
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            // load 8 int16 weights
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

            acc = _mm256_add_epi16(acc, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] += col[i];
#endif
        active_features.insert(feature_idx);
    }

    inline void remove_feature(int feature_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* col = W[feature_idx];
#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i));

            acc = _mm256_sub_epi32(acc, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else
        for (int i = 0; i < L1_SIZE; i++)
            vals[i] -= col[i];
#endif
        active_features.erase(feature_idx);
    }

    // typical moves perform both of these
    // so combine to best utilize SIMD
    inline void add_sub_feature(int add_idx, int sub_idx, int16_t (*W)[L1_SIZE]) {
        const int16_t* add_col = W[add_idx];
        const int16_t* sub_col = W[sub_idx];

#ifdef _WIN32
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals + i));
            __m256i add_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add_col + i));
            __m256i sub_w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub_col + i));

            acc = _mm256_add_epi32(acc, add_w);
            acc = _mm256_sub_epi32(acc, sub_w);

            _mm256_store_si256(reinterpret_cast<__m256i*>(vals + i), acc);
        }
#else 
        add_feature(add_idx, W);
        remove_feature(sub_idx, W);
#endif
    }

    void dump_active_features(const char* name) const {
        std::cout << "[ACTIVE FEATURES] " << name << " count=" << active_features.size() << "\n";
        int count = 0;
        for (int f : active_features) {
            std::cout << f << " ";
            if (++count % 16 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }
};


class DEBUG {
public:
    NNUE _nnue = NNUE();
    DEBUG_ACCUMULATOR debug_acc_stm, debug_acc_ntm;

    // accumulators
    void enable(NNUE& nnue);
    void disable(NNUE& nnue);

    // debugging
    void debug_simd(const Board& b);
    void debug_acc(const DEBUG_ACCUMULATOR& acc, const std::string& name) const;
    
    template <typename T>
    void debug_acc_full(const T& acc, const std::string& name) const;
    template <typename T>
    void debug_evaluate(const T& us, const T& them) const;

    void debug_on_move(const std::string& name, const Move& mv, int color, int moved_piece,
                         int f_from, int f_to) const;
    void on_make_move_debug(const Board& before, const Move& mv);
    void on_unmake_move_debug(const Board& board, const Move& mv);
    int evaluate_debug(bool is_white_move) const;
    void debug_check_incr_vs_full_after_make(
        const Board& before,
        const Move& mv,
        bool is_king_move
    );

    void debug_check_incr_vs_full_after_unmake(
        const Board& board_with_move,
        const Move& mv,
        bool is_king_move
    );
    void debug_replay_feature_changes(const Board& before,
                                        const Move& mv,
                                        const Board& after);
    void debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after);

    template <typename T>
    bool check_active_features_consistency(const T& incr,
                                              const T& full,
                                              const char* name,
                                              bool abort_on_mismatch = true);

    void debug_check_features_after_move(const Board& b);
};

#endif