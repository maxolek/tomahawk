#include <debug.h>

// accumulator helpers

void DEBUG::enable(NNUE& nnue)  { 
    nnue.set_accumulators(&debug_acc_stm, &debug_acc_ntm); 
}

void DEBUG::disable(NNUE& nnue) { 
    nnue.reset_accumulators(); 
}

// ============================================================
// Debug helpers
// ============================================================

#ifdef _WIN32
void DEBUG::debug_simd(const Board& b) {
    _nnue.build_halfka_accumulators(b);
    U64 occ = b.colorBitboards[0] | b.colorBitboards[1];
    const int bucket = output_bucket(occ);

    Accumulator* us   = b.is_white_move ? _nnue.acc_stm : _nnue.acc_ntm;
    Accumulator* them = b.is_white_move ? _nnue.acc_ntm : _nnue.acc_stm;

    // ============================================================
    // Stage 1: L1 activation (crelu)
    // ============================================================
    int32_t stm_act_s[L1_SIZE], ntm_act_s[L1_SIZE];
    alignas(32) int16_t stm_act_v[L1_SIZE], ntm_act_v[L1_SIZE];
    for (int i = 0; i < L1_SIZE; i++) {
        stm_act_s[i] = crelu(us->vals[i], QA);
        ntm_act_s[i] = crelu(them->vals[i], QA);
    }
    activate_crelu(us->vals,   stm_act_v, L1_SIZE, QA);
    activate_crelu(them->vals, ntm_act_v, L1_SIZE, QA);

    int mm_act1 = 0;
    for (int i = 0; i < L1_SIZE; i++) {
        if (stm_act_s[i] != (int32_t)stm_act_v[i] || ntm_act_s[i] != (int32_t)ntm_act_v[i]) {
            if (mm_act1 < 10)
                std::cerr << "L1 activation mismatch @" << i
                          << " stm scalar=" << stm_act_s[i] << " simd=" << stm_act_v[i]
                          << " ntm scalar=" << ntm_act_s[i] << " simd=" << ntm_act_v[i] << "\n";
            mm_act1++;
        }
    }
    std::cerr << "L1 activation mismatches: " << mm_act1 << " / " << L1_SIZE << "\n";

    // ============================================================
    // Stage 2: L1 pairwise multiply (+ concat)
    // ============================================================
    int32_t l1_pair_s[L1_SIZE];
    alignas(32) int32_t l1_pair_v[L1_SIZE];
    pairwise_mul(stm_act_s, ntm_act_s, l1_pair_s);
    pairwise_mul_simd(stm_act_v, ntm_act_v, l1_pair_v);

    int mm_pair = 0;
    for (int i = 0; i < L1_SIZE; i++) {
        if (l1_pair_s[i] != l1_pair_v[i]) {
            if (mm_pair < 10)
                std::cerr << "L1 pairwise_mul mismatch @" << i
                          << " scalar=" << l1_pair_s[i] << " simd=" << l1_pair_v[i] << "\n";
            mm_pair++;
        }
    }
    std::cerr << "L1 pairwise_mul mismatches: " << mm_pair << " / " << L1_SIZE << "\n";

    // ============================================================
    // Stage 3: L1 -> L2 (dot product with l1w, dequant + bias)
    // ============================================================
    int32_t l2_in_s[L2_SIZE], l2_in_v[L2_SIZE];
    for (int o = 0; o < L2_SIZE; o++) {
        const int8_t* w = _nnue.l1w[bucket * L2_SIZE + o];
        const int32_t bias = _nnue.l1b[bucket * L2_SIZE + o];

        int64_t sum_s = 0;
        for (int i = 0; i < L1_SIZE; i++)
            sum_s += (int64_t)l1_pair_s[i] * (int64_t)w[i];
        sum_s /= (int64_t)QA;
        sum_s += bias;
        l2_in_s[o] = (int32_t)sum_s;

        int64_t sum_v = dot_i32_i8_widen(l1_pair_v, w, L1_SIZE);
        sum_v /= QA;
        sum_v += bias;
        l2_in_v[o] = (int32_t)sum_v;
    }

    int mm_l2in = 0;
    for (int i = 0; i < L2_SIZE; i++) {
        if (l2_in_s[i] != l2_in_v[i]) {
            std::cerr << "L1->L2 dot mismatch @" << i
                       << " scalar=" << l2_in_s[i] << " simd=" << l2_in_v[i] << "\n";
            mm_l2in++;
        }
    }
    std::cerr << "L1->L2 dot mismatches: " << mm_l2in << " / " << L2_SIZE << "\n";

    // ============================================================
    // Stage 4: L2 activation (screlu)
    // ============================================================
    int32_t l2_act_s[L2_SIZE];
    alignas(32) int32_t l2_act_v[L2_SIZE];
    for (int i = 0; i < L2_SIZE; i++)
        l2_act_s[i] = screlu(l2_in_s[i], QA * QB);
    activate_screlu32(l2_in_v, l2_act_v, L2_SIZE, QA * QB);

    int mm_act2 = 0;
    for (int i = 0; i < L2_SIZE; i++) {
        if (l2_act_s[i] != l2_act_v[i]) {
            std::cerr << "L2 activation mismatch @" << i
                       << " scalar=" << l2_act_s[i] << " simd=" << l2_act_v[i] << "\n";
            mm_act2++;
        }
    }
    std::cerr << "L2 activation mismatches: " << mm_act2 << " / " << L2_SIZE << "\n";

    // ============================================================
    // Stage 5: L2 -> L3 (dot product with l2w, dequant + bias)
    // ============================================================
    int64_t l3_in_s[L3_SIZE];
    int32_t l3_in_v[L3_SIZE];
    for (int o = 0; o < L3_SIZE; o++) {
        const int8_t* w = _nnue.l2w[bucket * L3_SIZE + o];
        const int32_t bias = _nnue.l2b[bucket * L3_SIZE + o];

        int64_t sum_s = 0;
        for (int i = 0; i < L2_SIZE; i++)
            sum_s += (int64_t)l2_act_s[i] * (int64_t)w[i];
        sum_s /= (int64_t)(QA * QB);
        sum_s += bias;
        l3_in_s[o] = sum_s;

        int64_t sum_v = dot_i32_i8_widen(l2_act_v, w, L2_SIZE);
        sum_v /= (QA * QB);
        sum_v += bias;
        l3_in_v[o] = (int32_t)sum_v;
    }

    int mm_l3in = 0;
    for (int i = 0; i < L3_SIZE; i++) {
        if (l3_in_s[i] != (int64_t)l3_in_v[i]) {
            std::cerr << "L2->L3 dot mismatch @" << i
                       << " scalar=" << l3_in_s[i] << " simd=" << l3_in_v[i] << "\n";
            mm_l3in++;
        }
    }
    std::cerr << "L2->L3 dot mismatches: " << mm_l3in << " / " << L3_SIZE << "\n";

    // ============================================================
    // Stage 6: L3 activation (screlu, 64-bit)
    // ============================================================
    int64_t l3_act_s[L3_SIZE];
    alignas(32) int64_t l3_act_v[L3_SIZE];
    for (int i = 0; i < L3_SIZE; i++)
        l3_act_s[i] = screlu(l3_in_s[i], (int64_t)(QA * QB * QC));
    activate_screlu64(l3_in_v, l3_act_v, L3_SIZE, QA * QB * QC);

    int mm_act3 = 0;
    for (int i = 0; i < L3_SIZE; i++) {
        if (l3_act_s[i] != l3_act_v[i]) {
            std::cerr << "L3 activation mismatch @" << i
                       << " scalar=" << l3_act_s[i] << " simd=" << l3_act_v[i] << "\n";
            mm_act3++;
        }
    }
    std::cerr << "L3 activation mismatches: " << mm_act3 << " / " << L3_SIZE << "\n";

    // ============================================================
    // Stage 7: L3 -> output (dot product with l3w, dequant + bias + scale)
    // ============================================================
    int64_t out_s = 0;
    for (int i = 0; i < L3_SIZE; i++)
        out_s += (int64_t)l3_act_s[i] * (int64_t)_nnue.l3w[bucket][i];
    out_s /= (int64_t)(QA * QB * QC);
    out_s += _nnue.l3b[bucket];
    out_s *= SCALE;
    out_s /= (int64_t)(QA * QB * QC * QC);

    int64_t out_v = dot_i64_i8(l3_act_v, _nnue.l3w[bucket], L3_SIZE);
    out_v /= (int64_t)(QA * QB * QC);
    out_v += (int64_t)_nnue.l3b[bucket];
    out_v *= SCALE;
    out_v /= (int64_t)(QA * QB * QC * QC);

    std::cerr << "final output: scalar=" << out_s << " simd=" << out_v
               << (out_s != out_v ? "  <-- MISMATCH" : "") << "\n";

    // ============================================================
    // Summary
    // ============================================================
    int total = mm_act1 + mm_pair + mm_l2in + mm_act2 + mm_l3in + mm_act3 + (out_s != out_v ? 1 : 0);
    std::cerr << "=== debug_simd summary: " << total << " total mismatches across all stages ===\n";
}
#endif

template <typename T>
void DEBUG::debug_acc_full(const T& acc, const std::string& name) const {
    int32_t sum = 0, minv = acc.vals[0], maxv = acc.vals[0];
    for (int i = 0; i < L1_SIZE; ++i) {
        sum += acc.vals[i];
        if (acc.vals[i] < minv) minv = acc.vals[i];
        if (acc.vals[i] > maxv) maxv = acc.vals[i];
    }
    std::cout << "[DEBUG] Acc " << name << " sum=" << sum
              << " min=" << minv << " max=" << maxv << " first8=[";
    for (int i = 0; i < 8; ++i) std::cout << acc.vals[i] << (i < 7 ? "," : "");
    std::cout << "]\n";
}

int DEBUG::evaluate_debug(bool is_white_move) const {
    debug_acc_full(_nnue.acc_stm, "STM before screlu");
    debug_acc_full(_nnue.acc_ntm, "NTM before screlu");

    debug_evaluate(_nnue.acc_stm, _nnue.acc_ntm);

    //return _nnue.evaluate(is_white_move);
    return 0;
}


// Utility: Compare two accumulators and print differing feature indices and values

static void decode_halfka_feature(int f, bool ntm) {
    int bucket = f / 768;
    int rel = f % 768;

    int rank = bucket / 4;
    int file = bucket % 4;

    int flip = 0;

    // We don't know the original king file from the bucket alone.
    // The canonical bucket represents files a-d, with e-h folded
    // onto them.
    //
    // Therefore this decoder reports the canonical king square.
    int canonical_king_sq = rank * 8 + file;

    int color_block = rel >= 384 ? 1 : 0;

    int x = rel % 384;
    int piece = x / 64;
    int sq = x % 64;

    if (ntm) {
        color_block ^= 1;
        sq ^= 56;
    }

    std::cerr
        << "feature=" << f
        << " bucket=" << bucket
        << " canonical_king_sq=" << canonical_king_sq
        << " piece=" << piece
        << " sq=" << sq
        << " color=" << color_block
        << "\n";
}

template <typename T>
static void debug_diff_features_full(const T& incr,
                                     const T& full,
                                     const char* label) {
    std::set<int> incr_set(
        incr.active_features.begin(),
        incr.active_features.end());

    std::set<int> full_set(
        full.active_features.begin(),
        full.active_features.end());

    std::cerr << "   --- Feature Differences (" << label << ") ---\n";

    for (int f : incr_set) {
        if (full_set.find(f) == full_set.end())
            std::cerr << "      ONLY IN INCR: " << f << "\n";
    }

    for (int f : full_set) {
        if (incr_set.find(f) == incr_set.end())
            std::cerr << "      ONLY IN FULL: " << f << "\n";
    }

    decode_halfka_feature(46413, false);
    decode_halfka_feature(46069, true);

    std::cerr << "      incr feature count = " << incr_set.size() << "\n";
    std::cerr << "      full feature count = " << full_set.size() << "\n";
}


// ============================================================
// DEBUG AFTER MAKE
// ============================================================
void DEBUG::debug_check_incr_vs_full_after_make(
    const Board& before, const Move& mv, bool is_king_move)
{
    Board b_after = before;
    b_after.MakeMove(mv);

    // Remember the real production pointers.
    Accumulator* real_stm = _nnue.acc_stm;
    Accumulator* real_ntm = _nnue.acc_ntm;

    // Local copies to mutate incrementally.
    Accumulator acc_stm_incr = *real_stm;
    Accumulator acc_ntm_incr = *real_ntm;
    _nnue.set_accumulators(&acc_stm_incr, &acc_ntm_incr);

    if (is_king_move)
        _nnue.build_halfka_accumulators(b_after);
    else
        _nnue.on_make_move_halfka(before, mv);
    // acc_stm_incr / acc_ntm_incr are now updated -- no capture step needed

    _nnue.reset_accumulators();   // back to production pointers

    // Independent full rebuild, unchanged from before
    Accumulator acc_stm_full, acc_ntm_full;
    int w_king_sq = b_after.kingSquare(true);
    int b_king_sq = b_after.kingSquare(false);
    acc_stm_full.init_bias(_nnue.l0b);
    acc_ntm_full.init_bias(_nnue.l0b);

    U64 bb = b_after.colorBitboards[0] | b_after.colorBitboards[1];
    while (bb) {
        int sq = getLSB(bb); bb &= bb - 1;
        int pc = b_after.getMovedPiece(sq);
        int pc_c = b_after.getSideAt(sq);
        const int base_stm = stm_base(w_king_sq), base_ntm = ntm_base(b_king_sq);
        const int flip_stm = stm_flip(w_king_sq), flip_ntm = ntm_flip(b_king_sq);
        acc_stm_full.add_feature(feature_index_stm_halfka(sq, pc, pc_c, base_stm, flip_stm), _nnue.l0w);
        acc_ntm_full.add_feature(feature_index_ntm_halfka(sq, pc, pc_c, base_ntm, flip_ntm), _nnue.l0w);
    }

    _nnue.set_accumulators(&acc_stm_incr, &acc_ntm_incr);
    int incr_eval = _nnue.evaluate(b_after.is_white_move, b_after.colorBitboards[0] | b_after.colorBitboards[1]);

    _nnue.set_accumulators(&acc_stm_full, &acc_ntm_full);
    int full_eval = _nnue.evaluate(b_after.is_white_move, b_after.colorBitboards[0] | b_after.colorBitboards[1]);

    _nnue.reset_accumulators();  // restore production state before any abort()

    if (abs(full_eval - incr_eval) > 25) {
        std::cerr << "\n[NNUE DEBUG] MISMATCH AFTER MAKE: " << mv.uci()
                   << " full=" << full_eval << " incr=" << incr_eval << "\n";
        abort();
    }
}


// ============================================================
// DEBUG AFTER UNMAKE
// ============================================================
void DEBUG::debug_check_incr_vs_full_after_unmake(
    const Board& board_with_move,
    const Move& mv,
    bool is_king_move)
{
    // Production state on entry:
    //   board  = POST-MOVE
    //   acc    = POST-MOVE
    //
    // This function must leave the production state completely
    // untouched.

    Board b_before = board_with_move;
    b_before.UnmakeMove(mv);

    // Save the real production accumulator.
    Accumulator* acc_stm_real = _nnue.acc_stm;
    Accumulator* acc_ntm_real = _nnue.acc_ntm;

    // ------------------------------------------------------------
    // Simulate production incremental UNMAKE on LOCAL accumulators
    // ------------------------------------------------------------

    // Local copies to mutate incrementally.
    Accumulator acc_stm_incr = *acc_stm_real;
    Accumulator acc_ntm_incr = *acc_ntm_real;
    _nnue.set_accumulators(&acc_stm_incr, &acc_ntm_incr);

    if (is_king_move) {
        // Production:
        //   board.UnmakeMove()
        //   build_halfka_accumulators(before)

        _nnue.build_halfka_accumulators(b_before);
    } else {
        // Production:
        //   on_unmake_move_halfka(post_board, mv)
        //   board.UnmakeMove()

        _nnue.on_unmake_move_halfka(board_with_move, mv);
    }

    _nnue.reset_accumulators();   // back to production pointers

    // ------------------------------------------------------------
    // Build completely independent full PRE-MOVE accumulators
    // ------------------------------------------------------------

    Accumulator acc_stm_full;
    Accumulator acc_ntm_full;

    int w_king_sq = b_before.kingSquare(true);
    int b_king_sq = b_before.kingSquare(false);

    acc_stm_full.init_bias(_nnue.l0b);
    acc_ntm_full.init_bias(_nnue.l0b);

    U64 bb = b_before.colorBitboards[0] |
             b_before.colorBitboards[1];

    while (bb) {
        int sq = getLSB(bb);
        bb &= bb - 1;

        int pc    = b_before.getMovedPiece(sq);
        int color = b_before.getSideAt(sq);

        const int base_stm = stm_base(w_king_sq);
        const int base_ntm = ntm_base(b_king_sq);
        const int flip_stm = stm_flip(w_king_sq);
        const int flip_ntm = ntm_flip(b_king_sq);

        acc_stm_full.add_feature(
            feature_index_stm_halfka(
                sq, pc, color, base_stm, flip_stm),
            _nnue.l0w
        );

        acc_ntm_full.add_feature(
            feature_index_ntm_halfka(
                sq, pc, color, base_ntm, flip_ntm),
            _nnue.l0w
        );
    }

    // ------------------------------------------------------------
    // Evaluate simulated incremental result
    // ------------------------------------------------------------

    _nnue.set_accumulators(&acc_stm_incr, &acc_ntm_incr);
    int incr_eval = _nnue.evaluate(b_before.is_white_move, b_before.colorBitboards[0] | b_before.colorBitboards[1]);

    _nnue.set_accumulators(&acc_stm_full, &acc_ntm_full);
    int full_eval = _nnue.evaluate(b_before.is_white_move, b_before.colorBitboards[0] | b_before.colorBitboards[1]);

    _nnue.reset_accumulators();  // restore production state before any abort()

    // ------------------------------------------------------------
    // Compare
    // ------------------------------------------------------------

    if (abs(full_eval - incr_eval) > 25) {

        std::cerr
            << "\n[NNUE DEBUG] MISMATCH AFTER UNMAKE: "
            << mv.uci()
            << " full=" << full_eval
            << " incr=" << incr_eval
            << "\n";

        debug_diff_features_full(
            acc_stm_incr,
            acc_stm_full,
            "STM"
        );

        debug_diff_features_full(
            acc_ntm_incr,
            acc_ntm_full,
            "NTM"
        );

        abort();
    }
}

void DEBUG::debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after) {

    auto stm_before = before.move_color;
    auto stm_after  = after.move_color;

    int movingPiece = before.getMovedPiece(m.StartSquare());
    int capturedPiece = before.getCapturedPiece(m.TargetSquare());

    int w_king_sq = before.kingSquare(true);
    int b_king_sq = before.kingSquare(false);

    const int base_stm = stm_base(w_king_sq);
    const int base_ntm = ntm_base(b_king_sq);
    const int flip_stm = stm_flip(w_king_sq);
    const int flip_ntm = ntm_flip(b_king_sq);

    std::cerr << "\n[EXPECTED NNUE CHANGES]\n";

    // Remove from old STM (source)
    std::cerr << "Remove moved (STM old): "
              << feature_index_stm_halfka(m.StartSquare(), movingPiece, stm_before, base_stm, flip_stm) << "\n";

    // Add into old STM (target)
    std::cerr << "Add moved (STM old): "
              << feature_index_stm_halfka(m.TargetSquare(), movingPiece, stm_before, base_stm, flip_stm) << "\n";

    if (capturedPiece != -1) {
        std::cerr << "Remove captured (NTM old): "
                  << feature_index_ntm_halfka(m.TargetSquare(), capturedPiece, stm_before ^ 1, base_ntm, flip_ntm) << "\n";
    }

    // NEW STM perspective (flipped)
    std::cerr << "Re-add moved under new POV (NTM old side): "
              << feature_index_ntm_halfka(m.TargetSquare(), movingPiece, stm_after, base_ntm, flip_ntm) << "\n\n";
}

template <typename T>
bool DEBUG::check_active_features_consistency(const T& incr,
                                             const T& full,
                                             const char* name,
                                             bool abort_on_mismatch) {
    
    if constexpr (std::is_same_v<T, Accumulator>) {
        std::cout << "non-debug accumulators. active_features not set." << std::endl;
        return false;
    }

    // Convert unordered_set to sorted vector
    std::vector<int> incr_vec(incr.active_features.begin(), incr.active_features.end());
    std::vector<int> full_vec(full.active_features.begin(), full.active_features.end());
    std::sort(incr_vec.begin(), incr_vec.end());
    std::sort(full_vec.begin(), full_vec.end());

    if (incr_vec != full_vec) {
        std::cerr << "[NNUE FEATURE MISMATCH] Accumulator: " << name << "\n";

        std::set<int> incr_set(incr_vec.begin(), incr_vec.end());
        std::set<int> full_set(full_vec.begin(), full_vec.end());

        std::vector<int> only_in_incr, only_in_full;
        std::set_difference(incr_set.begin(), incr_set.end(),
                            full_set.begin(), full_set.end(),
                            std::back_inserter(only_in_incr));
        std::set_difference(full_set.begin(), full_set.end(),
                            incr_set.begin(), incr_set.end(),
                            std::back_inserter(only_in_full));

        // Decode features based on name (STM vs NTM)
        auto decode_feature = [&](int f) {
            int color, piece, sq;
            if (std::string(name) == "STM") {
                color = (f >= 384) ? 1 : 0;
                int idx_rel = (f >= 384) ? f - 384 : f;
                piece = idx_rel / 64;
                sq = idx_rel % 64;
            } else { // NTM
                color = (f < 384) ? 1 : 0;
                int idx_rel = (f < 384) ? f : f - 384;
                piece = idx_rel / 64;
                sq = idx_rel % 64;
                sq ^= 56;  // flip square for NTM perspective
            }
            return std::make_tuple(piece, sq, color);
        };

        auto print_decoded = [&](const std::vector<int>& vec) {
            for (int f : vec) {
                auto [piece, sq, color] = decode_feature(f);
                char file = 'a' + (sq % 8);
                char rank = '1' + (sq / 8);
                std::cerr << "  idx=" << f
                          << " piece=" << piece
                          << " color=" << (color ? "black" : "white")
                          << " square=" << file << rank
                          << "\n";
            }
        };

        if (!only_in_incr.empty()) {
            std::cerr << "  Features only in incremental build:\n";
            print_decoded(only_in_incr);
        }

        if (!only_in_full.empty()) {
            std::cerr << "  Features only in full build:\n";
            print_decoded(only_in_full);
        }

        std::cerr << "  Total incremental: " << incr_vec.size()
                  << ", total full: " << full_vec.size() << "\n";

        if (abort_on_mismatch) abort();

        return false;
    }

    return true;
}


// Usage example inside NNUE class:
// Call after make/unmake move
void DEBUG::debug_check_features_after_move(const Board& b) {
    NNUE nnue_full;
    nnue_full.load("../bin/halfka_1024.bin");
    nnue_full.build_halfka_accumulators(b);

    bool stm_correct; bool ntm_correct;

    stm_correct = check_active_features_consistency(_nnue.acc_stm, nnue_full.acc_stm, "STM", false);
    ntm_correct = check_active_features_consistency(_nnue.acc_ntm, nnue_full.acc_ntm, "NTM", false);

    if (!stm_correct || !ntm_correct) {
        b.allGameMoves.back().PrintMove();
        std::cout << "white pieces" << std::endl; print_bitboard(b.colorBitboards[0]); 
        std::cout << "black pieces" << std::endl; print_bitboard(b.colorBitboards[1]); 
        abort();
    }
}
