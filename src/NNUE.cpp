#include <nnue.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>

// ============================================================
// Helpers
// ============================================================

inline int feature_index_stm(int sq, int piece, int color) {
    // color: 0 = white, 1 = black
    return (color == 0 ? 0 : 384) + piece*64 + sq;
}

inline int feature_index_ntm(int sq, int piece, int color) {
    int sq_flip = sq ^ 56;
    return (color == 0 ? 384 : 0) + piece*64 + sq_flip;
}

inline int other_color(int c) { return c ^ 1; }

// ============================================================
// Load quantised.bin
// ============================================================

bool NNUE::load(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "NNUE: failed to open " << path << "\n";
        return false;
    }

    f.read((char*)l0w, sizeof(l0w));
    f.read((char*)l0b, sizeof(l0b));
    f.read((char*)l1w, sizeof(l1w));
    f.read((char*)&l1b, sizeof(l1b));

    if (!f) {
        std::cerr << "NNUE: file truncated or corrupted\n";
        return false;
    }

    //std::cout << "[DEBUG] NNUE loaded\n";
    return true;
}

// ============================================================
// Build accumulators fully from board (STM/NTM)
// ============================================================

void NNUE::build_accumulators(const Board& b) {
    acc_stm.init_bias(l0b);
    acc_ntm.init_bias(l0b);

    int stm = b.move_color;

    U64 bb = b.colorBitboards[0] | b.colorBitboards[1];
    int sq_idx;
    int pc; int pc_c;

    while (bb) {
        sq_idx = getLSB(bb);
        bb &= bb-1;
        
        pc = b.getMovedPiece(sq_idx);
        pc_c = b.getSideAt(sq_idx);

        acc_stm.add_feature(feature_index_stm(sq_idx, pc, pc_c), l0w);
        acc_ntm.add_feature(feature_index_ntm(sq_idx, pc, pc_c), l0w);
    }

    //acc_stm.dump_active_features("build_stm");
    //acc_ntm.dump_active_features("build_ntm");

    // Ensure STM corresponds to side to move
    //if (!b.is_white_move) std::swap(acc_stm, acc_ntm);
}

// ============================================================
// Evaluation
// ============================================================

int NNUE::evaluate(bool is_white_move) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif
    int64_t out64 = 0;

    Accumulator* us = is_white_move ? &acc_stm : &acc_ntm;
    Accumulator* them = is_white_move ? &acc_ntm : &acc_stm;

    // activate, then multiple by weight and add to output (node)
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        out64 += (int64_t)screlu(us->vals[i]) * (int32_t)l1w[i];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        out64 += (int64_t)screlu(them->vals[i]) * (int32_t)l1w[HIDDEN_SIZE + i];

    out64 /= (int64_t)QA;
    out64 += (int64_t)l1b;
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB);

    //std::cout << "eval: " << out64 << "\n\n";

    //us->dump_active_features("eval_stm");
    //them->dump_active_features("eval_ntm");

    return out64; //is_white_move ? out64 : -out64;
}

// lizard screlu
int NNUE::eval_simd(bool is_white_move) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif

    const __m256i vec_zero = _mm256_setzero_si256();
    const __m256i vec_qa   = _mm256_set1_epi32(QA);

    const int32_t* us = is_white_move ? acc_stm.vals : acc_ntm.vals;
    const int32_t* them = is_white_move ? acc_ntm.vals : acc_stm.vals;

    __m256i sum = _mm256_setzero_si256();

    // 8 neurons batched at once (~8x faster than scalar)
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        // Load 8 accumulator values (int32)
        __m256i us_acc = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(us + i)
        );
        __m256i them_acc = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(them + i)
        );

        // ScreLU = clamp(x, 0, QA)^2
        us_acc = _mm256_min_epi32(
            _mm256_max_epi32(us_acc, vec_zero),
            vec_qa
        );
        them_acc = _mm256_min_epi32(
            _mm256_max_epi32(them_acc, vec_zero),
            vec_qa
        );

        __m256i us_sq = _mm256_mullo_epi32(us_acc, us_acc);
        __m256i them_sq = _mm256_mullo_epi32(them_acc, them_acc);

        // Load weights (int16 extended to int32)
        __m128i us_w16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(l1w + i)
        );
        __m128i them_w16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(l1w + HIDDEN_SIZE + i)
        );

        __m256i us_w = _mm256_cvtepi16_epi32(us_w16);
        __m256i them_w = _mm256_cvtepi16_epi32(them_w16);

        // ------------------------------------------------------------
        // Multiply even lanes:
        // lanes 0,2,4,6
        // _mm256_mul_epi32 gives signed int32 * int32 -> int64
        // ------------------------------------------------------------

        __m256i us_even =
            _mm256_mul_epi32(us_sq, us_w);

        __m256i them_even =
            _mm256_mul_epi32(them_sq, them_w);

        sum = _mm256_add_epi64(sum, us_even);
        sum = _mm256_add_epi64(sum, them_even);

        // ------------------------------------------------------------
        // Multiply odd lanes:
        // shift each 64-bit pair right by 32 so that
        // elements 1,3,5,7 become the low 32 bits.
        // ------------------------------------------------------------

        __m256i us_sq_odd = _mm256_srli_epi64(us_sq, 32);
        __m256i us_w_odd = _mm256_srli_epi64(us_w, 32);

        __m256i them_sq_odd = _mm256_srli_epi64(them_sq, 32);
        __m256i them_w_odd = _mm256_srli_epi64(them_w, 32);

        __m256i us_odd = _mm256_mul_epi32(us_sq_odd, us_w_odd);
        __m256i them_odd = _mm256_mul_epi32(them_sq_odd, them_w_odd);

        sum = _mm256_add_epi64(sum, us_odd);
        sum = _mm256_add_epi64(sum, them_odd);
    }

    // ------------------------------------------------------------
    // Horizontal sum of 4 x int64
    // ------------------------------------------------------------

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);

    __m128i total128 = _mm_add_epi64(lo, hi);

    int64_t total =
        _mm_cvtsi128_si64(total128) +
        _mm_extract_epi64(total128, 1);

    // ------------------------------------------------------------
    // Match scalar evaluate() exactly
    // ------------------------------------------------------------

    int64_t out64 = total;

    out64 /= (int64_t)QA;
    out64 += (int64_t)l1b;
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB);

    return (int)out64;
}

int NNUE::full_eval(const Board& b) {
    build_accumulators(b);
    return evaluate(b.is_white_move);
}

// ============================================================
// Incremental updates (STM/NTM)
// ============================================================

// before board.makemove() 
// so board is in pre-move state (old state)
void NNUE::on_make_move(const Board& before, const Move& mv) {
    int moved_piece = before.getMovedPiece(mv.StartSquare());
    int piece_color = before.getSideAt(mv.StartSquare());
    bool promo = mv.IsPromotion();

    // Feature indices for moved piece (both accumulators)
    int f_from_stm = feature_index_stm(mv.StartSquare(), moved_piece, piece_color);
    int f_to_stm   = feature_index_stm(mv.TargetSquare(), moved_piece, piece_color);
    int f_from_ntm = feature_index_ntm(mv.StartSquare(), moved_piece, piece_color);
    int f_to_ntm   = feature_index_ntm(mv.TargetSquare(), moved_piece, piece_color);

    // ---- Update both POV accumulators for the moved piece ----
    acc_stm.remove_feature(f_from_stm, l0w);
    acc_stm.add_feature(f_to_stm, l0w);

    acc_ntm.remove_feature(f_from_ntm, l0w);
    acc_ntm.add_feature(f_to_ntm, l0w);

    // Promotion: replace pawn feature with promoted piece (both POVs)
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm(mv.TargetSquare(), promo_piece, piece_color);
        int f_promo_ntm = feature_index_ntm(mv.TargetSquare(), promo_piece, piece_color);

        // remove pawn entry we just added, then add promoted piece
        acc_stm.remove_feature(f_to_stm, l0w);
        acc_stm.add_feature(f_promo_stm, l0w);

        acc_ntm.remove_feature(f_to_ntm, l0w);
        acc_ntm.add_feature(f_promo_ntm, l0w);
    }

    // ---- Captured piece: remove from BOTH accumulators ----
    int captured_piece = before.getCapturedPiece(mv.TargetSquare());
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        // actual color of captured piece (should equal ntm but use board for safety)
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm(mv.TargetSquare(), captured_piece, cap_color);
        int f_cap_stm = feature_index_stm(mv.TargetSquare(), captured_piece, cap_color);
        acc_ntm.remove_feature(f_cap_ntm, l0w);
        acc_stm.remove_feature(f_cap_stm, l0w);
    }

    // ---- En passant: captured pawn is on cap_sq (remove from BOTH) ----
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8);
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm(cap_sq, pawn, cap_color);
        int f_cap_stm = feature_index_stm(cap_sq, pawn, cap_color);
        acc_ntm.remove_feature(f_cap_ntm, l0w);
        acc_stm.remove_feature(f_cap_stm, l0w);
    }

    // ---- Castling rook: update both POVs for rook from/to ----
    if (mv.MoveFlag() == Move::castleFlag) {
        int rank = (piece_color == 0 ? 0 : 7);
        int rook_from = (mv.TargetSquare() % 8 == 6 ? rank*8 + 7 : rank*8);
        int rook_to   = (mv.TargetSquare() % 8 == 6 ? rank*8 + 5 : rank*8 + 3);

        int f_r_from_stm = feature_index_stm(rook_from, rook, piece_color);
        int f_r_to_stm   = feature_index_stm(rook_to,   rook, piece_color);
        int f_r_from_ntm = feature_index_ntm(rook_from, rook, piece_color);
        int f_r_to_ntm   = feature_index_ntm(rook_to,   rook, piece_color);

        acc_stm.remove_feature(f_r_from_stm, l0w);
        acc_stm.add_feature(f_r_to_stm, l0w);

        acc_ntm.remove_feature(f_r_from_ntm, l0w);
        acc_ntm.add_feature(f_r_to_ntm, l0w);
    }

    //Board b_after = before; b_after.MakeMove(mv);
    //debug_check_features_after_move(b_after);

    // Finally swap so acc_stm always points to the side-to-move for the new board state
    //std::swap(acc_stm, acc_ntm);
}

// called before board.unmake_move() 
// so board is in post-move state
// called before board.unmake_move() 
// so board is in post-move state
void NNUE::on_unmake_move(const Board& board, const Move& mv) {
    bool promo = mv.IsPromotion();

    int moved_piece = board.getMovedPiece(mv.TargetSquare());
    int piece_color = board.getSideAt(mv.TargetSquare());

    int f_to_stm   = feature_index_stm(mv.TargetSquare(), moved_piece, piece_color);
    int f_from_stm = feature_index_stm(mv.StartSquare(), moved_piece, piece_color);
    int f_to_ntm   = feature_index_ntm(mv.TargetSquare(), moved_piece, piece_color);
    int f_from_ntm = feature_index_ntm(mv.StartSquare(), moved_piece, piece_color);

    //std::cout << "[NNUE DEBUG] UNMAKE MOVE " << mv.uci() << "\n";
    //std::cout << "  moved_piece: " << moved_piece
    //          << " color: " << piece_color
    //          << " from=" << mv.StartSquare()
    //          << " to=" << mv.TargetSquare() << "\n";

    // Undo promotion
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm(mv.TargetSquare(), promo_piece, piece_color);
        int f_pawn_stm  = feature_index_stm(mv.StartSquare(), pawn, piece_color);
        int f_promo_ntm = feature_index_ntm(mv.TargetSquare(), promo_piece, piece_color);
        int f_pawn_ntm  = feature_index_ntm(mv.StartSquare(), pawn, piece_color);

        // std::cout << "  Promotion: remove promo " << promo_piece
        //           << " add pawn\n";

        acc_stm.remove_feature(f_promo_stm, l0w);
        acc_stm.add_feature(f_pawn_stm, l0w);
        acc_ntm.remove_feature(f_promo_ntm, l0w);
        acc_ntm.add_feature(f_pawn_ntm, l0w);
    } else {
        // Normal move
        //std::cout << "  Normal: remove to_idx=" << f_to_stm << " add from_idx=" << f_from_stm << "\n";
        acc_stm.remove_feature(f_to_stm, l0w);
        acc_stm.add_feature(f_from_stm, l0w);
        acc_ntm.remove_feature(f_to_ntm, l0w);
        acc_ntm.add_feature(f_from_ntm, l0w);
    }

    // Undo captures
    int captured_piece = board.currentGameState.capturedPieceType;
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare();
        int cap_color = other_color(piece_color); // should be the captured piece color
        int f_cap_stm = feature_index_stm(cap_sq, captured_piece, cap_color);
        int f_cap_ntm = feature_index_ntm(cap_sq, captured_piece, cap_color);

        // std::cout << "  Capture: piece=" << captured_piece
        //           << " color=" << cap_color
        //           << " square=" << cap_sq
        //           << " f_stm=" << f_cap_stm
        //           << " f_ntm=" << f_cap_ntm << "\n";

        acc_stm.add_feature(f_cap_stm, l0w);
        acc_ntm.add_feature(f_cap_ntm, l0w);
    }

    // Undo en passant
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        // The captured pawn is behind the target square in the direction of the moving pawn
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8); 
        int cap_color = other_color(piece_color); // captured pawn color

        int f_cap_stm = feature_index_stm(cap_sq, pawn, cap_color);
        int f_cap_ntm = feature_index_ntm(cap_sq, pawn, cap_color);

        acc_stm.add_feature(f_cap_stm, l0w);
        acc_ntm.add_feature(f_cap_ntm, l0w);
    }


    // Undo castling rook
    if (mv.MoveFlag() == Move::castleFlag) {
        int rank = (piece_color == 0 ? 0 : 7);
        int rook_from = (mv.TargetSquare() % 8 == 6 ? rank*8 + 7 : rank*8);
        int rook_to   = (mv.TargetSquare() % 8 == 6 ? rank*8 + 5 : rank*8 + 3);
        int f_r_from_stm = feature_index_stm(rook_from, rook, piece_color);
        int f_r_to_stm   = feature_index_stm(rook_to,   rook, piece_color);
        int f_r_from_ntm = feature_index_ntm(rook_from, rook, piece_color);
        int f_r_to_ntm   = feature_index_ntm(rook_to,   rook, piece_color);

        //std::cout << "  Castling rook: from=" << rook_from << " to=" << rook_to
        //          << " f_stm_from=" << f_r_from_stm << " f_stm_to=" << f_r_to_stm << "\n";

        acc_stm.remove_feature(f_r_to_stm, l0w);
        acc_stm.add_feature(f_r_from_stm, l0w);
        acc_ntm.remove_feature(f_r_to_ntm, l0w);
        acc_ntm.add_feature(f_r_from_ntm, l0w);
    }
}



// ============================================================
// Debug helpers
// ============================================================

void NNUE::debug_acc_full(const Accumulator& acc, const std::string& name) const {
    int32_t sum = 0, minv = acc.vals[0], maxv = acc.vals[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        sum += acc.vals[i];
        if (acc.vals[i] < minv) minv = acc.vals[i];
        if (acc.vals[i] > maxv) maxv = acc.vals[i];
    }
    std::cout << "[DEBUG] Acc " << name << " sum=" << sum
              << " min=" << minv << " max=" << maxv << " first8=[";
    for (int i = 0; i < 8; ++i) std::cout << acc.vals[i] << (i < 7 ? "," : "");
    std::cout << "]\n";
}
/*
int NNUE::evaluate_debug(bool is_white_move) const {
    debug_acc_full(acc_stm, "STM before screlu");
    debug_acc_full(acc_ntm, "NTM before screlu");

    debug_evaluate(acc_stm, acc_ntm);

    return evaluate(is_white_move);
}
*/

// Utility: Compare two accumulators and print differing feature indices and values
static void debug_diff_features_full(const Accumulator& incr,
                                     const Accumulator& full,
                                     const char* label,
                                     int max_diffs = 40) {
    std::cerr << "   --- Feature Differences (" << label << ") ---\n";
    int count = 0;
    for (int i = 0; i < 256; ++i) {
        if (incr.vals[i] != full.vals[i]) {
            std::cerr << "      idx=" << i
                      << " incr=" << incr.vals[i]
                      << " full=" << full.vals[i] << "\n";

            // Attempt to decode piece/color/square if possible
            int color = (i >= 384) ? 1 : 0;
            int idx_rel = (i >= 384) ? i - 384 : i;
            int piece = idx_rel / 64;
            int sq = idx_rel % 64;
            std::cerr << "         decoded: piece=" << piece
                      << " sq=" << sq
                      << " color=" << color << "\n";

            if (++count >= max_diffs) {
                std::cerr << "      ... (more omitted)\n";
                break;
            }
        }
    }
}

// ============================================================
// DEBUG AFTER MAKE
// ============================================================
void NNUE::debug_check_incr_vs_full_after_make(const Board& before,
                                               const Move& mv,
                                               NNUE& nnue) {
    Board b_after = before;
    nnue.on_make_move(before, mv);
    b_after.MakeMove(mv);

    NNUE nnue_full;
    nnue_full.load("../bin/12x64_0.0.bin");
    nnue_full.build_accumulators(b_after);

    int full_eval = nnue_full.evaluate(b_after.is_white_move);
    int incr_eval = nnue.evaluate(b_after.is_white_move);

    if (abs(full_eval - incr_eval) > 25) {
        int stm = before.move_color;
        int ntm = stm ^ 1;
        int from = mv.StartSquare();
        int to   = mv.TargetSquare();
        int moved_piece = before.getMovedPiece(from);
        int captured_piece = before.getCapturedPiece(to);

        std::cerr << "\n[NNUE DEBUG] MISMATCH AFTER MAKE: "
                  << mv.uci() << " full=" << full_eval
                  << " incr=" << incr_eval << "\n";

        std::cerr << "   STM=" << stm << " NTM=" << ntm
                  << " moved=" << moved_piece
                  << " captured=" << captured_piece << "\n";

        int f_from_stm = feature_index_stm(from, moved_piece, stm);
        int f_to_stm   = feature_index_stm(to, moved_piece, stm);
        int f_cap_ntm  = (captured_piece != -1 ?
            feature_index_ntm(to, captured_piece, ntm) : -1);

        std::cerr << "   f_from_stm=" << f_from_stm
                  << " f_to_stm=" << f_to_stm
                  << " f_cap_ntm=" << f_cap_ntm << "\n";

        //debug_replay_feature_changes(before, mv, b_after);
        debug_expected_changes(before, mv, b_after);
        debug_diff_features_full(nnue.acc_stm, nnue_full.acc_stm, "STM");
        debug_diff_features_full(nnue.acc_ntm, nnue_full.acc_ntm, "NTM");

        abort();
    }
}

// ============================================================
// DEBUG AFTER UNMAKE
// ============================================================
void NNUE::debug_check_incr_vs_full_after_unmake(const Board& board_with_move,
                                                 const Move& mv,
                                                 NNUE& nnue) {
    Board b_before = board_with_move;
    nnue.on_unmake_move(board_with_move, mv);
    b_before.UnmakeMove(mv);

    NNUE nnue_full;
    nnue_full.load("../bin/12x64_0.0.bin");
    nnue_full.build_accumulators(b_before);

    int full_eval = nnue_full.evaluate(b_before.is_white_move);
    int incr_eval = nnue.evaluate(b_before.is_white_move);

    if (abs(full_eval - incr_eval) > 25) {
        int stm = b_before.move_color;
        int ntm = stm ^ 1;

        std::cerr << "\n[NNUE DEBUG] MISMATCH AFTER UNMAKE: "
                  << mv.uci() << " full=" << full_eval
                  << " incr=" << incr_eval << "\n";

        debug_diff_features_full(nnue.acc_stm, nnue_full.acc_stm, "STM");
        debug_diff_features_full(nnue.acc_ntm, nnue_full.acc_ntm, "NTM");

        abort();
    }
}

void NNUE::debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after) {

    auto stm_before = before.move_color;
    auto stm_after  = after.move_color;

    int movingPiece = before.getMovedPiece(m.StartSquare());
    int capturedPiece = before.getCapturedPiece(m.TargetSquare());

    std::cerr << "\n[EXPECTED NNUE CHANGES]\n";

    // Remove from old STM (source)
    std::cerr << "Remove moved (STM old): "
              << feature_index_stm(m.StartSquare(), movingPiece, stm_before) << "\n";

    // Add into old STM (target)
    std::cerr << "Add moved (STM old): "
              << feature_index_stm(m.TargetSquare(), movingPiece, stm_before) << "\n";

    if (capturedPiece != -1) {
        std::cerr << "Remove captured (NTM old): "
                  << feature_index_ntm(m.TargetSquare(), capturedPiece, stm_before ^ 1) << "\n";
    }

    // NEW STM perspective (flipped)
    std::cerr << "Re-add moved under new POV (NTM old side): "
              << feature_index_ntm(m.TargetSquare(), movingPiece, stm_after) << "\n\n";
}


void NNUE::debug_replay_feature_changes(const Board& before,
                                        const Move& mv,
                                        const Board& after) {

    std::cerr << "[NNUE FEATURE CHANGE LOG]\n";

    int stm_before = before.move_color;
    int stm_after  = after.move_color; // flipped after move

    int moved_piece = before.getMovedPiece(mv.StartSquare());
    int captured_piece = before.getCapturedPiece(mv.TargetSquare());

    int from = mv.StartSquare();
    int to   = mv.TargetSquare();

    //  moved piece
    std::cerr << "Deactivate: f_from_stm = "
              << feature_index_stm(from, moved_piece, stm_before) << "\n";
    std::cerr << "Activate:   f_to_stm   = "
              << feature_index_stm(to, moved_piece, stm_before) << "\n";

    //  captured piece (if any)
    if (captured_piece != -1) {
        std::cerr << "Deactivate captured: f_cap_ntm = "
                  << feature_index_ntm(to, captured_piece, stm_before ^ 1)
                  << "\n";
    }

    // After move, STM perspective flips → so these must flip too
    std::cerr << "Post-move should activate NTM: "
              << feature_index_ntm(to, moved_piece, stm_after) << "\n";
}

/*
bool NNUE::check_active_features_consistency(const Accumulator& incr,
                                             const Accumulator& full,
                                             const char* name,
                                             bool abort_on_mismatch) {
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
void NNUE::debug_check_features_after_move(const Board& b) {
    NNUE nnue_full;
    nnue_full.load("../bin/12x64_0.0.bin");
    nnue_full.build_accumulators(b);

    bool stm_correct; bool ntm_correct;

    stm_correct = check_active_features_consistency(acc_stm, nnue_full.acc_stm, "STM", false);
    ntm_correct = check_active_features_consistency(acc_ntm, nnue_full.acc_ntm, "NTM", false);

    if (!stm_correct || !ntm_correct) {
        b.allGameMoves.back().PrintMove();
        std::cout << "white pieces" << std::endl; print_bitboard(b.colorBitboards[0]); 
        std::cout << "black pieces" << std::endl; print_bitboard(b.colorBitboards[1]); 
        abort();
    }
}
*/