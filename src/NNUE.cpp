#include <nnue.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <string>


// ============================================================
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

// Bullet writes affine weights in column-major order for a matrix shaped
// [output_size, input_size]. In the trainer that means the flattened stream is
// effectively the row-major order of the transposed matrix. The C++ side stores
// the same logical matrix as l0w[input_feature][hidden], so we must reconstruct
// the feature-major representation from the flat Bullet stream before use.
static void decode_bullet_l0w(const uint8_t* p, int16_t (*out)[HIDDEN_SIZE]) {
    const int16_t* flat = reinterpret_cast<const int16_t*>(p);
    for (int feature = 0; feature < INPUT_SIZE; ++feature) {
        for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden) {
            //out[feature][hidden] = flat[(size_t)hidden * (size_t)INPUT_SIZE + (size_t)feature];
            out[feature][hidden] = flat[(size_t)feature * (size_t)HIDDEN_SIZE + (size_t)hidden];
        }
    }
}

static void decode_mirrored_bucket(int bucket, int& rank, int& file) {
    rank = bucket / 4;
    file = bucket % 4;
}

// ============================================================
// Load quantised.bin
// ============================================================

bool NNUE::load(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "NNUE: failed to open " << path << "\n";
        return false;
    }

    constexpr std::streamoff expected =
        (std::streamoff)sizeof(l0w) +
        (std::streamoff)sizeof(l0b) +
        (std::streamoff)sizeof(l1w) +
        (std::streamoff)sizeof(l1b);

    f.seekg(0, std::ios::end);
    const std::streamoff file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (file_size < expected) {
        std::cerr << "NNUE: file too small: "
                  << file_size << " bytes, expected at least "
                  << expected << "\n";
        return false;
    }

    // Read the tensor block.
    std::vector<uint8_t> data(static_cast<size_t>(expected));

    f.read(
        reinterpret_cast<char*>(data.data()),
        expected
    );

    if (!f) {
        std::cerr << "NNUE: failed to read network data\n";
        return false;
    }

    const uint8_t* p = data.data();

    // L0
    decode_bullet_l0w(p, l0w);
    p += sizeof(l0w);

    // L0 bias
    std::memcpy(l0b, p, sizeof(l0b));
    p += sizeof(l0b);

    // L1
    std::memcpy(l1w, p, sizeof(l1w));
    p += sizeof(l1w);

    // L1 bias
    std::memcpy(&l1b, p, sizeof(l1b));

    return true;
}

// ============================================================
// Build accumulators fully from board (STM/NTM)
// ============================================================

void NNUE::build_halfka_accumulators(const Board& b) {
    int w_king_sq = b.kingSquare(true);
    int b_king_sq = b.kingSquare(false);

    acc_stm.init_bias(l0b);
    acc_ntm.init_bias(l0b);

    U64 bb = b.colorBitboards[0] | b.colorBitboards[1];
    const int base_stm = mirrored_bucket(w_king_sq) * 768;
    const int base_ntm = mirrored_bucket(b_king_sq ^ 56) * 768;
    const int flip_stm = mirrored_flip(w_king_sq);
    const int flip_ntm = mirrored_flip(b_king_sq ^ 56);

    while (bb) {
        int sq_idx = getLSB(bb);
        bb &= bb - 1;

        int pc   = b.getMovedPiece(sq_idx);
        int pc_c = b.getSideAt(sq_idx);

        int stm_f = feature_index_stm_halfka(sq_idx, pc, pc_c, base_stm, flip_stm);
        int ntm_f = feature_index_ntm_halfka(sq_idx, pc, pc_c, base_ntm, flip_ntm);

        acc_stm.add_feature(stm_f, l0w);
        acc_ntm.add_feature(ntm_f, l0w);

    }

}
// ============================================================
// Evaluation
// ============================================================

int NNUE::evaluate(bool is_white_move, U64 occ) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif
    int64_t out64 = 0;

    Accumulator* us = is_white_move ? &acc_stm : &acc_ntm;
    Accumulator* them = is_white_move ? &acc_ntm : &acc_stm;

    const int bucket = output_bucket(occ);
    const int16_t* weights = l1w[bucket];
    const int32_t bias = l1b[bucket];

    // activate, then multiple by weight and add to output (node)
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        out64 += (int64_t)screlu(us->vals[i]) * (int32_t)weights[i];
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        out64 += (int64_t)screlu(them->vals[i]) * (int32_t)weights[HIDDEN_SIZE + i];

    out64 /= (int64_t)QA;
    out64 += (int64_t)bias;
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB);

    //std::cout << "eval: " << out64 << "\n\n";

    //us->dump_active_features("eval_stm");
    //them->dump_active_features("eval_ntm");

    return out64; //is_white_move ? out64 : -out64;
}

#ifdef _WIN32
// AVX2 -- 8 int32
int NNUE::eval_simd(bool is_white_move, U64 occ) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif

    const __m256i vec_zero = _mm256_setzero_si256();
    const __m256i vec_qa   = _mm256_set1_epi32(QA);

    const int32_t* us = is_white_move ? acc_stm.vals : acc_ntm.vals;
    const int32_t* them = is_white_move ? acc_ntm.vals : acc_stm.vals;

    const int bucket = output_bucket(occ);
    const int16_t* weights = l1w[bucket];
    const int32_t bias = l1b[bucket];

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
            reinterpret_cast<const __m128i*>(weights + i)
        );
        __m128i them_w16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weights + HIDDEN_SIZE + i)
        );
        __m256i us_w = _mm256_cvtepi16_epi32(us_w16);
        __m256i them_w = _mm256_cvtepi16_epi32(them_w16);

        // ------------------------------------------------------------
        // Multiply even lanes:
        // lanes 0,2,4,6
        // _mm256_mul_epi32 gives signed int32 * int32 -> int64
        // ------------------------------------------------------------

        __m256i us_even = _mm256_mul_epi32(us_sq, us_w);
        __m256i them_even = _mm256_mul_epi32(them_sq, them_w);

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
    out64 += (int64_t)bias;
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB);

    return (int)out64;
}
#endif

int NNUE::full_eval(const Board& b) {
    build_halfka_accumulators(b); // build_accumulators(b);
    return evaluate(
        b.is_white_move, 
        (b.colorBitboards[0] | b.colorBitboards[1])
    );
}

// ============================================================
// Incremental updates (STM/NTM)
// ============================================================

// before board.makemove() 
// so board is in pre-move state (old state)
void NNUE::on_make_move_halfka(const Board& before, const Move& mv) {
    int moved_piece = before.getMovedPiece(mv.StartSquare());
    int piece_color = before.getSideAt(mv.StartSquare());
    bool promo = mv.IsPromotion();

    // ONLY CALL FOR NON-KING MOVES
    // ----------------------------
    //if (moved_piece == king) {
    //    build_halfka_accumulators(before);
    //    return;
    //} 

    int w_king_sq = before.kingSquare(true);
    int b_king_sq = before.kingSquare(false);

    const int base_stm = mirrored_bucket(w_king_sq) * 768;
    const int base_ntm = mirrored_bucket(b_king_sq ^ 56) * 768;
    const int flip_stm = mirrored_flip(w_king_sq);
    const int flip_ntm = mirrored_flip(b_king_sq ^ 56);

    // Feature indices for moved piece (both accumulators)
    int f_from_stm = feature_index_stm_halfka(mv.StartSquare(), moved_piece, piece_color, base_stm, flip_stm);
    int f_to_stm   = feature_index_stm_halfka(mv.TargetSquare(), moved_piece, piece_color, base_stm, flip_stm);
    int f_from_ntm = feature_index_ntm_halfka(mv.StartSquare(), moved_piece, piece_color, base_ntm, flip_ntm);
    int f_to_ntm   = feature_index_ntm_halfka(mv.TargetSquare(), moved_piece, piece_color, base_ntm, flip_ntm);

    // ---- Update both POV accumulators for the moved piece ----
    //acc_stm.remove_feature(f_from_stm, l0w);
    //acc_stm.add_feature(f_to_stm, l0w);
    acc_stm.add_sub_feature(f_to_stm, f_from_stm, l0w);

    //acc_ntm.remove_feature(f_from_ntm, l0w);
    //acc_ntm.add_feature(f_to_ntm, l0w);
    acc_ntm.add_sub_feature(f_to_ntm, f_from_ntm, l0w);

    // Promotion: replace pawn feature with promoted piece (both POVs)
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_stm, flip_stm);
        int f_promo_ntm = feature_index_ntm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_ntm, flip_ntm);

        // remove pawn entry we just added, then add promoted piece
        //acc_stm.remove_feature(f_to_stm, l0w);
        //acc_stm.add_feature(f_promo_stm, l0w);
        acc_stm.add_sub_feature(f_promo_stm, f_to_stm, l0w);

        //acc_ntm.remove_feature(f_to_ntm, l0w);
        //acc_ntm.add_feature(f_promo_ntm, l0w);
        acc_ntm.add_sub_feature(f_promo_ntm, f_to_ntm, l0w);
    }

    // ---- Captured piece: remove from BOTH accumulators ----
    int captured_piece = before.getCapturedPiece(mv.TargetSquare());
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        // actual color of captured piece (should equal ntm but use board for safety)
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_stm, flip_stm);
        acc_ntm.remove_feature(f_cap_ntm, l0w);
        acc_stm.remove_feature(f_cap_stm, l0w);
    }

    // ---- En passant: captured pawn is on cap_sq (remove from BOTH) ----
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8);
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        acc_ntm.remove_feature(f_cap_ntm, l0w);
        acc_stm.remove_feature(f_cap_stm, l0w);
    }

    // ---- Castling rook: update both POVs for rook from/to ----
    /* castling=king_move == full build
    if (mv.MoveFlag() == Move::castleFlag) {
        int rank = (piece_color == 0 ? 0 : 7);
        int rook_from = (mv.TargetSquare() % 8 == 6 ? rank*8 + 7 : rank*8);
        int rook_to   = (mv.TargetSquare() % 8 == 6 ? rank*8 + 5 : rank*8 + 3);

        int f_r_from_stm = feature_index_stm_halfka(rook_from, rook, piece_color, base_stm, flip_stm);
        int f_r_to_stm   = feature_index_stm_halfka(rook_to,   rook, piece_color, base_stm, flip_stm);
        int f_r_from_ntm = feature_index_ntm_halfka(rook_from, rook, piece_color, base_ntm, flip_ntm);
        int f_r_to_ntm   = feature_index_ntm_halfka(rook_to,   rook, piece_color, base_ntm, flip_ntm);

        acc_stm.remove_feature(f_r_from_stm, l0w);
        acc_stm.add_feature(f_r_to_stm, l0w);

        acc_ntm.remove_feature(f_r_from_ntm, l0w);
        acc_ntm.add_feature(f_r_to_ntm, l0w);
    }
    */

    //Board b_after = before; b_after.MakeMove(mv);
    //debug_check_features_after_move(b_after);

    // Finally swap so acc_stm always points to the side-to-move for the new board state
    //std::swap(acc_stm, acc_ntm);
}


void NNUE::on_unmake_move_halfka(const Board& board, const Move& mv) {
    bool promo = mv.IsPromotion();

    int moved_piece = board.getMovedPiece(mv.TargetSquare());
    int piece_color = board.getSideAt(mv.TargetSquare());

    // ONLY CALL FOR NON-KING MOVES
    // ----------------------------
    //if (moved_piece == king) {
    //    build_halfka_accumulators(board);
    //   return;
    //} 

    int w_king_sq = board.kingSquare(true);
    int b_king_sq = board.kingSquare(false);

    const int base_stm = mirrored_bucket(w_king_sq) * 768;
    const int base_ntm = mirrored_bucket(b_king_sq ^ 56) * 768;
    const int flip_stm = mirrored_flip(w_king_sq);
    const int flip_ntm = mirrored_flip(b_king_sq ^ 56);

    int f_to_stm   = feature_index_stm_halfka(mv.TargetSquare(), moved_piece, piece_color, base_stm, flip_stm);
    int f_from_stm = feature_index_stm_halfka(mv.StartSquare(), moved_piece, piece_color, base_stm, flip_stm);
    int f_to_ntm   = feature_index_ntm_halfka(mv.TargetSquare(), moved_piece, piece_color, base_ntm, flip_ntm);
    int f_from_ntm = feature_index_ntm_halfka(mv.StartSquare(), moved_piece, piece_color, base_ntm, flip_ntm);

    //std::cout << "[NNUE DEBUG] UNMAKE MOVE " << mv.uci() << "\n";
    //std::cout << "  moved_piece: " << moved_piece
    //          << " color: " << piece_color
    //          << " from=" << mv.StartSquare()
    //          << " to=" << mv.TargetSquare() << "\n";

    // Undo promotion
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_stm, flip_stm);
        int f_pawn_stm  = feature_index_stm_halfka(mv.StartSquare(), pawn, piece_color, base_stm, flip_stm);
        int f_promo_ntm = feature_index_ntm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_ntm, flip_ntm);
        int f_pawn_ntm  = feature_index_ntm_halfka(mv.StartSquare(), pawn, piece_color, base_ntm, flip_ntm);

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
        int f_cap_stm = feature_index_stm_halfka(cap_sq, captured_piece, cap_color, base_stm, flip_stm);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, captured_piece, cap_color, base_ntm, flip_ntm);

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

        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);

        acc_stm.add_feature(f_cap_stm, l0w);
        acc_ntm.add_feature(f_cap_ntm, l0w);
    }


    // Undo castling rook
    /* castling=king_move == full build
    if (mv.MoveFlag() == Move::castleFlag) {
        int rank = (piece_color == 0 ? 0 : 7);
        int rook_from = (mv.TargetSquare() % 8 == 6 ? rank*8 + 7 : rank*8);
        int rook_to   = (mv.TargetSquare() % 8 == 6 ? rank*8 + 5 : rank*8 + 3);
        int f_r_from_stm = feature_index_stm_halfka(rook_from, rook, piece_color, w_king_sq);
        int f_r_to_stm   = feature_index_stm_halfka(rook_to,   rook, piece_color, w_king_sq);
        int f_r_from_ntm = feature_index_ntm_halfka(rook_from, rook, piece_color, b_king_sq);
        int f_r_to_ntm   = feature_index_ntm_halfka(rook_to,   rook, piece_color, b_king_sq);

        //std::cout << "  Castling rook: from=" << rook_from << " to=" << rook_to
        //          << " f_stm_from=" << f_r_from_stm << " f_stm_to=" << f_r_to_stm << "\n";

        acc_stm.remove_feature(f_r_to_stm, l0w);
        acc_stm.add_feature(f_r_from_stm, l0w);
        acc_ntm.remove_feature(f_r_to_ntm, l0w);
        acc_ntm.add_feature(f_r_from_ntm, l0w);
    }
    */
}

// ============================================================
// Debug helpers
// ============================================================

#ifdef _WIN32
void NNUE::debug_simd(const Board& b) {
    build_halfka_accumulators(b);
    int eval_scalar = evaluate(b.is_white_move, (b.colorBitboards[0] | b.colorBitboards[1]));
    int _eval_simd = eval_simd(b.is_white_move, (b.colorBitboards[0] | b.colorBitboards[1]));

    if (eval_scalar != _eval_simd) {
        std::cerr << "[NNUE DEBUG] SIMD mismatch: "
                  << "\n scalar=" << eval_scalar
                  << "\n simd=" << _eval_simd << "\n";
        abort();
    }
}
#endif

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

/*
static void debug_diff_features_full(const Accumulator& incr,
                                     const Accumulator& full,
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
void NNUE::debug_check_incr_vs_full_after_make(
    const Board& before,
    const Move& mv,
    bool is_king_move)
{
    // Production state on entry:
    //   board  = PRE-MOVE
    //   acc    = PRE-MOVE
    //
    // This function must leave the production state completely
    // untouched.

    Board b_after = before;
    b_after.MakeMove(mv);

    // Save the real production accumulator.
    Accumulator acc_stm_real = acc_stm;
    Accumulator acc_ntm_real = acc_ntm;

    // ------------------------------------------------------------
    // Simulate production incremental path on LOCAL accumulators
    // ------------------------------------------------------------

    Accumulator acc_stm_incr = acc_stm_real;
    Accumulator acc_ntm_incr = acc_ntm_real;

    // Temporarily install the local copies so the existing
    // production update functions operate on them.
    acc_stm = acc_stm_incr;
    acc_ntm = acc_ntm_incr;

    if (is_king_move) {
        // Production:
        //   board.MakeMove()
        //   build_halfka_accumulators(after)

        build_halfka_accumulators(b_after);
    } else {
        // Production:
        //   on_make_move_halfka(before, mv)
        //   board.MakeMove()

        on_make_move_halfka(before, mv);
    }

    // Capture the simulated production result.
    acc_stm_incr = acc_stm;
    acc_ntm_incr = acc_ntm;

    // Immediately restore the real production state.
    acc_stm = acc_stm_real;
    acc_ntm = acc_ntm_real;

    // ------------------------------------------------------------
    // Build completely independent full accumulators
    // ------------------------------------------------------------

    Accumulator acc_stm_full;
    Accumulator acc_ntm_full;

    int w_king_sq = b_after.kingSquare(true);
    int b_king_sq = b_after.kingSquare(false);

    acc_stm_full.init_bias(l0b);
    acc_ntm_full.init_bias(l0b);

    U64 bb = b_after.colorBitboards[0] |
             b_after.colorBitboards[1];

    while (bb) {
        int sq = getLSB(bb);
        bb &= bb - 1;

        int pc   = b_after.getMovedPiece(sq);
        int pc_c = b_after.getSideAt(sq);

        acc_stm_full.add_feature(
            feature_index_stm_halfka(
                sq, pc, pc_c, w_king_sq),
            l0w
        );

        acc_ntm_full.add_feature(
            feature_index_ntm_halfka(
                sq, pc, pc_c, b_king_sq),
            l0w
        );
    }

    // ------------------------------------------------------------
    // Evaluate simulated incremental result
    // ------------------------------------------------------------

    acc_stm = acc_stm_incr;
    acc_ntm = acc_ntm_incr;

    int incr_eval = evaluate(b_after.is_white_move);

    // ------------------------------------------------------------
    // Evaluate independent full result
    // ------------------------------------------------------------

    acc_stm = acc_stm_full;
    acc_ntm = acc_ntm_full;

    int full_eval = evaluate(b_after.is_white_move);

    // ------------------------------------------------------------
    // Restore real production state BEFORE reporting/aborting
    // ------------------------------------------------------------

    acc_stm = acc_stm_real;
    acc_ntm = acc_ntm_real;

    // ------------------------------------------------------------
    // Compare
    // ------------------------------------------------------------

    if (abs(full_eval - incr_eval) > 25) {

        std::cerr
            << "\n[NNUE DEBUG] MISMATCH AFTER MAKE: "
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


// ============================================================
// DEBUG AFTER UNMAKE
// ============================================================
void NNUE::debug_check_incr_vs_full_after_unmake(
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
    Accumulator acc_stm_real = acc_stm;
    Accumulator acc_ntm_real = acc_ntm;

    // ------------------------------------------------------------
    // Simulate production incremental UNMAKE on LOCAL accumulators
    // ------------------------------------------------------------

    Accumulator acc_stm_incr = acc_stm_real;
    Accumulator acc_ntm_incr = acc_ntm_real;

    // Temporarily install local copies.
    acc_stm = acc_stm_incr;
    acc_ntm = acc_ntm_incr;

    if (is_king_move) {
        // Production:
        //   board.UnmakeMove()
        //   build_halfka_accumulators(before)

        build_halfka_accumulators(b_before);
    } else {
        // Production:
        //   on_unmake_move_halfka(post_board, mv)
        //   board.UnmakeMove()

        on_unmake_move_halfka(board_with_move, mv);
    }

    // Capture simulated production result.
    acc_stm_incr = acc_stm;
    acc_ntm_incr = acc_ntm;

    // Immediately restore real production state.
    acc_stm = acc_stm_real;
    acc_ntm = acc_ntm_real;

    // ------------------------------------------------------------
    // Build completely independent full PRE-MOVE accumulators
    // ------------------------------------------------------------

    Accumulator acc_stm_full;
    Accumulator acc_ntm_full;

    int w_king_sq = b_before.kingSquare(true);
    int b_king_sq = b_before.kingSquare(false);

    acc_stm_full.init_bias(l0b);
    acc_ntm_full.init_bias(l0b);

    U64 bb = b_before.colorBitboards[0] |
             b_before.colorBitboards[1];

    while (bb) {
        int sq = getLSB(bb);
        bb &= bb - 1;

        int pc    = b_before.getMovedPiece(sq);
        int color = b_before.getSideAt(sq);

        acc_stm_full.add_feature(
            feature_index_stm_halfka(
                sq, pc, color, w_king_sq),
            l0w
        );

        acc_ntm_full.add_feature(
            feature_index_ntm_halfka(
                sq, pc, color, b_king_sq),
            l0w
        );
    }

    // ------------------------------------------------------------
    // Evaluate simulated incremental result
    // ------------------------------------------------------------

    acc_stm = acc_stm_incr;
    acc_ntm = acc_ntm_incr;

    int incr_eval = evaluate(b_before.is_white_move);

    // ------------------------------------------------------------
    // Evaluate independent full result
    // ------------------------------------------------------------

    acc_stm = acc_stm_full;
    acc_ntm = acc_ntm_full;

    int full_eval = evaluate(b_before.is_white_move);

    // ------------------------------------------------------------
    // Restore real production state BEFORE reporting/aborting
    // ------------------------------------------------------------

    acc_stm = acc_stm_real;
    acc_ntm = acc_ntm_real;

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

void NNUE::debug_expected_changes(const Board &before,
                            const Move &m,
                            const Board &after) {

    auto stm_before = before.move_color;
    auto stm_after  = after.move_color;

    int movingPiece = before.getMovedPiece(m.StartSquare());
    int capturedPiece = before.getCapturedPiece(m.TargetSquare());

    int w_king_sq = before.kingSquare(true);
    int b_king_sq = before.kingSquare(false);

    std::cerr << "\n[EXPECTED NNUE CHANGES]\n";

    // Remove from old STM (source)
    std::cerr << "Remove moved (STM old): "
              << feature_index_stm_halfka(m.StartSquare(), movingPiece, stm_before, w_king_sq) << "\n";

    // Add into old STM (target)
    std::cerr << "Add moved (STM old): "
              << feature_index_stm_halfka(m.TargetSquare(), movingPiece, stm_before, w_king_sq) << "\n";

    if (capturedPiece != -1) {
        std::cerr << "Remove captured (NTM old): "
                  << feature_index_ntm_halfka(m.TargetSquare(), capturedPiece, stm_before ^ 1, b_king_sq) << "\n";
    }

    // NEW STM perspective (flipped)
    std::cerr << "Re-add moved under new POV (NTM old side): "
              << feature_index_ntm_halfka(m.TargetSquare(), movingPiece, stm_after, b_king_sq) << "\n\n";
}


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
    nnue_full.load("../bin/halfka_1024.bin");
    nnue_full.build_halfka_accumulators(b);

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