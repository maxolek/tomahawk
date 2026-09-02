#include <nnue.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <string>

// Bullet writes affine weights in column-major order for a matrix shaped
// [output_size, input_size]. In the trainer that means the flattened stream is
// effectively the row-major order of the transposed matrix. The C++ side stores
// the same logical matrix as l0w[input_feature][hidden], so we must reconstruct
// the feature-major representation from the flat Bullet stream before use.
static void decode_bullet_l0w(const uint8_t* p, int16_t (*out)[L1_SIZE]) {
    const int16_t* flat = reinterpret_cast<const int16_t*>(p);
    for (int feature = 0; feature < INPUT_SIZE; ++feature) {
        for (int hidden = 0; hidden < L1_SIZE; ++hidden) {
            //out[feature][hidden] = flat[(size_t)hidden * (size_t)INPUT_SIZE + (size_t)feature];
            out[feature][hidden] = flat[(size_t)feature * (size_t)L1_SIZE + (size_t)hidden];
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
        (std::streamoff)sizeof(l1b) +
        (std::streamoff)sizeof(l2w) + 
        (std::streamoff)sizeof(l2b) + 
        (std::streamoff)sizeof(l3w) + 
        (std::streamoff)sizeof(l3b);

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
    p += sizeof(l1b);

    // L2
    std::memcpy(l2w, p, sizeof(l2w));
    p += sizeof(l2w);

    // L2 bias
    std::memcpy(&l2b, p, sizeof(l2b));
    p += sizeof(l2b);

    // L3 
    std::memcpy(l3w, p, sizeof(l3w));
    p += sizeof(l3w);

    // L3 bias
    std::memcpy(l3b, p, sizeof(l3b));

    return true;
}

// ACCUMULATOR HELPERS

void NNUE::set_accumulators(Accumulator* stm, Accumulator* ntm) {
    acc_stm = stm;
    acc_ntm = ntm;
}

void NNUE::reset_accumulators() {
    acc_stm = &acc_stm_storage;
    acc_ntm = &acc_ntm_storage;
}

// ============================================================
// Build accumulators fully from board (STM/NTM)
// ============================================================

void NNUE::build_halfka_accumulators(const Board& b) {
    int w_king_sq = b.kingSquare(true);
    int b_king_sq = b.kingSquare(false);

    acc_stm->init_bias(l0b);
    acc_ntm->init_bias(l0b);

    U64 bb = b.colorBitboards[0] | b.colorBitboards[1];
    const int base_stm = stm_base(w_king_sq);
    const int base_ntm = ntm_base(b_king_sq);
    const int flip_stm = stm_flip(w_king_sq);
    const int flip_ntm = ntm_flip(b_king_sq ^ 56);

    while (bb) {
        int sq_idx = getLSB(bb);
        bb &= bb - 1;

        int pc   = b.getMovedPiece(sq_idx);
        int pc_c = b.getSideAt(sq_idx);

        int stm_f = feature_index_stm_halfka(sq_idx, pc, pc_c, base_stm, flip_stm);
        int ntm_f = feature_index_ntm_halfka(sq_idx, pc, pc_c, base_ntm, flip_ntm);

        acc_stm->add_feature(stm_f, l0w);
        acc_ntm->add_feature(ntm_f, l0w);

    }

}
// ============================================================
// Evaluation
// ============================================================

int NNUE::evaluate(bool is_white_move, U64 occ) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif

    // ordering for dual-perspective network
    // concat( us, them )
    Accumulator* us = is_white_move ? acc_stm : acc_ntm;
    Accumulator* them = is_white_move ? acc_ntm : acc_stm;

    // pre-set layer array
    int32_t stm_activated[L1_SIZE];
    int32_t ntm_activated[L1_SIZE];
    int32_t l1_pair_mul[L1_SIZE];
    int32_t l2_in[L2_SIZE];
    int64_t l3_in[L3_SIZE];

    // pre-set output bucket
    const int bucket = output_bucket(occ);

    // ===== L1: accumulator --> hidden 1 =====

    // output vector loop
    for (int o = 0; o < L2_SIZE; o++) {
        int64_t sum = 0;
        const int8_t* w = l1w[bucket * L2_SIZE + o]; // NO PAIRWISE MULTIPLY: 2*LO_SIZE due to concat(dual_perspectives)
        const int32_t bias = l1b[bucket * L2_SIZE + o];

        // activate + pairwise multiply (+ concat)
        for (int i = 0; i < L1_SIZE; i++) {
            stm_activated[i] = crelu(us->vals[i], QA);
            ntm_activated[i] = crelu(them->vals[i], QA);
        }
        pairwise_mul(stm_activated, ntm_activated, l1_pair_mul);
        
        // feed forward to L2 activation (weight multiply, do not activate)
        for (int i = 0; i < L1_SIZE; i++) 
            sum += (int64_t)l1_pair_mul[i] * (int64_t)w[i];

        // dequantize, add bias, load to output vector
        sum /= (int64_t)QA;
        sum += bias;
        l2_in[o] = (int32_t)sum;
    }

    // ===== L2: hidden 1 --> hidden 2 =====

    // activate (in place)
    // do before to prevent redundant re-computes
    for (int i = 0; i < L2_SIZE; i++)
        l2_in[i] = screlu(l2_in[i], QA*QB);

    // weight loop
    for (int o = 0; o < L3_SIZE; o++) {
        int64_t sum = 0;
        const int8_t* w = l2w[bucket * L3_SIZE + o];
        const int32_t bias = l2b[bucket * L3_SIZE + o];

        // feed activates neurons forward to L3 pre-activation
        for (int i = 0; i < L2_SIZE; i++)
            sum += (int64_t)l2_in[i] * (int64_t)w[i];

        // dequantize, bias, and load
        sum /= (int64_t)(QA * QB); 
        sum += bias; 
        l3_in[o] = sum;
    }

    // ===== L3: hidden 2 --> output   =====

    // activate
    for (int i = 0; i < L3_SIZE; i++)
        l3_in[i] = screlu(l3_in[i], (int64_t)(QA*QB*QC));

    // weight loop
    int64_t out64 = 0;
    for (int i = 0; i < L3_SIZE; i++) 
        out64 += (int64_t)l3_in[i] * (int64_t)l3w[bucket][i];

    // dequantize, bias, and output
    out64 /= (int64_t)(QA * QB * QC); // (QA*QB*QC)*(QA*QB*QC)*QC --> QA*QB*QC*QC
    out64 += l3b[bucket];
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB * QC * QC);

    return out64;
}

#ifdef _WIN32
// AVX2 -- 32 int8
//      -- 16 int16
//      -- 8  int32
//      -- 4  int64
int NNUE::eval_simd(bool is_white_move, U64 occ) {
    #ifdef DEV
        ScopedTimer timer(T_NNUE);
    #endif

    // ordering for dual-perspective network
    // concat( us, them )
    Accumulator* us = is_white_move ? acc_stm : acc_ntm;
    Accumulator* them = is_white_move ? acc_ntm : acc_stm;

    // pre-set layer array
    alignas(32) int16_t us_act[L1_SIZE];
    alignas(32) int16_t them_act[L1_SIZE];
    alignas(32) int32_t l1_out[L1_SIZE];
    alignas(32) int32_t l2_in[L2_SIZE];
    alignas(32) int32_t l2_act[L2_SIZE];
    alignas(32) int32_t l3_in[L3_SIZE];
    alignas(32) int64_t l3_act[L3_SIZE];

    // pre-set output bucket
    const int bucket = output_bucket(occ);

    // ===== L1: accumulator --> hidden 1 =====

    // activate accumulator values
    // Crelu instead of SCrelu since pairwise mult gives us non-linearity
    //    and scale of (QA*QA)*(QA*QA) is huge
    activate_crelu(us->vals, us_act, L1_SIZE, QA);
    activate_crelu(them->vals, them_act, L1_SIZE, QA);

    // pairwise multiply + concat
    pairwise_mul_simd(us_act, them_act, l1_out);

    // weight transform
    //  AVX2 doesnt have int16 x int8 directly
    //  so widen weights to in16
    for (int o = 0; o < L2_SIZE; o++) {
        const int8_t* w = l1w[bucket * L2_SIZE + o];
        const int32_t bias = l1b[bucket * L2_SIZE + o];

        // create layer nodes (pre-activation)
        int64_t sum = dot_i32_i8_widen(l1_out, w, L1_SIZE);

        sum /= QA; // (QA*QA)*QB --> QA*QB
        sum += bias;
        l2_in[o] = sum;
    }

    // ===== L2: hidden 1 --> hidden 2 =====

    // activate
    activate_screlu32(l2_in, l2_act, L2_SIZE, QA*QB);

    // weight transform
    for (int o = 0; o < L3_SIZE; o++) {
        const int8_t* w = l2w[bucket * L3_SIZE + o];
        const int32_t bias = l2b[bucket * L3_SIZE + o];
        
        int64_t sum = dot_i32_i8_widen(l2_act, w, L2_SIZE);

        sum /= (QA * QB); // (QA*QB)*(QA*QB)*QC --> QA*QB*QC
        sum += bias;
        l3_in[o] = (int32_t)sum; 
    }

    // ===== L3: hidden 2 --> output =====

    // activate
    activate_screlu64(l3_in, l3_act, L3_SIZE, QA*QB*QC);

    // weight transform 
    int64_t out = (int64_t)dot_i64_i8(l3_act, l3w[bucket], L3_SIZE);

    out /= (int64_t)(QA * QB * QC); // (QA*QB*QC)*(QA*QB*QC)*QC --> QA*QB*QC*QC
    out += (int64_t)l3b[bucket];
    out *= SCALE;
    out /= (int64_t)(QA * QB * QC * QC);

    return (int)out;
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
    acc_stm->add_sub_feature(f_to_stm, f_from_stm, l0w);

    //acc_ntm.remove_feature(f_from_ntm, l0w);
    //acc_ntm.add_feature(f_to_ntm, l0w);
    acc_ntm->add_sub_feature(f_to_ntm, f_from_ntm, l0w);

    // Promotion: replace pawn feature with promoted piece (both POVs)
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_stm, flip_stm);
        int f_promo_ntm = feature_index_ntm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_ntm, flip_ntm);

        // remove pawn entry we just added, then add promoted piece
        //acc_stm.remove_feature(f_to_stm, l0w);
        //acc_stm.add_feature(f_promo_stm, l0w);
        acc_stm->add_sub_feature(f_promo_stm, f_to_stm, l0w);

        //acc_ntm.remove_feature(f_to_ntm, l0w);
        //acc_ntm.add_feature(f_promo_ntm, l0w);
        acc_ntm->add_sub_feature(f_promo_ntm, f_to_ntm, l0w);
    }

    // ---- Captured piece: remove from BOTH accumulators ----
    int captured_piece = before.getCapturedPiece(mv.TargetSquare());
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        // actual color of captured piece (should equal ntm but use board for safety)
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_stm, flip_stm);
        acc_ntm->remove_feature(f_cap_ntm, l0w);
        acc_stm->remove_feature(f_cap_stm, l0w);
    }

    // ---- En passant: captured pawn is on cap_sq (remove from BOTH) ----
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8);
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        acc_ntm->remove_feature(f_cap_ntm, l0w);
        acc_stm->remove_feature(f_cap_stm, l0w);
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

    const int base_stm = stm_base(w_king_sq);
    const int base_ntm = ntm_base(b_king_sq);
    const int flip_stm = stm_flip(w_king_sq);
    const int flip_ntm = ntm_flip(b_king_sq);

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

        //acc_stm.remove_feature(f_promo_stm, l0w);
        //acc_stm.add_feature(f_pawn_stm, l0w);
        acc_stm->add_sub_feature(f_pawn_stm, f_promo_stm, l0w);
        //acc_ntm.remove_feature(f_promo_ntm, l0w);
        //acc_ntm.add_feature(f_pawn_ntm, l0w);
        acc_ntm->add_sub_feature(f_pawn_ntm, f_promo_ntm, l0w);
    } else {
        // Normal move
        //std::cout << "  Normal: remove to_idx=" << f_to_stm << " add from_idx=" << f_from_stm << "\n";
        //acc_stm.remove_feature(f_to_stm, l0w);
        //acc_stm.add_feature(f_from_stm, l0w);
        acc_stm->add_sub_feature(f_from_stm, f_to_stm, l0w);
        //acc_ntm.remove_feature(f_to_ntm, l0w);
        //acc_ntm.add_feature(f_from_ntm, l0w);
        acc_ntm->add_sub_feature(f_from_ntm, f_to_ntm, l0w);
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

        acc_stm->add_feature(f_cap_stm, l0w);
        acc_ntm->add_feature(f_cap_ntm, l0w);
    }

    // Undo en passant
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        // The captured pawn is behind the target square in the direction of the moving pawn
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8); 
        int cap_color = other_color(piece_color); // captured pawn color

        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);

        acc_stm->add_feature(f_cap_stm, l0w);
        acc_ntm->add_feature(f_cap_ntm, l0w);
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

