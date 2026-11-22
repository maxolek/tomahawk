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

// the feature transformer is input-sparse, so input-major [feature][hidden]
// is better for simd as our transforms are in the hidden dim mainly
// fixed input neuron - all output weights are contiguous
static void decode_bullet_l0w(const uint8_t* p, int16_t (*out)[L1_SIZE]) {
    const int16_t* flat = reinterpret_cast<const int16_t*>(p);
    for (int feature = 0; feature < INPUT_SIZE; ++feature) {
        for (int hidden = 0; hidden < L1_SIZE; ++hidden) {
            //out[feature][hidden] = flat[(size_t)hidden * (size_t)INPUT_SIZE + (size_t)feature];
            out[feature][hidden] = flat[(size_t)feature * (size_t)L1_SIZE + (size_t)hidden];
        }
    }
}

// output-major [output][input] is preferred for simd in dense layouts
// fixed output neuron - all input weights are contiguous
template <typename T, int NUM_OUT, int NUM_IN>
static void decode_bullet_affine(const uint8_t* p, T* out) {
    const T* flat = reinterpret_cast<const T*>(p);
    for (int i = 0; i < NUM_IN; ++i)
        for (int o = 0; o < NUM_OUT; ++o)
            out[o * NUM_IN + i] = flat[(size_t)i * NUM_OUT + (size_t)o];
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

    // L0 (input-major column reader)
    decode_bullet_l0w(p, l0w);
    p += sizeof(l0w);
    std::memcpy(l0b, p, sizeof(l0b));
    p += sizeof(l0b);

    // L1  (in = L1_SIZE, out = L2_SIZE * NUM_OUTPUT_BUCKETS)
    decode_bullet_affine<int8_t, L2_SIZE * NUM_OUTPUT_BUCKETS, L1_SIZE>(p, &l1w[0][0]);
    p += sizeof(l1w);
    std::memcpy(&l1b, p, sizeof(l1b));   // bias is 1-D, memcpy is fine
    p += sizeof(l1b);

    // L2  (in = L2_SIZE, out = L3_SIZE * NUM_OUTPUT_BUCKETS)
    decode_bullet_affine<int8_t, L3_SIZE * NUM_OUTPUT_BUCKETS, L2_SIZE>(p, &l2w[0][0]);
    p += sizeof(l2w);
    std::memcpy(&l2b, p, sizeof(l2b));
    p += sizeof(l2b);

    // L3  (in = L3_SIZE, out = NUM_OUTPUT_BUCKETS)
    decode_bullet_affine<int8_t, NUM_OUTPUT_BUCKETS, L3_SIZE>(p, &l3w[0][0]);
    p += sizeof(l3w);
    std::memcpy(l3b, p, sizeof(l3b));

    build_inference_layouts();

    return true;
}

void NNUE::build_inference_layouts() {
    for (int b = 0; b < NUM_OUTPUT_BUCKETS; ++b) {

        for (int k = 0; k < L1_SIZE / 2; ++k) {
            for (int o = 0; o < L2_SIZE; ++o) {
                l1w_pairs[b][k][2 * o]     = l1w[b * L2_SIZE + o][2 * k];
                l1w_pairs[b][k][2 * o + 1] = l1w[b * L2_SIZE + o][2 * k + 1];
            }
        }

        for (int i = 0; i < L2_SIZE; i++) {
            for (int o = 0; o < L3_SIZE; o++) 
                l2w_by_input[b][i][o] = l2w[b * L3_SIZE + o][i];
        }
        
        for (int i = 0; i < L3_SIZE; i++) {
            l3w_i32[b][i] = l3w[b][i];
        }
    }
}

#ifdef SMALL_NET
bool NNUE::loadSmall(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "NNUE: failed to open " << path << "\n";
        return false;
    }

    constexpr std::streamoff small_head_weights_size_i8 =
        NUM_OUTPUT_BUCKETS * 2 * hl_size * static_cast<std::streamoff>(sizeof(int8_t));
    constexpr std::streamoff small_head_weights_size_i16 =
        NUM_OUTPUT_BUCKETS * 2 * hl_size * static_cast<std::streamoff>(sizeof(int16_t));
    constexpr std::streamoff small_head_bias_size_i8 =
        NUM_OUTPUT_BUCKETS * static_cast<std::streamoff>(sizeof(int32_t));
    constexpr std::streamoff small_head_bias_size_i16 =
        NUM_OUTPUT_BUCKETS * static_cast<std::streamoff>(sizeof(int16_t));
    constexpr std::streamoff tensor_base_size =
        (std::streamoff)sizeof(l0w) +
        (std::streamoff)sizeof(l0b);
    constexpr std::streamoff expected_i8 =
        tensor_base_size + small_head_weights_size_i8 + small_head_bias_size_i8;
    constexpr std::streamoff expected_i16 =
        tensor_base_size + small_head_weights_size_i16 + small_head_bias_size_i16;

    f.seekg(0, std::ios::end);
    const std::streamoff file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    const bool legacy_i16 = file_size >= expected_i16 && file_size < expected_i16 + 64;
    const bool current_i8 = file_size >= expected_i8 && file_size < expected_i8 + 64;
    if (!legacy_i16 && !current_i8) {
        std::cerr << "NNUE: file too small: "
                  << file_size << " bytes, expected at least "
                  << expected_i8 << " (int8 head) or " << expected_i16
                  << " (legacy int16 head), each padded to 64 bytes\n";
        return false;
    }

    // Read the tensor block.
    const std::streamoff tensor_size = legacy_i16 ? expected_i16 : expected_i8;
    std::vector<uint8_t> data(static_cast<size_t>(tensor_size));

    f.read(
        reinterpret_cast<char*>(data.data()),
        tensor_size
    );

    if (!f) {
        std::cerr << "NNUE: failed to read network data\n";
        return false;
    }

    const uint8_t* p = data.data();

    // L0 (input-major column reader)
    decode_bullet_l0w(p, l0w);
    p += sizeof(l0w);
    std::memcpy(l0b, p, sizeof(l0b));
    p += sizeof(l0b);

    if (legacy_i16) {
        std::memcpy(l1w_small_i16, p, sizeof(l1w_small_i16));
        small_head_format = SmallHeadFormat::I16_HEAD;
        p += small_head_weights_size_i16;
    } else {
        decode_bullet_affine<int8_t, NUM_OUTPUT_BUCKETS, 2 * hl_size>(
            p, &l1w_small[0][0]);
        small_head_format = SmallHeadFormat::I8_HEAD;
        p += small_head_weights_size_i8;
    }
    if (legacy_i16) {
        int16_t biases[NUM_OUTPUT_BUCKETS];
        std::memcpy(biases, p, sizeof(biases));
        for (int bucket = 0; bucket < NUM_OUTPUT_BUCKETS; ++bucket)
            l1b_small[bucket] = biases[bucket];
    } else {
        std::memcpy(l1b_small, p, sizeof(l1b_small));
    }

    return true;
}
#endif


// ============================================================
// Build accumulators fully from board (STM/NTM)
// ============================================================

void NNUE::build_halfka_accumulators(AccumulatorStack& stack, const Board& b) const {
    refresh_perspective(stack, b, 0);
    refresh_perspective(stack, b, 1);
}

void NNUE::refresh_perspective(AccumulatorStack& stack, const Board& b, int perspective) const {
    int features[64];
    int count = 0;
    U64 bb = b.colorBitboards[0] | b.colorBitboards[1];

    if (perspective == 0) {
        const int w_king_sq = b.kingSquare(true);
        const int base_stm = stm_base(w_king_sq);
        const int flip_stm = stm_flip(w_king_sq);
        while (bb) {
            const int sq_idx = getLSB(bb);
            bb &= bb - 1;
            features[count++] = feature_index_stm_halfka(sq_idx, b.getMovedPiece(sq_idx), b.getSideAt(sq_idx), base_stm, flip_stm);
        }
        stack.white().refresh(l0b, l0w, features, count);
    } else {
        const int b_king_sq = b.kingSquare(false);
        const int base_ntm = ntm_base(b_king_sq);
        const int flip_ntm = ntm_flip(b_king_sq ^ 56);
        while (bb) {
            const int sq_idx = getLSB(bb);
            bb &= bb - 1;
            features[count++] = feature_index_ntm_halfka(sq_idx, b.getMovedPiece(sq_idx), b.getSideAt(sq_idx), base_ntm, flip_ntm);
        }
        stack.black().refresh(l0b, l0w, features, count);
    }
}

// ============================================================
// Accumulator stack
// ============================================================

void NNUE::reset_stack(AccumulatorStack& stack, const Board& b) const {
    stack.ply = 0;
    build_halfka_accumulators(stack, b);
}

NNUE::AccumulatorDelta NNUE::move_delta(const Board& before, const Move& mv) const {
    const int from = mv.StartSquare();
    const int to = mv.TargetSquare();
    const int moved_piece = before.getMovedPiece(from);
    const int piece_color = before.getSideAt(from);
    const int cap_color = other_color(piece_color);
    // a promotion replaces the pawn by the promoted piece on the target square
    const int arriving = mv.IsPromotion() ? mv.PromotionPieceType() : moved_piece;

    const int w_king_sq = before.kingSquare(true);
    const int b_king_sq = before.kingSquare(false);
    const int base_stm = mirrored_bucket(w_king_sq) * 768;
    const int base_ntm = mirrored_bucket(b_king_sq ^ 56) * 768;
    const int flip_stm = mirrored_flip(w_king_sq);
    const int flip_ntm = mirrored_flip(b_king_sq ^ 56);

    const int add_stm = feature_index_stm_halfka(to, arriving, piece_color, base_stm, flip_stm);
    const int sub_stm = feature_index_stm_halfka(from, moved_piece, piece_color, base_stm, flip_stm);
    const int add_ntm = feature_index_ntm_halfka(to, arriving, piece_color, base_ntm, flip_ntm);
    const int sub_ntm = feature_index_ntm_halfka(from, moved_piece, piece_color, base_ntm, flip_ntm);

    int cap_stm = -1, cap_ntm = -1;
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        const int cap_sq = to + (piece_color == 0 ? -8 : 8);
        cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);
    } else {
        const int captured_piece = before.getCapturedPiece(to);
        if (captured_piece != -1) {
            cap_stm = feature_index_stm_halfka(to, captured_piece, cap_color, base_stm, flip_stm);
            cap_ntm = feature_index_ntm_halfka(to, captured_piece, cap_color, base_ntm, flip_ntm);
        }
    }

    return AccumulatorDelta{{add_stm, add_ntm}, {sub_stm, sub_ntm}, {cap_stm, cap_ntm}};
}

void NNUE::push_delta(AccumulatorStack& stack, const AccumulatorDelta& d) const {
    stack.reserve_next();
    const AccumulatorStack::Pair& src = stack.entries[static_cast<size_t>(stack.ply)];
    AccumulatorStack::Pair& dst = stack.entries[static_cast<size_t>(stack.ply) + 1];
    dst.acc[0].copy_update(src.acc[0], d.add[0], d.sub[0], d.cap[0], l0w);
    dst.acc[1].copy_update(src.acc[1], d.add[1], d.sub[1], d.cap[1], l0w);
    ++stack.ply;
}

void NNUE::push_king_move(AccumulatorStack& stack, const Board& after, const Move& mv) const {
    stack.reserve_next();
    ++stack.ply;

    // castling also moves a rook
    if (mv.MoveFlag() == Move::castleFlag) {
        build_halfka_accumulators(stack, after);
        return;
    }

    const int mover = after.is_white_move ? 1 : 0;
    refresh_perspective(stack, after, mover);

    // the other perspective keeps its king bucket: only the moving king and a
    // captured piece change
    const int other = 1 - mover;
    const int from = mv.StartSquare();
    const int to = mv.TargetSquare();
    const int captured = after.currentGameState.capturedPieceType;
    const Accumulator& src = stack.entries[static_cast<size_t>(stack.ply) - 1].acc[other];
    Accumulator& dst = stack.entries[static_cast<size_t>(stack.ply)].acc[other];

    if (other == 0) {
        const int w_king_sq = after.kingSquare(true);
        const int base = stm_base(w_king_sq);
        const int flip = stm_flip(w_king_sq);
        dst.copy_update(src,
                        feature_index_stm_halfka(to, king, mover, base, flip),
                        feature_index_stm_halfka(from, king, mover, base, flip),
                        captured != -1 ? feature_index_stm_halfka(to, captured, other, base, flip) : -1,
                        l0w);
    } else {
        const int b_king_sq = after.kingSquare(false);
        const int base = ntm_base(b_king_sq);
        const int flip = ntm_flip(b_king_sq ^ 56);
        dst.copy_update(src,
                        feature_index_ntm_halfka(to, king, mover, base, flip),
                        feature_index_ntm_halfka(from, king, mover, base, flip),
                        captured != -1 ? feature_index_ntm_halfka(to, captured, other, base, flip) : -1,
                        l0w);
    }
}


// ============================================================
// Evaluation
// ============================================================

int NNUE::evaluate(const AccumulatorStack& stack, bool is_white_move, U64 occ) const {
    #if defined(DEV) || defined(TIMING)
        ScopedTimer timer(T_NNUE);
    #endif

    // ordering for dual-perspective network
    // concat( us, them )
    const Accumulator* us = is_white_move ? &stack.white() : &stack.black();
    const Accumulator* them = is_white_move ? &stack.black() : &stack.white();

    // pre-set layer array
    int32_t stm_activated[L1_SIZE];
    int32_t ntm_activated[L1_SIZE];
    int32_t l1_pair_mul[L1_SIZE];
    int32_t l2_in[L2_SIZE];
    int64_t l3_in[L3_SIZE];

    // pre-set output bucket
    const int bucket = output_bucket(occ);


    // ===== L1: accumulator -. hidden 1 =====

    // output vector loop
    for (int o = 0; o < L2_SIZE; o++) {
        int64_t sum = 0;
        const int8_t* w = l1w[bucket * L2_SIZE + o]; 
        const int32_t bias = l1b[bucket * L2_SIZE + o];

        // activate + pairwise multiply (+ concat)
        for (int i = 0; i < L1_SIZE; i++) {
            stm_activated[i] = crelu<int32_t>(us->vals[i], QA);
            ntm_activated[i] = crelu<int32_t>(them->vals[i], QA);
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

    // ===== L2: hidden 1 -. hidden 2 =====

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

    // ===== L3: hidden 2 -. output   =====

    // activate
    for (int i = 0; i < L3_SIZE; i++)
        l3_in[i] = screlu(l3_in[i], (int64_t)(QA*QB*QC));

    // weight loop
    int64_t out64 = 0;
    for (int i = 0; i < L3_SIZE; i++) 
        out64 += (int64_t)l3_in[i] * (int64_t)l3w[bucket][i];

    // dequantize, bias, and output
    out64 /= (int64_t)(QA * QB * QC); // (QA*QB*QC)*(QA*QB*QC)*QC -. QA*QB*QC*QC
    out64 += l3b[bucket];
    out64 *= SCALE;
    out64 /= (int64_t)(QA * QB * QC * QC);

    return static_cast<int>(out64);
}

// AVX2 / NEON -- 32 int8
//      -- 16 int16
//      -- 8  int32
//      -- 4  int64
int NNUE::eval_simd(const AccumulatorStack& stack, bool is_white_move, U64 occ) const {
    #if defined(DEV) || defined(TIMING)
        ScopedTimer timer(T_NNUE);
    #endif

    // ordering for dual-perspective network
    // concat( us, them )
    const Accumulator* us = is_white_move ? &stack.white() : &stack.black();
    const Accumulator* them = is_white_move ? &stack.black() : &stack.white();

    // pre-set output bucket
    const int bucket = output_bucket(occ);

    alignas(32) int32_t l2_in[L2_SIZE];
    alignas(32) int32_t l2_act[L2_SIZE];
    alignas(32) int32_t l3_in[L3_SIZE];

#if defined(__AVX2__)

    // ===== L1: accumulator -. hidden 1 (CReLU, pairwise multiply, sparse affine) =====
    alignas(32) int32_t l1_sums[L2_SIZE];
    l1_sparse_avx2(us->vals, them->vals, l1w_pairs[bucket], l1_sums);
    for (int o = 0; o < L2_SIZE; o++) {
        int64_t sum = l1_sums[o];
        sum /= QA; // (QA*QA)*QB -. QA*QB
        sum += l1b[bucket * L2_SIZE + o];
        l2_in[o] = static_cast<int32_t>(sum);
    }

    // ===== L2: hidden 1 -. hidden 2 (SCReLU, dense affine) =====
    activate_screlu32(l2_in, l2_act, L2_SIZE, QA*QB);
    alignas(32) int64_t l2_sums[L3_SIZE];
    l2_avx2(l2_act, l2w_by_input[bucket], l2_sums);
    for (int o = 0; o < L3_SIZE; o++) {
        int64_t sum = l2_sums[o];
        sum /= (QA * QB); // (QA*QB)*(QA*QB)*QC -. QA*QB*QC
        sum += l2b[bucket * L3_SIZE + o];
        l3_in[o] = static_cast<int32_t>(sum);
    }

    // ===== L3: hidden 2 -. output (SCReLU fused into the affine transform / dot product) =====
    int64_t out = l3_screlu_avx2(l3_in, l3w_i32[bucket], QA*QB*QC);
#else

    alignas(32) int16_t us_act[L1_SIZE];
    alignas(32) int16_t them_act[L1_SIZE];
    alignas(32) int16_t l1_out[L1_SIZE];
    alignas(32) int64_t l3_act[L3_SIZE];

    // ===== L1: accumulator -. hidden 1 =====

    // activate accumulator values
    // Crelu instead of SCrelu since pairwise mult gives us non-linearity
    //    and scale of (QA*QA)*(QA*QA) is huge
    activate_crelu(us->vals, us_act, L1_SIZE, QA);
    activate_crelu(them->vals, them_act, L1_SIZE, QA);

    // pairwise multiply + concat
    pairwise_mul_simd(us_act, them_act, l1_out);

    // First-layer products and sums fit int16 and int32 respectively.
    for (int o = 0; o < L2_SIZE; o++) {
        const int8_t* w = l1w[bucket * L2_SIZE + o];
        const int32_t bias = l1b[bucket * L2_SIZE + o];

        // create layer nodes (pre-activation)
        int64_t sum = dot_pairwise_i16_i8(l1_out, w);

        sum /= QA; // (QA*QA)*QB -. QA*QB
        sum += bias;
        l2_in[o] = static_cast<int32_t>(sum);
    }

    // ===== L2: hidden 1 -. hidden 2 =====

    // activate
    activate_screlu32(l2_in, l2_act, L2_SIZE, QA*QB);

    // weight transform
    for (int o = 0; o < L3_SIZE; o++) {
        const int8_t* w = l2w[bucket * L3_SIZE + o];
        const int32_t bias = l2b[bucket * L3_SIZE + o];
        
        int64_t sum = dot_i32_i8_widen(l2_act, w, L2_SIZE);

        sum /= (QA * QB); // (QA*QB)*(QA*QB)*QC -. QA*QB*QC
        sum += bias;
        l3_in[o] = (int32_t)sum; 
    }

    // ===== L3: hidden 2 -. output =====

    // activate
    activate_screlu64(l3_in, l3_act, L3_SIZE, QA*QB*QC);

    // weight transform 
    int64_t out = (int64_t)dot_i64_i8(l3_act, l3w[bucket], L3_SIZE);
#endif

    out /= (int64_t)(QA * QB * QC); // (QA*QB*QC)*(QA*QB*QC)*QC -. QA*QB*QC*QC
    out += (int64_t)l3b[bucket];
    out *= SCALE;
    out /= (int64_t)(QA * QB * QC * QC);

    return (int)out;
}

int NNUE::full_eval(const Board& b) {
    build_halfka_accumulators(b); // build_accumulators(b);
    return evaluate(
        b.is_white_move, 
        (b.colorBitboards[0] | b.colorBitboards[1])
    );
}

// Direct output head over both accumulator perspectives. Division order
// matches the quantized SCReLU training formula (including negative truncation).
#ifdef SMALL_NET
int NNUE::eval_screlu(const AccumulatorStack& stack, bool is_white_move, U64 occ) const {
    const auto* us = is_white_move ? stack.white().vals : stack.black().vals;
    const auto* them = is_white_move ? stack.black().vals : stack.white().vals;
    const int bucket = output_bucket(occ);
    int64_t sum = 0;

    if (small_head_format == SmallHeadFormat::I16_HEAD) {
        for (int i = 0; i < hl_size; ++i) {
            sum += screlu<int64_t>(us[i], qa_small) * l1w_small_i16[bucket][i];
            sum += screlu<int64_t>(them[i], qa_small) * l1w_small_i16[bucket][hl_size + i];
        }
    } else {
        for (int i = 0; i < hl_size; ++i) {
            sum += screlu<int64_t>(us[i], qa_small) * l1w_small[bucket][i];
            sum += screlu<int64_t>(them[i], qa_small) * l1w_small[bucket][hl_size + i];
        }
    }

    sum /= (int64_t)(qa_small);
    sum += (int64_t)l1b_small[bucket];
    sum *= SCALE;
    sum /= int64_t(qa_small * QB);
    return (int)sum;
}

int NNUE::eval_screlu_simd(const AccumulatorStack& stack, bool is_white_move, U64 occ) const {
    const auto* us = is_white_move ? stack.white().vals : stack.black().vals;
    const auto* them = is_white_move ? stack.black().vals : stack.white().vals;
    const int bucket = output_bucket(occ);

    const int64_t sum = small_head_format == SmallHeadFormat::I16_HEAD
        ? dot_screlu_i16(us, l1w_small_i16[bucket], qa_small)
            + dot_screlu_i16(them, l1w_small_i16[bucket] + hl_size, qa_small)
        : dot_screlu_i8(us, l1w_small[bucket], qa_small)
            + dot_screlu_i8(them, l1w_small[bucket] + hl_size, qa_small);
    
    // equivalent to the explicit ordering above
    return static_cast<int>(
        (sum / qa_small + l1b_small[bucket]) * SCALE 
        / (int64_t(qa_small * QB))
    );
}
#endif

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
    //acc_stm().remove_feature(f_from_stm, l0w);
    //acc_stm().add_feature(f_to_stm, l0w);
    acc_stm().add_sub_feature(f_to_stm, f_from_stm, l0w);

    //acc_ntm().remove_feature(f_from_ntm, l0w);
    //acc_ntm().add_feature(f_to_ntm, l0w);
    acc_ntm().add_sub_feature(f_to_ntm, f_from_ntm, l0w);

    // Promotion: replace pawn feature with promoted piece (both POVs)
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_stm, flip_stm);
        int f_promo_ntm = feature_index_ntm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_ntm, flip_ntm);

        // remove pawn entry we just added, then add promoted piece
        //acc_stm().remove_feature(f_to_stm, l0w);
        //acc_stm().add_feature(f_promo_stm, l0w);
        acc_stm().add_sub_feature(f_promo_stm, f_to_stm, l0w);

        //acc_ntm().remove_feature(f_to_ntm, l0w);
        //acc_ntm().add_feature(f_promo_ntm, l0w);
        acc_ntm().add_sub_feature(f_promo_ntm, f_to_ntm, l0w);
    }

    // ---- Captured piece: remove from BOTH accumulators ----
    int captured_piece = before.getCapturedPiece(mv.TargetSquare());
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        // actual color of captured piece (should equal ntm but use board for safety)
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(mv.TargetSquare(), captured_piece, cap_color, base_stm, flip_stm);
        acc_ntm().remove_feature(f_cap_ntm, l0w);
        acc_stm().remove_feature(f_cap_stm, l0w);
    }

    // ---- En passant: captured pawn is on cap_sq (remove from BOTH) ----
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8);
        int cap_color = other_color(piece_color);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);
        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        acc_ntm().remove_feature(f_cap_ntm, l0w);
        acc_stm().remove_feature(f_cap_stm, l0w);
    }
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


    // Undo promotion
    if (promo) {
        int promo_piece = mv.PromotionPieceType();
        int f_promo_stm = feature_index_stm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_stm, flip_stm);
        int f_pawn_stm  = feature_index_stm_halfka(mv.StartSquare(), pawn, piece_color, base_stm, flip_stm);
        int f_promo_ntm = feature_index_ntm_halfka(mv.TargetSquare(), promo_piece, piece_color, base_ntm, flip_ntm);
        int f_pawn_ntm  = feature_index_ntm_halfka(mv.StartSquare(), pawn, piece_color, base_ntm, flip_ntm);

        //acc_stm().remove_feature(f_promo_stm, l0w);
        //acc_stm().add_feature(f_pawn_stm, l0w);
        acc_stm().add_sub_feature(f_pawn_stm, f_promo_stm, l0w);
        //acc_ntm().remove_feature(f_promo_ntm, l0w);
        //acc_ntm().add_feature(f_pawn_ntm, l0w);
        acc_ntm().add_sub_feature(f_pawn_ntm, f_promo_ntm, l0w);
    } else {
        //acc_stm().remove_feature(f_to_stm, l0w);
        //acc_stm().add_feature(f_from_stm, l0w);
        acc_stm().add_sub_feature(f_from_stm, f_to_stm, l0w);
        //acc_ntm().remove_feature(f_to_ntm, l0w);
        //acc_ntm().add_feature(f_from_ntm, l0w);
        acc_ntm().add_sub_feature(f_from_ntm, f_to_ntm, l0w);
    }

    // Undo captures
    int captured_piece = board.currentGameState.capturedPieceType;
    if (captured_piece != -1 && mv.MoveFlag() != Move::enPassantCaptureFlag) {
        int cap_sq = mv.TargetSquare();
        int cap_color = other_color(piece_color); // should be the captured piece color
        int f_cap_stm = feature_index_stm_halfka(cap_sq, captured_piece, cap_color, base_stm, flip_stm);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, captured_piece, cap_color, base_ntm, flip_ntm);

        acc_stm().add_feature(f_cap_stm, l0w);
        acc_ntm().add_feature(f_cap_ntm, l0w);
    }

    // Undo en passant
    if (mv.MoveFlag() == Move::enPassantCaptureFlag) {
        // The captured pawn is behind the target square in the direction of the moving pawn
        int cap_sq = mv.TargetSquare() + (piece_color == 0 ? -8 : 8); 
        int cap_color = other_color(piece_color); // captured pawn color

        int f_cap_stm = feature_index_stm_halfka(cap_sq, pawn, cap_color, base_stm, flip_stm);
        int f_cap_ntm = feature_index_ntm_halfka(cap_sq, pawn, cap_color, base_ntm, flip_ntm);

        acc_stm().add_feature(f_cap_stm, l0w);
        acc_ntm().add_feature(f_cap_ntm, l0w);
    }

}
