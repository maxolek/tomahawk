// Move Generator

// looks for valid:
// sliding moves, king moves, pins, captures, etc.

#include <movegen.h>

MoveGenerator::MoveGenerator(const Board& _board) {
    // load movegen at given state
    //board = _board;
    updateBitboards(_board);
    side = _board.is_white_move ? 0 : 1;
    in_check = false; // board.is_in_check;
    in_double_check = false; // gets updated during opponent gen moves
}

// look up kings last 
//      have to generate opponenent attack map
//          might not be that efficient for stuff like checks but it handles discoveries and everything
//      then look up opponent king first

// generate pins and legality
// easier code might be to generate it on iteration and not in this bool stuff

// fill in move info arrays
int MoveGenerator::generateMoves(const Board& _board, bool _quiescence) {
    #ifdef DEV
        ScopedTimer timer(T_MOVEGEN);
    #endif

    // load movegen at given state
    updateBitboards(_board);
    quiescence = _quiescence;

    // gen opponent moves
    // detect checks, pins, etc.
    generatePawnAttacks(false); // false, false normally
    generateKnightMoves(false);
    generateSlidingMoves(false);
    generateKingMoves(false);

    //if (check_ray_mask) {in_check = true;}

    if (in_double_check) { // cannot capture or block out of a double check
        generateKingMoves(true);
    } else {
        // gen moves
        // uses stored information from gen oppponent moves to determine legality
        generatePawnPushes(true);
        generatePawnAttacks(true);
        generateKnightMoves(true);
        generateSlidingMoves(true);
        generateKingMoves(true);
    }

    return std::min(count, MAX_MOVES);
}

// accelerated return for quick check
bool MoveGenerator::hasLegalMoves(const Board& _board) {
    // load movegen at given state
    //board = _board;
    updateBitboards(_board);
    quiescence = false;

    // gen opponent moves
    // detect checks, pins, etc.
    generatePawnAttacks(false); // false, false normally
    generateKnightMoves(false);
    generateSlidingMoves(false);
    generateKingMoves(false);

    //if (check_ray_mask) {in_check = true;}
    if (in_double_check) { // cannot capture or block out of a double check
        generateKingMoves(true);
    } else {
        // gen moves
        // uses stored information from gen oppponent moves to determine legality
        generatePawnPushes(true);
        if (count > 0) {return true;}
        generatePawnAttacks(true);
        if (count > 0) {return true;}
        generateKnightMoves(true);
        if (count > 0) {return true;}
        generateSlidingMoves(false);
        if (count > 0) {return true;}
        generateKingMoves(true);
        if (count > 0) {return true;}
    }

    return false;
}

// count number of pieces along pin_ray to determine if pinned 
// if only piece then pinned, otherwise movement is legal (not necessarily good tho)
bool MoveGenerator::isPinned(int square) {
    if (!((pin_rays >> square) & 1)) return false;

    // Full line through king and candidate in both directions
    // filtered to single direction @ slider_sq
    U64 full_line = PrecomputedMoveData::alignMasks[own_king_square][square];


    // Enemy sliders along this full line
    auto [dx, dy] = direction_map(own_king_square, square);
    bool ortho = (dx == 0 || dy == 0);
    U64 enemy_sliders = ortho ? (opp & (rooks | queens)) : (opp & (bishops | queens));
    U64 sliders_on_line = full_line & enemy_sliders;
    if (!sliders_on_line) return false;


    // Pick slider along the ray: MSB if candidate < king, LSB if candidate > king
    // pinning piece must be on 'more extreme' square idx than pinned_sq relative to king
    int slider_sq = (square < own_king_square)
                        ? getMSB(sliders_on_line & bitsBelow(square))
                        : getLSB(sliders_on_line & bitsAbove(square));

    // square is not between king and enemy sliding piece
    // solves UB of rayMask[king_sq][-1]
    if (slider_sq < 0) {return false;}

    // Count pieces between king and slider
    U64 between_king_and_slider = PrecomputedMoveData::rayMasks[own_king_square][slider_sq] & (own | opp);
    // remove end points (slider + king)
    U64 between = between_king_and_slider & ~(1ULL << own_king_square | 1ULL << slider_sq);

    // iff the only bit on between is the square we are checking, then it is pinned
    // !!! slow i think !!!
    return (between == (1ULL << square));
}


//need to make sure this doesnt result in a check 
//regular pins (e.g. file and diag) should already be handled via pin_rays
//but along rank, enpassant capture could result in check since both pawns are now gone
// whoops: can be diag too 
bool MoveGenerator::isEnpassantPinned(int start_square, int target_file) {
    // occ before ep
    enPassantMaskBlockers = (own|opp);

    // perform ep
    pop_bit(enPassantMaskBlockers, start_square); // dont re-init cause isnt relevant
    // remove opp
    if (side==0) pop_bit(enPassantMaskBlockers, 4*8 + target_file);
    else pop_bit(enPassantMaskBlockers, 3*8 + target_file);
    // occ after enpassant

    U64 relevant_opp_ep_sliders = opp & (rooks | queens | bishops) & PrecomputedMoveData::alignMasks[start_square][own_king_square];
    while (relevant_opp_ep_sliders) {
        int opp_sq = getLSB(relevant_opp_ep_sliders);
        pop_bit(relevant_opp_ep_sliders, opp_sq);

        // not in ray
        if (!(PrecomputedMoveData::alignMasks[own_king_square][opp_sq] & (1ULL << start_square))) {
            continue;
        }

        if ((rooks|queens) & (1ULL << opp_sq) && 
            (Magics::rookAttacks(opp_sq, enPassantMaskBlockers) & (own & kings)))
        {
            return true;
        }

        if ((bishops|queens) & (1ULL << opp_sq) && 
            (Magics::bishopAttacks(opp_sq, enPassantMaskBlockers) & (own & kings)))
        {
            return true;
        }
    }

    return false;
}

// -----------------------
// ---- sliding moves ----
// -----------------------

// magic bitboards
void MoveGenerator::generateSlidingMoves(bool ours) {
    U64 occ = own | opp;
    Move potential_move;

    struct SlidingData { U64 bb; int type; };
    std::vector<SlidingData> pieces;

    if (ours) {
        if (rooks & own)   pieces.push_back({rooks & own, rook});
        if (bishops & own) pieces.push_back({bishops & own, bishop});
        if (queens & own)  pieces.push_back({queens & own, queen});
    } else {
        if (rooks & opp)   pieces.push_back({rooks & opp, rook});
        if (bishops & opp) pieces.push_back({bishops & opp, bishop});
        if (queens & opp)  pieces.push_back({queens & opp, queen});
    }

    for (auto &piece : pieces) {
        while (piece.bb) {
            int start_square = getLSB(piece.bb);
            piece.bb &= piece.bb - 1;

            U64 potential_moves_bb = 0;
            U64 attacks = 0;

            // get attack bitboard
            if (piece.type == rook) attacks = Magics::rookAttacks(start_square, occ);
            else if (piece.type == bishop) attacks = Magics::bishopAttacks(start_square, occ);
            else attacks = Magics::rookAttacks(start_square, occ) | Magics::bishopAttacks(start_square, occ);

            updateAttackMapAndCheck(attacks, ours, start_square, piece.type);

            if (!ours) continue; // only need to update boards + check
            
            potential_moves_bb = attacks & ~own;

            potential_moves_bb = limitPinnedMoves(start_square, potential_moves_bb);
            potential_moves_bb = restrictCheckMoves(potential_moves_bb);

            addMovesFromBitboard(start_square, potential_moves_bb);
        }
    }
}

// sliding bitboards
/*
U64 MoveGenerator::odiff(U64 occ, SMasks pMask) {
    U64 lower, upper, ms1b, odiff_board;
    lower = pMask.lower & occ;
    upper = pMask.upper & occ;

    ms1b = U64(0x8000000000000000) >> __builtin_clzll(lower | 1); // ms1b of lower (at least bit zero)
    odiff_board = upper ^ (upper - ms1b);

    return pMask.lineEx & odiff_board;
}

void MoveGenerator::generateSlidingMoves(bool ours) {

    auto processSlider = [&](U64 pieces, int num_dirs, auto attack_fn) {
        while (pieces) {
            int start_square = getLSB(pieces);
            pieces &= pieces - 1;

            for (int dir = 0; dir < num_dirs; dir++) {
                const auto& atk = attack_fn(start_square, dir);
                U64 potential = Bits::fullSet;

                if (!ours) {
                     U64 o_diff = odiff(own | opp, atk);
                    potential &= o_diff & ~opp;
                    opponentAttackMap |= o_diff;

                    if (potential & (own & kings)) {
                        in_double_check = in_check;
                        check_ray_mask     |= PrecomputedMoveData::rayMasks[start_square][own_king_square];
                        check_ray_mask_ext |= PrecomputedMoveData::alignMasks[start_square][own_king_square];
                        in_check = true;
                    }
                    if (atk.lineEx & (own & kings))
                        pin_rays |= PrecomputedMoveData::rayMasks[start_square][own_king_square] & ~(1ULL << own_king_square);

                } else {
                    potential &= odiff(own | opp, atk) & ~own;
                    ownAttackMap |= potential;

                    potential = limitPinnedMoves(start_square, potential);
                    potential = restrictCheckMoves(potential);

                    addMovesFromBitboard(start_square, potential);
                }
            }
        }
    };

    processSlider(rooks   & (ours ? own : opp), 2, [](int s, int d) -> const auto& { return PrecomputedMoveData::blankRookAttacks[s][d];   });
    processSlider(bishops & (ours ? own : opp), 2, [](int s, int d) -> const auto& { return PrecomputedMoveData::blankBishopAttacks[s][d]; });
    processSlider(queens  & (ours ? own : opp), 4, [](int s, int d) -> const auto& { return PrecomputedMoveData::blankQueenAttacks[s][d];  });
}
*/

// ----------------------------
// -- static move generation --
// ----------------------------

void MoveGenerator::generateKnightMoves(bool ours) {
    U64 valid_knights = ours ? knights & own : knights & opp;
    Move potential_move;

    while (valid_knights) {
        int start_square = getLSB(valid_knights);
        valid_knights &= valid_knights - 1;

        // Start with the knight's attack mask
        U64 potential_moves_bb = PrecomputedMoveData::blankKnightAttacks[start_square];
        updateAttackMapAndCheck(potential_moves_bb, ours, start_square, knight);

        if (!ours) continue; // only need to update boards + check

        potential_moves_bb &= ~own;  // cannot capture own pieces

        potential_moves_bb = limitPinnedMoves(start_square, potential_moves_bb);
        potential_moves_bb = restrictCheckMoves(potential_moves_bb);

        addMovesFromBitboard(start_square, potential_moves_bb);
    }
}


void MoveGenerator::generatePawnPushes(bool ours) {
    // never called for opponent
    // doesnt have any "attacks"
    if (!ours) {return;}

    U64 valid_pawns = ours ? pawns & own : pawns & opp;
    U64 potential_moves_bb;
    Move potential_move;
    int start_square;

    bool on_starting_rank;
    int one_step, two_step;

    while (valid_pawns) {
        start_square = getLSB(valid_pawns);
        valid_pawns &= valid_pawns - 1;

        // Start with the pawn's forward move mask
        // do not update attacks as pawn pushes cannot be direct attacks
        potential_moves_bb = PrecomputedMoveData::blankPawnMoves[start_square][side];
        potential_moves_bb &= ~(own|opp);  // cannot push into pieces

        // Remove double push if blocked
        on_starting_rank = (side == 0) ? (((1ULL << start_square) & Bits::mask_rank_2) != 0) 
                                        : (((1ULL << start_square) & Bits::mask_rank_7) != 0);
        one_step = start_square + ((side == 0) ? 8 : -8);
        two_step = start_square + ((side == 0) ? 16 : -16);
        if (on_starting_rank && !get_bit(potential_moves_bb, one_step)) {
            potential_moves_bb &= ~(1ULL << two_step);
        }

        potential_moves_bb = limitPinnedMoves(start_square, potential_moves_bb);
        potential_moves_bb = restrictCheckMoves(potential_moves_bb);

        forEachBit(potential_moves_bb, [&](int target_square) {
            if (target_square == two_step) {
                Move m(start_square, target_square, Move::pawnTwoUpFlag);
                if (shouldAddMove(m)) moves[count++] = m;
            }
            else if (isPromotionPawn(start_square)) {
                generatePromotions(start_square, target_square);
            }
            else {
                Move m(start_square, target_square);
                if (shouldAddMove(m)) moves[count++] = m;
            }
        });
    }
}

void MoveGenerator::generatePawnAttacks(bool ours) {
    U64 valid_pawns = ours ? pawns & own : pawns & opp;
    Move potential_move;

    while (valid_pawns) {
        int start_square = getLSB(valid_pawns);
        valid_pawns &= valid_pawns - 1;

        // Start with full attack mask for opponent attack map
        U64 potential_moves_bb = PrecomputedMoveData::fullPawnAttacks[start_square][ours ? side : 1-side];
        updateAttackMapAndCheck(potential_moves_bb, ours, start_square, pawn);

        if (!ours) continue; // only need to update boards + check

        potential_moves_bb &= opp;

        potential_moves_bb = limitPinnedMoves(start_square, potential_moves_bb);
        potential_moves_bb = restrictCheckMoves(potential_moves_bb);

        // Handle en passant
        if (curr_gamestate.enPassantFile > -1) {
            int ep_square = (!side ? 5*8 : 2*8) + curr_gamestate.enPassantFile;   // destination
            int captured_pawn_sq = (!side ? 4*8 : 3*8) + curr_gamestate.enPassantFile;

            bool can_capture_ep = PrecomputedMoveData::fullPawnAttacks[start_square][side] & (1ULL << ep_square);
            bool ep_safe = !isEnpassantPinned(start_square, curr_gamestate.enPassantFile);
            bool ep_legal_in_check = !in_check || (check_ray_mask & (1ULL << captured_pawn_sq));

            if (can_capture_ep && ep_safe && ep_legal_in_check) {
                potential_moves_bb |= 1ULL << ep_square;
            }
        }

        // Add moves
        // must write out full function (addMoves only accepts 1 flag)
        forEachBit(potential_moves_bb, [&](int target_square) {
            if (isPromotionPawn(start_square)) {
                generatePromotions(start_square, target_square);
            } else if ((target_square % 8) == curr_gamestate.enPassantFile &&
                        ((side == 0 && target_square / 8 == 5) || (side == 1 && target_square / 8 == 2))) {
                Move m(start_square, target_square, Move::enPassantCaptureFlag);
                if (shouldAddMove(m)) moves[count++] = m;
            } else {
                Move m(start_square, target_square);
                if (shouldAddMove(m)) moves[count++] = m;
            }
        });
    }
}


void MoveGenerator::generatePromotions(int start_square, int target_square) { 
    int move_flags[4] = {Move::promoteToQueenFlag, Move::promoteToKnightFlag, Move::promoteToRookFlag, Move::promoteToBishopFlag};
    Move potential_move;

    for (int flag : move_flags) {
        potential_move = Move(start_square, target_square, flag);
        if (shouldAddMove(potential_move)) 
            moves[count++] = potential_move;
    }
}


void MoveGenerator::generateKingMoves(bool ours) {
    int king_square = ours ? own_king_square : sqidx(kings & opp);
    Move potential_move;
    U64 potential_moves_bb = PrecomputedMoveData::blankKingAttacks[king_square];

    // Update attack maps
    updateAttackMapAndCheck(potential_moves_bb, ours, king_square, king);
    if (!ours) return;

    // Remove friendly-occupied squares
    potential_moves_bb &= ~own;

    // Prevent king from moving into attack
    potential_moves_bb &= ~opponentAttackMap;

    // Limit moves if in check
    if (in_check) {
        //check_ray_mask_ext |= check_ray_mask;
        potential_moves_bb &= (~check_ray_mask_ext) | (opp & check_ray_mask);
    }

    addMovesFromBitboard(king_square, potential_moves_bb);

    // Castling
    if (ours && !in_check) {
        U64 castle_blockers = opponentAttackMap | own | opp;

        // Kingside
        if (curr_gamestate.HasKingsideCastleRight(side == 0)) {
            U64 mask = (side == 0) ? Bits::whiteKingsideMask : Bits::blackKingsideMask;
            if (!(mask & castle_blockers)) {
                int target = (side == 0) ? g1 : g8;
                Move m(king_square, target, Move::castleFlag);
                if (shouldAddMove(m)) moves[count++] = m;
            }
        }

        // Queenside
        if (curr_gamestate.HasQueensideCastleRight(side == 0)) {
            U64 mask = (side == 0) ? Bits::whiteQueensideMask : Bits::blackQueensideMask;
            U64 mask_ext = (side == 0) ? Bits::whiteQueensideMaskExt : Bits::blackQueensideMaskExt;
            if (!(mask & castle_blockers) && !((own | opp) & mask_ext)) {
                int target = (side == 0) ? c1 : c8;
                Move m(king_square, target, Move::castleFlag);
                if (shouldAddMove(m)) moves[count++] = m;
            }
        }
    }
}


bool MoveGenerator::isCheck(const Move move) {
    int piece = -1;
    int start_square = move.StartSquare();
    int target_square = move.TargetSquare();
    U64 opp_king = opp & kings;
    U64 new_occ = ((own | opp) & ~(1ULL << start_square)) | (1ULL << target_square);
    U64 new_own = ((own) & ~(1ULL << start_square)) | (1ULL << target_square);
    U64 discovery_ray = PrecomputedMoveData::rayMasks[start_square][sqidx(opp_king)]; // includes king
    bool is_direct_check = false;

    // get moved piece (could be replaced with board pointer functions)
    if (pawns & (1ULL << start_square)) piece = pawn;
    else if (knights & (1ULL << start_square)) piece = knight;
    else if (bishops & (1ULL << start_square)) piece = bishop;
    else if (rooks & (1ULL << start_square)) piece = rook;
    else if (queens & (1ULL << start_square)) piece = queen;
    else if (kings & (1ULL << start_square)) piece = king;

    // direct checks
    switch (piece) {
        case king:
            is_direct_check = false;
            break;
        case pawn:
            is_direct_check = (PrecomputedMoveData::fullPawnAttacks[target_square][side] & opp_king);
            break;
        case knight:
            is_direct_check = (PrecomputedMoveData::blankKnightAttacks[target_square] & opp_king);
            break;
        case bishop:
            is_direct_check = (Magics::bishopAttacks(target_square, new_occ) & opp_king);
            break;
        case rook:
            is_direct_check = (Magics::rookAttacks(target_square, new_occ) & opp_king);
            break;
        case queen:
            is_direct_check = ((Magics::bishopAttacks(target_square, new_occ) | Magics::rookAttacks(target_square, new_occ)) & opp_king);
            break;
    }
    if (is_direct_check) return true;

    // discovered checks
    
    // enpassant capture discovery check (rook + king along rank as both pawns)
    if (move.MoveFlag() == Move::enPassantCaptureFlag) {
        // remove ep captured pawn
        new_occ &= ~(1ULL << (target_square + (!side ? -8 : 8)));
    }

    U64 blockers = new_occ & discovery_ray; // includes discovery_sliders
    U64 discovery_sliders = (rooks | bishops | queens) & new_own & discovery_ray;
    
    blockers &= ~discovery_sliders; // only non-own sliding pieces along the ray
    if (blockers == opp_king)
        return true;

    // promotions (direct checks from new promotion piece)
    if (move.IsPromotion()) {
        switch (move.PromotionPieceType()) {
            case queen:
                return ((Magics::bishopAttacks(target_square, new_occ) | Magics::rookAttacks(target_square, new_occ)) & opp_king);
                break;
            case knight:
                return (PrecomputedMoveData::blankKnightAttacks[target_square] & opp_king);
                break;
            case rook:
                return (Magics::rookAttacks(target_square, new_occ) & opp_king);
                break;
            case bishop:
                return (Magics::bishopAttacks(target_square, new_occ) & opp_king);
                break;
        }
    }

    return false;
}

void MoveGenerator::updateBitboards(const Board& _board) {
    //board = _board;
    curr_gamestate = _board.currentGameState;
    side = _board.is_white_move ? 0 : 1;
    in_check = false; //board.is_in_check;
    in_double_check = false; // both get updated during gen opponent moves

    own = _board.colorBitboards[side];
    opp = _board.colorBitboards[1-side];
    pawns = _board.pieceBitboards[pawn];
    knights = _board.pieceBitboards[knight];
    bishops = _board.pieceBitboards[bishop];
    rooks = _board.pieceBitboards[rook];
    queens = _board.pieceBitboards[queen];
    kings = _board.pieceBitboards[king];

    check_ray_mask = 0ULL;
    check_ray_mask_ext = 0ULL;
    pin_rays = 0ULL;
    opponentAttackMap = ownAttackMap = 0ULL;
    own_king_square = sqidx(own&kings);

    count = 0;
}

U64 MoveGenerator::limitPinnedMoves(int square, U64 moves_bb) {
    if (isPinned(square)) {
        moves_bb &= PrecomputedMoveData::alignMasks[square][own_king_square];
    }
    return moves_bb;
}

U64 MoveGenerator::restrictCheckMoves(U64 moves_bb) {
    if (in_check) moves_bb &= check_ray_mask;
    return moves_bb;
}

void MoveGenerator::addMovesFromBitboard(int start_square, U64 moves_bb, int flag) {
    forEachBit(moves_bb, [&](int target_square){
        Move m(start_square, target_square, flag);
        if (shouldAddMove(m)) moves[count++] = m;
    });
}

bool MoveGenerator::isPromotionPawn(int square) {
    return (!side && get_bit(Bits::mask_rank_7, square)) || 
           (side && get_bit(Bits::mask_rank_2, square));
}

void MoveGenerator::updateAttackMapAndCheck(U64 attacks_bb, bool ours, int start_square, int piece_type) {
    if (ours) {
        ownAttackMap |= attacks_bb;
    } else {
        opponentAttackMap |= attacks_bb;

        // skewers/pinned pieces
        // a->b  inclusive
        if (piece_type == bishop || piece_type == rook || piece_type == queen) {
            pin_rays |= PrecomputedMoveData::rayMasks[start_square][sqidx(own&kings)];
        }


        // checks
        if (attacks_bb & (own & kings)) {
            in_double_check = in_check;
            in_check = true;

            if (piece_type == pawn || piece_type == knight) {
                // Only the attacking square matters
                check_ray_mask |= 1ULL << start_square;
            } else if (piece_type > -1 && piece_type < king) {
                // sliding piece
                check_ray_mask |= PrecomputedMoveData::rayMasks[start_square][own_king_square];
                check_ray_mask_ext |= PrecomputedMoveData::alignMasks[start_square][own_king_square];
            } else {
                // king attacking king? should not happen
            }
        }
    }
}




bool MoveGenerator::shouldAddMove(const Move& move) const {
    if (in_check) return true;              // all moves matter in check
    if (!quiescence) return true;           // in normal search, include everything
    if (move.IsPromotion()) return true;    // promotions are always interesting
    if (opp & (1ULL << move.TargetSquare())) return true; // captures
    return false;                           // otherwise, ignore in quiescence
}
