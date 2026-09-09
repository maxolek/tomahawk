// Precomputed blank board attack masks for all the pieces

#include <precomp.h>

// static member definitions
U64 PrecomputedMoveData::blankPawnMoves[64][2] = {};
U64 PrecomputedMoveData::fullPawnAttacks[64][2] = {};
U64 PrecomputedMoveData::blankKnightAttacks[64] = {};
U64 PrecomputedMoveData::blankKingAttacks[64] = {};

SMasks PrecomputedMoveData::blankBishopAttacks[64][2] = {};
SMasks PrecomputedMoveData::blankRookAttacks[64][2] = {};
SMasks PrecomputedMoveData::blankQueenAttacks[64][4] = {};

U64 PrecomputedMoveData::passedPawnMasks[64][2] = {};
U64 PrecomputedMoveData::rayMasks[64][64] = {};
U64 PrecomputedMoveData::alignMasks[64][64] = {};

int PrecomputedMoveData::distToEdge[64][8] = {};
int PrecomputedMoveData::kingMoveDistances[64][64] = {};

void PrecomputedMoveData::init() {
    generateFullPawnMoves();
    generateFullPawnAttacks();
    generateBlankKnightAttacks();
    generateBlankKingAttacks();

    generateBlankBishopAttacks();
    generateBlankRookAttacks();
    generateBlankQueenAttacks();

    generatePassedPawnsMasks();
    generateKingDistances();

    generateDistToEdge();
    generateAlignMasks();
    generateRayMasks();
}

void PrecomputedMoveData::generateFullPawnMoves() {
    U64 bitboard = 0ULL;
    bool isin_init_row;

    for (int side = 0; side < 2; side++) {
        for (int square = a2; square <= h7; square++) {
            set_bit(bitboard, square + ((side==0) ? 8 : -8));
            isin_init_row = (side==0) ? ((1ULL << square) & Bits::mask_rank_2) : ((1ULL << square) & Bits::mask_rank_7);
            if (isin_init_row) 
                set_bit(bitboard, square + ((side==0) ? 16 : -16));
            
            blankPawnMoves[square][side] = bitboard;
            bitboard = 0ULL;
        }
        bitboard = 0ULL;
    }
}

void PrecomputedMoveData::generateFullPawnAttacks() {
    U64 bitboard = 0ULL;
    U64 attacks = 0ULL;

    for (int side = 0; side < 2; side++) {
        for (int square = a1; square <= h8; square++) { // originally a2->h7 since pawns arent on 1st/8th rank, but since these are used to see if a king is in check they must cover all squares
            set_bit(bitboard,square);

            // white pawns
            if (!side){
                attacks |= ((bitboard) << 7);
                attacks |= (bitboard << 9);
            } else { // black pawns
                attacks |= (bitboard >> 7);
                attacks |= (bitboard >> 9);
            }

            // remove file warps
            if (bitboard & Bits::file_masks[0]) 
                attacks &= ~(Bits::file_masks[0] >> 1);
            else if (bitboard & Bits::file_masks[7])
                attacks &= ~(Bits::file_masks[7] << 1);

            fullPawnAttacks[square][side] = attacks;
            bitboard = 0ULL;
            attacks = 0ULL;
        }
    }
}

void PrecomputedMoveData::generateBlankKnightAttacks() { // quadrants are counterclockwise
    U64 bitboard = 0ULL;
    int knight_moves[8] = {-17, -15, -10, -6, 6, 10, 15, 17}; 
    // -17,-10 are q3    -15,-6 are q4    6,15 are q2     10,17 are q1 
    bool is_on_a_file; // no q2/q3
    bool is_on_b_file;  // limited q2/q3
    bool is_on_g_file;  // limited q1/q4
    bool is_on_h_file; // no q1/q4
    int target_square;

    for (int square = a1; square <= h8; square++) {
        is_on_a_file = (1ULL << square) & Bits::file_masks[0]; 
        is_on_b_file = (1ULL << square) & Bits::file_masks[1];
        is_on_g_file = (1ULL << square) & Bits::file_masks[6];
        is_on_h_file = (1ULL << square) & Bits::file_masks[7];

        for (int move : knight_moves) {
            target_square = square + move;

            // skip invalid moves
            if (target_square < a1 || target_square > h8)
                continue;

            // skip invalid moves
            if ((is_on_a_file && (move == -17 || move == -10 || move == 6 || move == 15))
                || (is_on_b_file && (move == 6 || move == -10))
                || (is_on_g_file && (move == -6 || move == 10))
                || (is_on_h_file && (move == 17 || move == 10 || move == -6 || move == -15))
                ) {
                continue;
            }

            set_bit(bitboard, target_square);
        }

        blankKnightAttacks[square] = bitboard;
        bitboard = 0ULL;
    }
}

void PrecomputedMoveData::generateBlankKingAttacks() {
    U64 bitboard = 0ULL;
    int king_moves[] = {-9,-8,-7,-1,1,7,8,9}; // if on a-file then only -8/-7/1/8/9, if on h-file then only -1/-8/-9/8/7
    int target_square;
    bool is_on_a_file, is_on_h_file;

    for (int square = a1; square <= h8; square++) {
        is_on_a_file = (1ULL << square) & Bits::file_masks[0];
        is_on_h_file = (1ULL << square) & Bits::file_masks[7];

        for (int move : king_moves){
            target_square = square + move;

            if (target_square >= a1 && target_square <= h8) {
                if ((is_on_a_file && (move == -9 || move == -1 || move == 7)) || (is_on_h_file && (move == -7 || move == 1 || move == 9)))
                    continue;
                set_bit(bitboard, target_square);
            }
        }
        blankKingAttacks[square] = bitboard;
        bitboard = 0ULL;
    }
}


void PrecomputedMoveData::generateBlankBishopAttacks() {
    SMasks mask;
    int file;
    int rank;
    int left_bound; int right_bound;
    U64 bitboard = 0ULL;
    
    for (int square = a1; square <= h8; square++) {
        file = square % 8;
        rank = square / 8; // int trunc in division gives multiple of 8
        for (int direction = 0; direction < 2; direction++) {
            if (direction) {
                    // BR -> UL    changes of 7
                left_bound = std::min(file, 7 - rank); // distance from left or top
                right_bound = std::min(rank, 7 - file); // distance from right or bottom
                for (int i = -right_bound; i <= left_bound; i++) {
                    if (i==0) {
                        mask.lower = bitboard;
                        continue;
                    }
                    set_bit(bitboard, square + 7*i);
                }
                mask.upper = bitboard ^ mask.lower;
            } else {
                    // BL -> UR    changes of 9
                left_bound = std::min(file,rank); // distance from left or bottom
                right_bound = 7 - std::max(file,rank); // distance from right or top
                for (int i = -left_bound; i <= right_bound; i++) {
                    if (i==0) {
                        mask.lower = bitboard;
                        continue;
                    }
                    set_bit(bitboard, square + 9*i);
                }
                mask.upper = bitboard ^ mask.lower;
            }
            
            mask.lineEx = mask.lower | mask.upper;
            blankBishopAttacks[square][direction] = mask;
            bitboard = 0ULL;
        }
    }
}

void PrecomputedMoveData::generateBlankRookAttacks() {
    SMasks mask;
    int file;
    int rank;
    U64 bitboard = 0ULL;

    for (int square = a1; square <= h8; square++) {
        file = square % 8;
        rank = square / 8;
        for (int direction = 0; direction < 2; direction++) {
            if (direction) {
                for (int i = -file; i < 8-file; i++) {
                    if (i==0) {
                        mask.lower = bitboard;
                        continue;
                    }
                    set_bit(bitboard, square + i);
                }
                mask.upper = bitboard ^ mask.lower;
            } else {
                for (int i = -rank; i < 8-rank; i++) {
                    if (i==0) {
                        mask.lower = bitboard;
                        continue;
                    }
                    set_bit(bitboard, square + 8*i);
                }
                mask.upper = bitboard ^ mask.lower;
            }

            mask.lineEx = mask.lower | mask.upper;
            blankRookAttacks[square][direction] = mask;
            bitboard = 0ULL;
        }
    }

}

void PrecomputedMoveData::generateBlankQueenAttacks() {
    SMasks mask;
    for (int square = a1; square <= h8; square++) {
        for (int direction = 0; direction < 4; direction++) {
            if (direction < 2)
                mask = blankBishopAttacks[square][direction];
            else
                mask = blankRookAttacks[square][direction-2];
            blankQueenAttacks[square][direction] = mask;
        }
    }
}


void PrecomputedMoveData::generatePassedPawnsMasks() {
    for (int sq = 0; sq < 64; sq++) {
        int file = sq % 8;
        int rank = sq / 8;

        U64 maskWhite = 0;
        U64 maskBlack = 0;

        // White: scan forward ranks
        for (int r = rank + 1; r < 8; r++) {
            for (int f = file - 1; f <= file + 1; f++) {
                if (f >= 0 && f < 8) {
                    int targetSq = r * 8 + f;
                    maskWhite |= (1ULL << targetSq);
                }
            }
        }

        // Black: scan backward ranks
        for (int r = rank - 1; r >= 0; r--) {
            for (int f = file - 1; f <= file + 1; f++) {
                if (f >= 0 && f < 8) {
                    int targetSq = r * 8 + f;
                    maskBlack |= (1ULL << targetSq);
                }
            }
        }

        passedPawnMasks[sq][0] = maskWhite; // For white pawn at sq: enemy pawns here block it
        passedPawnMasks[sq][1] = maskBlack; // For black pawn at sq: enemy pawns here block it
    }
}

void PrecomputedMoveData::generateKingDistances() {
    int file1, file2, rank1, rank2;
    int rankDistance, fileDistance;

    for (int sq1 = a1; sq1 <= h8; sq1++) {
        for (int sq2 = a1; sq2 <= h8; sq2++) {

            file1 = sq1  & 7;
            file2 = sq2  & 7;
            rank1 = sq1 >> 3;
            rank2 = sq2 >> 3;

            rankDistance = abs(rank2 - rank1);
            fileDistance = abs(file2 - file1);

            kingMoveDistances[sq1][sq2] = std::max(rankDistance, fileDistance);
        }
    }
}


// straight line mask that contains the entire line in the direction of a->b
void PrecomputedMoveData::generateAlignMasks() {
    int a_rank, b_rank, a_file, b_file;
    int rank_dir, file_dir; // {-1,0,1}
    int target_rank, target_file;
    int square;

    for (int a = a1; a <= h8; a++) {
        a_rank = a / 8;
        a_file = a % 8;
        for (int b = a1; b <= h8; b++) {
            b_rank = b / 8;
            b_file = b % 8;

            if (abs(b_file - a_file) == abs(b_rank - a_rank) || b_file == a_file || b_rank == a_rank) {
                rank_dir = (b_rank > a_rank) ? 1 : (b_rank < a_rank) ? -1 : 0;
                file_dir = (b_file > a_file) ? 1 : (b_file < a_file) ? -1 : 0;

                for (int i = -8; i < 8; i++) {
                    target_file = b_file + file_dir * i;
                    target_rank = b_rank + rank_dir * i;
                    square = target_rank * 8 + target_file;
                    if ((target_file > -1 && target_file < 8) && (target_rank > -1 && target_rank < 8))
                            alignMasks[a][b] |= 1ULL << square;
                }
            } 
        }
    }
}

// ray that goes from square -> edge in all 8 directions
void PrecomputedMoveData::generateRayMasks() {
    int a_rank, a_file, b_rank, b_file, rank_dir, file_dir, target_square, target_rank, target_file;
    for (int a = a1; a <= h8; a++) {
        for (int b = a1; b <= h8; b++) {
            a_rank = a / 8;
            a_file = a % 8;
            b_rank = b / 8;
            b_file = b % 8;

            rank_dir = 0;
            file_dir = 0;

            // check along array
            if (abs(b_file - a_file) == abs(b_rank - a_rank) || b_file == a_file || b_rank == a_rank) {
                
                rank_dir = (b_rank > a_rank) ? 1 : (b_rank < a_rank) ? -1 : 0;
                file_dir = (b_file > a_file) ? 1 : (b_file < a_file) ? -1 : 0;

                rayMasks[a][b] |= (1ULL << a); // include checking piece
                for (int i = 1; i < 8; i++) { 
                    target_rank = a_rank + rank_dir * i;
                    target_file = a_file + file_dir * i;

                    if (target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8) {
                        target_square = target_rank * 8 + target_file;
                        rayMasks[a][b] |= (1ULL << target_square);  
                    } else {
                        break;  
                    }
                    if (target_rank == b_rank && target_file == b_file) {
                        break;
                    }
                }
            } else {
                rayMasks[a][b] = 0ULL;
            }
        }
    }
}

// direction = N, NE, E, SE, S, SW, W, NW
void PrecomputedMoveData::generateDistToEdge() {
    int rank, file;

    for (int square = 0; square < 64; square++) {
        rank = square / 8;
        file = square % 8;

        distToEdge[square][0] = 7 - rank;
        distToEdge[square][1] = std::min(7-rank, 7-file);
        distToEdge[square][2] = 7 - file;
        distToEdge[square][3] = std::min(rank, 7-file);
        distToEdge[square][4] = rank;
        distToEdge[square][5] = std::min(rank,file);
        distToEdge[square][6] = file;
        distToEdge[square][7] = std::min(7-rank,file);
    }

}

