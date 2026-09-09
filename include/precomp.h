// Precomputed blank board attack masks for all the pieces
#ifndef MOVE_DATA_H
#define MOVE_DATA_H

#include "bits.h"
#include <cstdint>

class PrecomputedMoveData {
public:
    // ---------------- Static Bitboards ----------------
    static U64 blankPawnMoves[64][2];    // [square][white=0/black=1]
    static U64 fullPawnAttacks[64][2];   // [square][white=0/black=1]
    static U64 blankKnightAttacks[64];
    static U64 blankKingAttacks[64];

    static SMasks blankBishopAttacks[64][2]; // lower, upper, lineEx=lower|upper // direction
    static SMasks blankRookAttacks[64][2]; // lower, upper, lineEx=lower|upper // direction
    static SMasks blankQueenAttacks[64][4]; // bishop | rook // direction
    

    static U64 passedPawnMasks[64][2];   // [square][white/black]

    static U64 rayMasks[64][64];         // line connecting square_a -> square_b
    static U64 alignMasks[64][64];       // line including a & b

    static int distToEdge[64][8];        // square, direction (N,NE,E,...)
    static int kingMoveDistances[64][64]; // Chebyshev (king moves)

    // ---------------- Initialization -----------------
    // Call once at program startup
    static void init();

private:
    // Internal helpers to generate the arrays
    static void generateFullPawnMoves();
    static void generateFullPawnAttacks();
    static void generateBlankKnightAttacks();
    static void generateBlankBishopAttacks();
    static void generateBlankRookAttacks();
    static void generateBlankQueenAttacks();
    static void generateBlankKingAttacks();
    static void generatePassedPawnsMasks();
    static void generateKingDistances();
    static void generateAlignMasks();
    static void generateRayMasks();
    static void generateDistToEdge();

    // Prevent instance creation
    PrecomputedMoveData() = delete;
};

#endif
