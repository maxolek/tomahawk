// movegen.h
// Responsible for generating pseudo-legal and legal chess moves, including sliding moves,
// knight moves, king moves, pawn moves, captures, checks, pins, and en-passant legality.

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "helpers.h"
#include "magics.h"
#include "move_data.h"
#include "move.h"
#include "board.h"


class MoveGenerator {
public:
    // ------------------------
    // Public Move Storage
    // ------------------------

    // single move generation and storage
    // call (which clears prior state) generation to get those gens moves

    Move moves[MAX_MOVES];
    int count = 0;

    // -----------------------
    // Storage Type
    // -----------------------
    bool quiescence = false;

    // ------------------------
    // Constructor
    // ------------------------
    MoveGenerator(const Board& _board);

    // ------------------------
    // Public Move Generation
    // ------------------------
    // single pass full generation
    int generateMoves(const Board& _board, bool _quiescence);
    // quiescence limited generation (check-evasions, captures, promotions, limited quiet moves)
    // Check if side has any legal moves (accelerated generation)
    bool hasLegalMoves(const Board& _board);
    // classic move generation method
    //U64 odiff(U64 occ, SMasks pMask);

    // ------------------------
    // Individual Piece Move Generators
    // ------------------------
    void generateSlidingMoves(bool ours); // magic bitboards
    void generateKnightMoves(bool ours);
    void generatePawnPushes(bool ours);
    void generatePawnAttacks(bool ours);
    void generatePromotions(int start_square, int target_square);
    void generateKingMoves(bool ours);

    // ------------------------
    // Pins & En-passant
    // ------------------------
    bool isPinned(int square);
    bool isEnpassantPinned(int start_square, int target_file);

    // ------------------------
    // Move Classification
    // ------------------------
    bool shouldAddMove(const Move& move) const;
    bool isCheck(const Move move); // legal only (all - including discovered)

    // ------------------------
    // Bitboard Updates/Helpers
    // ------------------------
    void updateBitboards(const Board& board);
    // move restrictions based on board state
    U64 limitPinnedMoves(int square, U64 moves_bb);
    U64 restrictCheckMoves(U64 moves_bb);
    // bitboards -> moves list
    void addMovesFromBitboard(int start_square, U64 moves_bb, int flag=0);
    bool isPromotionPawn(int square);
    // opponent attack information
    void updateAttackMapAndCheck(U64 attacks_bb, bool ours, int start_square, int piece_type = -1);

    // ------------------------
    // Board & State References
    // ------------------------
    int side;                       // Side to move: 0 = white, 1 = black
    GameState curr_gamestate;
    int own_king_square = -1;

    U64 own = 0ULL, opp = 0ULL;
    U64 pawns = 0ULL, knights = 0ULL, bishops = 0ULL, rooks = 0ULL, queens = 0ULL, kings = 0ULL;

    // ------------------------
    // Game State Helpers
    // ------------------------
    bool in_check = false;
    bool in_double_check = false;

    // Mask boards for checks, pins, and attacks
    U64 check_ray_mask = 0ULL;
    U64 check_ray_mask_ext = 0ULL;
    U64 pin_rays = 0ULL;
    U64 opponentAttackMap = 0ULL;
    U64 ownAttackMap = 0ULL;

    // Post-en-passant attack maps for legality checks
    U64 postEnpassantOpponentAttackMap = 0ULL;
    U64 enPassantMaskBlockers = 0ULL;

private:
};

#endif
