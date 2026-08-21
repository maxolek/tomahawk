#ifndef BOARD_H
#define BOARD_H

#include "move.h"
#include "gamestate.h"
#include "zobrist.h"
#include "stats.h"
#include "timer.h"
//#include "NNUE.h"

/**
 * @class Board
 * @brief Represents the current state of a chess game, including piece positions, side to move,
 *        castling rights, en-passant square, Zobrist hash, and move history.
 *
 * The board can be initialized from a FEN string or a default starting position.
 * Moves can be made/unmade with MakeMove/UnmakeMove, updating the internal state.
 */
class Board {
public:
    // ==================== Bitboards ====================
    U64 colorBitboards[2];   ///< [0] = white, [1] = black
    U64 pieceBitboards[6];   ///< 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
    int sqToPiece[64];       ///< Maps square to piece index (0..11), -1 if empty

    // ==================== Game state ====================
    GameState currentGameState;          ///< Tracks castling, en-passant, fifty-move counter
    std::string fen;                     ///< Current board FEN
    int plyCount;                        ///< Number of half-moves played
    bool is_white_move;                  ///< True if white to move
    int move_color;                      ///< 0=white, 1=black
    bool is_in_check;                    ///< True if the side to move is in check
    bool pawn_endgame = false;          ///< True if only kings and pawns remain (for quick NMP verification)

    // ==================== Move & position history ====================
    std::vector<Move> allGameMoves;            ///< All moves played
    std::vector<GameState> gameStateHistory;  ///< Game state history for unmaking moves
    std::unordered_map<U64,int> hash_history; ///< Zobrist hash occurrences for repetition detection
    std::vector<U64> zobrist_history;

    // ==================== Castling trackers ====================
    bool white_castled = false; ///< True if white has castled
    bool black_castled = false; ///< True if black has castled

    // ==================== Zobrist hashing ====================
    U64 zobrist_hash;                ///< Current Zobrist hash
    U64 zobrist_table[12][64];       ///< Zobrist keys for 12 pieces x 64 squares
    // piece is encoded as follows
    // b-p = 0, w-p = 1, b-n = 2, w-n = 3, ... etc (black even, white odd .. incr val)
    U64 zobrist_side_to_move;        ///< Side to move key
    U64 zobrist_castling[4];         ///< Castling rights keys: KQkq
    U64 zobrist_enpassant[8];        ///< En passant file keys: a-h

    // -- polyglot book --
    //U64 polyglot_hash;                ///< Current Zobrist hash
    //U64 polyglot_table[12][64];       ///< Zobrist keys for 12 pieces x 64 squares
    // piece is encoded as follows
    // b-p = 0, w-p = 1, b-n = 2, w-n = 3, ... etc (black even, white odd .. incr val)
    //U64 polyglot_side_to_move;        ///< Side to move key
    //U64 polyglot_castling[4];         ///< Castling rights keys: KQkq
    //U64 polyglot_enpassant[8];        ///< En passant file keys: a-h

    // ==================== NNUE ====================
    //NNUE* nnue; // optional pointer ... allows incremental updates
    //void setNNUE(NNUE* nnue_ptr);

    // ==================== Constructors ====================
    Board(std::string _fen = STARTPOS_FEN); ///< Initialize from FEN
    Board(const Board& other); // deep copy search->game boards

    // ==================== Move execution ====================
    void MakeMove(Move move = false);               ///< Apply a move and update board state
    void UnmakeMove(Move move = false);             ///< Undo a move
    void MakeNullMove();                            ///< Apply a null move (pass turn)
    void UnmakeNullMove();  ///< Undo a null move

    // ==================== Piece manipulation ====================
    void putPiece(int pt12, int sq);         ///< Place a piece on a square
    void removePiece(int sq);                ///< Remove a piece from a square
    void MovePiece(int piece, int start, int target); ///< Move a piece between squares
    void CapturePiece(int piece, int target, bool is_enpassant, bool captured_is_moved_piece);
    void PromoteToPiece(int piece, int target); ///< Promote pawn to another piece

    // ==================== Piece queries ====================
    int getMovedPiece(int start_square) const;
    int getCapturedPiece(int target_square) const;
    int getSideAt(int square) const;
    int getPieceAt(int square, int side) const;
    int kingSquare(bool white) const;

    // ==================== Special move checks ====================
    bool canEnpassantCapture(int epFile) const; ///< Checks if en-passant is possible
    void updateFiftyMoveCounter(int moved_piece, bool isCapture);
    bool isThreefold();

    // ==================== Board state checks ====================
    bool inCheck(bool init); ///< True if the side-to-move is in check
    bool givesCheck(Move move); // see if a move will give a check if played

    // ==================== FEN handling ====================
    void setFromFEN(std::string _fen); ///< Initialize board from FEN
    void setBoardFEN();                 ///< Generate FEN string from current state
    std::string getBoardFEN();          ///< Return current board FEN

    // ==================== Zobrist helper functions ====================
    U64 randomU64();                           ///< Generate random 64-bit number
    void initZobristKeys();                    ///< Initialize Zobrist keys
    U64 computeZobristHash();                  ///< Compute current Zobrist hash
    U64 zobristCastlingHash(int castling_rights); ///< Hash from castling rights
    void auditZobrist(const Board &other, const std::string &label = "") const;
    void debugZobristDifference(uint64_t old_hash, uint64_t new_hash);
    void print_zobrist_history(int ply, const std::string& move_str);
    bool canEPCapture(); // check if ep is legal when possible - zobrist 
    bool isEpHashable(int epFile, bool stmIsWhite) const; // incremental version
    // polyglot
    //void initPolyglotKeys();                    ///< Initialize Zobrist keys
    //U64 computePolyglotHash();                  ///< Compute current Zobrist hash
    //U64 polyglotPieceHash(int piece, int color); // color^1 + 2*piece
    //U64 polyglotCastlingHash(int castling_rights); ///< Hash from castling rights

    // ==================== Debug ====================
    void print_board(); ///< Pretty-print board with FEN and additional info
};

#endif // BOARD_H
