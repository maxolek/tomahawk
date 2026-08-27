// engine.h
#ifndef ENGINE_H
#define ENGINE_H

#include "board.h"
#include "magics.h"
#include "moveGenerator.h"
#include "evaluator.h"
#include "searcher.h"
#include "tt.h"
#include "search_limits.h"
#include "logging.h"
#include "stats.h"
#include "timer.h"
#include "game_log.h"
#include "NNUE.h"
#include "book.h"

#include <filesystem>

class Searcher;
struct SearchLimits;
struct SearchResult;


// ---------------------
// -- Engine Settings --
// ---------------------

// CORE

enum class EngineMode {
    ANALYSIS,
    GAME
};

enum class EngineSide {
    WHITE,
    BLACK,
    UNKNOWN
};

struct GameTracker {
    GameResult result;
    GameEndReason reason;
    std::vector<Move> playedMoves;
    std::vector<int> evals;
    std::vector<Move> bestMoves;
    uint64_t lastPositionHash = 0;
    bool active = false;
};

// SEARCH

struct SearchSettings {
    int depth = 0;         // max depth (0 = no limit besides global engine limit)
    int nodes = 0;         // max nodes
    int movetime = 0;      // fixed search time (ms)
    int mate = 0;          // search for mate in N
    int wtime = 0;         // white time left (ms)
    int btime = 0;         // black time left (ms)
    int winc = 0;          // white increment (ms)
    int binc = 0;          // black increment (ms)
    int movestogo = 0;     // moves to next time control
    bool infinite = false; // search until "stop"
    bool ponder = false;   // pondering search
    bool send_eval = false; // send evaluation instead of move
};

struct EngineOptions {
    int  MOVE_OVERHEAD_MS = 10;
    int  MAX_THREADS      = 1;
    int  HASH_SIZE_MB     = 512;
    bool PONDERING        = false;
    bool UCI_SHOW_WDL     = false;

    fs::path opening_pst_path  = fs::path(PROJECT_ROOT) / "bin/pst/pst_opening.txt";
    fs::path endgame_pst_path  = fs::path(PROJECT_ROOT) / "bin/pst/pst_endgame.txt";
    fs::path nnue_weight_path  = fs::path(PROJECT_ROOT) / "bin/nnue_wgts/output_buckets_25wdl_1000.bin";
    fs::path opening_book_path = fs::path(PROJECT_ROOT) / "bin/Titans.bin";
    fs::path syzygy_path       = fs::path(PROJECT_ROOT);

    // 768_128x2
    // halfka_1024
    // 768x512
};

// ------------------
// -- Engine Class --
// ------------------

class Engine {
private:
public:
    // precomp data
    EngineOptions engine_options;
    PolyglotBook book;
    
    // search info
    int ply = 0;
    int search_depth = 0;
    int time_left[2] = {0, 0};          // [white, black] time left
    int increment[2] = {0, 0};          // [white, black] increment
    SearchSettings settings;
    SearchLimits limits;

    // constructors
    Engine();

    // search + eval
    std::unique_ptr<Searcher> searcher;
    NNUE nnue; 
    Evaluator evaluator;                // preload PST tables, eval
    TranspositionTable tt; // outside searcher for future multi-thread

    // boards
    Board game_board;        // main game board
    Board search_board;       // modifable copy of game board for searcher
    bool game_over = false;

    // movegen for current move
    std::unique_ptr<MoveGenerator> movegen;
    int legal_move_count = 0;
    Move legal_moves[MAX_MOVES];

    // output
    SearchResult result;
    Move bestMove = Move::NullMove();
    int bestEval = -MATE_SCORE;
    std::vector<Move> pv_line;

    // think on opponent time
    bool pondering = false;
    Move ponderMove = Move::NullMove();

    // --- State ---
    void clearState();

    // --- UCI Handlers ---
    void setOption(const std::string& name, const std::string& value);
    void setPosition(const std::string& fen, const std::vector<std::string>& uci_moves);
    void ponderHit();
    void print_info();

    // --- Search  ---
    void startSearch();
    void stopSearch();
    void computeSearchTime(const SearchSettings& settings); // defines the settings for it_dp
    //  Result
    void sendBestMove(Move bestMove, bool eval = false, bool ponder = false); // output of it_dp

    // --- Games ---
    void newGame();
    bool checkGameEnd();
    void trackGame();
    void finalizeGameLog();
    // --- Results ---
    bool isCheckmate();
    bool isStalemate();
    //bool isThreefold();

    // --- Tests ---
    uint64_t perft(int depth);
    void perftPrint(int depth); // same as perft but print instead of return
    void perftDivide(int depth);
    void SEETest(int capture_square);
    void staticEvalTest();
    void nnueEvalTest();
    void nnueSIMDTest();
    void moveOrderingTest(int depth);

    // --- Config ---
    void apply_config_file(const fs::path& path);
    void create_config_file(std::string config_name);
};

#endif // ENGINE_H
