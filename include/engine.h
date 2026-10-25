// engine.h
#ifndef ENGINE_H
#define ENGINE_H

#include "affinity.h"
#include "board.h"
#include "magics.h"
#include "movegen.h"
#include "threads.h"
#include "tt.h"
#include "search_limits.h"
#include "logging.h"
#include "search_stats.h"
#include "timer.h"
#include "game_stats.h"
#include "nnue.h"
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

// upper bound for the UCI Threads option
inline constexpr int MAX_SEARCH_THREADS = 256;

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
    int  HASH_SIZE_MB     = 512; // use higher int range incase multi-gb TT is tried (unlikely)
    bool PONDERING        = false;
    bool UCI_SHOW_WDL     = false;
    std::string CPU_AFFINITY = "auto"; // CPUs of the search threads: auto, none, or a list (affinity.h)

    // big      net : 1024_16_32_pairmul_screlu  
    // small    net : output_buckets_25wdl_1000  
    // original net : 768_128x
    fs::path nnue_weight_path  = fs::path(PROJECT_ROOT) / "nnue/weights/cirr_filt_0wdl_LR_001_000435_600batches-1200.bin"; // 1024_16_32_pairmul_screlu  output_buckets_25wdl_1000  768_128x
    fs::path opening_book_path = fs::path(PROJECT_ROOT) / "bin/Titans.bin";
    fs::path syzygy_path       = fs::path(PROJECT_ROOT);
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
    TranspositionTable tt; // outside searcher for future multi-thread
    SearchThreadPool threads; // lazy SMP helper threads
    affinity::Spec affinity_spec; // parsed engine_options.CPU_AFFINITY
    std::string affinity_report; // last thread placement reported (printed when it changes)
    int affinity_failures = 0;

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
    // threads
    void setThreads(int count); // request `count` search threads (1..MAX_SEARCH_THREADS), then fitThreads()
    bool fitThreads();          // run min(requested, threadLimit()) threads; true if that number changed
    int searchThreads() const;  // search threads that run: the UCI thread + helpers
    int threadLimit() const;    // one thread per available CPU (affinity::cpu_limit), at most MAX_SEARCH_THREADS
    void reportThreads() const; // "info string set Threads = N", noting a capped request
    bool setAffinity(const std::string& value); // Affinity option; an invalid value is reported and ignored
    //  Result
    void sendBestMove(Move bestMove, bool eval = false, bool ponder = false); // output of it_dp
    void sendSearchInfo(); // nodes, time, depth, eval, pv, etc (info string)

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
    void nnueEvalTest();
    void nnueSIMDTest();
    void moveOrderingTest(int depth);

    // --- Config ---
    void apply_config_file(const fs::path& path);
    void create_config_file(std::string config_name);
};

#endif // ENGINE_H
