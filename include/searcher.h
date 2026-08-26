#ifndef SEARCHER_H
#define SEARCHER_H

#include "engine.h"
#include "search_limits.h"
#include "helpers.h"
#include "stats.h"
#include "timer.h"
#include "NNUE.h"
#include "tt.h"
#include "moveGenerator.h"

class Engine;
class Evaluator;
class NNUE;
class MoveGenerator;

// ---- search constants ----
struct SearchParams {
    // pruning
    int   DELTA_PRUNE_THRESHOLD             = 1'000;
    int   SEE_PRUNE_THRESHOLD               = -50;
    int   REVERSE_FUTILITY_PRUNE_THRESHOLD  = 150;
    int   FUTILITY_PRUNE_MARGIN             = 100;
    int   FUTILITY_PRUNE_MOVE_THRESHOLD     = 5;
    // aspiration windows
    int   ASPIRATION_WINDOW      = 50;
    int   ASPIRATION_START_DEPTH = 6;
    int   ASPIRATION_DEPTH_SCALE = 10;
    float ASPIRATION_RESEARCH_SCALE = 2.0f;
    // positional
    int   DRAW_EVAL              = 0;
    int   CONTEMPT               = 0;
    // reductions
    //int   R_IID                  = 0.5f;     // internal iterative deepening (float = divide, int = subtract)
    int   IID_DEPTH_THRESHOLD    = 6;       // iid start depth (>=6  is standard)
    int   R_NMP                  = 3;      // null-move pruning
    float R_LMR_CONST            = 0.99f;  // late move reductions 
    float R_LMR_DENOM            = 3.14f;  //   = const + [log(depth) * log(move_order)] / denom
    int   LMR_MOVE_ORDER_THRESHOLD = 3; // minimum move order # to start using LMR
    int   LMR_DEPTH_THRESHOLD    = 3; // max search depth where LMR doesnt trigger
};

// ---- move ordering priorities ----
struct MoveScores {
    int TT_BASE       =  10'000'000;
    int PV_BASE       =   9'000'000;
    int PROMO_BASE    =   8'500'000;
    int GOOD_CAP_BASE =   8'000'000;
    int KILLER_BASE   =   7'000'000;
    int QUIET_BASE    =           0;
    int BAD_CAP_BASE  =  -1'000'000;
};

struct RootMoveScores {
    // Previous iteration information dominates
    int PV_BASE        = 10'000'000;
    int TT_BASE        =  9'000'000;
    // Tactical tiebreakers (else return eval)
    int PROMO_BASE     = 200;
    int GOOD_CAP_BASE  = 100;
    int BAD_CAP_BASE   = -100;
    // Quiet moves use previous scores/history as tiebreakers
    int QUIET_BASE     = 0;
};

class Searcher {
public:
    // ------------------------------- VARS -------------------------------

    // Object-owned state
    //Engine& engine;
    MoveGenerator& movegen; // = engine.movegen
    Board& board; //= engine.search_board;
    Evaluator& eval; // = engine.evaluator;
    NNUE& nnue; // = engine.nnue;
    TranspositionTable& tt; // = engine.tt

    SearchParams params;
    RootMoveScores root_scores;
    MoveScores move_scores;

    // iteration-local eval table
    int node_count_table[1 << 16];   // keyed by Move.Value()
    void store_last_node_counts(const SearchResult& res); // fill node_count_table
    int get_node_count(Move m) const; // retrieve from table

    bool stop = false;

    Move killerMoves[MAX_DEPTH][2] = {};
    int historyHeuristic[12][64] = {};

    std::vector<Move> best_line;
    std::vector<Move> best_quiescence_line;

    // ------------------------------- FUNCS -------------------------------

    Searcher(Board& b, MoveGenerator& mg, Evaluator& ev, NNUE& nn, TranspositionTable& _tt) 
        : board(b), 
          movegen(mg),
          eval(ev), 
          nnue(nn),
          tt(_tt) {}

    // ------------------------------- Main Search -------------------------------
    SearchResult iterativeDeepening(
        Move first_moves[MAX_MOVES],
        int move_count,
        SearchLimits limits
    );
    
    SearchResult search(
        RootMove root_moves[MAX_MOVES],
        int move_count,
        int depth,
        SearchLimits& limits,
        std::vector<Move>& previousPV,
        int previousEval
    );

    // --------------------------- Negamax & Quiescence --------------------------
    int negamax(
        int depth,
        int alpha,
        int beta,
        PV& pv,
        std::vector<Move>& previousPV,
        SearchLimits& limits,
        int ply,
        bool can_nmp
    );

    int quiescence(
        int alpha,
        int beta,
        PV& pv,
        SearchLimits& limits,
        int ply,
        int depth,
        int search_depth
    );

    // ------------------------------- Ordering & Scoring -------------------------------
    int rootMoveScore(
        const Move& move, 
        const Move& ttMove, 
        const Move& pvMove
    );
    
    int moveScore(
        const Move& move,
        const Board& board,
        int ply,
        const Move& ttMove,
        const std::vector<Move>& previousPV
    );

    void orderedMoves(
        Move moves[MAX_MOVES],
        size_t count,
        const Board& board,
        int ply,
        const Move ttMove,
        const std::vector<Move>& previousPV
    );

    int generateAndOrderMoves(
        Move moves[MAX_MOVES],
        int ply,
        const Move ttMove,
        const std::vector<Move>& previousPV
    );

    // ------------------------------- PV / pruning / helpers -------------------------------
    void updatePV(
        std::vector<Move>& pv, 
        const Move& move, 
        const std::vector<Move>& childPV
    );

    bool shouldPrune(
        Move& move,
        int standPat,
        int alpha,
        int search_depth,
        int ply
    );

    // -------------------------------- Search Reduction Parameters ----------------------------
    int R_lmr(
        int depth, 
        int move_order
    );

    // ----- nnue helpers -----
    bool update_kings(const bool& is_king_move, const Move& move);
    void perform_move(Board& board, const Move& move, const bool update_kings);
    void perform_unmove(Board& board, const Move& move, const bool update_kings);
};

#endif // SEARCHER_H
