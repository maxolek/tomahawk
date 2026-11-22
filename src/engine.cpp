// engine_options.cpp
#include <engine.h>
#include <tests.h>
#include "../nnue/tests.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <sstream>

// ------------------
// -- Constructors --
// ------------------

EngineMode mode = EngineMode::ANALYSIS;
EngineSide side = EngineSide::UNKNOWN;
GameTracker tracker;

Engine::Engine() : tt(engine_options.HASH_SIZE_MB) {
    game_board = Board();
    search_board = game_board; //Board(game_board);

    movegen = std::make_unique<MoveGenerator>(search_board);
    tt.clear();

#ifdef SMALL_NET
    nnue.loadSmall(engine_options.nnue_weight_path);
#else
    nnue.load(engine_options.nnue_weight_path);
#endif
    searcher = std::make_unique<Searcher>(search_board, *movegen, nnue, nnue.acc_state, tt);

    book.load(engine_options.opening_book_path);

    g_stats = SearchStats();
    fitThreads();
}


// ------------------
// -- Engine State --
// ------------------

void Engine::clearState() {
    pondering = false;
    bestMove = Move::NullMove();
    ponderMove = Move::NullMove();
    search_depth = 0;
    time_left[0] = time_left[1] = 0;
    increment[0] = increment[1] = 0;
    //stats = SearchStats();
    g_stats = SearchStats();
    tt.clear();
    game_board.setFromFEN(STARTPOS_FEN);
    search_board = game_board;
}

// -------------------
// -- UCI Handlers  --
// -------------------

void Engine::setOption(const std::string& name, const std::string& value) {
    auto boolFromString = [](const std::string &val) {
        return val == "true" || val == "True" || val == "1";
    };

    // --------- engine options ---------
    if (name == "Move Overhead") {
        limits.stopped = false;
        engine_options.MOVE_OVERHEAD_MS = std::stoi(value);
        std::cout << "info string set Move Overhead = " << engine_options.MOVE_OVERHEAD_MS << std::endl;
    } 
    else if (name == "Hash") {
        const int hash_size_mb = std::stoi(value);
        if (tt.resize(hash_size_mb)) {
            engine_options.HASH_SIZE_MB = hash_size_mb;
            std::cout << "info string set Hash = " << engine_options.HASH_SIZE_MB << std::endl;
        } else {
            std::cout << "info string keeping Hash = " << engine_options.HASH_SIZE_MB << std::endl;
        }
    } 
    else if (name == "Threads") {
        setThreads(std::stoi(value));
        reportThreads();
    } 
    else if (name == "Affinity") {
        if (setAffinity(value)) {
            std::cout << "info string set Affinity = " << engine_options.CPU_AFFINITY << std::endl;
            if (fitThreads()) reportThreads(); // the CPUs available changed
        }
    }

    else if (name == "Ponder") {
        engine_options.PONDERING = boolFromString(value);
        std::cout << "info string set Ponder = " << (engine_options.PONDERING ? "true" : "false") << std::endl;
    } 
    else if (name == "UCI_ShowWDL") {
        engine_options.UCI_SHOW_WDL = boolFromString(value);
        std::cout << "info string set UCI_ShowWDL = " << (engine_options.UCI_SHOW_WDL ? "true" : "false") << std::endl;
    } 

    // -------- search params --------
    else if (name == "delta_prune_threshold") {
        searcher->params.DELTA_PRUNE_THRESHOLD = std::stoi(value);
        std::cout << "info string set delta_prune_threshold = " << searcher->params.DELTA_PRUNE_THRESHOLD << std::endl;
    }
    else if (name == "see_prune_threshold") {
        searcher->params.SEE_PRUNE_THRESHOLD = std::stoi(value);
        std::cout << "info string set see_prune_threshold = " << searcher->params.SEE_PRUNE_THRESHOLD << std::endl;
    }
    else if (name == "aspiration_start_depth") {
        searcher->params.ASPIRATION_START_DEPTH = std::stoi(value);
        std::cout << "info string set aspiration_start_depth = " << searcher->params.ASPIRATION_START_DEPTH << std::endl;
    }
    else if (name == "aspiration_window") {
        searcher->params.ASPIRATION_WINDOW = std::stoi(value);
        std::cout << "info string set aspiration_window = " << searcher->params.ASPIRATION_WINDOW << std::endl;
    }
    else if (name == "aspiration_start_depth") {
        searcher->params.ASPIRATION_START_DEPTH = std::stoi(value);
        std::cout << "info string set aspiration_start_depth = " << searcher->params.ASPIRATION_START_DEPTH << std::endl;
    }
    else if (name == "aspiration_research_scale") {
        searcher->params.ASPIRATION_RESEARCH_SCALE = std::stof(value);
        std::cout << "info string set aspiration_research_scale = " << searcher->params.ASPIRATION_RESEARCH_SCALE << std::endl;
    }
    else if (name == "draw_eval") {
        searcher->params.DRAW_EVAL = std::stoi(value);
        std::cout << "info string set draw_eval = " << searcher->params.DRAW_EVAL << std::endl;
    }
    else if (name == "contempt") {
        searcher->params.CONTEMPT = std::stoi(value);
        std::cout << "info string set contempt = " << searcher->params.CONTEMPT << std::endl;
    }
    else if (name == "r_nmp") {
        searcher->params.R_NMP = std::stoi(value);
        std::cout << "info string set r_nmp = " << searcher->params.R_NMP << std::endl;
    }
    else if (name == "lmr_move_order_threshold") {
        searcher->params.LMR_MOVE_ORDER_THRESHOLD = std::stoi(value);
        std::cout << "info string set lmr_move_order_threshold = " << searcher->params.LMR_MOVE_ORDER_THRESHOLD << std::endl;
    }
    else if (name == "lmr_depth_threshold") {
        searcher->params.LMR_DEPTH_THRESHOLD = std::stoi(value);
        std::cout << "info string set lmr_depth_threshold = " << searcher->params.LMR_DEPTH_THRESHOLD << std::endl;
    }
    else if (name == "r_lmr_const") {
        searcher->params.R_LMR_CONST = std::stof(value)/100.0f;
        std::cout << "info string set r_lmr_const = " << static_cast<int>(100*searcher->params.R_LMR_CONST) << std::endl;
    }
    else if (name == "r_lmr_denom") {
        searcher->params.R_LMR_DENOM = std::stof(value)/100.0f;
        std::cout << "info string set r_lmr_denom = " << static_cast<int>(100*searcher->params.R_LMR_DENOM) << std::endl;
    }

    // -------- books / paths --------
    else if (name == "opening_book") {
        engine_options.opening_book_path = PROJECT_ROOT / fs::path("bin") / fs::path(value + ".bin"); // auto-construct full path

        if (!value.empty()) {
            if (book.load(engine_options.opening_book_path))
                std::cout << "info string Book loaded successfully: " << engine_options.opening_book_path << std::endl;
            else
                std::cout << "info string Failed to load book: " << engine_options.opening_book_path << std::endl;
        }
    }
    else if (name == "SyzygyPath") {
        engine_options.syzygy_path = value;
        //tablebases.load(value);
        std::cout << "info string set SyzygyPath = " << engine_options.syzygy_path << std::endl;
    }

    // -------- logging --------
    #if defined(DEV) || defined(STATS) || defined(TIMING)
    else if (name == "log_dir") {
        Logging::setLogDir(value);
        std::cout << "info string log_dir loaded successfully: " << Logging::log_dir << std::endl;
    }
    else if (name == "uci_logging") {
         if (Logging::track_uci != boolFromString(value)) {Logging::setTrackUCI(boolFromString(value));}
    }
    else {
        std::cout << "info string ignoring unknown option: " << name << std::endl;
    }
    #endif

    std::cout.flush();
}


void Engine::setPosition(const std::string& fen,
                         const std::vector<std::string>& moveStrs)
{
    // --- Reset board ---
    if (fen == "startpos")
        game_board.setFromFEN(STARTPOS_FEN);
    else
        game_board.setFromFEN(fen);

    // ply tracking
    if (game_board.is_white_move) ply = 1;
    else ply = 2;

    // --- Apply moves ---
    // must interpret uci in context of the board state for ep
    for (const auto& moveStr : moveStrs) {

        if (moveStr == "null" || moveStr == "NULL" || moveStr == "Null") {
            game_board.MakeMove();
            ply++;
            continue;
        }

        int start  = algebraic_to_square(moveStr.substr(0, 2));
        int target = algebraic_to_square(moveStr.substr(2, 2));
        int flag   = Move::noFlag;

        int movedPiece = game_board.getMovedPiece(start);
        bool isPawn = (movedPiece == pawn);

        // ---- Promotion ----
        if (moveStr.length() == 5) {
            switch (moveStr[4]) {
                case 'q': flag = Move::promoteToQueenFlag;  break;
                case 'r': flag = Move::promoteToRookFlag;   break;
                case 'b': flag = Move::promoteToBishopFlag; break;
                case 'n': flag = Move::promoteToKnightFlag; break;
            }
        }

        // ---- Castling ----
        if (movedPiece == king &&
            (moveStr == "e1g1" || moveStr == "e1c1" ||
             moveStr == "e8g8" || moveStr == "e8c8")) {
            flag = Move::castleFlag;
        }

        // ---- En-passant ----
        if (isPawn && game_board.currentGameState.enPassantFile != -1) {
            int epRank   = game_board.is_white_move ? 5 : 2;
            int epSquare = game_board.currentGameState.enPassantFile + epRank * 8;
            if (target == epSquare)
                flag = Move::enPassantCaptureFlag;
        }

        // ---- Double pawn push ----
        if (isPawn && flag == Move::noFlag) {
            int sr = start / 8;
            int tr = target / 8;
            if (std::abs(sr - tr) == 2)
                flag = Move::pawnTwoUpFlag;
        }

        Move m(start, target, flag);

        //if (nnue) nnue->on_make_move(game_board, m);
        game_board.MakeMove(m);
        ply++;
    }

    game_board.is_in_check = game_board.inCheck(true);

    // --- Sync search board ---
    search_board = game_board; //Board(game_board);

    // --- game handling ---
    if (mode == EngineMode::GAME) {
        side = game_board.is_white_move ? EngineSide::WHITE : EngineSide::BLACK;
        g_gamelog.side = game_board.is_white_move ? "white" : "black";
    }

    game_over = checkGameEnd();
}


// --------------------
// -- Search Control --
// --------------------

void Engine::computeSearchTime(const SearchSettings& search_settings) {
    // log
    if (g_run_context.is_new_game) {
        g_gamelog.wtime = search_settings.wtime;
        g_gamelog.btime = search_settings.btime;
        g_gamelog.winc = search_settings.winc;
        g_gamelog.binc = search_settings.binc;
        g_gamelog.movestogo = search_settings.movestogo;
        g_gamelog.depth = search_settings.depth;
        g_gamelog.nodes = search_settings.nodes;
        g_gamelog.movetime = search_settings.movetime;
        g_run_context.is_new_game = false;
    }

    // hard coded time/depth/nodes limits
    if (search_settings.infinite) {
        limits = SearchLimits(0, search_settings.depth, search_settings.nodes);
        return;
    }

    if (search_settings.movetime > 0 || search_settings.depth > 0 || search_settings.nodes > 0) {
        limits = SearchLimits(
            search_settings.movetime - engine_options.MOVE_OVERHEAD_MS,
            search_settings.depth,
            search_settings.nodes
        );
        return;
    }

    // game clock
    int side_index = game_board.is_white_move ? 0 : 1;
    int movesToGo = search_settings.movestogo > 0 ? search_settings.movestogo : 20;
    int myTime = (side_index == 0 ? search_settings.wtime : search_settings.btime);
    int myInc  = (side_index == 0 ? search_settings.winc  : search_settings.binc);

    // compute function
    double aggressiveness = 1.0;
    int timeBudget = static_cast<int>(
        (static_cast<double>(myTime) / movesToGo + myInc / 2.0)
        * aggressiveness
    );

    timeBudget = std::max(engine_options.MOVE_OVERHEAD_MS, timeBudget);
    limits = SearchLimits(timeBudget);
}


void Engine::startSearch() {
    resetSearchStats();
    result_depth = 0;

    // set nnue
    //nnue.build_accumulators(search_board);
    nnue.build_halfka_accumulators(search_board);

    // book probe
    if (!engine_options.opening_book_path.empty()) {
        uint64_t key = search_board.zobrist_hash;
        auto bookMoves = book.get_moves(key);

        if (!bookMoves.empty()) {
            // pick a move (currently first or weighted)
            std::string bestBookMove = book.pick_weighted_move(bookMoves);

            // Castling translation debug
            if (bestBookMove == "e1h1") { 
                bestBookMove = "e1g1"; 
            }
            else if (bestBookMove == "e1a1") { 
                bestBookMove = "e1c1"; 
            }
            else if (bestBookMove == "e8h8") { 
                bestBookMove = "e8g8"; 
            }
            else if (bestBookMove == "e8a8") { 
                bestBookMove = "e8c8"; 
            }

            bestMove = bestBookMove;
            g_stats.move = bestMove;
            g_stats.eval = bestEval;
            publishSearchStats({g_stats});
            return; // skip search entirely
        }
    }

    // search time / params
    search_depth = settings.depth;
    time_left[0] = settings.wtime;
    time_left[1] = settings.btime;
    increment[0] = settings.winc;
    increment[1] = settings.binc;
    pondering = settings.infinite;

    // stats tracking
    auto start_time = std::chrono::steady_clock::now();
    g_run_context.search_uuid = generate_uuid();
    
    // --- generate first moves once ---
    Move first_moves[MAX_MOVES];
    int count = movegen->generateMoves(game_board, false);
    std::copy_n(movegen->moves, count, first_moves);

    // --- run search ---
    // Helpers search the same root and share the TT; their best moves vote with the main thread.
    computeSearchTime(settings);

    // --- search ---
    limits.stop_signal = &threads.stop_signal;
    threads.start(search_board, limits, searcher->params);
    result = searcher->iterativeDeepening(first_moves, count, limits);

    if (count > 0 && Move::SameMove(result.bestMove, Move::NullMove()))
        result.bestMove = first_moves[0];
    threads.stop_and_wait();

    const Move main_move = result.bestMove;
    const int main_eval = result.eval;
    const std::vector<Move> main_pv = result.best_line.to_vector();
    g_stats.move = main_move;
    g_stats.eval = main_eval;
    g_stats.principal_variation = main_pv;

    ThreadVoteResult vote = threads.vote(result, g_stats.max_completed_depth);
    for (const SearchStats& thread_stats : vote.helper_stats)
        g_stats.max_seldepth = std::max(g_stats.max_seldepth, thread_stats.max_seldepth);
    if (!Move::SameMove(vote.best_move, Move::NullMove())) {
        result_depth = vote.depth;
        result.bestMove = vote.best_move;
        result.eval = vote.eval;
        result.best_line = vote.best_line;

        std::cout << "info string thread vote: " << result.bestMove.uci()
                  << " (weight " << vote.weight << "/" << vote.total_weight
                  << ", votes " << vote.votes << "/" << vote.total_votes << "; ";
        for (size_t i = 0; i < vote.tallies.size(); ++i) {
            if (i) std::cout << ", ";
            const ThreadVoteResult::Tally& tally = vote.tallies[i];
            std::cout << tally.move.uci() << "=" << tally.weight
                      << "/" << tally.votes;
        }
        std::cout << ")\n";
    } else {
        result_depth = g_stats.max_completed_depth;
    }

    #if defined(DEV) || defined(STATS)
        g_stats.tt_fill_ratio = tt.fillRatio();
    #endif
    bestMove = result.bestMove;
    bestEval = result.eval; 
    pv_line = result.best_line.to_vector();
    //sendBestMove(bestMove);

    // finalize cumulative stats
    g_stats.time_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();
    #if !defined(DEV) && !defined(STATS)
        g_stats.nodes = searcher->nodes;
    #endif
    uint64_t total_nodes = g_stats.nodes;
    #if defined(DEV) || defined(STATS)
        total_nodes += g_stats.qnodes;
    #endif
    total_nodes += threads.helper_nodes();
    g_stats.hashfull = tt.hashfull();
    g_stats.ebf = result_depth > 0 && total_nodes > 0
        ? std::pow(static_cast<double>(total_nodes), 1.0 / result_depth)
        : 0.0;
    #if defined(DEV) || defined(STATS)
        g_stats.game_ply = ply;

        tracker.bestMoves.push_back(bestMove);
        tracker.evals.push_back(bestEval);

        logSearchStats(game_board.getBoardFEN());  // JSON written once at end
        logTimingStats(game_board.getBoardFEN());
    #endif 
    std::vector<SearchStats> stats{g_stats};
    stats.insert(stats.end(), vote.helper_stats.begin(), vote.helper_stats.end());
    #if defined(DEV) || defined(STATS)
        for (SearchStats& thread_stats : stats)
            thread_stats.tt_fill_ratio = g_stats.tt_fill_ratio;
    #endif
    publishSearchStats(std::move(stats));
}

void Engine::resetSearchStop() {
    threads.stop_signal.store(false, std::memory_order_relaxed);
}

void Engine::stopSearch() {
    threads.stop_signal.store(true, std::memory_order_relaxed);
    tracker.result = GameResult::ABORTED;
    tracker.reason = GameEndReason::NONE;
}

std::vector<SearchStats> Engine::completedSearchStats() const {
    std::lock_guard<std::mutex> lock(search_stats_mutex);
    return completed_search_stats;
}

void Engine::publishSearchStats(std::vector<SearchStats> stats) {
    std::lock_guard<std::mutex> lock(search_stats_mutex);
    completed_search_stats = std::move(stats);
}

// -------------------
// -- threads --
// -------------------

void Engine::setThreads(int count) {
    engine_options.MAX_THREADS = std::clamp(count, 1, MAX_SEARCH_THREADS);
    fitThreads();
}

bool Engine::fitThreads() {
    const int target = std::min(engine_options.MAX_THREADS, threadLimit());
    const bool count_changed = target != searchThreads();
    affinity::Plan new_plan = affinity::make_plan(affinity_spec, target);
    if (!count_changed && affinity_plan.threads == new_plan.threads &&
        affinity_plan.unpinned == new_plan.unpinned)
        return false;

    affinity_plan = std::move(new_plan);
    threads.set_helpers(target - 1, nnue, tt, affinity_plan);
    return count_changed;
}

int Engine::searchThreads() const {
    return threads.helper_count() + 1;
}

int Engine::threadLimit() const {
    return std::min(MAX_SEARCH_THREADS, affinity::cpu_limit(affinity_spec));
}

void Engine::reportThreads() const {
    std::cout << "info string set Threads = " << searchThreads();
    if (searchThreads() < engine_options.MAX_THREADS)
        std::cout << " (requested " << engine_options.MAX_THREADS << "; at most one search thread per available cpu)";
    std::cout << std::endl;
}

bool Engine::setAffinity(const std::string& value) {
    affinity::Spec spec;
    std::string error;
    if (!affinity::parse(value, spec, error)) {
        std::cout << "info string invalid Affinity value '" << value << "': " << error
                  << "; keeping " << engine_options.CPU_AFFINITY << std::endl;
        return false;
    }
    affinity_spec = spec;
    engine_options.CPU_AFFINITY = affinity::to_string(spec);
    return true;
}

void Engine::applyMainThreadAffinity() const {
    static const std::vector<int> unpinned;
    const std::vector<int>& cpus = affinity_plan.threads.empty() ? unpinned : affinity_plan.threads[0];
    const std::vector<int>& fallback = affinity_plan.unpinned.empty() ? unpinned : affinity_plan.unpinned[0];
    const bool pin_failed = cpus.empty()
        ? !affinity::apply_to_current_thread(fallback)
        : !affinity::apply_to_current_thread(cpus);
    if (pin_failed) affinity::apply_to_current_thread(fallback);

    int helper_failures = threads.placement_failures();
    std::string report = affinity_plan.summary;
    report += "; main search thread currently on " + affinity::describe_current_core();
    if (pin_failed) report += " (the operating system refused the main-thread placement)";
    if (helper_failures > 0)
        report += "; the operating system refused to pin " + std::to_string(helper_failures) +
                  " helper thread(s)";
    std::cout << "info string " << report << std::endl;
}

void Engine::ponderHit() {
    pondering = false;
}


// -------------------
// -- Communication --
// -------------------

void Engine::sendBestMove(Move best, bool eval, bool ponder) {
    game_board.MakeMove(best);
    search_board.MakeMove(best);
    game_over = checkGameEnd();

    if (mode == EngineMode::GAME) {trackGame();}

    //resetSearchStats();
    Logging::logUCIout(best.uci());

    if (eval) std::cout << "eval " << bestEval << " ";
    std::cout << "bestmove " << best.uci();
    if (ponder) std::cout << " ponder";
    std::cout << std::endl;
}

void Engine::sendSearchInfo() {
    uint64_t total_nodes = g_stats.nodes;
    #if defined(DEV) || defined(STATS)
        total_nodes += g_stats.qnodes;
    #endif
    total_nodes += threads.helper_nodes();
    const int seldepth = std::max(g_stats.max_seldepth, threads.helper_seldepth());

    std::cout << "info";
    std::cout << " depth " << result_depth;
    std::cout << " seldepth " << seldepth;
    std::cout << " nodes " << total_nodes;
    std::cout << " time " << g_stats.time_ms;
    std::cout << " hashfull " << tt.hashfull();
    // Approximate EBF as the geometric mean nodes per completed ply.
    if (result_depth > 0 && total_nodes > 0)
        std::cout << " ebf " << std::pow(static_cast<double>(total_nodes), 1.0 / result_depth);
    std::cout << " score cp " << result.eval;
    std::cout << " pv";
    for (const auto& move : result.best_line.to_vector()) {
        std::cout << " " << move.uci();
    }
    std::cout << std::endl;
}


// GAME LOGGING

void Engine::newGame() {
    mode = EngineMode::GAME;
    startNewSession();
    clearState();
    g_run_context.game_uuid = generate_uuid();
    g_run_context.is_new_game = true;
    g_game_start_time = std::chrono::steady_clock::now();

    tracker.playedMoves.clear();
    tracker.lastPositionHash = 0;
    tracker.active = true;

    if (Logging::log_dir == Logging::DEFAULT_LOG_DIR) Logging::setLogDir(Logging::san_jacinto_logs / "game_logs");
    g_gamelog = GameLog{};
    g_gamelog.startFEN = game_board.getBoardFEN();


    game_board.setFromFEN(STARTPOS_FEN);
    search_board = game_board;
}

bool Engine::checkGameEnd() {
    if (g_gamelog.finalized) return true;

    if (isCheckmate()) {
        g_gamelog.outcome =
            !game_board.is_white_move ? GameResult::WHITE_WIN
                                     : GameResult::BLACK_WIN;
        g_gamelog.reason = GameEndReason::CHECKMATE;
    }
    else if (isStalemate()) {
        g_gamelog.outcome = GameResult::DRAW;
        g_gamelog.reason = GameEndReason::STALEMATE;
    }
    else if (game_board.currentGameState.fiftyMoveCounter >= 50) {
        g_gamelog.outcome = GameResult::DRAW;
        g_gamelog.reason = GameEndReason::FIFTY_MOVE;
    }
    else if (game_board.isThreefold()) {
        g_gamelog.outcome = GameResult::DRAW;
        g_gamelog.reason = GameEndReason::THREEFOLD;
    }
    else {
        return false; // game not over yet
    }

    // finalize
    g_game_end_time = std::chrono::steady_clock::now();
    g_gamelog.totalTimeSeconds = std::chrono::duration<double>(g_game_end_time - g_game_start_time).count();
    g_gamelog.moves = game_board.allGameMoves;
    g_gamelog.finalEval = nnue.full_eval(game_board);
    logGameLog();
    g_gamelog.finalized = true;

    //tt.clear();

    return true;
}


bool Engine::isCheckmate() {
    return !movegen->hasLegalMoves(game_board) && game_board.is_in_check;
}

bool Engine::isStalemate() {
    return !movegen->hasLegalMoves(game_board) && !game_board.is_in_check;
}


void Engine::trackGame() {
    tracker.active = true;
    size_t old_count = tracker.playedMoves.size();
    size_t new_count = game_board.allGameMoves.size();

    if (new_count > old_count) {
        for (size_t i = old_count; i < new_count; i++) {
            tracker.playedMoves.push_back(game_board.allGameMoves[i]);
        }
    }
}

void Engine::finalizeGameLog() {
    g_gamelog.finalEval = nnue.full_eval(game_board);
    logGameLog();
}
