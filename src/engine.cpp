// engine_options.cpp
#include <searcher.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>


// ------------------
// -- Constructors --
// ------------------

EngineMode mode = EngineMode::ANALYSIS;
EngineSide side = EngineSide::UNKNOWN;
GameTracker tracker;

Engine::Engine() {
    game_board = Board();
    search_board = game_board; //Board(game_board);

    movegen = std::make_unique<MoveGenerator>(search_board);
    tt.clear();

    nnue.load(engine_options.nnue_weight_path);
    evaluator.loadOpeningPST(engine_options.opening_pst_path);
    evaluator.loadEndgamePST(engine_options.endgame_pst_path);

    searcher = std::make_unique<Searcher>(search_board, *movegen, evaluator, nnue, tt);

    book.load(engine_options.opening_book_path);

    g_stats = SearchStats();
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
    //evaluator = Evaluator(&precomp);
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
        engine_options.HASH_SIZE_MB = std::stoi(value);
        tt.resize(engine_options.HASH_SIZE_MB);
        std::cout << "info string set Hash = " << engine_options.HASH_SIZE_MB << std::endl;
    } 
    else if (name == "Threads") {
        engine_options.MAX_THREADS = std::stoi(value);
        // Apply threads to searcher
        std::cout << "info string set Threads = " << engine_options.MAX_THREADS << std::endl;
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
        searcher->params.ASPIRATION_RESEARCH_SCALE = std::stoi(value);
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
        searcher->params.R_LMR_CONST = std::stof(value)/100.000;
        std::cout << "info string set r_lmr_const = " << static_cast<int>(100*searcher->params.R_LMR_CONST) << std::endl;
    }
    else if (name == "r_lmr_denom") {
        searcher->params.R_LMR_DENOM = std::stof(value)/100.000;
        std::cout << "info string set r_lmr_denom = " << static_cast<int>(100*searcher->params.R_LMR_DENOM) << std::endl;
    }

    // -------- books / paths --------
    else if (name == "opening_pst_file") {
        engine_options.opening_pst_path = PROJECT_ROOT / fs::path("bin") / fs::path(value + ".txt");
        if(evaluator.loadEndgamePST(engine_options.opening_pst_path)) {
            std::cout << "info string OpeningPST loaded successfully: " << engine_options.opening_pst_path << std::endl;
        } else {
            std::cout << "info string Failed to load OpeningPST: " << engine_options.opening_pst_path << std::endl;
        }
    }
    else if (name == "endgame_pst_file") {
        engine_options.endgame_pst_path = PROJECT_ROOT / fs::path("bin") / fs::path(value + ".txt");
        if(evaluator.loadEndgamePST(engine_options.endgame_pst_path)) {
            std::cout << "info string EndgamePST loaded successfully: " << engine_options.endgame_pst_path << std::endl;
        } else {
            std::cout << "info string Failed to load EndgamePST: " << engine_options.endgame_pst_path << std::endl;
        }
    }
    else if (name == "nnue_weight_file") {
        engine_options.nnue_weight_path = PROJECT_ROOT / fs::path("bin/nnue_wgts") / fs::path(value + ".bin");
        if(nnue.load(engine_options.nnue_weight_path)) {
            std::cout << "info string NNUE loaded successfully: " << engine_options.nnue_weight_path << std::endl;
        } else {
            std::cout << "info string Failed to load NNUE: " << engine_options.nnue_weight_path << std::endl;
        }
    }
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
    #ifdef DEV
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
            Move m = Move::NullMove();
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

void Engine::computeSearchTime(const SearchSettings& settings) {
    if (g_run_context.is_new_game) {
        g_gamelog.wtime = settings.wtime; 
        g_gamelog.btime = settings.btime; 
        g_gamelog.winc = settings.winc; 
        g_gamelog.binc = settings.binc; 
        g_gamelog.movestogo = settings.movestogo;
        g_gamelog.depth = settings.depth;
        g_gamelog.nodes = settings.nodes;
        g_gamelog.movetime = settings.movetime;
        g_run_context.is_new_game = false;
    }

    int movesToGo = settings.movestogo > 0 ? settings.movestogo : 20;

    // hard coded time/depth
    if (settings.movetime > 0 || settings.depth > 0) {
        limits = SearchLimits(
            settings.movetime - engine_options.MOVE_OVERHEAD_MS,
            settings.depth
        );
        return;
    }

    // game clock
    int side = game_board.is_white_move ? 0 : 1;
    int myTime = (side == 0 ? settings.wtime : settings.btime);
    int myInc  = (side == 0 ? settings.winc  : settings.binc);

    // compute function
    double aggressiveness = 1.0;
    int timeBudget = static_cast<int>(
        (static_cast<double>(myTime) / movesToGo + myInc / 2.0)
        * aggressiveness
    );

    timeBudget = std::max(10, timeBudget - engine_options.MOVE_OVERHEAD_MS);

    limits = SearchLimits(timeBudget);
}


void Engine::startSearch() {
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
    resetSearchStats();  
    
    // --- generate first moves once ---
    Move first_moves[MAX_MOVES];
    int count = movegen->generateMoves(game_board, false);
    std::copy_n(movegen->moves, count, first_moves);

    // --- run search ---
    computeSearchTime(settings);
    result = searcher->iterativeDeepening(first_moves, count, limits);
    bestMove = result.bestMove;
    bestEval = result.eval; 
    pv_line = result.best_line.line;
    //sendBestMove(bestMove);

    // finalize cumulative stats
    g_stats.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count();
    #ifdef DEV
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count();
        g_stats.game_ply = ply;
        //g_stats.nps = elapsed > 0 ? (1000.0 * g_stats.nodes / elapsed) : 0.0;
        g_stats.eval = result.eval;
        g_stats.move = result.bestMove;
        g_stats.principal_variation = result.best_line.line;

        //g_stats.ebf = pow(double(g_stats.nodes) / 1.0, 1.0 / g_stats.maxDepth); // rough estimate
        //g_stats.qratio = double(g_stats.qnodes) / g_stats.nodes;

        tracker.bestMoves.push_back(bestMove);
        tracker.evals.push_back(bestEval);

        logSearchStats(game_board.getBoardFEN());  // JSON written once at end
        logTimingStats(game_board.getBoardFEN());
    #endif 
}


void Engine::stopSearch() {
    limits.stopped = true;
    tracker.result = GameResult::ABORTED;
    tracker.reason = GameEndReason::NONE;
}

void Engine::ponderHit() {
    pondering = false;
}


// -------------------
// -- Communication --
// -------------------

void Engine::sendBestMove(Move best, bool eval, bool ponder) {
    if (eval) std::cout << "eval " << bestEval << " ";
    std::cout << "bestmove " << best.uci();
    if (ponder) std::cout << " ponder";
    std::cout << std::endl;

    game_board.MakeMove(best);
    game_over = checkGameEnd();

    if (mode == EngineMode::GAME) {trackGame();}

    //resetSearchStats();
    Logging::logUCIout(best.uci());
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

    if (Logging::log_dir == Logging::DEFAULT_LOG_DIR) Logging::setLogDir(Logging::project_root / "logs/game_logs");
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

// CONFIG

void Engine::apply_config_file(const fs::path& path) {
    auto kv = parse_kv(path);
    auto get = [&](const std::string& k) -> const std::string* {
        auto it = kv.find(k); return it != kv.end() ? &it->second : nullptr;
    };
    auto b = [](const std::string& v) { return v == "true" || v == "1"; };

    // SearchParams
    if (auto* v = get("delta_prune_threshold"))    searcher->params.DELTA_PRUNE_THRESHOLD               = std::stoi(*v);
    if (auto* v = get("see_prune_threshold"))      searcher->params.SEE_PRUNE_THRESHOLD      = std::stoi(*v);
    if (auto* v = get("aspiration_window"))        searcher->params.ASPIRATION_WINDOW        = std::stoi(*v);
    if (auto* v = get("aspiration_start_depth"))   searcher->params.ASPIRATION_START_DEPTH   = std::stoi(*v);
    if (auto* v = get("aspiration_depth_scale"))   searcher->params.ASPIRATION_DEPTH_SCALE  = std::stoi(*v);
    if (auto* v = get("aspiration_research_scale"))searcher->params.ASPIRATION_RESEARCH_SCALE = std::stof(*v);
    if (auto* v = get("draw_eval"))                searcher->params.DRAW_EVAL                = std::stoi(*v);
    if (auto* v = get("contempt"))                 searcher->params.CONTEMPT                 = std::stoi(*v);
    if (auto* v = get("r_nmp"))                    searcher->params.R_NMP                    = std::stoi(*v);
    if (auto* v = get("r_lmr_const"))              searcher->params.R_LMR_CONST              = std::stof(*v);
    if (auto* v = get("r_lmr_denom"))              searcher->params.R_LMR_DENOM              = std::stof(*v);
    if (auto* v = get("lmr_move_order_threshold")) searcher->params.LMR_MOVE_ORDER_THRESHOLD              = std::stof(*v);
    if (auto* v = get("lmr_depth_threshold"))      searcher->params.LMR_DEPTH_THRESHOLD              = std::stof(*v);

    // EngineOptions
    if (auto* v = get("move_overhead_ms")) engine_options.MOVE_OVERHEAD_MS= std::stoi(*v);
    if (auto* v = get("hash_size_mb"))     engine_options.HASH_SIZE_MB     = std::stoi(*v);
    if (auto* v = get("max_threads"))     engine_options.MAX_THREADS     = std::stoi(*v);
    if (auto* v = get("pondering"))        engine_options.PONDERING         = b(*v);
    if (auto* v = get("nnue_weight_path"))  engine_options.nnue_weight_path  = Logging::project_root / *v;
    if (auto* v = get("opening_book_path")) engine_options.opening_book_path = Logging::project_root / *v;
    if (auto* v = get("syzygy_path"))       engine_options.syzygy_path       = Logging::project_root / *v;  
}

void Engine::create_config_file(std::string config_name) {
    // ensure .ini extension
    if (config_name.size() < 4 || config_name.substr(config_name.size() - 4) != ".ini")
        config_name += ".ini";

    fs::path dir = fs::path(PROJECT_ROOT) / "bin/configs";
    fs::create_directories(dir);  // ensure directory exists
    fs::path out_path = dir / config_name;

    std::ofstream f(out_path);
    if (!f) {
        std::cerr << "Error: could not open " << out_path << " for writing.\n";
        return;
    }

    auto b = [](bool v) -> std::string { return v ? "true" : "false"; };

    // helper: write a relative path (strip PROJECT_ROOT prefix if present)
    auto rel = [](const fs::path& p) -> std::string {
        const fs::path root = fs::path(PROJECT_ROOT);
        std::error_code ec;
        fs::path r = fs::relative(p, root, ec);
        return (!ec && !r.empty()) ? r.generic_string() : p.generic_string();
    };

    f << "# =========================================================\n"
      << "# " << fs::path(config_name).stem().string() << " Engine Configuration\n"
      << "# =========================================================\n\n";

    // ── [search] ─────────────────────────────────────────────────────────────
    f << "[search]\n"
      << "delta_prune_threshold      = " << searcher->params.DELTA_PRUNE_THRESHOLD      << "\n"
      << "see_prune_threshold        = " << searcher->params.SEE_PRUNE_THRESHOLD        << "\n\n"
      << "aspiration_window          = " << searcher->params.ASPIRATION_WINDOW          << "\n"
      << "aspiration_start_depth     = " << searcher->params.ASPIRATION_START_DEPTH     << "\n"
      << "aspiration_depth_scale     = " << searcher->params.ASPIRATION_DEPTH_SCALE     << "\n"
      << "aspiration_research_scale  = " << searcher->params.ASPIRATION_RESEARCH_SCALE  << "\n\n"
      << "draw_eval                  = " << searcher->params.DRAW_EVAL                  << "\n"
      << "contempt                   = " << searcher->params.CONTEMPT                   << "\n\n"
      << "r_nmp                      = " << searcher->params.R_NMP                      << "\n"
      << "r_lmr_const                = " << searcher->params.R_LMR_CONST                << "\n"
      << "r_lmr_denom                = " << searcher->params.R_LMR_DENOM                << "\n\n"
      << "lmr_move_order_threshold   = " << searcher->params.LMR_MOVE_ORDER_THRESHOLD   << "\n"
      << "lmr_depth_threshold        = " << searcher->params.LMR_DEPTH_THRESHOLD        << "\n\n\n";

    // ── [engine] ──────────────────────────────────────────────────────────────
    f << "[engine]\n"
      << "move_overhead_ms  = " << engine_options.MOVE_OVERHEAD_MS   << "\n"
      << "max_threads       = " << engine_options.MAX_THREADS        << "\n"
      << "hash_size_mb      = " << engine_options.HASH_SIZE_MB       << "\n"
      << "pondering         = " << b(engine_options.PONDERING)       << "\n\n"
      << "nnue_weight_path  = " << fs::relative(engine_options.nnue_weight_path,  Logging::project_root).generic_string() << "\n"
      << "opening_book_path = " << fs::relative(engine_options.opening_book_path, Logging::project_root).generic_string() << "\n"
      << "syzygy_path       = " << fs::relative(engine_options.syzygy_path,       Logging::project_root).generic_string() << "\n";

    std::cout << "Config written to " << out_path << "\n";
}

// TESTING

uint64_t Engine::perft(int depth) {
    //ScopedTimer timer(T_PERFT);
    if (depth == 0) {return 1;}

    uint64_t nodes = 0;

    Move moves[MAX_MOVES];
    int count = movegen->generateMoves(search_board, false);
    std::copy_n(movegen->moves, count, moves);

    // we can skip the last make/unmake move by utilizing
    // that the num_moves at depth=1 is the perft value for that branch
    // its a decent speed up
    //if (depth == 1) return count; 

    Move* end = moves + count;
    for (Move* m = moves; m < end; ++m) {
        search_board.MakeMove(*m);
        nodes += perft(depth - 1);
        search_board.UnmakeMove(*m);
    }

    return nodes;
}

void Engine::perftPrint(int depth) {   
    if (depth == 0) {std::cout << "1" << std::endl;}

    unsigned long long nodes = 0;

    Move moves[MAX_MOVES];
    int count = movegen->generateMoves(search_board, false);
    std::copy_n(movegen->moves, count, moves);

    Move* end = moves + count;
    for (Move* m = moves; m < end; ++m) {
        search_board.MakeMove(*m);
        nodes += perft(depth - 1);
        search_board.UnmakeMove(*m);
    }

    std::cout << "Nodes searched: " << nodes << std::endl;
    logTimingStats(game_board.getBoardFEN());
}

void Engine::perftDivide(int depth) {
    int count = movegen->generateMoves(search_board, false);
    Move* moves = movegen->moves;

    unsigned long long total = 0;

    for (int i = 0; i < count; i++) {
        search_board.MakeMove(moves[i]);
        uint64_t n = perft(depth - 1);
        search_board.UnmakeMove(moves[i]);

        std::cout << moves[i].uci() << ": " << n << "\n";
        total += n;
    }

    std::cout << "Total: " << total << "\n";
}

void Engine::SEETest(int capture_square) {
    int count = movegen->generateMoves(search_board, true);

    for (int i = 0; i < count; i++) {
        Move m = movegen->moves[i];
        // check if capture (otherwise will incl checks and other qsearch stuff)
        if (m.TargetSquare() == capture_square) {
            int see = evaluator.SEE(search_board, m);
            std::cout << "SEE (" << m.uci() << ") = " << see << std::endl;
        }
    }
}

void Engine::staticEvalTest() {
    // Compute opening and endgame reports
    TaperedEvalReport report = evaluator.Evaluate(search_board, evaluator.PST_opening);

    // For clarity, also compute endgame separately
    TaperedEvalReport endgame_report = evaluator.Evaluate(search_board, evaluator.PST_endgame);

    // Merge opening/endgame into one tapered report
    report.opening = report.opening;
    report.endgame = endgame_report.opening; // endgame components
    int phase = evaluator.gamePhase(search_board);
    report.computeTapered(phase, evalWeights);

    std::cout << "=== Static Evaluation ===\n";
    std::cout << "Phase: " << phase << ") / 256\n";
    report.printDetailed(evalWeights);
}


void Engine::nnueEvalTest() {
    //nnue.build_accumulators(search_board);
    nnue.build_halfka_accumulators(search_board);

    int eval = nnue.evaluate(search_board.is_white_move);
    int simd_eval = nnue.eval_simd(search_board.is_white_move);

    if (!search_board.is_white_move) eval = -eval;

    std::cout << "=== NNUE Evaluation ===\n";
    std::cout << "NNUE Eval: " << eval << " centipawns\n";
    std::cout << "SIMD Eval: " << simd_eval << " centipawns\n";
}

void Engine::nnueSIMDTest() {
    std::ifstream file("../san-jacinto/bin/test_positions/perft.epd");

    if (!file) {
        std::cerr << "Could not open perft positions file\n";
        return;
    }

    std::string line;
    int position = 0;
    int errors = 0;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        // FEN is everything before the first ';'
        std::string fen = line.substr(0, line.find(';'));
        Board b = Board(fen);
        ++position;

        nnue.build_accumulators(b);
        int scalar = nnue.evaluate(b.is_white_move);
        int simd = nnue.eval_simd(b.is_white_move);

        if (scalar != simd) {
            ++errors;
            std::cerr << "\n[SIMD MISMATCH]"
                      << "\nPosition: " << position
                      << "\nFEN:      " << fen
                      << "\nScalar:   " << scalar
                      << "\nSIMD:     " << simd
                      << "\n";
        }
    }

    std::cout << "\nSIMD NNUE TEST COMPLETE\n"
              << "Positions: " << position
              << "\nErrors:    " << errors
              << "\n";
}

void Engine::moveOrderingTest(int depth) {
    std::cout << "=== Move Ordering Test ===\n";

    // Generate moves at root
    int moveCount = movegen->generateMoves(search_board, false);
    Move* moves = movegen->moves;

    std::vector<std::pair<Move, int>> scoredMoves;

    // Use empty TT and PV for testing at root
    Move ttMove;               // assume default-constructed = no move
    std::vector<Move> previousPV;

    for (int i = 0; i < moveCount; ++i) {
        int score = searcher->moveScore(
            moves[i], 
            search_board, 
            0,        // ply 0 at root
            ttMove, 
            previousPV
        );
        scoredMoves.emplace_back(moves[i], score);
    }

    // Sort by score descending
    std::sort(scoredMoves.begin(), scoredMoves.end(), [](auto& a, auto& b){
        return a.second > b.second;
    });

    // Print
    for (auto& [move, score] : scoredMoves) {
        std::cout << move.uci() << ": " << score << "\n";
    }
}

