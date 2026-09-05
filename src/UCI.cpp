#include <UCI.h>
#include <vector>
#include <string>
#include <iostream>
#include <stats.h>


namespace fs = std::filesystem;

UCI::UCI(Engine& eng) {
    engine=&eng;
}  

void UCI::loop() {
    std::string line;
    bool isRunning = true;

    while (isRunning && std::getline(std::cin, line)) {
        if (line.empty()) continue;

        handleCommand(line);

        if (line == "quit") {
            isRunning = false;
            exit(0);
        }
    }
}

void UCI::handleCommand(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    Logging::logUCIin(line); // cmd -> engine

    // standard commands
    if ((token == "uci") || (token == "uci_dev")) {
            std::cout << "id name tomahawk\n";
            std::cout << "id version " << ENGINE_ID << "\n";
            std::cout << "id author max oleksa\n";

            // engine options
            std::cout << "option name Move Overhead type spin default " << engine->engine_options.MOVE_OVERHEAD_MS << " min 0 max 1000\n";
            std::cout << "option name Threads type spin default " << engine->engine_options.MAX_THREADS<< " min 1 max 1\n";
            std::cout << "option name Hash type spin default " << engine->engine_options.HASH_SIZE_MB<< " min 1 max 1024\n";
            std::cout << "option name Ponder type check default " << engine->engine_options.PONDERING << "\n";
            if (token == "uci_dev") {std::cout << std::endl;} // line break
            // Required for lichess
            std::cout << "option name UCI_ShowWDL type check default " << engine->engine_options.UCI_SHOW_WDL << "\n";
            
            if (token == "uci_dev") {
                std::cout << std::endl; // line break
                // books
                std::cout << "option name nnue_weight_file type string default " << engine->engine_options.nnue_weight_path << "\n";
                std::cout << "option name opening_book type string default " << engine->engine_options.opening_book_path << "\n";
                std::cout << "option name syzygy type string default " << engine->engine_options.syzygy_path << "\n\n";
                // search params
                std::cout << "option name delta_prune_threshold type spin default " << engine->searcher->params.DELTA_PRUNE_THRESHOLD << " min 0 max 10000\n";
                std::cout << "option name see_prune_threshold type spin default " << engine->searcher->params.SEE_PRUNE_THRESHOLD << " min -1000 max 1000\n";
                //      aspiration
                std::cout << "option name aspiration_start_depth type spin default " << engine->searcher->params.ASPIRATION_START_DEPTH << " min 1 max 32\n";
                std::cout << "option name aspiration_window type spin default " << engine->searcher->params.ASPIRATION_WINDOW << " min 1 max 1000\n";
                std::cout << "option name aspiration_depth_scale type spin default " << engine->searcher->params.ASPIRATION_DEPTH_SCALE << " min 0 max 100\n";
                std::cout << "option name aspiration_research_scale type spin default " << engine->searcher->params.ASPIRATION_RESEARCH_SCALE << " min 0 max 1000\n";
                //       draw/contempt
                std::cout << "option name draw_eval type spin default " << engine->searcher->params.DRAW_EVAL << " min 0 max 1000\n";
                std::cout << "option name contempt type spin default " << engine->searcher->params.CONTEMPT << " min 0 max 10\n";
                //      reductions
                std::cout << "option name r_nmp type spin default " << engine->searcher->params.R_NMP << " min 0 max 32\n";
                //          lmr
                std::cout << "option name lmr_move_order_threshold type spin default " << engine->searcher->params.LMR_MOVE_ORDER_THRESHOLD << " min 0 max 256\n";
                std::cout << "option name lmr_depth_threshold type spin default " << engine->searcher->params.LMR_DEPTH_THRESHOLD << " min 0 max 32\n";
                std::cout << "option name r_lmr_const type spin default " << static_cast<int>(100*engine->searcher->params.R_LMR_CONST) << " min 0 max 200\n";
                std::cout << "option name r_lmr_denom type spin default " << static_cast<int>(100*engine->searcher->params.R_LMR_DENOM) << " min 100 max 500\n";
                // line break
                std::cout << std::endl; 
            }

            // logging
            std::cout << "option name log_dir type string default " << Logging::log_dir << "\n";
            std::cout << "option name uci_logging type check default " << Logging::track_uci << "\n";

            std::cout << "uciok\n";
    }
    else if (token == "isready") {
        std::cout << "readyok\n";
    }
    else if (token == "setoption") {
        handleSetOption(iss);
    }
    else if (token == "ucinewgame") {
        engine->newGame();
    }
    else if (token == "position") { // start all search/tests/everything after set_position
        handlePosition(iss);
    }
    else if (token == "go") {
        handleGo(iss);
    }
    else if (token == "stop") {
        engine->stopSearch();
        std::string best = engine->bestMove.uci();
        std::cout << "bestmove " << best << "\n";
    }
    else if (token == "quit") {
        engine->stopSearch();
    }
    else if (token == "config") { // see config options and apply
        fs::path dir = fs::path(PROJECT_ROOT) / "bin/configs";

        // print config options
        std::cout << "Available config files:\n\n";

        int idx = 1;
        std::vector<fs::path> configs;

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".ini") continue;

            configs.push_back(entry.path());

            std::cout << "  [" << idx++ << "] "
                    << entry.path().stem().string()
                    << "\n";
        }

        if (configs.empty()) {
            std::cout << "  (no config files found)\n";
            return;
        }

        // user select config file
        std::cout << "\nSelect config number: ";

        int choice;
        std::cin >> choice;

        if (choice < 1 || choice > (int)configs.size()) {
            std::cout << "Invalid selection.\n";
            return;
        }

        // apply config
        engine->apply_config_file(configs[choice - 1]);

        // apply specifics (e.g. tt.resize)
        engine->tt.resize(engine->engine_options.HASH_SIZE_MB);
    }
    else if (token == "apply_config") { // apply config option (without seeing options)
        std::string name; 
        iss >> name;

        fs::path path = fs::path(PROJECT_ROOT) / "bin/configs" / name;
        if (path.extension() != ".ini") path += ".ini";
        
        engine->apply_config_file(path);

        // apply specifics (e.g. tt.resize)
        engine->tt.resize(engine->engine_options.HASH_SIZE_MB);
    }
    else if (token == "save_config") { // save current config
        std::string name;
        iss >> name;

        engine->create_config_file(name);
    }
    else if (token == "ponderhit") {
        //
    }
    // custom commands
    else if (token == "static_eval") {
        // static or tapered evaluation test
        engine->staticEvalTest();
    }
    else if (token == "nnue_eval") {
        engine->nnueEvalTest();
    }
#ifdef _WIN32
    else if (token == "nnue_test") {
        engine->nnueSIMDTest();
    }
#endif
    else if (token == "perft") {
        int depth;
        iss >> depth;
        engine->perftPrint(depth);
    }
    else if (token == "see") {
        std::string target_sq;
        if (iss >> target_sq) {
            engine->SEETest(algebraic_to_square(target_sq));
        }
    }
    else if (token == "dumpzobrist") {
        std::cout << "\nCurrent Hash: 0x" << std::hex 
          << engine->game_board.zobrist_hash << "\n";
        std::cout << "\nLast 10 hashes" << std::endl;
        for (int i = 0; i < std::min(10,(int)engine->game_board.zobrist_history.size()); i++) {
            const auto& elem = engine->game_board.zobrist_history[engine->game_board.zobrist_history.size() - 1 - i];
            std::cout << elem << "\n";
        }
        std::cout << "\n--- Rep Stack ---\n";

        int sum = 0;
        for (const auto& entry : engine->game_board.hash_history) {
            uint64_t hash = entry.first;
            int count = entry.second;
            std::cout << "Hash: 0x" << std::hex << hash 
                    << "  Count: " << std::dec << count << "\n";
            sum += count;
        }
        std::cout << "--- End of Hash History ---\n";
        std::cout << "allgamemoves.size: " << engine->game_board.allGameMoves.size() << "\tzobrist_history.sum: " << sum << "\tzobrist_vec.size: " << engine->search_board.zobrist_history.size() << std::endl;
    }
    else if (token == "dump_tt") {
        #ifdef DEV
            std::cout << "\n(Last search) Stores:     " << g_stats.tt_stores << std::endl;
            std::cout << "\n(Last search) Hits:       " << g_stats.tt_hits << std::endl;
            std::cout << "\n(Last search) Overwrites: " << g_stats.tt_overwritten << std::endl;
            std::cout << "\n(Full game)   Fill %:     " << round_to_n_decimals(100*engine->tt.fillRatio(),2) << " %" << std::endl;
        #else 
            std::cout << "\nPROD engine ... no stats tracking" << std::endl;
        #endif
    }
    else if (token == "dumpstats") {
        dumpSearchStats(); // print collected stats to console for last search
    }
    #ifdef DEV
        else if (token == "dumpmoves") {
            dumpRootMoves(engine->result);
        }
    #endif
    else if (token == "clear_tt") {
        std::cout << "\nClearing ... " << engine->tt.filledCount << " / " << engine->tt.entriesCount << std::endl;
        engine->tt.clear();
        std::cout << "\nCleared!\n" << std::endl;
    }
    else if (token == "bench") {
        //engine->bench(depth);  // implement bench mode
    }
    else if (token == "speedtest") {
        //engine->speedTest(); // implement speed test
    }
    else if (token == "flip") {
        engine->search_board.is_white_move = !engine->search_board.is_white_move;
    }
    else if (token == "print_board") {
        engine->game_board.print_board();
    }
    // else ignore unknown commands or add more handlers
}

// --- handle "setoption" ---
void UCI::handleSetOption(std::istringstream& iss) {
    std::string token;
    std::string name;
    std::string value;

    // First token should be "name"
    iss >> token;
    if (token != "name") {
        return; // invalid format
    }

    // Read option name until "value" or end
    while (iss >> token) {
        if (token == "value") {
            // Everything after "value" is the value string
            std::getline(iss, value);
            // Trim leading whitespace
            size_t first = value.find_first_not_of(" \t");
            if (first != std::string::npos) {
                value = value.substr(first);
            } else {
                value.clear();
            }
            break;
        } else {
            if (!name.empty()) name += " ";
            name += token;
        }
    }

    // Pass parsed option to engine
    engine->setOption(name, value);
}


void UCI::handlePosition(std::istringstream& iss)
{
    std::string token;
    iss >> token;

    std::string fen;
    std::vector<std::string> moves;

    if (token == "startpos") {
        fen = "startpos";
    }
    else if (token == "fen") {
        std::string part;
        for (int i = 0; i < 6; ++i) {
            iss >> part;
            fen += part + " ";
        }
        fen.pop_back();
    }

    if (iss >> token && token == "moves") {
        std::string m;
        while (iss >> m)
            moves.push_back(m);
    }

    engine->setPosition(fen, moves);
    //engine->nnue.debug_simd(engine->game_board);
}

void UCI::handleGo(std::istringstream& iss) {
    // Reset defaults
    sendEval = false;
    wtime = btime = 0;
    winc = binc = 0;
    movetime = -1;
    depth = -1;

    std::string token;
    int movestogo = 0;
    int nodes = -1;
    bool infinite = false;

    while (iss >> token) {
        if (token == "eval") {
            sendEval = true;            // mark that we want evaluation only
        }
        else if (token == "wtime") iss >> wtime;
        else if (token == "btime") iss >> btime;
        else if (token == "winc") iss >> winc;
        else if (token == "binc") iss >> binc;
        else if (token == "movestogo") iss >> movestogo;
        else if (token == "depth") iss >> depth;
        else if (token == "nodes") iss >> nodes;
        else if (token == "movetime") iss >> movetime;
        else if (token == "infinite") infinite = true;
    }

    // Apply settings to engine
    engine->settings.wtime = wtime;
    engine->settings.btime = btime;
    engine->settings.winc = winc;
    engine->settings.binc = binc;
    engine->settings.movestogo = movestogo;
    engine->settings.depth = depth;
    engine->settings.nodes = nodes;
    engine->settings.movetime = movetime;
    engine->settings.infinite = infinite;

    // Otherwise start a normal search
    engine->trackGame();
    engine->startSearch();
    engine->sendBestMove(engine->bestMove, sendEval);
}

