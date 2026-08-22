#include <searcher.h>
#include <limits>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

// ============================================================================
// Construction / small helpers
// ============================================================================

void Searcher::updatePV(std::vector<Move>& pv, const Move& move, const std::vector<Move>& childPV) {
    pv.clear();
    pv.push_back(move);
    pv.insert(pv.end(), childPV.begin(), childPV.end());
}

// leaf node pruning
bool Searcher::shouldPrune(Move& move, int standPat, int alpha, int search_depth, int ply) {
    const int captured = board.getCapturedPiece(move.TargetSquare());
    const bool is_promo = move.IsPromotion();
    const bool in_check = board.is_in_check;

    
    if (captured != -1 
        && !is_promo
        && !in_check
    ) {
        // see pruning (bad captures ... excl promo and checks)
        int seeGain = eval.SEE(board, move);
        if (seeGain < params.SEE_PRUNE_THRESHOLD) {
            #ifdef DEV
                STATS_SEE_PRUNE(search_depth, ply);
            #endif
            return true;
        }

        // delta purning (quiet moves that wont raise alpha anyway)
        if (standPat + params.DELTA_PRUNE_THRESHOLD < alpha) {
            #ifdef DEV
                STATS_DELTA_PRUNE(search_depth, ply);
            #endif
            return true;
        }
    }

    return false;
}

// ============================================================================
// ROOT EVAL STORAGE
// ============================================================================

void Searcher::store_last_node_counts(const SearchResult& res) {
    // clear table
    std::fill(std::begin(node_count_table), std::end(node_count_table), 0);

    // write moves/evals from last depth
    for (int i = 0; i < res.root_count; ++i) {
        node_count_table[res.root_moves[i].move.Value()] = res.root_moves[i].nodes;
    }
}

inline int Searcher::get_node_count(Move m) const {
    return node_count_table[m.Value()];
}


// ============================================================================
// MOVE ORDERING
// ============================================================================

int Searcher::rootMoveScore(const Move& move, const Move& ttMove, const Move& pvMove) {
    int score = 0;

    if (Move::SameMove(move, pvMove))
        score += root_scores.PV_BASE;

    if (Move::SameMove(move, ttMove))
        score += root_scores.TT_BASE;

    score += get_node_count(move);

    if (move.IsPromotion())
        score += root_scores.PROMO_BASE;

    int see_score = eval.SEE(board, move);
    if (see_score > 0)
        score += root_scores.GOOD_CAP_BASE;
    else 
        score += root_scores.BAD_CAP_BASE;

    return score;
}

int Searcher::moveScore(const Move& move, const Board& boardRef,
                        int ply, const Move& ttMove, const std::vector<Move>& previousPV) {
    int seeScore;

    // TT + PV
    if (Move::SameMove(ttMove, move)) return move_scores.TT_BASE;
    if (ply < (int)previousPV.size() && Move::SameMove(move, previousPV[ply])) return move_scores.PV_BASE;

    // promotions
    if (move.IsPromotion()) {
        int promoBonus = 0;
        switch (move.PromotionPieceType()) {
            case queen:  promoBonus = 900; break;
            case knight: promoBonus = 300; break;
            case rook:   promoBonus = 100; break;
            case bishop: promoBonus = 100; break;
        }
        int captured = boardRef.getCapturedPiece(move.TargetSquare());
        seeScore = 0;
        if (captured != -1) {
            seeScore = eval.SEE(boardRef, move);
            return seeScore >= 0 ? move_scores.GOOD_CAP_BASE + seeScore + promoBonus
                                 : move_scores.BAD_CAP_BASE  + seeScore + promoBonus;
        }
        return move_scores.PROMO_BASE + promoBonus + seeScore;
    }

    // captures (scored via SEE or MVVLVA)
    int captured = boardRef.getCapturedPiece(move.TargetSquare());
    if (captured != -1) {
        // SEE
        seeScore = eval.SEE(boardRef, move);
        return seeScore >= 0 ? move_scores.GOOD_CAP_BASE + seeScore : move_scores.BAD_CAP_BASE + seeScore;
        // MVVLVA
        //int attacker = board.getMovedPiece(move.StartSquare());
        //int victim = board.getMovedPiece(move.TargetSquare());
        //return 100000 + 10 * victim - attacker;
    }

    // killer moves
    if (Move::SameMove(killerMoves[ply][0], move)) return move_scores.KILLER_BASE;
    if (Move::SameMove(killerMoves[ply][1], move)) return move_scores.KILLER_BASE - 1;

    // quiet moves (history heuristic)
    int piece = boardRef.getMovedPiece(move.StartSquare());
    int sidePiece = boardRef.is_white_move ? piece : piece + 6;

    return move_scores.QUIET_BASE + historyHeuristic[sidePiece][move.TargetSquare()] / 16;
}

void Searcher::orderedMoves(Move moves[MAX_MOVES], size_t count,
                            const Board& boardRef, int ply, 
                            const Move ttMove, const std::vector<Move>& previousPV) {
    
    #ifdef DEV
        ScopedTimer timer(T_SCORE_ORDER);
    #endif
    U64 hash = boardRef.zobrist_hash;

    // sorting
    std::pair<int, Move> scored[MAX_MOVES];
    for (size_t i = 0; i < count; ++i)
        scored[i] = {moveScore(moves[i], boardRef, ply, ttMove, previousPV), moves[i]};

    std::sort(scored, scored + count,
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < count; ++i) moves[i] = scored[i].second;
}

int Searcher::generateAndOrderMoves(Move moves[MAX_MOVES], int ply, const Move ttMove, const std::vector<Move>& previousPV) {
    int count = movegen.generateMoves(board, false); // the bool flag for captures-only or not — adapt if signature differs
    std::copy_n(movegen.moves, count, moves);

    orderedMoves(moves, static_cast<size_t>(count), board, ply, ttMove, previousPV);
    
    return count;
}

// ============================================================================
// QUIESCENCE SEARCH
// ============================================================================

int Searcher::quiescence(int alpha, int beta, PV& pv, SearchLimits& limits, int ply, int depth, int search_depth) {
    if (limits.out_of_time()) return alpha;

    // draw detection
    if (board.isThreefold() || board.currentGameState.fiftyMoveCounter >= 50) {
        //tt.store(board.zobrist_hash, depth, ply, 0, EXACT, Move::NullMove());
        return params.DRAW_EVAL;
    }


    // --- TT probe ---
    /*
    TTEntry* ttEntry = tt.probe(board.zobrist_hash);
    if (ttEntry && ttEntry->key == board.zobrist_hash) {
        STATS_TT_HIT(search_depth, ply);
        int ttScore = ttEntry->eval;

        if (ttEntry->flag == EXACT) return ttEntry->eval;
        else if (ttEntry->flag == UPPERBOUND && ttEntry->eval <= alpha) return ttScore;
        else if (ttEntry->flag == LOWERBOUND && ttEntry->eval >= beta)  return ttScore;

        //if (alpha >= beta) return ttEntry->eval;
    }
    */

    // Use incremental NNUE output (accumulators must be kept in sync)
    //boardallGameMoves.back().PrintMove();
    int standPat = nnue.eval_simd(board.is_white_move); //eval.taperedEval(board);
    if (standPat >= beta) return standPat;
    if (standPat > alpha) alpha = standPat;

    //return standPat;

    #ifdef DEV
        STATS_QNODE(search_depth, ply); 
    #else 
        g_stats.nodes++;
    #endif

    // generate only captures/promotions 
    int count = movegen.generateMoves(board, true);
    Move moves[MAX_MOVES];
    if (count == 0) {
        if (board.is_in_check) { return -MATE_SCORE + ply; }
        return standPat;
    }
    std::copy_n(movegen.moves, count, moves);

    int bestEval = standPat;
    Move bestMove = Move::NullMove();

    for (int i = 0; i < count; i++) {
        if (limits.out_of_time()) break;

        Move m = moves[i];
        if (shouldPrune(m, standPat, alpha, search_depth, ply)) continue;

        // Apply NNUE incremental update then board move
        nnue.on_make_move(board, m);
        board.MakeMove(m);

        int score; PV childPV;
        score = -quiescence(-beta, -alpha, childPV, limits, ply+1, depth, search_depth);

        // Undo board & NNUE (capture before must be reconstructed from states)
        nnue.on_unmake_move(board, m);
        board.UnmakeMove(m);

        if (score >= beta) { 
            #ifdef DEV
                STATS_FAILHIGH(search_depth, ply, i);
            #endif
            return score; 
        }
        if (score > bestEval) {
            bestEval = score;
            bestMove = m;
            alpha = std::max(alpha, score);
            pv.set(m, childPV);
        }
    }

    // store best capture or draw
    //tt.store(board.zobrist_hash, depth, ply, bestEval, EXACT, bestMove);
    //STATS_TT_STORE(search_depth, search_depth);

    return bestEval;
}

// ============================================================================
// NEGAMAX SEARCH
// ============================================================================

int Searcher::negamax(int depth, int alpha, int beta, PV& pv,
                      std::vector<Move>& previousPV, SearchLimits& limits, int ply, 
                      bool can_nmp) {
    int static_eval;
    int  f_prune = 0; // flag for futility pruning
    #ifdef DEV
        STATS_NODE(depth+ply, ply); // track node per depth
    #else 
        g_stats.nodes++;
    #endif 
    if (limits.out_of_time()) return alpha;

    // --- end of search conditions ---

    if (board.isThreefold() || board.currentGameState.fiftyMoveCounter >= 50) {
        //tt.store(board.zobrist_hash, depth, ply, 0, EXACT, Move::NullMove());
        return params.DRAW_EVAL;
    }

    if (depth == 0) {
        return quiescence(alpha, beta, pv, limits, ply, depth, ply);
    }

    // --- TT probe ---

    int alphaOrig = alpha;
    TTEntry* ttEntry = tt.probe(board.zobrist_hash);
    Move ttMove = Move::NullMove();

    if (ttEntry && ttEntry->key == board.zobrist_hash) {
        #ifdef DEV
            STATS_TT_HIT(depth+ply, ply);
        #endif
        // grab move for move ordering
        ttMove = ttEntry->bestMove;

        // return score if tt-move's search depth is >= than current search depth
        if (ttEntry->depth >= depth) {
            #ifdef DEV 
                STATS_TT_RETURN(depth+ply, ply);
            #endif 

            int ttScore = ttEntry->eval;
            if (ttEntry->flag == EXACT) return ttScore;
            else if (ttEntry->flag == UPPERBOUND && ttScore <= alpha) return ttScore;
            else if (ttEntry->flag == LOWERBOUND && ttScore >= beta)  return ttScore;
        }
    } 

    // --- internal iterative deepening ---
    // no usable TT move, PV node, enough depth to bother
    // run reduced search to replace TT move
    /*
    if (ttMove.IsNull() && depth >= params.IID_DEPTH_THRESHOLD) {
        #ifdef DEV
            STATS_IID(depth+ply, ply);
        #endif 
        PV iidPV;

        // no flip in negamax func call cause we are not making a move yet (+ dont save eval)
        // currently, scaled reduction
        negamax(depth * params.R_IID, alpha, beta, iidPV, previousPV, limits, ply, can_nmp);
        
        ttEntry = tt.probe(board.zobrist_hash);
        if (ttEntry && ttEntry->key == board.zobrist_hash) {
            ttMove = ttEntry->bestMove;
        }
    }
    */

    // ---------------------------------------------------
    // reverse futility pruning (aka static null move pruning)
    // ---------------------------------------------------
    // at pre-frontier nodes (2nd-parent of leaf node)
    // skip the move loop if static eval > beta+margin
    // similar to NMP but no search, just a fail-high
    if (
        depth < 3
        && !(beta-alpha>1)
        && !board.is_in_check
        && beta < MATE_SCORE - 100
    ) {
        static_eval = nnue.eval_simd(board.is_white_move);
        int eval_margin = depth * params.REVERSE_FUTILITY_PRUNE_THRESHOLD;

        if (static_eval - eval_margin >= beta) {
            #ifdef DEV 
                STATS_RFP(depth+ply, ply);
            #endif 
            // padded score prevents overestimating unexpected tactical replies
            return static_eval - eval_margin; 
        }
    }

    // -----------------------------
    // Null Move Pruning
    // -----------------------------
    // making a move is almost always better than passing
    // so we pass the move (null move) and perform a null-window search with reduced depth around beta
    // if the null move fails high then we can assume the best move will also fail high
    // certain position restrictions must be used for the assumptions to hold
    if ( 
        // depth condition
        (depth - params.R_NMP > 0)
        &&
        // 2x null-moves not allowed
        can_nmp
        && 
        // board conditions (not in-check .. not pawn-endgame)
        !(
            board.is_in_check 
            || board.pawn_endgame 
        )
        && 
        // static eval > beta (use cached if available)
        ((static_eval ? static_eval : nnue.eval_simd(board.is_white_move)
         ) > beta)
    ) {
        #ifdef DEV
            ScopedTimer timer(T_NMP_SEARCH);
            STATS_NMP(depth+ply, ply);
        #endif
        PV emptyPV;

        // null moves just change the side to move (and last-move cache)
        board.MakeNullMove();
        int null_score = -negamax(depth - params.R_NMP, -beta, -(beta - 1), emptyPV, previousPV, limits, ply + 1, false);
        board.UnmakeNullMove();

        // null window around beta so if if null move fails high 
        // then so should the real move with a full window search
        if (null_score >= beta) {
            #ifdef DEV
                STATS_NMP_FAILHIGH(depth+ply, ply);
            #endif
            return null_score;
        }
    }

    // --- search ---

    Move moves[MAX_MOVES];
    int count = generateAndOrderMoves(moves, ply, ttMove, previousPV);
    if (count == 0) return board.is_in_check ? -(MATE_SCORE - ply) : 0;

    int bestEval = -MATE_SCORE;
    Move bestMove = Move::NullMove();

    int _lmr_R = 0;

    Move m; int score; PV childPV; PV emptyPV;
    bool gives_check, is_capture;

    // current board state info
    bool in_check = board.is_in_check;
    bool is_pawn_endgame = board.pawn_endgame;
    bool was_capture = board.currentGameState.capturedPieceType != -1;

    /*
    FUTILITY PRUNING  
        not in check
        not searching for a checkmate 
        eval is below (alpha - margin)
    it might  mean that searching non-tactical moves at  low depths    
    is futile, so we set a flag allowing this pruning               
     */
    if (depth <= 3
        && !(alpha-beta > 1)
        && !in_check
        && abs(alpha) < 9000
        && ((static_eval ? static_eval : nnue.eval_simd(board.is_white_move)) 
                + (depth * params.FUTILITY_PRUNE_MARGIN) <= alpha)
    )
        f_prune = 1;
    

    // --- move loop ---

    for (int i = 0; i < count; i++) {
        if (limits.out_of_time()) break;

        m = moves[i];

        // future board state info
        gives_check = board.givesCheck(m);
        is_capture = board.getCapturedPiece(m.TargetSquare()) != -1;
        
        score = 0; childPV = {}; emptyPV = {};

        // ------------------------------------
        // Futility Pruning (aka late move pruning)
        // ------------------------------------
        if (f_prune                          // Applied at leaf and pre-leaf 
            && i > params.FUTILITY_PRUNE_MOVE_THRESHOLD  // Only applied to late moves
            && !is_capture                      // Never prune captures
            && !m.IsPromotion()                 // Never prune promotions
            && !gives_check             // Never prune moves that give check
        ) {
            #ifdef DEV
                STATS_FUTILITY_PRUNE(depth+ply, ply);
            #endif
            continue; // Skip searching this unpromising quiet move!
        }

        // -----------------------------
        // Late Move Reduction
        // -----------------------------
        // if a move is late in the move ordering and non-tactical then we can reduce its search depth
        // if the reduced search returns > alpha then we can re-search it with full depth
        // since we expect it not to raise alpha due to our good move ordering we expect these researches to be rare
        // integration with PVS: use PVS+LMR concurrently
        //    if the move returns > alpha then do full-search (depth+window)
        if (
            // positional conditions apply
            !is_capture
            && !m.IsPromotion()
            && !in_check
        ) {
            // obsidian log formula
            _lmr_R = R_lmr(depth, i);
        } else {
            _lmr_R = 0;
        }

        //score = -negamax(depth - 1 - _lmr_R, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
        //if (_lmr_R > 0 && score > alpha) {
        //    childPV = {}; // dont let teh reduced-search line leak into the full-depth result
        //    score = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply+1, true);
        //}

        // Apply NNUE/update & board
        nnue.on_make_move(board, m);
        board.MakeMove(m);

        // -----------------------------
        // principal variation search 
        // -----------------------------
        // run a null-window search around alpha
        // after generating a PV move (and thus alpha=eval) from a full-window search
        // if a move fails high then we can re-search it with a full window 
        // else it fails low and is not going to be a better move than what has been found
        // null window searches are cheap and so the re-searches are worth the speedup
        if (i == 0) {
            score = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
        } else {
            // null-window search
            // lmr =0 OR >0
            score = -negamax(depth - 1 - _lmr_R, -(alpha+1), -alpha, emptyPV, previousPV, limits, ply + 1, true);

            // if lmr_r > 0 then re-search with null-window at full depth
            // if score > alpha from this re-search,
            // then full-window gets triggered below with lmr_R == 0
            // if re-search not triggered then neither will trigger as first null-window failed low
            if (_lmr_R > 0 && score > alpha) {
                #ifdef DEV
                    STATS_PVS_RESEARCH(depth+ply, ply, 0);
                #endif
                emptyPV = {};
                score = -negamax(depth - 1, -(alpha+1), -alpha, emptyPV, previousPV, limits, ply + 1, true);
            }

            // if lmr_R == 0 then search with null window
            // if score > alpha then research with full window
            if (score > alpha && beta-alpha > 1) {
                // beta-alpha > 1 prevents reduandant research of non-PV nodes
                // if beta = alpha+1, then score > alpha --> score >= beta --> no need for re-search
                #ifdef DEV
                    STATS_PVS_RESEARCH(depth+ply, ply, 1);
                #endif
                emptyPV = {};
                score = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
            }
        }


        nnue.on_unmake_move(board, m);
        board.UnmakeMove(m);

        if (score > bestEval) {
            bestEval = score;
            bestMove = m;
            pv.set(m, childPV);
        }

        alpha = std::max(alpha, bestEval);
        if (alpha >= beta) {
            #ifdef DEV
                STATS_FAILHIGH(depth+ply, ply, i);
            #endif

            if (!Move::SameMove(killerMoves[ply][0], m) && !m.IsPromotion() &&
                !(1ULL << m.TargetSquare() & board.colorBitboards[1 - board.move_color])) {
                killerMoves[ply][1] = killerMoves[ply][0];
                killerMoves[ply][0] = m;
            }
            int piece = board.getMovedPiece(m.StartSquare());
            historyHeuristic[board.is_white_move ? piece : piece + 6][m.TargetSquare()] += depth * depth;
            break;
        }
    }

    // --- tt-store ---

    BoundType flag = EXACT;
    if (bestEval <= alphaOrig) { 
        flag = UPPERBOUND; 
        #ifdef DEV
            STATS_FAILLOW(depth+ply, ply); 
        #endif
    }
    else if (bestEval >= beta) { 
        flag = LOWERBOUND; 
    }
    tt.store(board.zobrist_hash, depth, ply, bestEval, flag, bestMove);
    //STATS_TT_STORE(depth+ply, ply);

    return bestEval;
}

// ============================================================================
// ROOT SEARCH
// ============================================================================

SearchResult Searcher::search(RootMove root_moves[MAX_MOVES], int count, int depth, SearchLimits& limits, std::vector<Move>& previousPV, int previousEval) {
    int eval;
    int ply = 0; // root moves are depth=0, ply+1 in arg call makes made moves depth=1
    SearchResult result;
    int _lmr_R = 0;
    int nodes_before = 0;
    bool exact = false;

    // current board state info
    bool in_check = board.is_in_check;
    bool is_pawn_endgame = board.pawn_endgame;
    bool was_capture = board.currentGameState.capturedPieceType != -1;
    bool is_capture;

    // --- aspiration search ---
    
    int delta = params.ASPIRATION_WINDOW;
    int alpha = -MATE_SCORE;
    int beta = MATE_SCORE;
    
    if (depth >= params.ASPIRATION_START_DEPTH) {
        delta = std::max(delta, params.ASPIRATION_WINDOW + depth * params.ASPIRATION_DEPTH_SCALE);
        alpha = previousEval - delta;
        beta  = previousEval + delta;
    }

    #ifdef DEV
        STATS_NODE(depth, ply);  
    #else 
        g_stats.nodes++;
    #endif   

    // root move loop for both aspiration + regular search
    while (true) {
        SearchResult iter_result;
        int aspAlpha = alpha;
        int aspBeta = beta;

        // explore (our) root moves
        for (int i = 0; i < count; ++i) {
            if (limits.out_of_time()) break;
            Move m = root_moves[i].move;
            if (Move::SameMove(m, Move::NullMove())) continue;

            // root move stat tracking
            exact = false;
            nodes_before = g_stats.nodes;
            #ifdef DEV
                auto move_start = std::chrono::steady_clock::now();
            #endif

            nnue.on_make_move(board, m);
            board.MakeMove(m);
  
            PV childPV; PV emptyPV; // must be declared per root_move

            // --- LMR ---
            is_capture = board.getCapturedPiece(m.TargetSquare()) != -1;

            if (
                // positional conditions apply
                !is_capture
                && !m.IsPromotion()
                && !in_check
            ) {
                // obsidian log formula
                _lmr_R = R_lmr(depth, i);
            } else {
                _lmr_R = 0;
            }
            
            //eval = -negamax(depth - 1 - _lmr_R, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
            //if (_lmr_R > 0 && eval > alpha) { // research at full depth if move raises alpha
            //    childPV = {};   // don't let the reduced-search line leak into the full-depth result
            //    eval = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply+1, true);
            //}

            // --- PVS ---
            if (i == 0) {
                eval = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
                exact = true;
            } else {
                // null-window search
                // lmr =0 OR >0
                eval = -negamax(depth - 1 - _lmr_R, -(alpha+1), -alpha, emptyPV, previousPV, limits, ply + 1, true);

                // if lmr_r > 0 then re-search with null-window at full depth
                // if score > alpha from this re-search,
                // then full-window gets triggered below with lmr_R == 0
                // if re-search not triggered then neither will trigger as first null-window failed low
                if (_lmr_R > 0 && eval > alpha) {
                    #ifdef DEV
                        STATS_PVS_RESEARCH(depth+ply, ply, 0); // lmr
                    #endif
                    emptyPV = {};
                    eval = -negamax(depth - 1, -(alpha+1), -alpha, emptyPV, previousPV, limits, ply + 1, true);
                }

                // if lmr_R == 0 then search with null window
                // if score > alpha then research with full window
                if (eval > alpha && beta-alpha > 1) {
                    // beta-alpha > 1 prevents reduandant research of non-PV nodes
                    // if beta = alpha+1, then score > alpha --> score >= beta --> no need for re-search
                    #ifdef DEV
                        STATS_PVS_RESEARCH(depth+ply, ply, 2); // root full
                    #endif
                    emptyPV = {};
                    eval = -negamax(depth - 1, -beta, -alpha, childPV, previousPV, limits, ply + 1, true);
                    exact = true;
                }
            }
            
            nnue.on_unmake_move(board, m);
            board.UnmakeMove(m);

            #ifdef DEV
                auto move_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - move_start).count();
                iter_result.root_moves[i].time_ms = move_ms;
            #endif

            // time out is propagated up the tree, so eval and move cannot be trusted
            if (limits.out_of_time() && i > 0) break;

            iter_result.root_moves[i].move = m;
            iter_result.root_moves[i].eval = eval;
            iter_result.root_moves[i].exact = exact;
            iter_result.root_moves[i].nodes = g_stats.nodes - nodes_before;
            iter_result.root_count++;

            alpha = std::max(alpha, eval);
            if (i == 0 || eval > iter_result.eval) {
                iter_result.eval = eval;
                iter_result.bestMove = m;
                iter_result.best_line.set(m, childPV); 
            }
        }

        // aspiration check
        if (!limits.should_stop(depth) && (depth >= params.ASPIRATION_START_DEPTH)) {
            if (iter_result.eval <= aspAlpha) {
                #ifdef DEV
                    STATS_ASPIRATION_FAILLOW(depth);
                #endif
                alpha = aspAlpha - delta;
                beta = aspBeta;
                delta *= params.ASPIRATION_RESEARCH_SCALE;
                continue;
            } else if (iter_result.eval >= aspBeta) {
                #ifdef DEV
                    STATS_ASPIRATION_FAILHIGH(depth);
                #endif
                beta = aspBeta + delta;
                alpha = aspAlpha;
                delta *= params.ASPIRATION_RESEARCH_SCALE;
                continue;
            }
        } 

        result = iter_result;
        break;
    }

    return result;
}

SearchResult Searcher::iterativeDeepening(Move first_moves[MAX_MOVES], int move_count, SearchLimits limits) {
    SearchResult last_result;
    SearchResult result;

    // nodes from prior iteration INIT
    std::fill(std::begin(node_count_table), std::end(node_count_table), 0);

    int depth = 1;
    Move prevBest = Move::NullMove();

    // Build NNUE accumulators for root position
    nnue.build_accumulators(board);

    // --- iterative deepening loop ---
    while (!limits.should_stop(depth)) {
        auto depth_start = std::chrono::steady_clock::now();
        g_stats.max_depth = depth;

        // --- move ordering ---
        if (depth > 1) {
            Move pvMove = last_result.bestMove;
            RootMove* root_moves = last_result.root_moves;

            // Force PV move to the front.
            auto pv_it = std::find_if(root_moves, root_moves + move_count,
                                    [&](const RootMove& rm) { return Move::SameMove(rm.move, pvMove); });
            if (pv_it != root_moves + move_count) {
                std::iter_swap(root_moves, pv_it);
            }

            // Sort the rest:
            // - exact scores (PV re-searches / fail-highs) rank by score
            // - fail-low bounds (unresolved) fall back to node count as effort proxy
            std::sort(root_moves + 1, root_moves + move_count,
                [](const RootMove& a, const RootMove& b) {
                    if (a.exact != b.exact)
                        return a.exact > b.exact;
                    if (a.exact && b.exact && a.eval != b.eval)
                        return a.eval > b.eval;
                    return a.nodes > b.nodes;
                });
        } else {
            // first search: order like typical mid-tree ordering   
            // tt probe
            TTEntry* ttEntry = tt.probe(board.zobrist_hash);
            Move ttMove = Move::NullMove();

            if (ttEntry && ttEntry->key == board.zobrist_hash) {
                #ifdef DEV
                    STATS_TT_HIT(depth, 0);
                #endif
                ttMove = ttEntry->bestMove;
            }
            orderedMoves(first_moves, move_count, board, 0, ttMove, {});

            for (int i = 0; i < move_count; ++i) {
                last_result.root_moves[i].move = first_moves[i];
            }
        }

        // --- search ---
        result = search(last_result.root_moves, move_count, depth, limits,
                                     last_result.best_line.line, last_result.eval);


        // --- store results ---
        if (!Move::SameMove(result.bestMove, Move::NullMove())) {
            last_result = result;
            store_last_node_counts(result);
            g_stats.max_completed_depth = depth; 
        }

        // --- logging --- 
        #ifdef DEV
            auto depth_end = std::chrono::steady_clock::now();
            //g_stats.max_completed_depth = depth;
            g_stats.it_depth_eval[depth] = result.eval;
            g_stats.it_depth_move[depth] = result.bestMove;
            g_stats.it_depth_time_ms[depth] = std::chrono::duration<double, std::milli>(depth_end - depth_start).count();
            //if (Move::SameMove(bestMove, prevBest)) g_stats.bestmoveStable++;

            // log per-root-move timing data
            logRootMoves(result, depth);
        #endif

        if (std::abs(result.eval) >= MATE_SCORE - 10) break;
        depth++;
        // if only 1 legal move, perform depth 2 search then play move
        // depth 2 to get fair eval with recaptures, etc. (very fast)
        // but quit early since we know what were going to play anyway
        // still want fair eval for continuation metrics, etc.
        if ((move_count == 1) && (depth == 3)) break;
    }

    return last_result;
}

// -----------------------
// LMR reduction formula
// -----------------------
// log formula: constant + [log(depth) * log(move_order)] / denominator
int Searcher::R_lmr(int depth, int move_order) {
    if (move_order <= params.LMR_MOVE_ORDER_THRESHOLD) return 0; // no reduction for first 3 moves
    if (depth <= params.LMR_DEPTH_THRESHOLD) return 0;      // no reduction for shallow depths

    // logarithmic reduction formula
    float log_depth = std::log(static_cast<float>(depth));
    float log_order = std::log(static_cast<float>(move_order));
    // unlikely to exceed 4 without heavy parameter tuning (or very deep search)
    int reduction = static_cast<int>(params.R_LMR_CONST + (log_depth * log_order) / params.R_LMR_DENOM);
    
    return std::min(reduction, depth-2);
}