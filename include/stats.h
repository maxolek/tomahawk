// ---------------------
// -- Data Collection --
// ---------------------

#ifndef STATS_H
#define STATS_H

#include "helpers.h"
#include "move.h"
#include "session.h"
#include "logging.h"
#include "search_result.h"
#include <vector>
#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

constexpr int STATS_MAX_PLY = 64;
constexpr int STATS_MAX_ITER_DEPTH = 32;
constexpr int STATS_FH_BUCKETS = 6; // move index buckets: 0, 1, 2, 3, 4-7, 8+
constexpr int STATS_PVS_RESEARCH_TYPES = 3; // [lmr w/ full depth, full, root full]

#define STATS_BOUNDS_CHECK(it_d, ply) \
    assert((it_d) >= 0 && (it_d) <= STATS_MAX_ITER_DEPTH && \
            (ply) >= 0 && (ply) <= STATS_MAX_PLY)

#define STATS_BOUNDS_CHECK_1D(d) \
    assert((d) >= 0 && (d) <= STATS_MAX_ITER_DEPTH)

// -------------------------
//      search stats
// -------------------------

// 3 layer structure
//  1. full search summary 
//      total nodes, total ttHits, final eval/move
//  2. iterative deepening breakdown
//      nodes, etc. on it_d = 4,5,6
//  3. tree ply breakdown
//      nodes, ttHits, etc. on tree ply (depth in search tree)
//      multiple counts lower plies as it_deep will lead to searching ply < it_deep multiple times
//      Qnodes are assigned to the search_depth ply 
//          (e.g. if Qsearch is triggered at ply=6, all Q-stats are assigned to ply=6 even though Qsearch technically increases ply)
struct SearchStats {
    //  results / env
    int game_ply = 1;
    int eval = 0;
    Move move = Move::NullMove();
    std::vector<Move> principal_variation;
    uint64_t time_ms = 0;

    // full summary
    int max_completed_depth = 0;
    int max_depth = 0;
    int max_qdepth = 0;
    uint64_t nodes = 0;
    #ifdef DEV
    uint64_t qnodes = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_returns = 0;                              
    uint64_t tt_stores = 0;
    double tt_fill_ratio = 0.0; // snapshot
    uint64_t tt_overwritten = 0;
    //uint64_t iid = 0; // internal iterative deepening
    uint64_t fail_highs = 0;
    uint64_t fail_lows = 0;
    // fail-high move index histogram: buckets [0, 1, 2, 3, 4-7, 8+]
    std::array<uint64_t, STATS_FH_BUCKETS> fail_high_index{};
    uint64_t aspiration_fail_low_researches = 0;
    uint64_t aspiration_fail_high_researches = 0;
    std::array<uint64_t, STATS_PVS_RESEARCH_TYPES> pvs_researches{};
    uint64_t see_prunes = 0;
    uint64_t delta_prunes = 0;
    uint64_t nmp = 0;
    uint64_t nmp_failhigh = 0;
    uint64_t rfp = 0;
    uint64_t futility_prune = 0;

    // iterative depth (stats per depth of single search)
    // tracks performance improvements across depths
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_time_ms{};
    std::array<int, STATS_MAX_ITER_DEPTH> it_depth_eval{};
    std::array<Move, STATS_MAX_ITER_DEPTH> it_depth_move{};
    //std::vector<std::vector<Move>> it_depth_pv;
    std::array<int, STATS_MAX_ITER_DEPTH> it_depth_qdepth{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_nodes{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_qnodes{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_ttstores{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_tthits{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_tt_returns{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_ttfill{};
    //std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_iid{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_fail_highs{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_fail_lows{};
    // per-iteration-depth fail-high index histogram
    std::array<std::array<uint64_t, STATS_FH_BUCKETS>, STATS_MAX_ITER_DEPTH> it_depth_fh_index{};
    std::array<std::array<uint64_t, STATS_PVS_RESEARCH_TYPES>, STATS_MAX_ITER_DEPTH> it_depth_pvs_researches{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_aspiration_failhigh_researches{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_aspiration_faillow_researches{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_see_prunes{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_delta_prunes{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_nmp{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_nmp_failhigh{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_rfp{};
    std::array<uint64_t, STATS_MAX_ITER_DEPTH> it_depth_futility_prune{};

    // tree depth (stats per depth of search tree for single search)
    //  this leads to multi-counting as it_depth searches d=1..it_depth tree depths
    //  but this is where the actual work is done
    //      so search improvements based on data may begin here
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_nodes{}; 
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_qnodes{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_ttstores{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_tthits{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_tt_returns{};
    //std::array<uint64_t, STATS_MAX_PLY> tree_depth_iid{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_fail_highs{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_fail_lows{};
    // per-tree-depth fail-high index histogram
    std::array<std::array<uint64_t, STATS_FH_BUCKETS>, STATS_MAX_PLY> tree_depth_fh_index{};
    std::array<std::array<uint64_t, STATS_PVS_RESEARCH_TYPES>, STATS_MAX_PLY> tree_depth_pvs_researches{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_see_prunes{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_delta_prunes{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_nmp{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_nmp_failhigh{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_rfp{};
    std::array<uint64_t, STATS_MAX_PLY> tree_depth_futility_prune{};
    #endif
};

// --- single header-only global instance ---
inline SearchStats g_stats{};

// -------------------------
//      Reset
// -------------------------
inline void resetSearchStats() {
    g_stats = SearchStats{};
}


// --- macros for tracking ---

#ifdef DEV

#define STATS_NODE(it_d, ply)                                \
    do {                                                             \
        /*STATS_BOUNDS_CHECK(it_d, ply);    */               \
        g_stats.nodes++;                                \
        g_stats.it_depth_nodes[it_d]++;                \
        g_stats.tree_depth_nodes[ply]++;                 \
    } while (0)

#define STATS_QNODE(it_d, ply)                               \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);*/                    \
        g_stats.qnodes++;                               \
        g_stats.it_depth_qnodes[it_d]++;                \
        g_stats.it_depth_qdepth[it_d] = std::max(g_stats.it_depth_qdepth[it_d], ply); \
        g_stats.tree_depth_qnodes[ply]++;               \
        g_stats.max_qdepth = std::max(g_stats.max_qdepth, ply); \
    } while (0)

#define STATS_ASPIRATION_FAILLOW(depth) \
    do { \
        /*STATS_BOUNDS_CHECK_1D(depth); */\
        g_stats.aspiration_fail_low_researches++; \
        g_stats.it_depth_aspiration_faillow_researches[depth]++; \
    } while(0)

#define STATS_ASPIRATION_FAILHIGH(depth) \
    do { \
        /*STATS_BOUNDS_CHECK_1D(depth); */\
        g_stats.aspiration_fail_high_researches++; \
        g_stats.it_depth_aspiration_failhigh_researches[depth]++; \
    } while(0)

#define STATS_PVS_RESEARCH(it_d, ply, type) \
    do { \
        /*STATS_BOUNDS_CHECK(it_d, ply);*/ \
        g_stats.pvs_researches[type]++; \
        g_stats.it_depth_pvs_researches[it_d][type]++; \
        g_stats.tree_depth_pvs_researches[ply][type]++; \
    } while(0)

// Helper to map move index to bucket [0,1,2,3,4-7,8+]
inline int fh_bucket(int move_idx) {
    if (move_idx <= 3) return move_idx;
    if (move_idx <= 7) return 4;
    return 5;
}

#define STATS_FAILHIGH(it_d, ply, move_idx)                  \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);  */                  \
        g_stats.fail_highs++;                           \
        g_stats.it_depth_fail_highs[it_d]++;            \
        g_stats.tree_depth_fail_highs[ply]++;           \
        int _fh_b = fh_bucket(move_idx);                \
        g_stats.fail_high_index[_fh_b]++;               \
        g_stats.it_depth_fh_index[it_d][_fh_b]++;       \
        g_stats.tree_depth_fh_index[ply][_fh_b]++;      \
    } while (0)

#define STATS_FAILLOW(it_d, ply)                             \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);   */                 \
        g_stats.fail_lows++;                            \
        g_stats.it_depth_fail_lows[it_d]++;             \
        g_stats.tree_depth_fail_lows[ply]++;            \
    } while (0)

#define STATS_TT_HIT(it_d, ply)                              \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);     */               \
        g_stats.tt_hits++;                              \
        g_stats.it_depth_tthits[it_d]++;                \
        g_stats.tree_depth_tthits[ply]++;               \
    } while (0)

#define STATS_TT_RETURN(it_d, ply)                              \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);     */               \
        g_stats.tt_returns++;                              \
        g_stats.it_depth_tt_returns[it_d]++;                \
        g_stats.tree_depth_tt_returns[ply]++;               \
    } while (0)

#define STATS_TT_STORE(it_d, ply)                            \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);     */               \
        g_stats.tt_stores++;                            \
        g_stats.it_depth_ttstores[it_d]++;              \
        g_stats.tree_depth_ttstores[ply]++;             \
    } while (0)

#define STATS_SEE_PRUNE(it_d, ply)                           \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);    */                \
        g_stats.see_prunes++;                           \
        g_stats.it_depth_see_prunes[it_d]++;            \
        g_stats.tree_depth_see_prunes[ply]++;           \
    } while (0)

#define STATS_DELTA_PRUNE(it_d, ply)                         \
    do {                                                    \
        /*STATS_BOUNDS_CHECK(it_d, ply);    */                \
        g_stats.delta_prunes++;                         \
        g_stats.it_depth_delta_prunes[it_d]++;          \
        g_stats.tree_depth_delta_prunes[ply]++;         \
    } while (0)

#define STATS_NMP(it_d, ply) \
    do { \
        /*STATS_BOUNDS_CHECK(it_d, ply);*/ \
        g_stats.nmp++; \
        g_stats.it_depth_nmp[it_d]++; \
        g_stats.tree_depth_nmp[ply]++; \
    } while(0)

#define STATS_NMP_FAILHIGH(it_d, ply) \
    do { \
        /*STATS_BOUNDS_CHECK(it_d, ply);*/ \
        g_stats.nmp_failhigh++; \
        g_stats.it_depth_nmp_failhigh[it_d]++; \
        g_stats.tree_depth_nmp_failhigh[ply]++; \
    } while(0)

#define STATS_RFP(it_d, ply) \
    do { \
        g_stats.rfp++; \
        g_stats.it_depth_rfp[it_d]++; \
        g_stats.tree_depth_rfp[ply]++; \
    } while(0)

#define STATS_FUTILITY_PRUNE(it_d, ply) \
    do { \
        g_stats.futility_prune++; \
        g_stats.it_depth_futility_prune[it_d]++; \
        g_stats.tree_depth_futility_prune[ply]++; \
    } while(0)
            

    /*
#define STATS_IID(it_d, ply) \
    do { \
        

        }
    } while(0)
     */

#endif

// -------------------------
//      Logging Functions
// -------------------------

template <typename T, size_t N>
std::string array_to_json(const std::array<T, N>& v, size_t len)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 1; i < len; ++i) // start from 1 to skip depth 0 which is init to 0
    {
        if (i-1) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
};

template <size_t N>
std::string moves_to_json(const std::array<Move, N>& v, size_t len)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 1; i < len; ++i) // start from 1 to skip depth 0 which is init to null_move
    {
        if (i-1) oss << ",";
        oss << "\"" << v[i].uci() << "\"";
    }
    oss << "]";
    return oss.str();
};

// Serialize a fixed-size array as JSON (no skipping index 0)
template <typename T, size_t N>
std::string bucket_array_to_json(const std::array<T, N>& v)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
};

// Serialize per-depth bucket histogram as JSON array of arrays
template <size_t D, size_t B>
std::string depth_buckets_to_json(const std::array<std::array<uint64_t, B>, D>& v, size_t len)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 1; i < len; ++i) {
        if (i-1) oss << ",";
        oss << "[";
        for (size_t j = 0; j < B; ++j) {
            if (j) oss << ",";
            oss << v[i][j];
        }
        oss << "]";
    }
    oss << "]";
    return oss.str();
};

// serialize one bucket from 2d array
template <size_t D, size_t B>
std::string depth_bucket_to_json(
    const std::array<std::array<uint64_t, B>, D>& v,
    size_t len,
    size_t bucket)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 1; i < len; ++i) {
        if (i > 1) oss << ",";
        oss << v[i][bucket];
    }
    oss << "]";
    return oss.str();
}

inline void logRootMoves(const SearchResult& result, int depth) {
    static std::ofstream root_log(Logging::log_file_name("root_moves.jsonl"), std::ios::app);
    if (root_log.is_open()) {
        root_log << "{\"search_uuid\":\"" << g_run_context.search_uuid << "\","
                 << "\"depth\":" << depth << ",\"moves\":[";
        for (int rm = 0; rm < result.root_count; ++rm) {
            if (rm) root_log << ",";
            root_log << "{\"move\":\"" << result.root_moves[rm].move.uci() << "\","
                     << "\"eval\":" << result.root_moves[rm].eval << ","
                     << "\"time_ms\":" << result.root_moves[rm].time_ms << ","
                     << "\"nodes\":" << result.root_moves[rm].nodes << "}";
        }
        root_log << "]}\n";
        root_log.flush();
    }
}

inline void logSearchStats(const std::string& fen = "") {
    static std::ofstream out(Logging::log_file_name("search.jsonl"), std::ios::app);
    if (!out.is_open()) return;

    size_t n = g_stats.max_depth + 1;
    size_t q_n = g_stats.max_qdepth + 1;

    auto moves_list_to_json = [](const std::vector<Move>& v) { 
        std::ostringstream oss; 
        oss << "["; 
        for (size_t i = 0; i < v.size(); ++i) { 
            if (i) oss << ","; 
            oss << "\"" << v[i].uci() << "\""; 
        } oss << "]"; 
        return oss.str(); 
    };

    out << "{"
        << "\"engine_id\":\"" << ENGINE_ID << "\","
        << "\"instance_id\":" << instanceID() << ","
        << "\"type\":\"search\","
        << "\"session\":" << currentSession() << ","
        << "\"game_uuid\":\"" << g_run_context.game_uuid << "\","
        << "\"search_uuid\":\"" << g_run_context.search_uuid << "\","
        << "\"fen\":\"" << fen << "\","
        << "\"ply\":" << g_stats.game_ply << ","

        << "\"nodes\":" << g_stats.nodes << ","
        #ifdef DEV
        << "\"qnodes\":" << g_stats.qnodes << ","

        << "\"max_depth\":" << g_stats.max_depth << ","
        << "\"max_qdepth\":" << g_stats.max_qdepth << ","
        << "\"completed_depth\":" << g_stats.max_completed_depth << ","
        << "\"time_ms\":" << g_stats.time_ms << ","
        << "\"root_eval\":" << g_stats.eval << ","
        << "\"best_move\":\"" << g_stats.move.uci() << "\","
        << "\"principal_variation\":" << moves_list_to_json(g_stats.principal_variation) << ","

        << "\"tt_hits\":" << g_stats.tt_hits << ","
        << "\"tt_returns\":" << g_stats.tt_returns << ","
        << "\"tt_stores\":" << g_stats.tt_stores << ","
        << "\"tt_fill\":" << g_stats.tt_fill_ratio << ","
        << "\"tt_overwritten\":" << g_stats.tt_overwritten << ","

        << "\"fail_highs\":" << g_stats.fail_highs << ","
        << "\"fail_lows\":" << g_stats.fail_lows << ","
        << "\"fail_high_index\":" << bucket_array_to_json(g_stats.fail_high_index) << ","

        << "\"delta_prunes\":" << g_stats.delta_prunes << ","
        << "\"see_prunes\":" << g_stats.see_prunes << ","
        << "\"aspiration_fail_low_researches\":" << g_stats.aspiration_fail_low_researches << ","
        << "\"aspiration_fail_high_researches\":" << g_stats.aspiration_fail_high_researches << ","

        << "\"pvs_researches_depth\":" << g_stats.pvs_researches[0] << ","
        << "\"pvs_researches_full\":" << g_stats.pvs_researches[1] << ","
        << "\"pvs_researches_root\":" << g_stats.pvs_researches[2] << ","
        << "\"nmp\":" << g_stats.nmp << ","
        << "\"nmp_failhigh\":" << g_stats.nmp_failhigh << ","
        << "\"rfp\":" << g_stats.rfp << ","
        << "\"futility_prune\":" << g_stats.futility_prune << ","

        // iteration stats
        << "\"itdepth_time_ms\":" << array_to_json(g_stats.it_depth_time_ms, n) << ","
        << "\"itdepth_eval\":" << array_to_json(g_stats.it_depth_eval, n) << ","
        << "\"itdepth_move\":" << moves_to_json(g_stats.it_depth_move, n) << ","
        << "\"itdepth_nodes\":" << array_to_json(g_stats.it_depth_nodes, n) << ","
        << "\"itdepth_qnodes\":" << array_to_json(g_stats.it_depth_qnodes, n) << ","
        << "\"itdepth_qdepth\":" << array_to_json(g_stats.it_depth_qdepth, n) << ","
        << "\"itdepth_tthits\":" << array_to_json(g_stats.it_depth_tthits, n) << ","
        << "\"itdepth_tt_returns\":" << array_to_json(g_stats.it_depth_tt_returns, n) << ","
        << "\"itdepth_ttstores\":" << array_to_json(g_stats.it_depth_ttstores, n) << ","
        << "\"itdepth_fail_highs\":" << array_to_json(g_stats.it_depth_fail_highs, n) << ","
        << "\"itdepth_fail_lows\":" << array_to_json(g_stats.it_depth_fail_lows, n) << ","
        << "\"itdepth_fh_index\":" << depth_buckets_to_json(g_stats.it_depth_fh_index, n) << ","
        << "\"itdepth_aspiration_failhigh_researches\":" << array_to_json(g_stats.it_depth_aspiration_failhigh_researches, n) << ","
        << "\"itdepth_aspiration_faillow_researches\":" << array_to_json(g_stats.it_depth_aspiration_faillow_researches, n) << ","
        << "\"itdepth_see_prunes\":" << array_to_json(g_stats.it_depth_see_prunes, n) << ","
        << "\"itdepth_delta_prunes\":" << array_to_json(g_stats.it_depth_delta_prunes, n) << ","
        << "\"itdepth_pvs_researches_depth\":" << depth_bucket_to_json(g_stats.it_depth_pvs_researches, n, 0) << ","
        << "\"itdepth_pvs_researches_full\":" << depth_bucket_to_json(g_stats.it_depth_pvs_researches, n, 1) << ","
        << "\"itdepth_pvs_researches_root\":" << depth_bucket_to_json(g_stats.it_depth_pvs_researches, n, 2) << ","
        << "\"itdepth_nmp\":" << array_to_json(g_stats.it_depth_nmp, n) << ","
        << "\"itdepth_nmp_failhigh\":" << array_to_json(g_stats.it_depth_nmp_failhigh, n) << ","
        << "\"itdepth_rfp\":" << array_to_json(g_stats.it_depth_rfp, n) << ","
        << "\"itdepth_futility_prune\":" << array_to_json(g_stats.it_depth_futility_prune, n) << ","

        // tree ply stats
        << "\"treedepth_nodes\":" << array_to_json(g_stats.tree_depth_nodes, q_n) << ","
        << "\"treedepth_qnodes\":" << array_to_json(g_stats.tree_depth_qnodes, q_n) << ","
        << "\"treedepth_tt_hits\":" << array_to_json(g_stats.tree_depth_tthits, q_n) << ","
        << "\"treedepth_tt_returns\":" << array_to_json(g_stats.tree_depth_tt_returns, q_n) << ","
        << "\"treedepth_tt_stores\":" << array_to_json(g_stats.tree_depth_ttstores, q_n) << ","
        << "\"treedepth_fail_highs\":" << array_to_json(g_stats.tree_depth_fail_highs, q_n) << ","
        << "\"treedepth_fail_lows\":" << array_to_json(g_stats.tree_depth_fail_lows, q_n) << ","
        << "\"treedepth_fh_index\":" << depth_buckets_to_json(g_stats.tree_depth_fh_index, q_n) << ","
        << "\"treedepth_see_prunes\":" << array_to_json(g_stats.tree_depth_see_prunes, q_n) << ","
        << "\"treedepth_delta_prunes\":" << array_to_json(g_stats.tree_depth_delta_prunes, q_n) << ","
        << "\"treedepth_pvs_researches_depth\":" << depth_bucket_to_json(g_stats.tree_depth_pvs_researches, n, 0) << ","
        << "\"treedepth_pvs_researches_full\":" << depth_bucket_to_json(g_stats.tree_depth_pvs_researches, n, 1) << ","
        << "\"treedepth_pvs_researches_root\":" << depth_bucket_to_json(g_stats.tree_depth_pvs_researches, n, 2) << ","
        << "\"treedepth_nmp\":" << array_to_json(g_stats.tree_depth_nmp, q_n) << ","
        << "\"treedepth_nmp_failhigh\":" << array_to_json(g_stats.tree_depth_nmp_failhigh, q_n) << ","
        << "\"treedepth_rfp\":" << array_to_json(g_stats.tree_depth_rfp, q_n) << ","
        << "\"treedepth_futility_prune\":" << array_to_json(g_stats.tree_depth_futility_prune, q_n)
        #endif
        << "}\n";

    out.flush();

    //resetSearchStats();
}


inline void dumpSearchStats()
{
    using std::cout;
    using std::setw;
    using std::left;
    using std::right;
 
    #ifdef DEV
        const uint64_t total_nodes = g_stats.nodes + g_stats.qnodes;
    #else 
        const uint64_t total_nodes = g_stats.nodes; 
    #endif
 
    const double nps =
        g_stats.time_ms > 0
            ? (1000.0 * static_cast<double>(total_nodes)) / static_cast<double>(g_stats.time_ms)
            : 0.0;

    #ifdef DEV
    const double tt_return_pct =
        (g_stats.tt_returns + g_stats.tt_stores) > 0
            ? 100.0 * static_cast<double>(g_stats.tt_returns) /
              static_cast<double>(g_stats.tt_returns + g_stats.tt_stores)
            : 0.0;
 
    const double tt_hit_pct =
        (g_stats.tt_hits + g_stats.tt_stores) > 0
            ? 100.0 * static_cast<double>(g_stats.tt_hits) /
              static_cast<double>(g_stats.tt_hits + g_stats.tt_stores)
            : 0.0;
 
    const double nmp_fh_pct =
        g_stats.nmp > 0
            ? 100.0 * static_cast<double>(g_stats.nmp_failhigh) / static_cast<double>(g_stats.nmp)
            : 0.0;
    

    // Effective index values for the FH histogram buckets: 0,1,2,3,5.5(avg 4-7),10(avg 8+)
    static const double bucket_midpoints[STATS_FH_BUCKETS] = {0.0, 1.0, 2.0, 3.0, 5.5, 10.0};
 
    // mean / median / 90 / %first, computed from a raw bucket-count array
    auto fhStats = [](const auto& counts, uint64_t total) {
        struct { double mean, median, p90, first_pct; } r{0.0, 0.0, 0.0, 0.0};
        if (total == 0) return r;
        const double ftotal = static_cast<double>(total);
        for (int b = 0; b < STATS_FH_BUCKETS; ++b)
            r.mean += bucket_midpoints[b] * static_cast<double>(counts[b]);
        r.mean /= ftotal;
 
        uint64_t cum = 0;
        bool found_median = false, found_p90 = false;
        const uint64_t median_target = (total + 1) / 2;
        const uint64_t p90_target = (total * 9 + 9) / 10;
        for (int b = 0; b < STATS_FH_BUCKETS; ++b) {
            cum += counts[b];
            if (!found_median && cum >= median_target) { r.median = bucket_midpoints[b]; found_median = true; }
            if (!found_p90 && cum >= p90_target) { r.p90 = bucket_midpoints[b]; found_p90 = true; }
        }
        r.first_pct = 100.0 * static_cast<double>(counts[0]) / ftotal;
        return r;
    };
 
    const auto fh = fhStats(g_stats.fail_high_index, g_stats.fail_highs);
    #endif
 
    const size_t n   = g_stats.max_depth + 1;   // iterative-deepening rows
    const size_t q_n = g_stats.max_qdepth + 1;  // tree-ply rows
 
    constexpr int W = 160; // total box width
    auto rule = [&](char c) { cout << std::string(W, c) << "\n"; };
    auto section = [&](const std::string& title) {
        cout << "\n";
        rule('-');
        cout << " " << title << "\n";
        rule('-');
    };
    auto row = [&](const std::string& label, auto value, const char* suffix = "") {
        cout << "  " << left << setw(20) << label << right << setw(14) << value << suffix << "\n";
    };
    auto _pv = [&](const std::vector<Move>) {
        cout << "  " << left << setw(20) << "PV" << right << setw(14);
        for (const Move& move : g_stats.principal_variation)
            cout << move << " ";
        cout << "\n";
    };
 
    cout << std::fixed << std::setprecision(2);
 
    // ===================== HEADER =====================
    rule('-');
    rule('-');
    cout << "SEARCH STATS\n";
    rule('-');
    rule('-');
 
    // ===================== SEARCH SUMMARY =====================
    section("SEARCH");
    row("Best Move", g_stats.move.uci());
    row("Eval", g_stats.eval);
    _pv(g_stats.principal_variation);
    row("Time (ms)", g_stats.time_ms);
    row("Depth", g_stats.max_depth);
    row("Completed Depth", g_stats.max_completed_depth);
    row("Max QDepth", g_stats.max_qdepth);
 
    // ===================== NODES =====================
    section("NODES");
    row("Nodes", g_stats.nodes);
    #ifdef DEV
    row("QNodes", g_stats.qnodes);
    #endif
    row("Total", total_nodes);
    row("NPS", static_cast<uint64_t>(nps));
 
    #ifdef DEV
    // ===================== TT =====================
    section("TRANSPOSITION TABLE");
    row("Returns", g_stats.tt_returns);
    row("Hits", g_stats.tt_hits);
    row("Stores", g_stats.tt_stores);
    row("Overwritten", g_stats.tt_overwritten);
    row("Fill Ratio", g_stats.tt_fill_ratio * 100.0, "%");
    row("Return Rate", tt_return_pct, "%");
    row("Hit Rate", tt_hit_pct, "%");
 
    // ===================== CUTOFFS =====================
    section("CUTOFFS");
    row("Fail Highs", g_stats.fail_highs);
    row("Fail Lows", g_stats.fail_lows);
    row("FH Index [0]", g_stats.fail_high_index[0]);
    row("FH Index [1]", g_stats.fail_high_index[1]);
    row("FH Index [2]", g_stats.fail_high_index[2]);
    row("FH Index [3]", g_stats.fail_high_index[3]);
    row("FH Index [4-7]", g_stats.fail_high_index[4]);
    row("FH Index [8+]", g_stats.fail_high_index[5]);
    row("FH % at [0]", fh.first_pct, "%");
    row("FH Mean Index", fh.mean);
    row("FH Median Index", fh.median);
    row("FH p90 Index", fh.p90);
 
    // ===================== RESEARCHES / PRUNING / NULL MOVE / ASPIRATION ====
    section("RESEARCHES & PRUNING");
    row("Aspiration FH Re", g_stats.aspiration_fail_high_researches);
    row("Aspiration FL Re", g_stats.aspiration_fail_low_researches);
    row("PVS w/ LMR", g_stats.pvs_researches[0]);
    row("PVS full", g_stats.pvs_researches[1]);
    row("PVS @ root", g_stats.pvs_researches[2]);
    row("SEE Prunes", g_stats.see_prunes);
    row("Delta Prunes", g_stats.delta_prunes);
    row("Rvrs Futil Prunes", g_stats.rfp);
    row("Futility Prunes", g_stats.futility_prune);
    row("NMP Attempts", g_stats.nmp);
    row("NMP Fails", g_stats.nmp_failhigh);
    row("NMP FH %", nmp_fh_pct, "%");
 
    // ===================== PV =====================
    section("PRINCIPAL VARIATION");
    cout << "  ";
    for (const Move& m : g_stats.principal_variation)
        cout << m.uci() << " ";
    cout << "\n";
 
    // ===================== ITERATIVE DEEPENING =====================
    section("ITERATIVE DEEPENING (per completed depth)");
    cout << left
         << setw(4)  << "D"
         << setw(7)  << "Eval"
         << setw(7)  << "Move"
         << setw(9)  << "Time"
         << setw(11)  << "Nodes"
         << setw(11)  << "QNodes"
         << setw(5)  << "QD"
         << setw(9)  << "TTRet"
         << setw(9)  << "TTHit"
         << setw(9)  << "TTSt"
         << setw(8)  << "FH"
         << setw(8)  << "FL"
         << setw(7)  << "FHavg"
         << setw(7)  << "NMP"
         << setw(7)  << "NMPf"
         << setw(8)  << "SEE"
         << setw(8)  << "Delta"
         << setw(8)  << "RFP"
         << setw(8)  << "FP"
         << setw(7)  << "PVSl"
         << setw(7)  << "PVSf"
         << setw(7)  << "PVSr"
         << setw(7)  << "AspFH"
         << setw(7)  << "AspFL"
         << "\n";
    rule('-');
 
    for (size_t d = 0; d < n; ++d) {
        const auto dfh = fhStats(g_stats.it_depth_fh_index[d], g_stats.it_depth_fail_highs[d]);
        cout << left
             << setw(4)  << d
             << setw(7)  << g_stats.it_depth_eval[d]
             << setw(7)  << g_stats.it_depth_move[d].uci()
             << setw(9)  << g_stats.it_depth_time_ms[d]
             << setw(11)  << g_stats.it_depth_nodes[d]
             << setw(11)  << g_stats.it_depth_qnodes[d]
             << setw(5)  << g_stats.it_depth_qdepth[d]
             << setw(9)  << g_stats.it_depth_tt_returns[d]
             << setw(9)  << g_stats.it_depth_tthits[d]
             << setw(9)  << g_stats.it_depth_ttstores[d]
             << setw(8)  << g_stats.it_depth_fail_highs[d]
             << setw(8)  << g_stats.it_depth_fail_lows[d]
             << setw(7)  << dfh.mean
             << setw(7)  << g_stats.it_depth_nmp[d]
             << setw(7)  << g_stats.it_depth_nmp_failhigh[d]
             << setw(8)  << g_stats.it_depth_see_prunes[d]
             << setw(8)  << g_stats.it_depth_delta_prunes[d]
             << setw(8)  << g_stats.it_depth_rfp[d]
             << setw(8)  << g_stats.it_depth_futility_prune[d]
             << setw(7)  << g_stats.it_depth_pvs_researches[d][0]
             << setw(7)  << g_stats.it_depth_pvs_researches[d][1]
             << setw(7)  << g_stats.it_depth_pvs_researches[d][2]
             << setw(7)  << g_stats.it_depth_aspiration_failhigh_researches[d]
             << setw(7)  << g_stats.it_depth_aspiration_faillow_researches[d]
             << "\n";
    }
 
    // ===================== TREE DEPTH (BY PLY) =====================
    section("TREE STATS (per ply, includes quiescence)");
    cout << left
         << setw(4)  << "Ply"
         << setw(11) << "Nodes"
         << setw(10) << "QNodes"
         << setw(9)  << "TTRet"
         << setw(9)  << "TTHit"
         << setw(9)  << "TTSt"
         << setw(8)  << "FH"
         << setw(8)  << "FL"
         << setw(7)  << "FHavg"
         << setw(8)  << "SEE"
         << setw(8)  << "Delta"
         << setw(8)  << "RFP"
         << setw(8)  << "FP"
         << setw(7)  << "PVSl"
         << setw(7)  << "PVSf"
         << setw(7)  << "PVSr"
         << setw(7)  << "NMP"
         << setw(7)  << "NMPf"
         << "\n";
    rule('-');

    for (size_t p = 0; p < q_n; ++p) {
        const auto pfh = fhStats(g_stats.tree_depth_fh_index[p], g_stats.tree_depth_fail_highs[p]);
        cout << left
             << setw(4)  << p
             << setw(11) << g_stats.tree_depth_nodes[p]
             << setw(10) << g_stats.tree_depth_qnodes[p]
             << setw(9)  << g_stats.tree_depth_tt_returns[p]
             << setw(9)  << g_stats.tree_depth_tthits[p]
             << setw(9)  << g_stats.tree_depth_ttstores[p]
             << setw(8)  << g_stats.tree_depth_fail_highs[p]
             << setw(8)  << g_stats.tree_depth_fail_lows[p]
             << setw(7)  << pfh.mean
             << setw(8)  << g_stats.tree_depth_see_prunes[p]
             << setw(8)  << g_stats.tree_depth_delta_prunes[p]
             << setw(8)  << g_stats.tree_depth_rfp[p]
             << setw(8)  << g_stats.tree_depth_futility_prune[p]
             << setw(7)  << g_stats.tree_depth_pvs_researches[p][0]
             << setw(7)  << g_stats.tree_depth_pvs_researches[p][1]
             << setw(7)  << g_stats.tree_depth_pvs_researches[p][2]
             << setw(7)  << g_stats.tree_depth_nmp[p]
             << setw(7)  << g_stats.tree_depth_nmp_failhigh[p]
             << "\n";
    }
    #endif
 
    rule('=');
    cout << "\n";
}


#endif // STATS_H
