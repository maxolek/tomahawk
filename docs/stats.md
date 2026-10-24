# Build modes, statistics, and debugging

[Overview](../README.md) · [Engine architecture](engine.md) · [UCI commands](uci.md) · [Validation](validation.md)

Production builds are for playing and representative speed measurements. **DEV** is for understanding search behavior through statistics. **DEBUG** is for inspecting correctness, especially NNUE features, accumulators, and board/Zobrist state. These are separate purposes: enabling DEBUG does not enable DEV statistics, and an optimized DEV build still pays instrumentation costs.

## Build configuration and compile definitions

[CMakeLists.txt](../CMakeLists.txt) selects the following explicit flags and definitions. CMake/toolchain defaults can add flags; the table describes this project's configuration.

| CMake build type | Project definition | Purpose and instrumentation | Project flags | Output directory |
|---|---|---|---|---|
| `Release` (default), `RELEASE`, `Prod`, `PROD` | Neither `DEV` nor `DEBUG` | Production search; aggregate node counting and basic diagnostics remain | `-O3 -ffast-math -march=native -mbmi2 -mavx2 -flto` | `engines/prod/` |
| `DEV` | `DEV` | Detailed search counters, depth/ply breakdowns, scoped timers, search/timing JSONL, root diagnostics | Same optimization flags as production | `engines/dev/` |
| `Debug` | `DEBUG` | NNUE active-feature bookkeeping and compiled debug helpers; symbols for board/hash debugging | `-O1 -g -mbmi2 -mavx2 -fno-optimize-sibling-calls` | `engines/dev/` |
| `Profile` | `DEBUG` | Optimized build with symbols and frame pointers; also includes DEBUG bookkeeping | `-O3 -ffast-math -march=native -mbmi2 -mavx2 -flto -g -fno-omit-frame-pointer` | `engines/dev/` |

Use the exact spelling **`Debug`** for `CMAKE_BUILD_TYPE`; `DEBUG` is the preprocessor definition it enables. Uppercase `-DCMAKE_BUILD_TYPE=DEBUG` currently falls through to optimized flags without that definition. There is no `PROD` preprocessor definition. `VERSION=dev` controls naming/identity, not statistics, and the `uci_dev` command does not switch build modes.

The supported project modes are now Release/PROD, DEV, Debug, and Profile. Asan and RelWithDebInfo no longer have dedicated branches. CMake does not reject these removed names or other unknown names: they enter the optimized fallback, without project DEBUG/DEV definitions, and write to `engines/dev/`. CMake may still add its own configuration defaults for recognized standard names. In particular, selecting `Asan` no longer enables sanitizers.

For example, configure with `cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=DEV` and build with `cmake --build build-dev --parallel`. Use `Debug` or `Release` in a separate build directory for the other modes. These directories separate build intermediates, but Debug and DEV still share the default output filename under `engines/dev/`; building one can overwrite the other's executable. See the [build guide](../README.md#build) for toolchain and `VERSION` naming details.

### Compiler flags and portability

`-mavx2` and `-mbmi2` explicitly enable the instruction sets used by Windows NNUE SIMD and PEXT attack lookup. They are compatible with `-march=native`, and normally redundant when the build CPU already supports both. They do not restrict native code generation to those two features or provide runtime CPU detection. A machine without either feature is not a valid runtime target; a native build can also require additional features of its build CPU. These are x86 flags, so the configuration is not currently an ARM build recipe. See [GCC's x86 options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

`-O3`, `-ffast-math`, and `-flto` remain the optimized-build policy. Fast math relaxes floating-point semantics; the integer NNUE path does not make the whole engine integer-only, since LMR uses floating-point logarithms. Treat search equivalence as something to measure. GCC can automatically link LTO objects through its linker plugin, but the project currently supplies these options through `target_compile_options()` only. Explicit link options or CMake IPO configuration would make LTO intent clearer across toolchains; a GCC link succeeding does not verify Clang's LTO setup. See [GCC's optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).

`Profile` retains DEBUG feature bookkeeping, so it profiles that workload rather than a completely production-equivalent evaluator. `-g` and frame pointers help inspection, while optimization/LTO can still inline or reorganize functions. `Debug` keeps its existing `-O1` policy. These GCC/Clang-style flags remain incompatible with a native MSVC configuration despite the separate `/W4` warning branch.

## DEBUG: inspecting correctness

[accumulator.h](../nnue/accumulator.h) maintains an `active_features` set under DEBUG: clearing an accumulator clears the set, adding a feature inserts its index, and removing it erases the index. This makes missing, extra, or incorrectly mirrored features inspectable alongside the numeric vector. It adds memory and work to feature updates, including in `Profile` builds.

[nnue.h](../nnue/nnue.h) and [nnue.cpp](../nnue/nnue.cpp) expose DEBUG helpers for accumulator inspection, scalar/SIMD stages, and incremental/full comparisons. Their calls inside `Searcher::perform_move()` and `perform_unmove()` are currently commented out. Selecting Debug compiles the facilities; it does not automatically audit every searched position.

Current build limitation: `NNUE::evaluate_debug()` calls `debug_evaluate()`, which is declared but has no definition. A MinGW GCC Debug build fails at link time on that symbol. An optimized Profile build can discard the unused helper through LTO and link successfully, so Profile succeeding does not establish that all DEBUG helpers are usable. This is independent of the added AVX2/BMI2 flags.

Board and Zobrist debugging uses `print_board`, `dumpzobrist`, `computeZobristHash()`, `auditZobrist()`, and field comparisons around make/unmake. These commands and board helpers are not all DEBUG-gated; several audit/assert call sites are commented out. DEBUG is the intended working mode for investigating those invariants, not a guarantee that they are continuously checked. Likewise, `nnue_test` is gated by Windows, not by DEBUG. See [validation](validation.md#state-restoration-and-hashing) for what to compare.

## DEV: what the statistics measure

[stats.h](../include/stats.h) owns the global `g_stats` record and `STATS_*` macros. Search updates counters where an event occurs; TT storage also records events inside [tt.h](../include/tt.h). There are three views of the work:

| View | Intended question | Interpretation |
|---|---|---|
| Search totals | How much work did this search perform? | Nodes, qnodes, elapsed time, depths, result/PV, TT events, cutoffs, pruning and re-searches |
| `it_depth_*` / JSON `itdepth_*` | How does work change across iterative-deepening depths? | Includes retries and re-searches; many interior counters use `remaining depth + ply`, which can differ from the outer iteration after reductions |
| `tree_depth_*` / JSON `treedepth_*` | Where in the recursive tree is work happening? | Aggregates visits by root-relative ply across iterations and repeated searches; these are visits, not unique positions |

In DEV, `nodes` and `qnodes` are separate; the console summary's total is their sum. In non-DEV builds, both normal-search and quiescence visits increment `nodes`, so adding an assumed qnode count would double-count work. Root-move node records use differences in `g_stats.nodes`, which means DEV root records exclude the separate qnode counter. Depth-zero negamax entry and its subsequent quiescence entry can both count as visits.

Quiescence increments its counter after the early time/draw and stand-pat return gates, so it does not count every invocation. Its recursive calls increase `ply`; tree qnode rows therefore reflect actual recursive ply, despite an older comment in `stats.h` describing grouping at the entry ply.

| Counter family | What it helps explain |
|---|---|
| TT hits, returns, stores, overwrites | A matching entry is distinct from a score usable for cutoff; storage/replacement measures another event |
| Fail highs/lows and fail-high move-index histogram | Whether ordering finds cutoffs early; buckets are indices 0, 1, 2, 3, 4–7, 8+ |
| Aspiration fail-high/fail-low retries | How often the previous score estimate needs a wider window |
| PVS re-search types | Full-depth verification after reduction, full-window interior re-search, and full-window root re-search |
| SEE, delta, reverse-futility, quiet-futility pruning | Which selective gates eliminate work |
| NMP attempts and fail highs | How often passing produces a cutoff; the console label `NMP Fails` refers to fail highs |

Use the raw counts with their actual denominators. For example, `dumpstats` computes its TT hit percentage as `hits / (hits + stores)`, not hits divided by all probes. Fail-high histogram mean/percentiles use representative values for grouped buckets, so they are estimates. A declared field need not be populated: `tt_fill_ratio` and `tt_overwritten` in `g_stats` should not be assumed to match the TT object's live occupancy and separate counters.

## Collection lifetime and output

For the normal tree-search path, `Engine::startSearch()` creates a search UUID, resets `g_stats`, runs iterative deepening, then records elapsed time and PV. DEV also fills the summary move/evaluation and writes search and timing records. The opening-book return occurs before this reset and logging, so diagnostics after a book move can still describe the previous search. Non-DEV `dumpstats` has a reduced summary, and fields populated only by DEV can retain default values.

| Output | Producer and availability | Contents |
|---|---|---|
| `dumpstats` | All builds; detailed sections in DEV | Console summary, plus DEV counter families and depth/ply tables |
| `dumpmoves` | DEV only | Last `SearchResult` root records: move, evaluation, time, nodes |
| `dump_tt` | Detailed output in DEV | Search TT counters and current live fill percentage; other builds print a no-stats message |
| `search_<instance>.jsonl` | End of a normal DEV tree search | Search result, context, counters and arrays |
| `timing_<instance>.jsonl` | End of a normal DEV tree search; also written by `perft` in all builds | Per-timer total milliseconds, calls, and average milliseconds |
| `game_<instance>.jsonl` | Game logging, not DEV-gated | Game outcome/reason, moves, elapsed time, start position and request settings |
| `uci_<instance>.log` | When UCI tracking is enabled; runtime setters are DEV-only | Incoming commands and explicitly logged outgoing messages; not a complete capture of stdout |

[logging.h](../include/logging.h), [session.h](../include/session.h), and [game_log.h](../include/game_log.h) attach engine, instance, session, game, and search identity as appropriate. The initial directory is `PROJECT_ROOT/../san-jacinto/logs/test_logs`. `ucinewgame` changes it to `PROJECT_ROOT/logs/game_logs` if it still equals that default. In DEV, set `log_dir` before the first write to choose an experiment directory. Search/game/timing writers keep function-local static streams: changing the setting later does not migrate streams already opened.

## Timing and analysis boundaries

[timer.h](../include/timer.h) uses scoped timers around search, move generation, board changes, NNUE, ordering, SEE, TT operations, and other call sites. The member named `cycles` actually stores elapsed **nanoseconds**. Recursive/nested timers measure overlapping durations, so their totals should not be summed into a wall-clock budget. `g_timing` is not reset with `g_stats`; its records can accumulate across searches and perft calls.

There are current implementation issues to account for before using this telemetry quantitatively:

- `ScopedTimer::active` is read in the destructor but never initialized by its constructor. Timing collection is unreliable until that is fixed.
- Two timer IDs serialize with the same `PVS_ROOT_SEARCH` name, producing duplicate JSON keys rather than distinct root-search/root-research fields.
- Stats arrays have 32 iteration slots and 64 ply slots, but use depth/ply directly as indices; the commented bounds checks also allow the out-of-range endpoints. Depth 32 or ply 64 can exceed storage. Qsearch has no separate hard ply cap.
- JSON depth-array helpers skip index zero, including tree-ply arrays. The console includes row zero. Some tree PVS arrays use the iteration-length variable when serializing, so JSON arrays need not share a common length.

Treat DEV data as evidence about observed work under these instrumentation limits. Use production builds for representative speed/strength measurements and preserve the build mode with every record. The [validation guide](validation.md#search-correctness-and-performance) explains how timing, correctness, and match evidence complement one another.
