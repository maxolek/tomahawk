# UCI and diagnostic command reference

[Overview](../README.md) · [Engine architecture](engine.md) · [Build modes and stats](stats.md) · [Validation](validation.md)

This is the complete command inventory implemented by [UCI::handleCommand()](../src/UCI.cpp), including diagnostic extensions and recognized placeholders. Names are case-sensitive. Unknown commands are silently ignored. “All builds” means the handler exists, not that every advertised capability is complete.

## Protocol and search commands

| Command | Availability | Current behavior |
|---|---|---|
| `uci` | All | Print engine identity, standard/logging options, then `uciok` |
| `uci_dev` | All | Print the same handshake plus network/book paths and tuning options; does not enable DEV instrumentation |
| `isready` | All | Print `readyok` when the listener can process the command |
| `setoption name <name> value <value>` | All; some options DEV-only | Pass the name/value to `Engine::setOption()`; see the option table below |
| `ucinewgame` | All | Start a new game/session, clear engine state and TT, reset boards; existing searcher killer/history tables remain |
| `position startpos [moves <move> ...]` | All | Rebuild the starting position and replay coordinate moves |
| `position fen <six FEN fields> [moves <move> ...]` | All | Rebuild the supplied position and replay moves |
| `go [arguments]` | All | Search, emit `bestmove`, then apply the chosen move to both boards |
| `stop` | All; incomplete | Set the engine stop flag and print its current best move when processed; cannot interrupt synchronous search |
| `ponderhit` | All; placeholder | Recognized with an empty handler |
| `quit` | All; synchronous | Request stop; the listener exits for the exact input line `quit` once it can read it |

`go` runs on the input listener itself. Commands queued during search, including `isready`, `stop`, and `quit`, wait until it returns. Search limits are also passed by value. See [implementation boundaries](engine.md#current-implementation-boundaries).

| `go` argument | Meaning and support |
|---|---|
| `depth <plies>` | Requested search depth, subject to the internal depth cap |
| `movetime <ms>` | Requested move time |
| `wtime <ms>`, `btime <ms>` | Remaining clocks |
| `winc <ms>`, `binc <ms>` | Clock increments |
| `movestogo <count>` | Used when allocating clock time |
| `nodes <count>` | Parsed and stored; no enforced node budget |
| `infinite` | Parsed/stored; does not establish true unlimited, interruptible search |
| `eval` | Custom extension: prepend `eval <score>` to the bestmove output; still searches and applies the move |

`searchmoves`, `mate`, and `ponder` are not parsed as search controls. `debug` and `register` have no command handlers. A bare `go` uses the engine's defaults; supply an explicit depth or time for reproducible diagnostics.

## Configuration and diagnostics

| Command | Availability | Current behavior |
|---|---|---|
| `config` | All | List `.ini` files in `PROJECT_ROOT/bin/configs`, then read a numeric selection from stdin and apply it; interactive, so unsuitable as a normal GUI exchange |
| `apply_config <name>` | All | Read `bin/configs/<name>.ini` (suffix optional) and resize the TT from the loaded setting |
| `save_config <name>` | All | Save configuration through `Engine::create_config_file()` |
| `nnue_eval` | All | Rebuild and evaluate the search board |
| `nnue_test` | Windows only | Compare scalar/SIMD results using the external San Jacinto position file described in [validation](validation.md#nnue-agreement) |
| `perft <depth>` | All | Print the total legal move-tree leaf count; use a positive depth |
| `see <square>` | All | Diagnose exchanges on a target such as `e4` |
| `dumpzobrist` | All | Print game-board hash, recent hashes, repetition counts, and history sizes |
| `dump_tt` | All | DEV prints detailed TT counters and fill; other builds print a no-stats message |
| `dumpstats` | All | Print last collected search stats; detailed breakdowns require DEV |
| `dumpmoves` | DEV only | Print the last result's root-move records |
| `clear_tt` | All | Clear the transposition table and report occupancy before/after |
| `print_board` | All | Print the game board |
| `flip` | All; unsafe diagnostic | Toggle only `search_board.is_white_move`; does not synchronize `move_color`, hash, or other state. Reset with `position` before further work |
| `bench` | All; placeholder | Recognized but does nothing |
| `speedtest` | All; placeholder | Recognized but does nothing |

Run `position` before an independent diagnostic. After `go`, the engine has already applied its selected move. Configuration loading updates stored paths but does not automatically reload NNUE/book resources; see [weights loading](../nnue/weights/readme.md#loading-and-compatibility). Diagnostic helpers without dispatcher entries, such as `perftDivide()`, are not UCI commands.

## Option names and actual support

The handshake shows defaults and advertised ranges from the current executable. [Engine::setOption()](../src/engine.cpp) defines what is actually accepted; advertising an option does not enforce its range. Values are parsed directly, so use valid numeric values. Boolean true values accepted by the setter are `true`, `True`, and `1`.

| Option name | Availability / advertisement | Setter behavior |
|---|---|---|
| `Move Overhead` | All / `uci` | Set clock-allocation overhead in milliseconds |
| `Hash` | All / `uci` | Resize TT using the requested MB setting |
| `Threads` | All / `uci` | Store the value; engine still has one search worker, advertised range 1–1 |
| `Ponder` | All / `uci` | Store the flag; pondering protocol is incomplete |
| `UCI_ShowWDL` | All / `uci` | Store the flag; no corresponding WDL output implementation |
| `nnue_weight_file` | All / `uci_dev` | Load `PROJECT_ROOT/bin/nnue_wgts/<value>.bin`; value is a stem, despite the advertised default being a path |
| `opening_book` | All / `uci_dev` | Construct `PROJECT_ROOT/bin/<value>.bin` and load for nonempty values |
| `SyzygyPath` | All / advertised as `syzygy` | Store path only; no tablebase loading. The advertised spelling is not a setter alias |
| `log_dir` | DEV setter / advertised in all builds | Set logging directory; configure before writers open their streams |
| `uci_logging` | DEV setter / advertised in all builds | Toggle UCI logging |
| `delta_prune_threshold`, `see_prune_threshold` | All / `uci_dev` | Set quiescence pruning thresholds |
| `aspiration_start_depth`, `aspiration_window`, `aspiration_research_scale` | All / `uci_dev` | Set aspiration parameters; research scale is parsed as an integer |
| `aspiration_depth_scale` | Advertised in `uci_dev` only | No setter branch; the runtime command does not update it |
| `draw_eval` | All / `uci_dev` | Set the search draw score |
| `contempt` | All / `uci_dev` | Store the parameter; no active search use |
| `r_nmp` | All / `uci_dev` | Set null-move reduction |
| `lmr_move_order_threshold`, `lmr_depth_threshold` | All / `uci_dev` | Set reduction eligibility thresholds |
| `r_lmr_const`, `r_lmr_denom` | All / `uci_dev` | Parse value and divide by 100; for example, `99` becomes `0.99` |

Unknown options produce an “ignoring unknown option” message in DEV; non-DEV builds do not have that fallback message. DEV-only setters cannot be enabled by first issuing `uci_dev`. See [build modes](stats.md#build-configuration-and-compile-definitions) for the exact CMake selections.
