<div align="center">

<img src="bin/logos/logo.png" alt="Tomahawk logo" width="200">

# Tomahawk

An NNUE chess engine written in C++.

![UCI protocol](https://img.shields.io/badge/protocol-UCI-16a34a)
![C++17](https://img.shields.io/badge/C%2B%2B-17-2563eb)

[Play Tomahawk on Lichess](https://lichess.org/@/tomahawkBOT)

</div>

Tomahawk combines alpha-beta search with incrementally updated neural network evaluation. It communicates through the Universal Chess Interface (UCI), so a chess GUI or bot controller can supply positions and request moves.

Development, strength testing, tuning, and analytics are supported by the companion **San Jacinto** framework.

## Documentation

| Guide | What it covers |
|---|---|
| [Engine architecture](docs/engine.md) | Startup, ownership, UCI commands, board state, and the complete search lifecycle |
| [Position and move generation](docs/position.md) | Board encoding, attacks, legal moves, reversible state, hashing, and special moves |
| [Search](docs/search.md) | Iterative deepening, negamax, quiescence, pruning, reductions, move ordering, and the transposition table |
| [NNUE architecture](docs/nnue.md) | Color-coded v1/v2/v3 network diagrams, feature encoding, layers, and inference |
| [Network weights](nnue/weights/readme.md) | Bundled nets, compatibility, and profiles for training data and test results |
| [Validation and experimentation](docs/validation.md) | Correctness contracts, diagnostic coverage, performance measurements, and reproducible strength tests |
| [Build modes and statistics](docs/stats.md) | PROD/DEV/DEBUG differences, search counters, timing, log files, and debugging facilities |
| [UCI command reference](docs/uci.md) | Every protocol/diagnostic command, search arguments, options, and current support limits |

## Features

- **Search:** iterative deepening, aspiration windows, Principal Variation Search (PVS), and quiescence search.
- **Selective search:** late move reductions, null move pruning, reverse futility pruning, late quiet-move futility pruning, and quiescence SEE/delta pruning.
- **Move ordering:** transposition-table and previous-PV moves, promotions, Static Exchange Evaluation (SEE), killers, and history scores; later root iterations reuse scores and node counts.
- **Evaluation:** dual-perspective NNUE accumulators, mirrored king buckets, material-based output buckets, and a `1024 -> 16 -> 32 -> 1` big net with pairwise multiplication and SCReLU. Windows uses the explicit SIMD inference path; other builds use scalar evaluation.
- **Position handling:** bitboards, sliding-piece attack tables, reversible moves, Zobrist hashing, and repetition tracking.
- **Development:** perft, NNUE/SEE diagnostics, and detailed search, timing, and game telemetry in `DEV` builds.

Search currently uses one worker. Some advertised or parsed UCI controls are incomplete; see [current implementation boundaries](docs/engine.md#current-implementation-boundaries).

## Build

The checked-in CMake configuration requires **CMake 3.16+** and **C++17**. Its compiler flags target GCC/Clang-style toolchains; on Windows, use MinGW-w64. A native MSVC build is not configured consistently yet.

From the repository root, with CMake, Ninja, and a compiler on `PATH`:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

For MinGW Makefiles, replace `-G Ninja` with `-G "MinGW Makefiles"`. Use a separate build directory when changing generators. When invoking Windows CMake from PowerShell, use a native Windows Ninja executable; MSYS `/usr/bin/ninja` uses shell path handling that can break compiler invocation. MinGW Makefiles works with `mingw32-make`.

The default executable is `engines/prod/tomahawk` (`tomahawk.exe` on Windows). Set `-DVERSION=1.2.3` during configuration to change the non-Debug executable name and compiled version. Other build configurations write to `engines/dev/`.

| Configuration | Purpose |
|---|---|
| `Release` / `PROD` | Optimized production build; `Release` is the default |
| `DEV` | Optimized build with detailed search telemetry |
| `Debug` | Symbols and `DEBUG` NNUE feature/debug facilities; mode for board/Zobrist investigation |
| `Profile` | Optimized build with symbols, frame pointers, and `DEBUG` bookkeeping |

All four build modes explicitly request AVX2 and BMI2. Release/PROD, DEV, and Profile additionally use `-march=native`, which can enable instructions beyond that baseline. Build and run on a compatible x86 CPU; these are not portable binaries for arbitrary AVX2 machines. The removed Asan and RelWithDebInfo modes no longer have dedicated project configuration.

See [build modes and statistics](docs/stats.md) for exact compile definitions, instrumentation, and log behavior. `DEV` collects detailed stats; `Debug` enables correctness-debugging facilities. The CMake spelling is `Debug`, not `DEBUG`.

### Runtime assets

CMake embeds the source checkout path as `PROJECT_ROOT`. The default network is:

```text
nnue/weights/1024_16_32_pairmul_screlu_T60T70Farseer_filtered.bin
```

Keep the checkout available when running the executable. Copying only the executable does not bundle its weights. The configured opening-book path is `bin/Titans.bin`; that file is not tracked in this repository. Logging defaults to `../san-jacinto/logs/test_logs` relative to the source root, and startup creates that directory. See the [weights guide](nnue/weights/readme.md#loading-and-compatibility) for the current network-loading paths.

## Run

Launch `./engines/prod/tomahawk` on Linux or `.\engines\prod\tomahawk.exe` in PowerShell. Enter:

```text
uci
isready
setoption name Hash value 128
ucinewgame
position startpos moves e2e4 e7e5
go depth 6
```

Wait for `bestmove`, then enter `quit` to exit. A GUI can send the same protocol commands. For clock-based play, it supplies `wtime`, `btime`, increments, and optionally `movestogo` with `go`.

For a move-generation check, start a fresh process or reset the position before running:

```text
position startpos
perft 4
```

The starting-position depth-4 leaf count is **197,281**. Other diagnostics include `nnue_eval`, `see <square>`, `print_board`, and `clear_tt`; `nnue_test` is Windows-only. See the [complete UCI command reference](docs/uci.md) for syntax, build availability, and unsupported placeholders.

## Repository layout

```text
src/             UCI, engine, search, board, move generation, opening book
include/         Interfaces, search data, hashing, limits, and telemetry
nnue/            Network definition, accumulators, inference, and SIMD
nnue/weights/    Bundled network files and their profiles
docs/            Architecture, position, search, NNUE, and validation guides
bin/             Logos and local runtime assets/configuration
engines/         Build outputs and local reference engines (ignored by Git)
CMakeLists.txt   Build configuration
```

## Development

Use position and perft checks for move-generation changes, NNUE diagnostics for evaluation changes, and controlled strength tests for search changes. San Jacinto supports the broader workflow: automated games and SPRT testing, SPSA/CMA-ES tuning, telemetry processing, and analysis. Record the engine revision, network checksum, build configuration, and test conditions together so results remain reproducible.
