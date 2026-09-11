# Validation and experimentation

[Overview](../README.md) · [Engine architecture](engine.md) · [Position](position.md) · [Search](search.md) · [NNUE](nnue.md)

Engine changes need different kinds of evidence. A move generator can be fast and wrong; an evaluator can agree with its reference implementation and still be weak; a search can visit more nodes per second and play worse. Establish state and algorithm correctness before interpreting performance or game results.

This guide describes available diagnostics and the checks worth applying to each subsystem. It is not a claim that the repository already provides an automated suite for every contract below. The current [CMake configuration](../CMakeLists.txt) builds the engine without registering a CTest suite.

Use [build modes and statistics](stats.md) to choose production measurements, DEV search analysis, or DEBUG correctness inspection. It also records current timer and counter limitations. The [UCI reference](uci.md) gives the complete command/option inventory and availability.

## Match the evidence to the change

| Change | Useful evidence | What it does not establish |
|---|---|---|
| Board or move generation | Perft counts, exact move-set comparisons, special-move positions, make/unmake restoration | Correct NNUE updates or stronger play |
| Hashes, metadata, history | Incremental/full hash agreement, restoration checks, repetition and fifty-move cases | Safe reuse of every history-sensitive TT result |
| NNUE feature updates | Both accumulators equal full rebuilds after make and undo, including bucket/mirror crossings | Agreement of dense inference implementations or network strength |
| Quantization or SIMD | Layer-by-layer scalar/SIMD comparison and final-score agreement across buckets | Training quality or Elo improvement |
| Search or TT | Tactical/terminal cases, bound and mate-distance checks, controlled games | A universal strength improvement from one suite |
| Performance optimization | Repeatable timings on fixed positions with fixed builds and hardware | Stronger play from NPS alone |
| Weights or tuning | Controlled baseline/candidate matches with complete provenance | Generalization beyond the tested conditions |

## Position and move-generation checks

The UCI diagnostic `perft <depth>` calls `Engine::perftPrint()`, which recursively counts legal move-tree leaves using board make/unmake and full move generation. It does not call NNUE or selective search, and it does not stop branches for search draw rules.

For a basic check, start the engine built using the [README instructions](../README.md#build) and enter:

```text
position startpos
perft 4
```

The expected count is **197,281**. Use positive depths: the current `perftPrint(0)` prints a result but does not return before continuing into recursion. Reset with `position` before a separate diagnostic, especially after a search that applies its selected move.

Starting-position counts are only a baseline. A useful position set exercises castling rights lost through rook moves/captures, attacked castling transit squares, pins, single and double check, all four promotions, en passant exposing a slider, and en passant while in check. Compare exact legal move sets as well as totals: missing and extra moves can cancel in a count. Compare `hasLegalMoves()` against whether full generation returns a nonzero count.

A per-root-move breakdown helps isolate the first divergent branch. `Engine::perftDivide()` exists, but the UCI `perft` command invokes the total-count routine. The divide helper currently retains the generator's shared move buffer across recursive calls; copy the parent moves before relying on it for diagnostics.

## State restoration and hashing

For each selected legal move, snapshot semantic board state, make the move, compare the incremental hash with `computeZobristHash()`, undo, and compare against the snapshot. Check color/type bitboards, square lookup, side, check state, castling/en-passant metadata, fifty-move counter, ply, and histories. Include nested move sequences; a single round trip can miss corruption exposed only by siblings or deeper undo.

Run null-move checks separately from real moves because repetition bookkeeping intentionally differs. Include positions with an en-passant file before the pass. Check copy construction separately from assignment; the explicit copy constructor has a different field list.

Hash recomputation is a consistency check against the engine's hash definition. It does not prove that the definition handles repetition equivalence correctly. Exercise repeated positions with different histories and en-passant availability, plus positions near the fifty-move boundary. `dumpzobrist` and `print_board` help inspect state; they are not automated equality assertions.

## NNUE agreement

`nnue_eval` rebuilds accumulators and evaluates the current search board. It is useful for inspecting a position, but rebuilding can hide an incremental-update error. Capture incremental accumulators before rebuilding, then compare every element of both perspectives against independently rebuilt values. Comparing only the final evaluation can miss differences hidden by clipping or integer division.

Cover quiet moves, captures, promotions, castling, en passant, and king moves that stay within or cross a bucket/mirror boundary. Check after make and after undo, with both sides to move. Piece-count changes should also exercise transitions between output buckets.

On Windows, `nnue_test` invokes the SIMD diagnostic, which reads positions from `../san-jacinto/bin/test_positions/perft.epd` relative to the process working directory. That external file is required; this is not a self-contained checked-in test suite. [nnue.cpp](../nnue/nnue.cpp) also contains stage comparisons and incremental/full debug helpers; the calls to the latter inside the search move helpers are currently commented out. Do not assume a normal search or Debug build automatically runs these checks. For inference changes, require scalar/SIMD agreement under the same integer arithmetic, clipping, tensor layout, and weights; inspect the first differing layer before the final score.

## Search correctness and performance

Use explicit checkmate, stalemate, in-check quiescence, repetition, and fifty-move positions before game testing. For TT changes, test exact/lower/upper-bound use at different windows and depths, and mate scores reached at different plies. A cached bound is not always an exact evaluation. The [search guide](search.md) records current limitations, including stand pat in check and mate-score storage.

For timing comparisons, fix the position set, weights, compiler and flags, CPU, hash size, and process reuse policy. Run repeated measurements and keep variability alongside the central result. Debug and `DEV` instrumentation change costs, so compare like builds; use production builds for representative speed results. Profile also includes DEBUG feature bookkeeping. There is no longer a dedicated Asan configuration: memory/undefined-behavior diagnostics require an explicitly configured sanitizer build with compile and link flags appropriate to the toolchain.

Fixed-depth runs reveal changes in both work and speed. Record elapsed time, node count, evaluation, and selected move together: equal depth is not equal work when pruning changes. For an implementation-only optimization, unexplained move/score or node-count changes need investigation. A search heuristic may legitimately change all of them.

NPS measures the cost of the nodes actually visited. A heuristic that eliminates cheap nodes can reduce NPS while reducing total time; one that searches more irrelevant nodes can raise NPS without helping play. Current `go nodes` parsing does not enforce a node budget, so do not label those runs fixed-node benchmarks. See [limit boundaries](engine.md#current-implementation-boundaries).

## Strength experiments and provenance

Choose a baseline engine and baseline net explicitly. Keep openings, color pairing, time control, hardware, hash, thread settings, tablebases, and adjudication comparable. Record games and W/D/L or paired pentanomial counts, the Elo estimate and confidence interval, and the statistical method. A small positive estimate with a wide interval is inconclusive evidence.

When using SPRT, record hypotheses, error rates, and the stopping decision configured for that experiment. Tuning and final validation should use distinct position/opening sets where practical, so selecting a parameter against one sample is not mistaken for independent confirmation. Tactical suites provide targeted regression evidence; game tests measure the combined engine under their specific conditions.

Keep a result record containing engine revision **plus any uncommitted patch**, compiler/build flags, executable identity, network SHA-256, test-runner revision, exact command/configuration, conditions, and raw logs. A revision alone cannot reproduce working-tree edits. Record whether runs use fresh processes: `ucinewgame` clears the TT but leaves the existing searcher's killer/history tables alive.

Tomahawk supplies the engine, diagnostics, and `DEV` telemetry. San Jacinto is the companion framework for orchestration, matches/SPRT, tuning, and analysis; its runner commands and external assets are not defined by this checkout. Record its actual revision and configuration rather than assuming that a Tomahawk revision fixes the entire experiment. Use the [weight profiles](../nnue/weights/readme.md#network-comparison) to retain net-specific results and provenance.
