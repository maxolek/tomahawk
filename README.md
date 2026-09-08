<div align="center">

<img src="bin/logos/logo.png" width="200">

# Tomahawk

### A NNUE chess engine written in C++

[![UCI](https://img.shields.io/badge/protocol-UCI-green.svg)]()

[![C++](https://img.shields.io/badge/C++-20-blue.svg)]()

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()

<br>

Play against Tomahawk!

https://lichess.org/@/tomahawkBOT

<br>

The development, testing, tuning, and analytics framework used for Tomahawk can be found in the sister repository:

**San Jacinto**

</div>

---

# Overview

Tomahawk is a UCI chess engine written in C++ focused on high-performance search, neural network evaluation, and continuous improvement through data-driven development.

The engine combines modern alpha-beta search techniques with NNUE evaluation to provide a strong and efficient chess playing system.

Development is supported by the companion framework **San Jacinto**, which provides automated testing, SPRT analysis, parameter tuning, telemetry processing, and analytics.

---

# Features

## Search

Tomahawk implements a modern chess search framework including:

- Iterative deepening
- Alpha-beta search
- Aspiration windows
- Principal Variation Search (PVS)
- Late Move Reductions (LMR)
- Null Move Pruning (NMP)
- Futility Pruning
- Reverse Futility Pruning (RFP)
- Move ordering heuristics
- Quiescence search

Search improvements are validated through automated strength testing using SPRT.

---

## NNUE Evaluation

Tomahawk uses a neural network evaluation system based on NNUE architecture.

Features include:

- Incremental accumulator updates
    - Compute first forward weight pass (pre-activation)
- Network
    - 3 hidden layers
    - crelu + screlu activations
    - pairwise multiplication + concatination (hidden 1)
- Efficient CPU inference (SIMD)

Neural network files are stored within:

```
/nnue/
```

---

# Repository Structure

```
Tomahawk/
│
├── src/
│   └── Engine source code
|
├── nnue/
│   └── NNUE source code, weights, and SIMD
|
├── include/
│   └── Header files
│
├── bin/
│   └── Runtime assets and resources (including nnue weights)
│
├── docs/
│   └── Documentation
│
└── CMakeLists.txt
```

---

# Development Workflow

Tomahawk is developed alongside the San Jacinto testing framework.

```
Modify Engine
      |
      v
Build Tomahawk
      |
      v
Run San Jacinto Tests
      |
      v
Analyze Results
      |
      v
Tune Parameters
```

San Jacinto provides:

- SPRT strength testing
- Automated tournaments
- SPSA/CMA-ES tuning
- Engine telemetry processing
- Search analysis
- Performance dashboards

---

# Building

## Requirements

- C++20 compatible compiler
- CMake
- Ninja or Make

## Build

Example:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

The resulting executable is a standard UCI chess engine.

---

# Playing

Tomahawk communicates through the Universal Chess Interface (UCI) protocol.

Compatible GUIs include:

- Arena
- Cute Chess
- Banksia GUI
- Lichess Bot

---

# Documentation

Additional information can be found in:

```
/docs/
```

including:

- architecture notes
- search details
- development notes
- release information

---

# Related Projects

## San Jacinto

Development framework for Tomahawk.

Provides:

- automated testing
- SPRT evaluation
- tuning infrastructure
- analytics pipelines
- visualization tools

---

<div align="center">

Built with C++ and a lot of chess.

</div>