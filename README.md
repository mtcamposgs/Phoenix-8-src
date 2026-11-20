<div align="center">
  <img src="assets/phoenix-icon.png" alt="Phoenix Chess Engine icon" width="180">
  <h1>Phoenix</h1>
  <p><strong>Phoenix</strong> · A UCI chess engine written in C</p>
  <p>
    <a href="https://github.com/mtcamposgs/Phoenix-Chess-Engine">Source code</a>
  </p>
</div>

---

## Overview

**Phoenix** is a chess engine written in C by **Matheus Campos**. It communicates through the Universal Chess Interface (UCI), so it can be used with a UCI-compatible chess GUI or driven from a terminal.

The engine combines bitboard-based move generation, alpha-beta search techniques, a transposition table, Zobrist hashing, and a quantized neural evaluation network called **SHRA**. The evaluation network is embedded into the executable by the supplied assembly file and weight data.

Phoenix is a developing engine.

## Features

- UCI communication for use with compatible chess GUIs.
- Bitboard board representation and magic-bitboard sliding-piece attacks.
- Legal move handling, including castling, en passant, and promotions.
- Iterative deepening with Negamax and alpha-beta pruning.
- Principal Variation Search (PVS) and aspiration windows.
- Quiescence search and Static Exchange Evaluation (SEE).
- Search heuristics including transposition-table move ordering, killer moves, history heuristics, null-move pruning, late-move reductions, and singular extensions.
- Zobrist hashing and a configurable transposition table.
- Multi-threaded search through OpenMP.
- Quantized SHRA neural evaluation with incremental accumulator updates and AVX2 operations.
- Embedded default network, with an optional `EvalFile` UCI option for a compatible external network.
- Built-in `bench` command covering six positions.

## Requirements

The current Makefile is configured for an **x86-64 CPU** and enables these instruction sets unconditionally:

- AVX2
- FMA
- POPCNT
- BMI1
- BMI2

You also need:

- GCC (or a compatible GNU toolchain).
- GNU Make.
- OpenMP support in the compiler and runtime.
- A toolchain capable of assembling `weights_embed.S`.

**Important:** a CPU without the required instruction-set support may fail to run the resulting executable, potentially with an illegal-instruction error. The current Makefile is not a portable, runtime-dispatched build for arbitrary CPUs.

## Build

Clone the repository and compile it from its root directory:

```bash
git clone https://github.com/mtcamposgs/Phoenix-Chess-Engine.git
cd Phoenix-Chess-Engine
make
```

The Makefile currently declares the output target as:

```text
phoenix-windows-x86-64-avx2
```

Keep `shra_weights.bin` and `weights_embed.S` in the repository root: the build uses both to embed the default evaluation network.

To remove generated object files and the executable before rebuilding:

```bash
make clean
make
```

The supplied Makefile uses GCC flags such as `-O3`, `-mavx2`, `-mfma`, `-fopenmp`, and `-flto`. Other operating systems or toolchains may require Makefile adjustments; no cross-platform build matrix is published in this repository.

## UCI options

The engine advertises the following options:

| Option | Default | Range / purpose |
| --- | ---: | --- |
| `Threads` | `1` | Number of search threads; 1–32 |
| `Hash` | `128` MB | Transposition-table size; 1–16384 MB |
| `Clear Hash` | — | Clears the transposition table |
| `Move Overhead` | `30` ms | Time reserved as move overhead; 0–5000 ms |
| `EvalFile` | `<internal>` | Loads an external SHRA network file compatible with the engine's expected binary layout |

The default neural network is embedded in the executable. An external `EvalFile` is optional and must match the network format expected by this version of Phoenix.

## Built-in benchmark

Phoenix includes a small benchmark command that searches six built-in FEN positions:

```text
bench
bench 12
```

The default benchmark depth is 10. The accepted depth is clamped to the range 1–20. The command reports the elapsed time for the six-position run.

This built-in benchmark is useful for checking that a build runs and for rough local comparisons. It is **not** an independently verified Elo rating, a standardized engine-strength test, or a substitute for a controlled match-testing framework.

## Training data

The SHRA evaluation network was trained using chess position evaluations from the [Lichess Chess Position Evaluations dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations). The dataset is made available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## License

Phoenix Chess Engine is licensed under the **GNU General Public License v3.0**. See the [`LICENSE`](LICENSE) file for details.

## Author

**Matheus Campos**  
Engine: **Phoenix**  
Latest version released: 9

- Repository: [mtcamposgs/Phoenix-Chess-Engine](https://github.com/mtcamposgs/Phoenix-Chess-Engine)
