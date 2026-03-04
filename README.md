# High-Performance Nagel-Schreckenberg Traffic Simulation

A highly optimized, multi-threaded implementation of the Nagel-Schreckenberg (NaSch) cellular automaton model for freeway traffic simulation.

This implementation achieved **1st place on the performance leaderboard** against a large pool of concurrent entries by leveraging aggressively optimized data structures, OpenMP parallelization, and memory-conscious algorithms.

## Overview

The Nagel-Schreckenberg model represents traffic flow on a single lane or multi-lane circular road. The simulation runs in discrete time steps, and cars move according to four basic rules:

1.  **Acceleration**: If the velocity $v$ of a vehicle is lower than $v_{max}$ and the distance to the next car ahead is larger than $v + 1$, the speed is advanced by 1.
2.  **Slowing down (due to other cars)**: If a vehicle at site $i$ sees the next vehicle at site $i + j$ (with $j \le v$), it reduces its speed to $j - 1$.
3.  **Randomization**: With probability $p$, the velocity of a moving vehicle ($v \ge 1$) is decreased by 1.
4.  **Car motion**: Each vehicle is advanced $v$ sites.

This specific repository extends the basic model to handle multi-lane dynamics, allowing cars to change lanes based on gap acceptance and surrounding vehicle speeds.

## Performance & Technical Highlights

To achieve peak computational efficiency, this implementation employs several advanced optimization strategies:

- **OpenMP Multithreading**: The entire discrete timestep loop is fully parallelized to distribute vehicle updates and lane changes across available CPU cores efficiently.
- **Dense vs. Sparse Lookup Modes**: The simulation dynamically toggles between a table-based dense lookup approach (saving next/previous pointers) and a standard scan algorithm depending on the calculated road density.
- **Compact Velocity Representations**: Array structures for vehicle speeds dynamically select the smallest possible primitive data type (`uint8_t`, `uint16_t`, or `int32_t`) based on $v_{max}$. This drastically reduces cache misses and memory bandwidth overhead.
- **Deferred Grid Writes**: To avoid unnecessary memory invalidation, position updates are batched and deferred when vehicles are stationary or moving predictably.
- **Custom PRNG Handling**: Deterministic but localized pseudo-random number generators ensure thread-safe randomization without introducing lock contention or synchronization barriers.

## Build and Run Instructions

### Prerequisites

- C++20 compliant compiler (e.g., GCC 10+, Clang 11+)
- OpenMP support (`-fopenmp`)
- `make` utility

### Compilation

To compile the highly optimized release (performance) build:

```bash
make sim.perf
```

For the debug build with asserts enabled:

```bash
make sim.debug
```

### Usage

After compilation, execute the binary:

```bash
./sim.perf
```

_(Ensure you have valid input data or test parameters piped into the simulation standard input when executing.)_
