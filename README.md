# GPUXRay — Phase 1: NCCL Benchmarking

Phase 1 establishes the performance baseline for GPUXRay.

## Objective

Build a small, reproducible NCCL benchmark that measures:

- AllReduce
- AllGather
- ReduceScatter
- message-size scaling
- warm-up vs measured iterations
- per-rank latency
- effective bandwidth

The initial target is 2 GPUs, followed by 4 and 8 GPUs.

## Requirements

- NVIDIA GPU(s)
- CUDA toolkit
- NCCL
- CMake >= 3.18
- C++17 compiler

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

For a single-node 2-GPU run:

```bash
./build/gpuxray_nccl_bench --gpus 2 --collective allreduce
```

Run a message-size sweep:

```bash
./build/gpuxray_nccl_bench \
  --gpus 2 \
  --collective allreduce \
  --min-bytes 1024 \
  --max-bytes 67108864 \
  --factor 2 \
  --warmup 20 \
  --iters 100
```

Output is CSV so Phase 2 can consume it directly.

## Phase 1 acceptance criteria

1. The benchmark initializes all requested ranks successfully.
2. Warm-up iterations are excluded from measured latency.
3. Each rank measures the same collective operation.
4. Results include message size, rank, latency, and bandwidth.
5. The same benchmark can be repeated with 2, 4, and 8 GPUs.
6. Results can be compared across message sizes.

## Important

This is a benchmark, not yet the GPUXRay observability agent. We intentionally keep Phase 1 narrow: first establish a trustworthy communication baseline.
