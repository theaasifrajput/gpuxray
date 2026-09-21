# GPUXRay Phase 1 — NCCL Benchmarking

## Objective

Establish a trustworthy baseline for GPU-to-GPU collective communication before implementing observability and diagnosis.

## Questions Phase 1 must answer

1. How does AllReduce latency scale with message size?
2. How consistent is latency across ranks?
3. How does effective bandwidth scale with message size?
4. Do different collective types show different scaling behavior?
5. Can the benchmark be repeated consistently across 2, 4, and 8 GPUs?

## Measurements

For every run:

- collective
- world size
- rank
- message size
- warm-up iterations
- measured iterations
- average latency
- effective bandwidth

## Why rank-level output?

GPUXRay's later straggler detector depends on comparing ranks participating in the same collective. Phase 1 therefore records one result per rank instead of reporting only a single aggregate number.

## Methodology

1. Initialize one NCCL communicator per rank.
2. Bind each rank to a GPU.
3. Allocate device buffers once.
4. Run warm-up collectives.
5. Run measured collectives using CUDA events.
6. Synchronize before recording the result.
7. Emit CSV output.
8. Repeat for increasing message sizes.

## Important measurement rule

The benchmark reports **effective collective bandwidth**, not raw physical network bandwidth. The formula depends on the collective and world size. Later phases can add topology-aware metrics.

## Next step

Once the benchmark is validated on the available GPU system, add:

- median and percentile latency
- run-to-run variance
- JSON output
- host/GPU metadata
- topology capture

Those additions should happen only after the baseline numbers are trusted.
