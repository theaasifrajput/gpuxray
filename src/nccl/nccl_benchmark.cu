#include "nccl/nccl_benchmark.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define NCCL_CHECK(call)                                                   \
    do {                                                                   \
        ncclResult_t err__ = (call);                                       \
        if (err__ != ncclSuccess) {                                        \
            std::cerr << "NCCL error: "                                    \
                      << ncclGetErrorString(err__)                          \
                      << " at " << __FILE__ << ":" << __LINE__             \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)


struct RankContext {
    int rank = 0;
    int device = 0;

    ncclComm_t comm{};
    cudaStream_t stream{};

    float* send = nullptr;
    float* recv = nullptr;

    cudaEvent_t start{};
    cudaEvent_t stop{};
};


// ------------------------------------------------------------
// Argument parsing
// ------------------------------------------------------------

BenchmarkConfig parse_benchmark_args_impl(int argc, char** argv) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {

        const std::string arg = argv[i];

        if (arg == "--gpus" && i + 1 < argc) {
            config.gpus = std::stoi(argv[++i]);
        }
        else if (arg == "--collective" && i + 1 < argc) {
            config.collective = argv[++i];
        }
        else if (arg == "--min-bytes" && i + 1 < argc) {
            config.min_bytes =
                static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--max-bytes" && i + 1 < argc) {
            config.max_bytes =
                static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--factor" && i + 1 < argc) {
            config.factor = std::stod(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        }
        else if (arg == "--iters" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (config.collective != "allreduce" &&
        config.collective != "allgather" &&
        config.collective != "reducescatter") {

        std::cerr << "Unsupported collective: "
                  << config.collective << "\n";

        std::exit(EXIT_FAILURE);
    }

    if (config.gpus < 2 ||
        config.warmup < 0 ||
        config.iterations <= 0 ||
        config.min_bytes == 0 ||
        config.max_bytes < config.min_bytes ||
        config.factor <= 1.0) {

        std::cerr << "Invalid benchmark options.\n";
        std::exit(EXIT_FAILURE);
    }

    return config;
}


// ------------------------------------------------------------
// Collective helpers
// ------------------------------------------------------------

static std::size_t elements_for_collective(
    const BenchmarkConfig& config,
    std::size_t bytes
) {
    if (config.collective == "reducescatter") {
        return std::max<std::size_t>(
            1,
            bytes / sizeof(float)
        );
    }

    return std::max<std::size_t>(
        1,
        bytes / sizeof(float)
    );
}


static void run_collective(
    const BenchmarkConfig& config,
    RankContext& ctx,
    std::size_t bytes,
    int iterations,
    bool warmup
) {
    const std::size_t count =
        elements_for_collective(config, bytes);

    for (int i = 0; i < iterations; ++i) {

        if (!warmup) {
            CUDA_CHECK(
                cudaEventRecord(ctx.start, ctx.stream)
            );
        }

        if (config.collective == "allreduce") {

            NCCL_CHECK(
                ncclAllReduce(
                    ctx.send,
                    ctx.recv,
                    count,
                    ncclFloat,
                    ncclSum,
                    ctx.comm,
                    ctx.stream
                )
            );

        } else if (config.collective == "allgather") {

            NCCL_CHECK(
                ncclAllGather(
                    ctx.send,
                    ctx.recv,
                    count,
                    ncclFloat,
                    ctx.comm,
                    ctx.stream
                )
            );

        } else if (config.collective == "reducescatter") {

            NCCL_CHECK(
                ncclReduceScatter(
                    ctx.send,
                    ctx.recv,
                    count / static_cast<std::size_t>(
                        config.gpus
                    ),
                    ncclFloat,
                    ncclSum,
                    ctx.comm,
                    ctx.stream
                )
            );
        }

        if (!warmup) {
            CUDA_CHECK(
                cudaEventRecord(ctx.stop, ctx.stream)
            );
        }
    }

    CUDA_CHECK(
        cudaStreamSynchronize(ctx.stream)
    );
}


// ------------------------------------------------------------
// Measurement
// ------------------------------------------------------------

static float measured_latency_us(
    const BenchmarkConfig& config,
    RankContext& ctx,
    std::size_t bytes
) {
    run_collective(
        config,
        ctx,
        bytes,
        config.warmup,
        true
    );

    double total_us = 0.0;

    for (int i = 0; i < config.iterations; ++i) {

        CUDA_CHECK(
            cudaEventRecord(ctx.start, ctx.stream)
        );

        const std::size_t count =
            elements_for_collective(config, bytes);

        if (config.collective == "allreduce") {

            NCCL_CHECK(
                ncclAllReduce(
                    ctx.send,
                    ctx.recv,
                    count,
                    ncclFloat,
                    ncclSum,
                    ctx.comm,
                    ctx.stream
                )
            );

        } else if (config.collective == "allgather") {

            NCCL_CHECK(
                ncclAllGather(
                    ctx.send,
                    ctx.recv,
                    count,
                    ncclFloat,
                    ctx.comm,
                    ctx.stream
                )
            );

        } else if (config.collective == "reducescatter") {

            NCCL_CHECK(
                ncclReduceScatter(
                    ctx.send,
                    ctx.recv,
                    count / static_cast<std::size_t>(
                        config.gpus
                    ),
                    ncclFloat,
                    ncclSum,
                    ctx.comm,
                    ctx.stream
                )
            );
        }

        CUDA_CHECK(
            cudaEventRecord(ctx.stop, ctx.stream)
        );

        CUDA_CHECK(
            cudaEventSynchronize(ctx.stop)
        );

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                ctx.start,
                ctx.stop
            )
        );

        total_us +=
            static_cast<double>(ms) * 1000.0;
    }

    return static_cast<float>(
        total_us /
        static_cast<double>(config.iterations)
    );
}


static double effective_bandwidth_gbps(
    const BenchmarkConfig& config,
    std::size_t bytes,
    double latency_us
) {
    if (latency_us <= 0.0) {
        return 0.0;
    }

    double multiplier = 1.0;

    if (config.collective == "allreduce") {

        multiplier =
            2.0 *
            (config.gpus - 1.0) /
            config.gpus;

    } else if (config.collective == "allgather") {

        multiplier =
            (config.gpus - 1.0) /
            config.gpus;

    } else if (config.collective == "reducescatter") {

        multiplier =
            (config.gpus - 1.0) /
            config.gpus;
    }

    const double seconds =
        latency_us / 1e6;

    const double payload_bytes =
        static_cast<double>(bytes) * multiplier;

    return (payload_bytes / seconds) / 1e9;
}

} // namespace


// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

BenchmarkConfig parse_benchmark_args(
    int argc,
    char** argv
) {
    return parse_benchmark_args_impl(argc, argv);
}


int run_nccl_benchmark(
    const BenchmarkConfig& config
) {
    int device_count = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&device_count)
    );

    if (device_count < config.gpus) {

        std::cerr
            << "Requested "
            << config.gpus
            << " GPUs, but only "
            << device_count
            << " are visible.\n";

        return EXIT_FAILURE;
    }

    std::vector<RankContext> ranks(
        config.gpus
    );

    ncclUniqueId id;

    NCCL_CHECK(
        ncclGetUniqueId(&id)
    );

    // --------------------------------------------------------
    // Initialize NCCL ranks
    // --------------------------------------------------------

    for (int r = 0; r < config.gpus; ++r) {

        ranks[r].rank = r;
        ranks[r].device = r;

        CUDA_CHECK(
            cudaSetDevice(r)
        );

        NCCL_CHECK(
            ncclCommInitRank(
                &ranks[r].comm,
                config.gpus,
                id,
                r
            )
        );

        CUDA_CHECK(
            cudaStreamCreateWithFlags(
                &ranks[r].stream,
                cudaStreamNonBlocking
            )
        );

        CUDA_CHECK(
            cudaEventCreate(&ranks[r].start)
        );

        CUDA_CHECK(
            cudaEventCreate(&ranks[r].stop)
        );

        const std::size_t max_count =
            elements_for_collective(
                config,
                config.max_bytes
            ) *
            static_cast<std::size_t>(
                config.collective == "allgather"
                    ? config.gpus
                    : 1
            );

        CUDA_CHECK(
            cudaMalloc(
                &ranks[r].send,
                max_count * sizeof(float)
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &ranks[r].recv,
                max_count * sizeof(float)
            )
        );

        CUDA_CHECK(
            cudaMemsetAsync(
                ranks[r].send,
                0,
                max_count * sizeof(float),
                ranks[r].stream
            )
        );

        CUDA_CHECK(
            cudaMemsetAsync(
                ranks[r].recv,
                0,
                max_count * sizeof(float),
                ranks[r].stream
            )
        );

        CUDA_CHECK(
            cudaStreamSynchronize(
                ranks[r].stream
            )
        );
    }

    // --------------------------------------------------------
    // Benchmark output
    // --------------------------------------------------------

    std::cout
        << "collective,world_size,rank,"
           "message_bytes,latency_us,effective_gbps\n";

    for (
        std::size_t bytes = config.min_bytes;
        bytes <= config.max_bytes;
    ) {

        std::vector<float> latency_us(
            config.gpus
        );

        for (int r = 0; r < config.gpus; ++r) {

            CUDA_CHECK(
                cudaSetDevice(r)
            );

            latency_us[r] =
                measured_latency_us(
                    config,
                    ranks[r],
                    bytes
                );
        }

        for (int r = 0; r < config.gpus; ++r) {

            const double gbps =
                effective_bandwidth_gbps(
                    config,
                    bytes,
                    latency_us[r]
                );

            std::cout
                << config.collective << ","
                << config.gpus << ","
                << r << ","
                << bytes << ","
                << std::fixed
                << std::setprecision(3)
                << latency_us[r] << ","
                << std::setprecision(3)
                << gbps
                << "\n";
        }

        if (bytes > config.max_bytes / config.factor) {
            break;
        }

        bytes = static_cast<std::size_t>(
            static_cast<double>(bytes) *
            config.factor
        );
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    for (int r = 0; r < config.gpus; ++r) {

        CUDA_CHECK(
            cudaSetDevice(r)
        );

        if (ranks[r].send) {
            CUDA_CHECK(
                cudaFree(ranks[r].send)
            );
        }

        if (ranks[r].recv) {
            CUDA_CHECK(
                cudaFree(ranks[r].recv)
            );
        }

        CUDA_CHECK(
            cudaEventDestroy(ranks[r].start)
        );

        CUDA_CHECK(
            cudaEventDestroy(ranks[r].stop)
        );

        CUDA_CHECK(
            cudaStreamDestroy(ranks[r].stream)
        );

        NCCL_CHECK(
            ncclCommDestroy(ranks[r].comm)
        );
    }

    return EXIT_SUCCESS;
}