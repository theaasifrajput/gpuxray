#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t err__ = (call);                                           \
        if (err__ != ncclSuccess) {                                            \
            std::cerr << "NCCL error: " << ncclGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

struct Options {
    int gpus = 2;
    int warmup = 20;
    int iters = 100;
    std::size_t min_bytes = 1 << 10;
    std::size_t max_bytes = 1 << 26;
    double factor = 2.0;
    std::string collective = "allreduce";
};

static void usage(const char* name) {
    std::cout
        << "Usage: " << name << " [options]\\n"
        << "  --gpus N              Number of local GPUs (default: 2)\\n"
        << "  --collective NAME     allreduce|allgather|reducescatter\\n"
        << "  --min-bytes N         Minimum message size (default: 1024)\\n"
        << "  --max-bytes N         Maximum message size (default: 67108864)\\n"
        << "  --factor X            Message-size multiplier (default: 2)\\n"
        << "  --warmup N            Warm-up iterations (default: 20)\\n"
        << "  --iters N             Measured iterations (default: 100)\\n";
}

static std::size_t parse_size(const char* s) {
    std::size_t value = 0;
    std::stringstream ss(s);
    ss >> value;
    return value;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (a == "--gpus") o.gpus = std::atoi(require_value("--gpus"));
        else if (a == "--collective") o.collective = require_value("--collective");
        else if (a == "--min-bytes") o.min_bytes = parse_size(require_value("--min-bytes"));
        else if (a == "--max-bytes") o.max_bytes = parse_size(require_value("--max-bytes"));
        else if (a == "--factor") o.factor = std::atof(require_value("--factor"));
        else if (a == "--warmup") o.warmup = std::atoi(require_value("--warmup"));
        else if (a == "--iters") o.iters = std::atoi(require_value("--iters"));
        else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << a << "\\n";
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (o.gpus < 2 || o.warmup < 0 || o.iters <= 0 ||
        o.min_bytes == 0 || o.max_bytes < o.min_bytes || o.factor <= 1.0) {
        std::cerr << "Invalid benchmark options.\\n";
        std::exit(EXIT_FAILURE);
    }

    return o;
}

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

static std::size_t elements_for_collective(const Options& o, std::size_t bytes) {
    if (o.collective == "reducescatter") {
        // Each rank contributes bytes and receives bytes / world_size.
        return std::max<std::size_t>(1, bytes / sizeof(float));
    }
    return std::max<std::size_t>(1, bytes / sizeof(float));
}

static void run_collective(
    const Options& o,
    RankContext& ctx,
    std::size_t bytes,
    int iterations,
    bool warmup
) {
    const std::size_t count = elements_for_collective(o, bytes);

    for (int i = 0; i < iterations; ++i) {
        if (!warmup) CUDA_CHECK(cudaEventRecord(ctx.start, ctx.stream));

        if (o.collective == "allreduce") {
            NCCL_CHECK(ncclAllReduce(
                ctx.send, ctx.recv, count, ncclFloat, ncclSum,
                ctx.comm, ctx.stream));
        } else if (o.collective == "allgather") {
            NCCL_CHECK(ncclAllGather(
                ctx.send, ctx.recv, count, ncclFloat,
                ctx.comm, ctx.stream));
        } else if (o.collective == "reducescatter") {
            NCCL_CHECK(ncclReduceScatter(
                ctx.send, ctx.recv, count / static_cast<std::size_t>(o.gpus),
                ncclFloat, ncclSum, ctx.comm, ctx.stream));
        }

        if (!warmup) CUDA_CHECK(cudaEventRecord(ctx.stop, ctx.stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

static float measured_latency_us(
    const Options& o,
    RankContext& ctx,
    std::size_t bytes
) {
    run_collective(o, ctx, bytes, o.warmup, true);

    double total_us = 0.0;

    for (int i = 0; i < o.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(ctx.start, ctx.stream));

        const std::size_t count = elements_for_collective(o, bytes);

        if (o.collective == "allreduce") {
            NCCL_CHECK(ncclAllReduce(
                ctx.send, ctx.recv, count, ncclFloat, ncclSum,
                ctx.comm, ctx.stream));
        } else if (o.collective == "allgather") {
            NCCL_CHECK(ncclAllGather(
                ctx.send, ctx.recv, count, ncclFloat,
                ctx.comm, ctx.stream));
        } else if (o.collective == "reducescatter") {
            NCCL_CHECK(ncclReduceScatter(
                ctx.send, ctx.recv, count / static_cast<std::size_t>(o.gpus),
                ncclFloat, ncclSum, ctx.comm, ctx.stream));
        }

        CUDA_CHECK(cudaEventRecord(ctx.stop, ctx.stream));
        CUDA_CHECK(cudaEventSynchronize(ctx.stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ctx.start, ctx.stop));
        total_us += static_cast<double>(ms) * 1000.0;
    }

    return static_cast<float>(total_us / static_cast<double>(o.iters));
}

static double effective_bandwidth_gbps(
    const Options& o,
    std::size_t bytes,
    double latency_us
) {
    if (latency_us <= 0.0) return 0.0;

    // This is an effective payload bandwidth, not a claim about physical link bandwidth.
    double multiplier = 1.0;
    if (o.collective == "allreduce") {
        multiplier = 2.0 * (o.gpus - 1.0) / o.gpus;
    } else if (o.collective == "allgather") {
        multiplier = (o.gpus - 1.0) / o.gpus;
    } else if (o.collective == "reducescatter") {
        multiplier = (o.gpus - 1.0) / o.gpus;
    }

    const double seconds = latency_us / 1e6;
    const double payload_bytes = static_cast<double>(bytes) * multiplier;
    return (payload_bytes / seconds) / 1e9;
}


static void print_gpu_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "GPUXRay GPU Information\n";
    std::cout << "=======================\n";
    std::cout << "Visible GPUs: " << device_count << "\n\n";

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};

        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "GPU " << device << "\n";
        std::cout << "  Name               : " << prop.name << "\n";
        std::cout << "  Compute Capability : "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  SMs                : "
                  << prop.multiProcessorCount << "\n";
        std::cout << "  Global Memory      : "
                  << static_cast<double>(prop.totalGlobalMem) /
                         (1024.0 * 1024.0 * 1024.0)
                  << " GB\n";
        std::cout << "  Memory Bus Width   : "
                  << prop.memoryBusWidth << " bits\n";
        std::cout << "  Clock Rate         : "
                  << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  Memory Clock       : "
                  << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "info") {
        print_gpu_info();
        return EXIT_SUCCESS;
    }

    Options o = parse_args(argc, argv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count < o.gpus) {
        std::cerr << "Requested " << o.gpus << " GPUs, but only "
                  << device_count << " are visible.\\n";
        return EXIT_FAILURE;
    }

    std::vector<RankContext> ranks(o.gpus);
    ncclUniqueId id;

    NCCL_CHECK(ncclGetUniqueId(&id));

    for (int r = 0; r < o.gpus; ++r) {
        ranks[r].rank = r;
        ranks[r].device = r;

        CUDA_CHECK(cudaSetDevice(r));
        NCCL_CHECK(ncclCommInitRank(
            &ranks[r].comm, o.gpus, id, r));

        CUDA_CHECK(cudaStreamCreateWithFlags(&ranks[r].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&ranks[r].start));
        CUDA_CHECK(cudaEventCreate(&ranks[r].stop));

        const std::size_t max_count =
            elements_for_collective(o, o.max_bytes) * static_cast<std::size_t>(
                o.collective == "allgather" ? o.gpus : 1);

        CUDA_CHECK(cudaMalloc(&ranks[r].send, max_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ranks[r].recv, max_count * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(
            ranks[r].send, 0, max_count * sizeof(float), ranks[r].stream));
        CUDA_CHECK(cudaMemsetAsync(
            ranks[r].recv, 0, max_count * sizeof(float), ranks[r].stream));
        CUDA_CHECK(cudaStreamSynchronize(ranks[r].stream));
    }

    std::cout << "collective,world_size,rank,message_bytes,latency_us,effective_gbps\\n";

    for (std::size_t bytes = o.min_bytes; bytes <= o.max_bytes;) {
        std::vector<float> latency_us(o.gpus);

        for (int r = 0; r < o.gpus; ++r) {
            CUDA_CHECK(cudaSetDevice(r));
            latency_us[r] = measured_latency_us(o, ranks[r], bytes);
        }

        for (int r = 0; r < o.gpus; ++r) {
            const double gbps =
                effective_bandwidth_gbps(o, bytes, latency_us[r]);

            std::cout << o.collective << ","
                      << o.gpus << ","
                      << r << ","
                      << bytes << ","
                      << std::fixed << std::setprecision(3)
                      << latency_us[r] << ","
                      << std::setprecision(3)
                      << gbps << "\\n";
        }

        if (bytes > static_cast<std::size_t>(
                static_cast<double>(o.max_bytes) / o.factor)) {
            break;
        }
        bytes = static_cast<std::size_t>(static_cast<double>(bytes) * o.factor);
    }

    for (auto& ctx : ranks) {
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaFree(ctx.send));
        CUDA_CHECK(cudaFree(ctx.recv));
        CUDA_CHECK(cudaEventDestroy(ctx.start));
        CUDA_CHECK(cudaEventDestroy(ctx.stop));
        CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        NCCL_CHECK(ncclCommDestroy(ctx.comm));
    }

    return EXIT_SUCCESS;
}
