#include "gpu/gpu_info.h"
#include "p2p/p2p.h"
#include "nccl/nccl_benchmark.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

static void print_gpu_info() {
    const auto devices = discover_gpus();

    std::cout << "GPUXRay GPU Information\n";
    std::cout << "=======================\n\n";
    std::cout << "Visible GPUs: " << devices.size() << "\n\n";

    for (const auto& gpu : devices) {
        std::cout << "GPU " << gpu.id << "\n";
        std::cout << "  Name               : " << gpu.name << "\n";
        std::cout << "  Compute Capability : "
                  << gpu.compute_major << "."
                  << gpu.compute_minor << "\n";
        std::cout << "  SMs                : "
                  << gpu.sm_count << "\n";
        std::cout << "  Global Memory      : "
                  << static_cast<double>(gpu.global_memory_bytes) /
                         (1024.0 * 1024.0 * 1024.0)
                  << " GB\n";
        std::cout << "  Memory Bus Width   : "
                  << gpu.memory_bus_width_bits << " bits\n";
        std::cout << "  Clock Rate         : "
                  << gpu.clock_mhz << " MHz\n";
        std::cout << "  Memory Clock       : "
                  << gpu.memory_clock_mhz << " MHz\n\n";
    }
}

static void print_p2p_matrix() {
    const auto links = discover_p2p_links();

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout << "GPUXRay P2P Capability\n";
    std::cout << "======================\n\n";
    std::cout << "Visible GPUs: " << device_count << "\n\n";

    if (device_count < 2) {
        std::cout << "P2P testing requires at least 2 visible GPUs.\n";
        return;
    }

    std::cout << "      ";

    for (int gpu = 0; gpu < device_count; ++gpu) {
        std::cout << "GPU" << gpu << "   ";
    }

    std::cout << "\n";

    for (int source = 0; source < device_count; ++source) {
        std::cout << "GPU" << source << "  ";

        for (int destination = 0;
             destination < device_count;
             ++destination) {

            if (source == destination) {
                std::cout << "  --   ";
                continue;
            }

            const auto it = std::find_if(
                links.begin(),
                links.end(),
                [source, destination](const P2PLink& link) {
                    return link.source == source &&
                           link.destination == destination;
                }
            );

            std::cout << (
                it != links.end() && it->supported
                    ? " YES   "
                    : " NO    "
            );
        }

        std::cout << "\n";
    }
}

int main(int argc, char** argv) {

    if (argc > 1 && std::string(argv[1]) == "info") {
        print_gpu_info();
        return EXIT_SUCCESS;
    }

    if (argc > 1 && std::string(argv[1]) == "p2p") {
        print_p2p_matrix();
        return EXIT_SUCCESS;
    }

    BenchmarkConfig config = parse_benchmark_args(argc, argv);

    return run_nccl_benchmark(config);
}