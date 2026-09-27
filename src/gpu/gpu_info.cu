#include "gpu/gpu_info.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::cerr << "CUDA error: "                                    \
                      << cudaGetErrorString(err__)                          \
                      << " at " << __FILE__ << ":" << __LINE__             \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

} // namespace

std::vector<GpuDevice> discover_gpus() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::vector<GpuDevice> devices;
    devices.reserve(device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};

        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        GpuDevice gpu;

        gpu.id = device;
        gpu.name = prop.name;

        gpu.compute_major = prop.major;
        gpu.compute_minor = prop.minor;

        gpu.sm_count = prop.multiProcessorCount;

        gpu.global_memory_bytes = prop.totalGlobalMem;

        gpu.memory_bus_width_bits = prop.memoryBusWidth;

        gpu.clock_mhz = prop.clockRate / 1000;
        gpu.memory_clock_mhz = prop.memoryClockRate / 1000;

        devices.push_back(std::move(gpu));
    }

    return devices;
}