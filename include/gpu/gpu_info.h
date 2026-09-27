#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct GpuDevice {
    int id;
    std::string name;

    int compute_major;
    int compute_minor;

    int sm_count;

    std::size_t global_memory_bytes;

    int memory_bus_width_bits;

    int clock_mhz;
    int memory_clock_mhz;
};

std::vector<GpuDevice> discover_gpus();