#pragma once

#include <cstddef>
#include <string>

struct BenchmarkConfig {
    int gpus = 2;

    std::string collective = "allreduce";

    std::size_t min_bytes = 1024;
    std::size_t max_bytes = 64ULL * 1024ULL * 1024ULL;

    double factor = 2.0;

    int warmup = 20;
    int iterations = 100;
};

BenchmarkConfig parse_benchmark_args(int argc, char** argv);

int run_nccl_benchmark(const BenchmarkConfig& config);