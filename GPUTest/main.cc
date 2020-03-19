// Main file for GPU Benchmarks to run
#include "common.h"
#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>

#include "launchKernel.h"
#include "malloc.h"
#include "rtc.h"
#include "vectorTypes.h"

BENCHMARK_MAIN();
