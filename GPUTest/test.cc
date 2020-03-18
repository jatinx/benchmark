#define ENABLEGPU
#include <benchmark/benchmark.h>

__global__ void add(int* a) { *a += 3; }

static void BM_CudaMalloc(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();
    int *d_a;
    hipMalloc(&d_a, sizeof(int));
    hipFree(d_a);
    BENCHMARK_GPU_STOP();
  }
}
// Register the function as a benchmark
GPUBENCHMARK(BM_CudaMalloc);

// Define another benchmark
static void BM_LaunchKernel(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();
    int *d_a;
    hipMalloc(&d_a, sizeof(int));
    add<<<1,1>>>(d_a);
    hipFree(d_a);
    BENCHMARK_GPU_STOP();
  }
}
GPUBENCHMARK(BM_LaunchKernel);

BENCHMARK_MAIN();
