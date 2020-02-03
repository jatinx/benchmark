#define ENABLEGPU
#include <benchmark/benchmark.h>

__global__ void add(int* a) { *a += 3; }

static void BM_CudaMalloc(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_DECLARE();
    BENCHMARK_GPU_PRE_KERNEL();
    int *d_a;
    cudaMalloc(&d_a, sizeof(int));
    cudaFree(d_a);
    BENCHMARK_GPU_POST_KERNEL();
    BENCHMARK_GPU_SET_TIME();
    BENCHMARK_GPU_CLEANUP();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_CudaMalloc)->UseManualTime();

// Define another benchmark
static void BM_LaunchKernel(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_DECLARE();
    BENCHMARK_GPU_PRE_KERNEL();
    int *d_a;
    cudaMalloc(&d_a, sizeof(int));
    add<<<1,1>>>(d_a);
    cudaFree(d_a);
    BENCHMARK_GPU_POST_KERNEL();
    BENCHMARK_GPU_SET_TIME();
    BENCHMARK_GPU_CLEANUP();
  }
}
BENCHMARK(BM_LaunchKernel)->UseManualTime();

BENCHMARK_MAIN();
