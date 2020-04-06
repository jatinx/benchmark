#pragma once
__global__ void add(int* a) { *a += 3; }

// Define another benchmark
static void BM_LaunchKernel(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    int *d_a;
    hipMalloc(&d_a, sizeof(int));
    hipMemset(d_a, 0, sizeof(int));
    BENCHMARK_GPU_START();
    hipLaunchKernelGGL(add, 1, 1, 0, 0, d_a);
    BENCHMARK_GPU_STOP();
    hipFree(d_a);
  }
}
GPUBENCHMARK(BM_LaunchKernel);

static void BM_LaunchKernel_B(benchmark::State& state) {
  BENCHMARK_GPU_DECLARE();
  for (auto _ : state) {
    int* d_a;
    hipMalloc(&d_a, sizeof(int));
    hipMemset(d_a, 0, sizeof(int));
    BENCHMARK_GPU_BEGIN();
    hipLaunchKernelGGL(add, 1, 1, 0, 0, d_a);
    BENCHMARK_GPU_END();
    hipFree(d_a);
  }
  BENCHMARK_GPU_CLEANUP();
}
GPUBENCHMARK(BM_LaunchKernel_B);
