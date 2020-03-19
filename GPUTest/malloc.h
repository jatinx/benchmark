#pragma once

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
