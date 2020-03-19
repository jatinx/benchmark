#pragma once
__global__ void add(int* a) { *a += 3; }

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
