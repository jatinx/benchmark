#pragma once

static void BM_hip_malloc(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();
    int *d_a;
    hipMalloc(&d_a, sizeof(int));
    BENCHMARK_GPU_STOP();
    hipFree(d_a);
  }
}
GPUBENCHMARK(BM_hip_malloc);

static void BM_hip_memcpy_h2d(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    int size = 80;
    int *h_a = new int[size];
    int *d_a;
    hipMalloc(&d_a, sizeof(int) * size);
    BENCHMARK_GPU_START();
    hipMemcpy(d_a, h_a, sizeof(int) * size, hipMemcpyHostToDevice);
    BENCHMARK_GPU_STOP();
    hipFree(d_a);
    delete[] h_a;
  }
}
GPUBENCHMARK(BM_hip_memcpy_h2d);

static void BM_hip_memcpy_d2h(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    int size = 80;
    int *h_a = new int[size];
    int *d_a;
    hipMalloc(&d_a, sizeof(int) * size);
    BENCHMARK_GPU_START();
    hipMemcpy(h_a, d_a, sizeof(int) * size, hipMemcpyDeviceToHost);
    BENCHMARK_GPU_STOP();
    hipFree(d_a);
    delete[] h_a;
  }
}
GPUBENCHMARK(BM_hip_memcpy_d2h);
