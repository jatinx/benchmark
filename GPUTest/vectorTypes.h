#pragma once

template <typename T>
__global__ void addVector(T *a, T *b, T* c) {
  *c = *a + *b;
}

// Define another benchmark
static void BM_host_add_int4(benchmark::State& state) {
  for (auto _ : state) {
    int4 h_a, h_b, h_c;
    h_a = h_b + h_c;
    benchmark::DoNotOptimize(h_a);
  }
}
BENCHMARK(BM_host_add_int4);

static void BM_device_add_int4(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    
    int4 *d_a, *d_b, *d_c;
    hipMalloc(&d_a, sizeof(int4));
    hipMalloc(&d_b, sizeof(int4));
    hipMalloc(&d_c, sizeof(int4));

    BENCHMARK_GPU_START();

    addVector<<<1, 1>>>(d_a, d_b, d_c);

    BENCHMARK_GPU_STOP();

    int4 res;
    hipMemcpy(&res, d_c, sizeof(int4), hipMemcpyDeviceToHost);
    benchmark::DoNotOptimize(res);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
  }
}
GPUBENCHMARK(BM_device_add_int4);
