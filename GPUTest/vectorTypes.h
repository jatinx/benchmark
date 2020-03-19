#pragma once

template <typename T>
__global__ void addVector(T *a, T *b, T *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

template<typename T>
static void BM_device_add(benchmark::State& state) {
  for (auto _ : state) {
    T *d_a, *d_b, *d_c;
    hipMalloc(&d_a, sizeof(T) * 32);
    hipMalloc(&d_b, sizeof(T) * 32);
    hipMalloc(&d_c, sizeof(T) * 32);

    auto kernel = addVector<T>;

    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();

    hipLaunchKernelGGL(kernel, 1, 32, 0, 0, d_a, d_b, d_c);

    BENCHMARK_GPU_STOP();

    T res[32];
    hipMemcpy(res, d_c, sizeof(T) * 32, hipMemcpyDeviceToHost);
    benchmark::DoNotOptimize(res);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
  }
}

auto BM_add_int4 = BM_device_add<int4>;
auto BM_add_int3 = BM_device_add<int3>;
auto BM_add_int2 = BM_device_add<int2>;
auto BM_add_int1 = BM_device_add<int1>;
auto BM_add_float4 = BM_device_add<float4>;
auto BM_add_float3 = BM_device_add<float3>;
auto BM_add_float2 = BM_device_add<float2>;
auto BM_add_float1 = BM_device_add<float1>;
auto BM_add_double4 = BM_device_add<double4>;
auto BM_add_double3 = BM_device_add<double3>;
auto BM_add_double2 = BM_device_add<double2>;
auto BM_add_double1 = BM_device_add<double1>;
auto BM_add_long4 = BM_device_add<long4>;
auto BM_add_long3 = BM_device_add<long3>;
auto BM_add_long2 = BM_device_add<long2>;
auto BM_add_long1 = BM_device_add<long1>;
auto BM_add_longlong4 = BM_device_add<longlong4>;
auto BM_add_longlong3 = BM_device_add<longlong3>;
auto BM_add_longlong2 = BM_device_add<longlong2>;
auto BM_add_longlong1 = BM_device_add<longlong1>;

GPUBENCHMARK(BM_add_int4);
GPUBENCHMARK(BM_add_int3);
GPUBENCHMARK(BM_add_int2);
GPUBENCHMARK(BM_add_int1);
GPUBENCHMARK(BM_add_float4);
GPUBENCHMARK(BM_add_float3);
GPUBENCHMARK(BM_add_float2);
GPUBENCHMARK(BM_add_float1);
GPUBENCHMARK(BM_add_double4);
GPUBENCHMARK(BM_add_double3);
GPUBENCHMARK(BM_add_double2);
GPUBENCHMARK(BM_add_double1);
GPUBENCHMARK(BM_add_long4);
GPUBENCHMARK(BM_add_long3);
GPUBENCHMARK(BM_add_long2);
GPUBENCHMARK(BM_add_long1);
GPUBENCHMARK(BM_add_longlong4);
GPUBENCHMARK(BM_add_longlong3);
GPUBENCHMARK(BM_add_longlong2);
GPUBENCHMARK(BM_add_longlong1);
