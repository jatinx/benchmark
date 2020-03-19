#pragma once
#include <hip/hiprtc.h>

static constexpr auto saxpy{
    R"(
    #include <hip/hip_runtime.h>

    extern "C"
    __global__
    void saxpy(float a, float* x, float* y, float* out, size_t n)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n)
           out[tid] = a * x[tid] + y[tid];
    }
    )"};


static void BM_rtc_declare(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, saxpy, "saxpy.cu", 0, nullptr, nullptr);

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {sarg.c_str()};

    benchmark::DoNotOptimize(prog);
    benchmark::DoNotOptimize(props);
    benchmark::DoNotOptimize(options);

    BENCHMARK_GPU_STOP();
  }
}
GPUBENCHMARK(BM_rtc_declare);

// Define another benchmark
static void BM_rtc_compile(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    BENCHMARK_GPU_START();

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, saxpy, "saxpy.cu", 0, nullptr, nullptr);

    hipDeviceProp_t props;
    int device = 0;
    hipGetDeviceProperties(&props, device);
    std::string gfxName = "gfx" + std::to_string(props.gcnArch);
    std::string sarg = "--gpu-architecture=" + gfxName;
    const char* options[] = {sarg.c_str()};

    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
    benchmark::DoNotOptimize(compileResult);

    BENCHMARK_GPU_STOP();
  }
}
GPUBENCHMARK(BM_rtc_compile);
