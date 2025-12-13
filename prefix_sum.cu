#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

#define CHECK_CUDA(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " at line " << __LINE__ << std::endl; exit(1); }

// -----------------------------------------------------------
// GPU KERNELS (same as before)
// -----------------------------------------------------------
__global__ void upsweep(int *data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride * 2 - 1 < n) {
        data[idx + stride * 2 - 1] += data[idx + stride - 1];
    }
}

__global__ void downsweep(int *data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride * 2 - 1 < n) {
        int t = data[idx + stride - 1];
        data[idx + stride - 1] = data[idx + stride * 2 - 1];
        data[idx + stride * 2 - 1] += t;
    }
}

// Convert Exclusive → Inclusive
__global__ void make_inclusive(int *output, int *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] += input[idx];
    }
}

// -----------------------------------------------------------
// CPU inclusive prefix sum
// -----------------------------------------------------------
void cpu_prefix_sum(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main() {
    // --------------------------------------------------------------------
    // CHANGE THIS VALUE FOR BENCHMARKING
    // MUST be power of 2 for this Blelloch version
    // Examples: 1<<20 = 1M, 1<<24 = 16M, 1<<25 = 33M, etc.
    // --------------------------------------------------------------------
    int n = 1 << 20;  // 1 million elements

    std::cout << "\nBenchmarking prefix sum for n = " << n << " elements\n";

    // Random input
    std::vector<int> h_data(n);
    std::vector<int> cpu_out(n);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(1, 10);

    for (int i = 0; i < n; i++) h_data[i] = dist(rng);

    // -----------------------------------------------------------
    // CPU Benchmark
    // -----------------------------------------------------------
    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_prefix_sum(h_data, cpu_out);
    auto t2 = std::chrono::high_resolution_clock::now();

    double cpu_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double cpu_gbs = (n * sizeof(int)) / (cpu_us * 1e-6) / 1e9;

    std::cout << "CPU Time: " << cpu_us << " us,  Throughput: "
              << cpu_gbs << " GB/s\n";

    // -----------------------------------------------------------
    // GPU Setup
    // -----------------------------------------------------------
    int *d_data, *d_input;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input, h_data.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ---------------- UPSWEEP ----------------
    for (int stride = 1; stride < n; stride *= 2) {
        upsweep<<<grid, block>>>(d_data, n, stride);
        cudaDeviceSynchronize();
    }

    int zero = 0;
    cudaMemcpy(d_data + n - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // ---------------- DOWNSWEEP ----------------
    for (int stride = n/2; stride >= 1; stride /= 2) {
        downsweep<<<grid, block>>>(d_data, n, stride);
        cudaDeviceSynchronize();
    }

    // Convert exclusive → inclusive
    make_inclusive<<<grid, block>>>(d_data, d_input, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    double gpu_gbs = (n * sizeof(int)) / (gpu_ms / 1000.0) / 1e9;

    std::vector<int> gpu_out(n);
    cudaMemcpy(gpu_out.data(), d_data, n*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU Time: " << gpu_ms * 1000 << " us,  Throughput: "
              << gpu_gbs << " GB/s\n";

    cudaFree(d_data);
    cudaFree(d_input);
}
