#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

#define TILE 32   // tile size for shared memory kernel

// --------------------------
// CPU TRANSPOSE (BASELINE)
// --------------------------
void transposeCPU(const float* A, float* B, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

// --------------------------
// GPU NAIVE KERNEL
// --------------------------
__global__ void transposeNaive(float* B, const float* A, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        B[x * N + y] = A[y * N + x];
    }
}

// --------------------------
// GPU TILE + SHARED MEMORY
// (+1 for bank conflict avoidance)
// --------------------------
__global__ void transposeTiled(float* B, const float* A, int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load into shared memory
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];

    __syncthreads();

    // Transposed indices
    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;

    if (tx < N && ty < N)
        B[ty * N + tx] = tile[threadIdx.x][threadIdx.y];
}

// --------------------------
// GPU LAUNCH WRAPPERS
// --------------------------
float benchmarkNaive(float* dB, const float* dA, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeNaive<<<grid, block>>>(dB, dA, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

float benchmarkTiled(float* dB, const float* dA, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeTiled<<<grid, block>>>(dB, dA, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// --------------------------
// VERIFICATION FUNCTION
// --------------------------
bool verify(const float* A, const float* B, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (B[j * N + i] != A[i * N + j])
                return false;
    return true;
}

// --------------------------
// MAIN
// --------------------------
int main(int argc, char** argv)
{
    int N = (argc > 1) ? std::stoi(argv[1]) : 2048;

    std::cout << "\nMatrix Transpose Benchmark\n";
    std::cout << "Size: " << N << " x " << N << "\n";

    size_t bytes = N * N * sizeof(float);

    // Host allocation
    std::vector<float> h_A(N * N), h_B_cpu(N * N), h_B_gpu(N * N);

    // Initialize A
    for (int i = 0; i < N * N; i++)
        h_A[i] = float(i % 100);

    // --------------------------
    // CPU BENCHMARK
    // --------------------------
    auto start = std::chrono::high_resolution_clock::now();
    transposeCPU(h_A.data(), h_B_cpu.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\nCPU transpose: " << cpu_ms << " ms\n";

    // --------------------------
    // GPU ALLOCATION
    // --------------------------
    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);

    // --------------------------
    // GPU NAIVE BENCHMARK
    // --------------------------
    float naive_ms = benchmarkNaive(d_B, d_A, N);

    cudaMemcpy(h_B_gpu.data(), d_B, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU naive transpose: " << naive_ms << " ms ";
    std::cout << (verify(h_A.data(), h_B_gpu.data(), N) ? "(OK)" : "(FAIL)") << "\n";

    // --------------------------
    // GPU TILED BENCHMARK
    // --------------------------
    float tiled_ms = benchmarkTiled(d_B, d_A, N);

    cudaMemcpy(h_B_gpu.data(), d_B, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU tiled transpose: " << tiled_ms << " ms ";
    std::cout << (verify(h_A.data(), h_B_gpu.data(), N) ? "(OK)" : "(FAIL)") << "\n";

    // --------------------------
    // SPEEDUP STATS
    // --------------------------
    std::cout << "\nSpeedups:\n";
    std::cout << "Naive GPU vs CPU: " << cpu_ms / naive_ms << "x\n";
    std::cout << "Tiled GPU vs CPU: " << cpu_ms / tiled_ms << "x\n";
    std::cout << "Tiled vs Naive GPU: " << naive_ms / tiled_ms << "x\n";

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
