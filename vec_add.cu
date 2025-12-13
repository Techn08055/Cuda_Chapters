#include <cuda_runtime.h>
#include <iostream>

__global__ void VecAdd(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1000000) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
   
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}