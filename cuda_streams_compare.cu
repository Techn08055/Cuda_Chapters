#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)        // ~16M elements
#define BLOCK 256
#define STREAMS 4

__global__ void square_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * in[idx];
    }
}

float run_single_stream(float* h_in, float* h_out, int n) {
    float *d_in, *d_out;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    int grid = (n + BLOCK - 1) / BLOCK;
    square_kernel<<<grid, BLOCK>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float run_multi_stream(float* h_in, float* h_out, int n) {
    int chunk = n / STREAMS;
    size_t chunk_bytes = chunk * sizeof(float);

    float *d_in[STREAMS], *d_out[STREAMS];
    cudaStream_t streams[STREAMS];

    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_in[i], chunk_bytes);
        cudaMalloc(&d_out[i], chunk_bytes);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < STREAMS; i++) {
        int offset = i * chunk;

        cudaMemcpyAsync(
            d_in[i],
            h_in + offset,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[i]
        );

        int grid = (chunk + BLOCK - 1) / BLOCK;
        square_kernel<<<grid, BLOCK, 0, streams[i]>>>(
            d_in[i], d_out[i], chunk
        );

        cudaMemcpyAsync(
            h_out + offset,
            d_out[i],
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    for (int i = 0; i < STREAMS; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    size_t bytes = N * sizeof(float);

    // Pinned host memory (important!)
    float *h_in, *h_out;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    float t1 = run_single_stream(h_in, h_out, N);
    float t2 = run_multi_stream(h_in, h_out, N);

    printf("Single stream time: %.3f ms\n", t1);
    printf("Multi-stream time : %.3f ms\n", t2);
    printf("Speedup           : %.2fx\n", t1 / t2);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
