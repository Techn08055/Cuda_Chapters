# CUDA Fundamentals — Phase 1 Foundations

This directory contains the core implementations and benchmarks for **Phase 1: Foundations** of the AI Systems Engineer curriculum.

The goal of this phase is to build a mental model of compute pipelines, memory movement, and GPU architecture.

---

## 🗓️ Phase 1 Roadmap & Implementations

### Week 1: Environment + CUDA Fundamentals
- **Objective:** Write first GPU kernel and compare with CPU.
- **Implementations:**
    - [vec_add.cu](./vec_add.cu): C++ Kernel adding 1M elements using 256 threads/block.
    - [ved_add.py](./ved_add.py): Python (Numba) equivalent adding 10M elements to demonstrate Python-to-GPU integration.
- **Theory:** High-level abstraction of the `cudaMalloc` -> `cudaMemcpy` -> `Kernel<<<>>>` -> `cudaMemcpy` flow.

### Week 2: Profiling + Benchmarking
- **Objective:** Measure performance accurately, identify bottlenecks.
- **Implementations:**
    - [prefix_sum.cu](./prefix_sum.cu): Parallel **Blelloch Scan** (Work-Efficient) implementation using Upsweep and Downsweep phases.
    - [Profiling Reports](./report1.nsys-rep): Nsight Systems reports visualizing the timeline of memmoves vs compute.
- **Theory:** Benchmark $2^{20}$ elements. Notice how `cudaDeviceSynchronize` is used between scan phases to ensure global consistency.

### Week 3: Memory Hierarchy
- **Objective:** Understand Global, Shared, and Register memory.
- **Implementations:**
    - [matrix_transpose.cu](./matrix_transpose.cu): Compares **Naive** vs **Tiled** (Shared Memory) transpose.
- **Theory:** 
    - **Tiling:** Uses $32 \times 32$ shared memory blocks to reorder data.
    - **Bank Conflict Avoidance:** Uses `__shared__ float tile[TILE][TILE + 1]` (padding) to ensure optimal throughput during shared memory access.

### Week 4: Streams + System Thinking
- **Objective:** Overlap computation and data transfer.
- **Implementations:**
    - [cuda_streams_compare.cu](./cuda_streams_compare.cu): Compares a single stream vs **4 asynchronous streams** for a square kernel on 16M elements.
- **Theory:** Demonstrates the necessity of **Pinned Host Memory** (`cudaMallocHost`) to achieve true H2D/Compute/D2H overlap.

### Week 5: TensorRT Introduction
- **Objective:** Inference optimization using TensorRT.
- **Implementations:**
    - [pytorch_onnx.py](../AI_list/pytorch_onnx.py): Export ResNet18 to ONNX.
    - **Commands:**
      ```bash
      # Export to ONNX
      python ../AI_list/pytorch_onnx.py
      
      # Build engine via trtexec
      trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.engine --fp16
      ```
- **Theory:** Static graph optimization, layer fusion, and precision calibration.

### Week 6: INT8 + Phase Review
- **Objective:** Quantization and Performance Evaluation.
- **Implementations:**
    - [calibrate_int8.py](../AI_list/calibrate_int8.py): Custom Int8 Calibration logic.
    - [benchmark_models.py](../AI_list/benchmark_models.py): Latency and throughput benchmarking tool.
- **Commands:**
    ```bash
    # Run INT8 Calibration (requires archive.zip)
    python ../AI_list/calibrate_int8.py
    
    # Run Benchmarks
    python ../AI_list/benchmark_models.py --archive archive.zip
    ```
- **Theory:** Quantization theory—how 8-bit integers can represent 32-bit floats through scale factors.

---

## 📊 Performance Principles
1. **Memory Bound vs Compute Bound:** Most CUDA kernels are limited by VRAM bandwidth.
2. **Coalesced Access:** Tiled matrix transpose demonstrates how shared memory can "fix" non-coalesced global memory writes.
3. **Latency Hiding:** CUDA Streams allow the GPU to work on one data chunk while another is being moved over the PCIe bus.

---
*Reference Repository: [Cuda_Chapters](https://github.com/Techn08055/Cuda_Chapters)*
