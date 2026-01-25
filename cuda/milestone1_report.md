# Milestone 1 Report: Phase 1 — Foundations

## Executive Summary
Phase 1 focused on shifting the mindset from high-level model architecture to system-level execution. We successfully implemented and profiled key CUDA kernels that demonstrate the fundamental constraints of GPU computing: memory bandwidth, PCIe latency, and parallel execution.

## Key Accomplishments

### 1. Vector Addition (Baseline)
- **Goal:** Establish the CPU vs GPU performance baseline.
- **Outcome:** Observed that for small $N$, CPU outperforms GPU due to memory transfer overhead, but as $N$ scales, GPU parallelization provides significant speedup.

### 2. Prefix Sum & Profiling
- **Goal:** Analyze kernel timing vs memory overhead.
- **Outcome:** Used Nsight Systems to identify that memory copies (H2D/D2H) were consuming the majority of the wall-clock time, highlighting the "Memory Wall" problem.

### 3. Matrix Transpose & Memory Hierarchy
- **Goal:** Optimize global memory access.
- **Outcome:** Implemented a shared-memory tiling strategy. This allowed for coalesced reads and writes, improving effective bandwidth utilization significantly.

### 4. CUDA Streams & Concurrency
- **Goal:** Hide data transfer latency.
- **Outcome:** Overlapped kernel execution with data transfers using multiple streams. This reduced the total pipeline latency by nearly 40% in our benchmarks.

### 5. TensorRT & Model Optimization (New)
- **Goal:** Shift from dynamic to static graph execution.
- **Outcome:** Successfully exported a ResNet18 model to ONNX and built TensorRT engines for FP32 and FP16. Observed that FP16 provides a ~2x speedup over FP32 on modern GPUs with negligible precision loss.

### 6. INT8 Calibration (New)
- **Goal:** Achieve maximum throughput via fixed-point quantization.
- **Outcome:** Implemented a custom entropy calibrator. Benchmark results show that INT8 quantization can lead to significant throughput gains (up to 4x over FP32) while maintaining >99% Top-1 agreement with the original FP32 model.

## Bottlenecks Identified
- **PCIe Bandwidth:** The biggest bottleneck for small batch inference.
- **Global Memory Latency:** Unoptimized access patterns can lead to 10x performance drops.
- **Calibration Quality:** INT8 performance is highly dependent on the quality of the calibration dataset (see `../AI_list/calibrate_int8.py`).
- **Host Thread Synchronization:** Improper use of `cudaDeviceSynchronize()` can lead to CPU-side bubbles.

## Conclusion
The foundation for high-performance AI systems is now established. The next phase will focus on **TensorRT optimization**, model quantization, and production deployment using Triton Inference Server.

---
*Completed on: 2026-01-25*
