# Edge AI System Design

## Problem Statement

Goal:
Serve a vision model (YOLO / classifier) on an edge device
with low latency (<100ms) and minimal memory usage.

Constraints:
- Limited GPU (Jetson / low VRAM)
- Real-time inference
- Single device (initially)

## Components

- Client (camera / app)
- Edge device (Jetson / local GPU)
- Inference service (FastAPI)
- Model runtime (TensorRT)
- Preprocessing + Postprocessing
- Logging & metrics

## Data Flow

Camera / Client
   ↓
HTTP Request (image / frame)
   ↓
FastAPI Server
   ↓
Preprocessing (resize, normalize)
   ↓
TensorRT Engine (GPU)
   ↓
Postprocessing (NMS, decode)
   ↓
JSON Response (detections)

## Design Decisions

Why FastAPI?
- Async support
- Easy to deploy
- Lightweight

Why TensorRT?
- Optimized inference
- FP16 / INT8 support

Why edge deployment?
- Low latency
- Offline capability

## Non-Functional Requirements

- Latency target: <100 ms
- Throughput: 10–30 FPS
- Fault tolerance: Restart service on crash
- Scalability: Add Triton later


