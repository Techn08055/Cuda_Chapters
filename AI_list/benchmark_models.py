import os
import time
import zipfile
from io import BytesIO
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
from collections import defaultdict
import json

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ImageNet normalization
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


class TensorRTInference:
    """Wrapper for TensorRT engine inference"""
    
    def __init__(self, engine_path: str, precision: str):
        """
        Initialize TensorRT inference engine
        
        Args:
            engine_path: Path to TensorRT engine file
            precision: Precision type ('fp32', 'fp16', 'int8')
        """
        self.precision = precision
        self.engine_path = engine_path
        
        # Load engine
        print(f"Loading {precision} engine from {engine_path}...")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load {precision} engine")
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output shapes
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_dtype = dtype
                self.input_name = name
            else:
                self.output_shape = shape
                self.output_dtype = dtype
                self.output_name = name
        
        # Allocate buffers
        self._allocate_buffers()
        
        # Create CUDA stream for async execution
        self.stream = cuda.Stream()
        
        print(f"✓ {precision} engine loaded: input={self.input_shape}, output={self.output_shape}")
    
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs"""
        # Input buffer
        input_size = int(np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.d_input = cuda.mem_alloc(input_size)
        
        # Output buffer
        output_size = int(np.prod(self.output_shape) * np.dtype(self.output_dtype).itemsize)
        self.d_output = cuda.mem_alloc(output_size)
        
        # Host buffers
        self.h_input = np.empty(self.input_shape, dtype=self.input_dtype)
        self.h_output = np.empty(self.output_shape, dtype=self.output_dtype)
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference
        
        Args:
            input_data: Input data as numpy array (NCHW format)
        
        Returns:
            Output predictions
        """
        # Copy input to host buffer
        np.copyto(self.h_input, input_data.astype(self.input_dtype))
        
        # Copy input to device
        cuda.memcpy_htod(self.d_input, self.h_input)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Synchronize stream before copying output
        self.stream.synchronize()
        
        # Copy output from device
        cuda.memcpy_dtoh(self.h_output, self.d_output)
        
        return self.h_output.copy()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'd_input'):
            del self.d_input
        if hasattr(self, 'd_output'):
            del self.d_output


def load_image_from_archive(archive_path: str, image_path: str) -> np.ndarray:
    """Load and preprocess an image from archive"""
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        image_data = zip_ref.read(image_path)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        tensor = IMAGENET_TRANSFORM(image)
        return tensor.numpy().astype(np.float32)


def get_test_images(archive_path: str, num_images: int = 100) -> List[Tuple[str, np.ndarray]]:
    """Get test images from archive"""
    image_paths = []
    image_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        for file_path in all_files:
            if file_path.lower().endswith(image_extensions):
                image_paths.append(file_path)
                if len(image_paths) >= num_images:
                    break
    
    print(f"Loading {len(image_paths)} test images...")
    images = []
    for img_path in image_paths:
        try:
            img_array = load_image_from_archive(archive_path, img_path)
            images.append((img_path, img_array))
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    return images


def benchmark_latency(model: TensorRTInference, images: List[np.ndarray], 
                     num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark latency (inference time per image)
    
    Returns:
        Dictionary with latency statistics
    """
    # Warmup
    for _ in range(num_warmup):
        model.infer(images[0])
    
    cuda.Context.synchronize()
    
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        img = images[np.random.randint(0, len(images))]
        start = time.perf_counter()
        model.infer(img)
        cuda.Context.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'std': np.std(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }


def benchmark_throughput(model: TensorRTInference, images: List[np.ndarray],
                        duration_seconds: float = 10.0) -> Dict[str, float]:
    """
    Benchmark throughput (images per second)
    
    Returns:
        Dictionary with throughput statistics
    """
    # Warmup
    for _ in range(10):
        model.infer(images[0])
    
    cuda.Context.synchronize()
    
    # Measure throughput
    num_images = 0
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    
    while time.perf_counter() < end_time:
        img = images[np.random.randint(0, len(images))]
        model.infer(img)
        num_images += 1
    
    cuda.Context.synchronize()
    elapsed = time.perf_counter() - start_time
    throughput = num_images / elapsed
    
    return {
        'throughput': throughput,
        'total_images': num_images,
        'duration': elapsed
    }


def compare_precision(models: Dict[str, TensorRTInference], 
                     images: List[Tuple[str, np.ndarray]],
                     reference_model: str = 'fp32') -> Dict[str, float]:
    """
    Compare precision/accuracy between models
    
    Args:
        models: Dictionary of model_name -> TensorRTInference
        images: List of (image_path, image_array) tuples
        reference_model: Model to use as reference (default: 'fp32')
    
    Returns:
        Dictionary with precision metrics
    """
    if reference_model not in models:
        raise ValueError(f"Reference model '{reference_model}' not found")
    
    print(f"\nComparing precision using {reference_model} as reference...")
    
    ref_model = models[reference_model]
    results = {}
    
    for model_name, model in models.items():
        if model_name == reference_model:
            continue
        
        print(f"  Comparing {model_name} vs {reference_model}...")
        
        mse_values = []
        cosine_similarities = []
        top1_agreements = []
        top5_agreements = []
        
        for img_path, img_array in images[:50]:  # Use subset for precision comparison
            # Get predictions
            ref_output = ref_model.infer(img_array)
            test_output = model.infer(img_array)
            
            # Flatten outputs to 1D (handle batch dimension if present)
            ref_flat = ref_output.flatten()
            test_flat = test_output.flatten()
            
            # MSE
            mse = np.mean((ref_flat - test_flat) ** 2)
            mse_values.append(mse)
            
            # Cosine similarity
            cosine_sim = np.dot(ref_flat, test_flat) / (
                np.linalg.norm(ref_flat) * np.linalg.norm(test_flat)
            )
            cosine_similarities.append(cosine_sim)
            
            # Top-1 and Top-5 agreement (use flattened arrays)
            ref_top1 = int(np.argmax(ref_flat))
            ref_top5 = [int(x) for x in np.argsort(ref_flat)[-5:][::-1]]
            test_top1 = int(np.argmax(test_flat))
            test_top5 = [int(x) for x in np.argsort(test_flat)[-5:][::-1]]
            
            top1_agreements.append(1 if ref_top1 == test_top1 else 0)
            top5_agreements.append(1 if len(set(ref_top5) & set(test_top5)) > 0 else 0)
        
        results[model_name] = {
            'mse': np.mean(mse_values),
            'cosine_similarity': np.mean(cosine_similarities),
            'top1_agreement': np.mean(top1_agreements),
            'top5_agreement': np.mean(top5_agreements)
        }
    
    return results


def print_results(latency_results: Dict[str, Dict], 
                 throughput_results: Dict[str, Dict],
                 precision_results: Dict[str, Dict]):
    """Print formatted benchmark results"""
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Latency results
    print("\n📊 LATENCY (ms per image):")
    print("-" * 80)
    print(f"{'Model':<10} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'P95':<10} {'P99':<10}")
    print("-" * 80)
    for model_name in ['fp32', 'fp16', 'int8']:
        if model_name in latency_results:
            r = latency_results[model_name]
            print(f"{model_name:<10} {r['mean']:<10.3f} {r['median']:<10.3f} "
                  f"{r['min']:<10.3f} {r['max']:<10.3f} {r['p95']:<10.3f} {r['p99']:<10.3f}")
    
    # Throughput results
    print("\n🚀 THROUGHPUT (images/second):")
    print("-" * 80)
    print(f"{'Model':<10} {'Throughput':<15} {'Total Images':<15} {'Duration (s)':<15}")
    print("-" * 80)
    for model_name in ['fp32', 'fp16', 'int8']:
        if model_name in throughput_results:
            r = throughput_results[model_name]
            print(f"{model_name:<10} {r['throughput']:<15.2f} {r['total_images']:<15} {r['duration']:<15.2f}")
    
    # Precision results
    print("\n🎯 PRECISION (vs FP32):")
    print("-" * 80)
    print(f"{'Model':<10} {'MSE':<15} {'Cosine Sim':<15} {'Top-1 Agree':<15} {'Top-5 Agree':<15}")
    print("-" * 80)
    for model_name in ['fp16', 'int8']:
        if model_name in precision_results:
            r = precision_results[model_name]
            print(f"{model_name:<10} {r['mse']:<15.6f} {r['cosine_similarity']:<15.4f} "
                  f"{r['top1_agreement']*100:<14.2f}% {r['top5_agreement']*100:<14.2f}%")
    
    # Speedup comparison
    print("\n⚡ SPEEDUP (vs FP32):")
    print("-" * 80)
    if 'fp32' in latency_results:
        fp32_latency = latency_results['fp32']['mean']
        for model_name in ['fp16', 'int8']:
            if model_name in latency_results:
                speedup = fp32_latency / latency_results[model_name]['mean']
                print(f"{model_name}: {speedup:.2f}x faster")
    
    print("\n" + "="*80)


def main():
    """Main benchmarking function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark TensorRT models')
    parser.add_argument('--archive', type=str, default='archive.zip',
                       help='Path to archive.zip with test images')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of test images to use')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of runs for latency benchmark')
    parser.add_argument('--throughput-duration', type=float, default=10.0,
                       help='Duration in seconds for throughput benchmark')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Engine files
    engine_files = {
        'fp32': 'resnet18_fp32.engine',
        'fp16': 'resnet18_fp16.engine',
        'int8': 'resnet18_int8.engine'
    }
    
    # Check if engines exist
    missing_engines = [name for name, path in engine_files.items() if not os.path.exists(path)]
    if missing_engines:
        print(f"Warning: Missing engine files: {missing_engines}")
        engine_files = {k: v for k, v in engine_files.items() if os.path.exists(v)}
    
    if not engine_files:
        print("ERROR: No engine files found!")
        return
    
    # Load models
    models = {}
    for name, path in engine_files.items():
        try:
            models[name] = TensorRTInference(path, name)
        except Exception as e:
            print(f"ERROR: Failed to load {name} engine: {e}")
            continue
    
    if not models:
        print("ERROR: No models loaded!")
        return
    
    # Load test images
    if not os.path.exists(args.archive):
        print(f"ERROR: Archive not found at {args.archive}")
        return
    
    test_images = get_test_images(args.archive, args.num_images)
    if not test_images:
        print("ERROR: No test images loaded!")
        return
    
    image_arrays = [img for _, img in test_images]
    
    # Benchmark latency
    print("\n" + "="*80)
    print("BENCHMARKING LATENCY...")
    print("="*80)
    latency_results = {}
    for name, model in models.items():
        print(f"\nBenchmarking {name} latency...")
        latency_results[name] = benchmark_latency(model, image_arrays, num_runs=args.num_runs)
    
    # Benchmark throughput
    print("\n" + "="*80)
    print("BENCHMARKING THROUGHPUT...")
    print("="*80)
    throughput_results = {}
    for name, model in models.items():
        print(f"\nBenchmarking {name} throughput...")
        throughput_results[name] = benchmark_throughput(model, image_arrays, 
                                                       duration_seconds=args.throughput_duration)
    
    # Compare precision
    if 'fp32' in models and len(models) > 1:
        print("\n" + "="*80)
        print("COMPARING PRECISION...")
        print("="*80)
        precision_results = compare_precision(models, test_images, reference_model='fp32')
    else:
        precision_results = {}
    
    # Print results
    print_results(latency_results, throughput_results, precision_results)
    
    # Save results to JSON if requested
    if args.output:
        results = {
            'latency': latency_results,
            'throughput': throughput_results,
            'precision': precision_results
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

