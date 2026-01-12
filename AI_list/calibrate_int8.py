import os
import zipfile
from io import BytesIO
import numpy as np
import tensorrt as trt
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List

# Import pycuda for CUDA memory management
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    raise ImportError(
        "pycuda is required for TensorRT calibration. "
        "Install it with: conda install -c conda-forge pycuda --prefix ./venv"
    )

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Calibrator class for TensorRT int8 quantization using images from archive.zip
    """
    def __init__(self, archive_path: str, batch_size: int = 1, num_images: int = 200):
        """
        Initialize the calibrator
        
        Args:
            archive_path: Path to archive.zip
            batch_size: Batch size for calibration
            num_images: Number of images to use for calibration (default: 200)
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.archive_path = archive_path
        self.batch_size = batch_size
        self.num_images = num_images
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Extract image paths from archive
        self.image_paths = self._extract_image_paths()
        self.current_index = 0
        
        # Allocate memory for calibration data
        self.device_input = None
        
    def _extract_image_paths(self) -> List[str]:
        """Extract image file paths from archive.zip"""
        image_paths = []
        image_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            for file_path in all_files:
                if file_path.lower().endswith(image_extensions):
                    image_paths.append(file_path)
                    if len(image_paths) >= self.num_images:
                        break
        
        print(f"Found {len(image_paths)} images for calibration")
        return image_paths
    
    def _load_image_from_archive(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image from the archive"""
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            # Read image from archive
            image_data = zip_ref.read(image_path)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Convert to numpy array (CHW format)
            numpy_array = tensor.numpy().astype(np.float32)
            
            return numpy_array
    
    def get_batch_size(self):
        """Return the batch size"""
        return self.batch_size
    
    def get_batch(self, names):
        """
        Get a batch of calibration data
        
        Args:
            names: List of input tensor names
            
        Returns:
            List of numpy arrays or None if no more data
        """
        if self.current_index >= len(self.image_paths):
            return None
        
        batch_images = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.image_paths):
                break
            
            image_path = self.image_paths[self.current_index]
            image_array = self._load_image_from_archive(image_path)
            batch_images.append(image_array)
            self.current_index += 1
        
        if not batch_images:
            return None
        
        # Stack images into batch (NCHW format)
        batch = np.stack(batch_images, axis=0)
        
        # Pad batch if necessary
        if batch.shape[0] < self.batch_size:
            padding = np.zeros((self.batch_size - batch.shape[0],) + batch.shape[1:], 
                             dtype=batch.dtype)
            batch = np.concatenate([batch, padding], axis=0)
        
        # Allocate device memory if not already done
        if not PYCUDA_AVAILABLE:
            # Use TensorRT's built-in memory allocation
            # For TensorRT 8.0+, we can return numpy arrays directly in some cases
            # But for calibration, we need device memory
            # Try to allocate using ctypes/CUDA runtime
            try:
                import ctypes
                cudart = ctypes.CDLL("libcudart.so")
                
                # Allocate device memory
                if self.device_input is None:
                    self.device_input_size = batch.nbytes
                    self.device_input = ctypes.c_void_p()
                    cudart.cudaMalloc(ctypes.byref(self.device_input), self.device_input_size)
                
                # Copy to device
                cudart.cudaMemcpy(
                    self.device_input,
                    batch.ctypes.data_as(ctypes.c_void_p),
                    self.device_input_size,
                    1  # cudaMemcpyHostToDevice
                )
                
                return [int(self.device_input.value)]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to allocate CUDA memory. pycuda is required for calibration. "
                    f"Error: {e}\n"
                    f"Please install pycuda: conda install -c conda-forge pycuda"
                )
        
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)
        
        return [self.device_input]
    
    def read_calibration_cache(self):
        """Read calibration cache if it exists"""
        cache_file = 'calibration_cache.cache'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to disk"""
        cache_file = 'calibration_cache.cache'
        with open(cache_file, 'wb') as f:
            f.write(cache)


def build_int8_engine(onnx_path: str, archive_path: str, 
                     num_calibration_images: int = 200,
                     max_batch_size: int = 1,
                     max_workspace_size: int = 1 << 30):
    """
    Build a TensorRT int8 engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model file
        archive_path: Path to archive.zip containing calibration images
        num_calibration_images: Number of images to use for calibration
        max_batch_size: Maximum batch size for the engine
        max_workspace_size: Maximum workspace size in bytes (default: 1GB)
    
    Returns:
        Serialized TensorRT engine (bytes)
    """
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"Loading ONNX model from {onnx_path}...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX model parsed successfully")
    
    # Configure builder
    config = builder.create_builder_config()
    # TensorRT 10.x uses set_memory_pool_limit instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.set_flag(trt.BuilderFlag.INT8)
    
    # Create calibrator
    print(f"Creating calibrator with {num_calibration_images} images from {archive_path}...")
    calibrator = ImageCalibrator(archive_path, batch_size=1, num_images=num_calibration_images)
    config.int8_calibrator = calibrator
    
    # Build engine (TensorRT 10.x uses build_serialized_network)
    print("Building TensorRT int8 engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    print("Engine built successfully!")
    return serialized_engine


def save_engine(serialized_engine, output_path: str):
    """Save TensorRT engine to file"""
    print(f"Saving engine to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    # Check if pycuda is available
    if not PYCUDA_AVAILABLE:
        print("ERROR: pycuda is required for TensorRT calibration")
        print("Install it with: pip install pycuda")
        sys.exit(1)
    
    # Configuration
    ONNX_MODEL_PATH = "resnet18.onnx"
    ARCHIVE_PATH = "archive.zip"
    OUTPUT_ENGINE_PATH = "resnet18_int8.engine"
    NUM_CALIBRATION_IMAGES = 200
    
    # Check if files exist
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(ARCHIVE_PATH):
        print(f"ERROR: Archive not found at {ARCHIVE_PATH}")
        sys.exit(1)
    
    # Build int8 engine
    engine = build_int8_engine(
        onnx_path=ONNX_MODEL_PATH,
        archive_path=ARCHIVE_PATH,
        num_calibration_images=NUM_CALIBRATION_IMAGES
    )
    
    if engine:
        # Save engine
        save_engine(engine, OUTPUT_ENGINE_PATH)
        print(f"\n✓ Int8 calibration complete!")
        print(f"✓ Engine saved to {OUTPUT_ENGINE_PATH}")
    else:
        print("\n✗ Failed to build int8 engine")
        sys.exit(1)

