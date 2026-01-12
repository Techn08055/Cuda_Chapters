import numpy as np
from numba import cuda

@cuda.jit
def vector_add_cuda_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < len(c):
        c[idx] = a[idx] + b[idx]

# Example usage
size = 10**7
a_host = np.random.rand(size)
b_host = np.random.rand(size)
c_host = np.empty_like(a_host)

# Transfer data to GPU
a_device = cuda.to_device(a_host)
b_device = cuda.to_device(b_host)
c_device = cuda.device_array_like(c_host)

# Configure and launch the kernel
threads_per_block = 256
blocks_per_grid = (size + (threads_per_block - 1)) // threads_per_block
vector_add_cuda_kernel[blocks_per_grid, threads_per_block](a_device, b_device, c_device)

# Transfer results back to CPU
c_device.copy_to_host(c_host)
print(c_host[:5])