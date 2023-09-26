import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse

def fft_benchmark(n, fft_type):
    if fft_type == "complex":
        # Complex tensor generation
        T = cp.random.randn(n, n, n) + 1j * cp.random.randn(n, n, n)
        op = lambda: cp.fft.fftn(T)
        ops_scale = 5
    elif fft_type == "real":
        # Real tensor generation
        T = cp.random.randn(n, n, n)
        op = lambda: cp.fft.rfftn(T)
        ops_scale = 2.5
    else:
        raise ValueError(f"Invalid FFT type: {fft_type}")
    
    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)
    
    # Calculating the operations based on rules from https://www.fftw.org/speed/method.html
    ops = ops_scale * (n**3) * np.log2(n**3)
    mflops = (ops / compute_time) / 1e9  # Return GFLOPs
    
    return compute_time, mflops

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark 3D FFTs on GPU using CuPy.')
    parser.add_argument('N', type=int, help='Size of the tensor side to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the tensor

    if(n < 1):
        print(f"Error: invalid tensor size N = {n}!")
        exit()

    print("\n" + "="*50 + "\n")  # Separator for clarity
    for fft_type in ["real", "complex"]:
        compute_time, mflops = fft_benchmark(n, fft_type)
        
        print(f"Benchmarking GPU 3D {fft_type.upper()} FFT")
        print(f"Dimension of Tensor: {n}x{n}x{n}")
        print(f"Time taken: {compute_time:.5f} seconds")
        print(f"GFLOPs: {mflops:.2f}")
        print("\n" + "="*50 + "\n")  # Separator for clarity
