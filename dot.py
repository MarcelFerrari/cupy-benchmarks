import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse

def dot_product_benchmark(n):
    # Random matrix and vector generation
    A = cp.random.randn(n, n)
    x = cp.random.randn(n)
    
    # Define the dot product operation
    op = lambda: cp.dot(A, x)
    
    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)
    
    # GFLOPs calculation: for matrix-vector multiplication
    gflops = ((2 * n**2 - n) / compute_time) / 1e9
    
    return compute_time, gflops


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark matrix-vector dot product on GPU using CuPy.')
    parser.add_argument('N', type=int,
                        help='Size of the matrix to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the system
    
    if(n < 2):
        print(f"Error: invalid system size N = {n}!")
        exit()
    
    compute_time, gflops = dot_product_benchmark(n)
    
    print("\n" + "="*50 + "\n") 
    print("Benchmarking GPU dot product")
    print(f"Dimension of System: {n}x{n}")
    print(f"Time taken: {compute_time:.5f} seconds")
    print(f"GFLOPs: {gflops:.2f}")
    print("\n" + "="*50 + "\n") 