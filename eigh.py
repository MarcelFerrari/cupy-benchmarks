import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse

def symmetric_eigenvalue_benchmark(n):
    # Generate a random symmetric matrix
    M = cp.random.randn(n, n)
    A = M + M.T  # Symmetric matrix generation
    
    # Define the symmetric eigenvalue decomposition operation
    op = lambda: cp.linalg.eigh(A)
    
    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)
    
    # The complexity of the algorithm is approximately 4/3 * n^3 + n^2 for tridiagonalization + MRRR
    flops = ((4./3.)*n**3 + n**2) / compute_time
    gflops = flops / 1e9
    
    return compute_time, gflops

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark symmetric eigenvalue decomposition on GPU using CuPy.')
    parser.add_argument('N', type=int,
                        help='Size of the matrix side to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the matrix
    
    if(n < 1):
        print(f"Error: invalid matrix size N = {n}!")
        exit()
    
    compute_time, gflops = symmetric_eigenvalue_benchmark(n)
    
    print("\n" + "="*50 + "\n") 
    print(f"Benchmarking GPU symmetric eigenvalue decomposition")
    print(f"Dimension of Matrix: {n}x{n}")
    print(f"Time taken: {compute_time:.5f} seconds")
    print(f"GFLOPs: {gflops:.2f}")
    print("\n" + "="*50 + "\n")
