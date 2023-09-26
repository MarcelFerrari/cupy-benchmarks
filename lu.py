import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
import argparse

# GPU Functions
def compute_residual(A, x, b):
    residual_vector = cp.dot(A, x) - b
    return cp.max(cp.abs(residual_vector))  # L-infinity norm for a vector

def megaflops(compute_time, n):
    flops = (2.0/3.0) * (n**3) + (3./2.) * n**2
    return (flops / compute_time) / 1e9  # Return GFLOPs

def gpu_benchmark(n):
    A = cp.random.randn(n, n)
    b = cp.random.randn(n)
    
    # Benchmark system
    results = benchmark(cp.linalg.solve, (A, b), n_repeat=5, n_warmup=3)

    # Compute solution explicitly to evaluate error
    x = cp.linalg.solve(A, b)

    compute_time = np.median(results.gpu_times)

    mflops = megaflops(compute_time, n)
    residual = compute_residual(A, x, b)

    return compute_time, mflops, residual

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('N', type=int,
                        help='Size of the linear system to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the system
    
    if(n < 2):
        print("Error: invalid system size N = {n}!")
        exit()

    # Print results
    compute_time, mflops, residual = gpu_benchmark(n)
    
    print("\n" + "="*50 + "\n") 
    print("Benchmarking GPU partial pivot LU system solver")
    print(f"Dimension of System: {n}")
    print(f"Time taken: {compute_time:.5f} seconds")
    print(f"GFLOPs: {mflops:.2f}")
    print(f"Residual L∞ error: {residual:.6e}")  # scientific notation with 6 digit precision
    print("\n" + "="*50 + "\n") 