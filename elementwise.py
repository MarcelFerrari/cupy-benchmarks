import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse
from random import uniform

def elementwise_benchmark(n):
    # Random matrix generation
    A = cp.random.randn(n, n)
    x = cp.random.uniform()  # Generate a random scalar on GPU
    
    # Define the operation
    op = lambda: A * x
    
    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)
    
    # GFLOPs calculation: for every element of the matrix, there's one multiplication
    gflops = (n**2 / compute_time) / 1e9
    
    return compute_time, gflops


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('N', type=int,
                        help='Size of the matrix to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the system
    
    if(n < 2):
        print("Error: invalid system size N = {n}!")
        exit()
    
    compute_time, gflops = elementwise_benchmark(n)
    
    print("\n" + "="*50 + "\n") 
    print("Benchmarking GPU elementwise multiplication")
    print(f"Dimension of System: {n}x{n}")
    print(f"Time taken: {compute_time:.5f} seconds")
    print(f"GFLOPs: {gflops:.2f}")
    print("\n" + "="*50 + "\n") 
