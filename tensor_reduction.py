import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse

def reduction_benchmark(n, operation='sum'):
    # Random tensor generation
    T = cp.random.randn(n, n, n)
    
    # Define the reduction operation based on the input
    if operation == 'sum':
        op = lambda: cp.sum(T)
    elif operation == 'multiply':
        op = lambda: cp.prod(T)
    elif operation == 'max':
        op = lambda: cp.max(T)
    else:
        raise ValueError(f"Unsupported operation {operation}")
    
    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)
    
    # The total operations for each reduction is n^3 (for every element in the tensor)
    gflops = (n**3 / compute_time) / 1e9
    
    return compute_time, gflops

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark tensor reductions on GPU using CuPy.')
    parser.add_argument('N', type=int,
                        help='Size of the tensor side to benchmark.')

    args = parser.parse_args()

    n = int(args.N)  # Dimension of the tensor
    
    if(n < 1):
        print(f"Error: invalid tensor size N = {n}!")
        exit()
    
    for operation in ['sum', 'multiply', 'max']:
        compute_time, gflops = reduction_benchmark(n, operation=operation)
        
        print("\n" + "="*50 + "\n") 
        print(f"Benchmarking GPU tensor {operation} reduction")
        print(f"Dimension of Tensor: {n}x{n}x{n}")
        print(f"Time taken: {compute_time:.5f} seconds")
        print(f"GFLOPs: {gflops:.2f}")
        print("\n" + "="*50 + "\n") 
