import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import argparse

def stream_benchmark(n, op_type):
    # Create large vectors
    a = cp.zeros(n, dtype=cp.float64)
    b = cp.random.rand(n).astype(cp.float64)
    c = cp.random.rand(n).astype(cp.float64)
    q = cp.float64(0.5)  # scalar for SCALE and TRIAD

    # Define the operations based on op_type
    if op_type == "COPY":
        def op():
            a = cp.copy(b)
    elif op_type == "SCALE":
        def op():
            a = q * b
    elif op_type == "SUM":
        def op():
            a = b + c
    else:
        raise ValueError(f"Invalid operation type: {op_type}")

    # Benchmark system
    cublas_times = benchmark(op, n_repeat=5, n_warmup=3)
    compute_time = np.median(cublas_times.gpu_times)

    # Memory operations: 2 reads and 1 write for SUM and TRIAD, 1 read and 1 write for COPY and SCALE.
    if op_type in ["SUM"]:
        num_ops = 3 * n
    else:
        num_ops = 2 * n

    bandwidth = (num_ops * 8) / compute_time / 1e9  # Using float64, so 8 bytes per operation

    return compute_time, bandwidth

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark STREAM operations on GPU using CuPy in double precision.')
    parser.add_argument('N', type=int, help='Number of elements in the array to benchmark (in millions).')

    args = parser.parse_args()

    n = int(args.N)*1000000  # Number of elements in the array

    if(n < 1):
        print(f"Error: invalid array size N = {n}!")
        exit()

    print("\n" + "="*50 + "\n")
    for op_type in ["COPY", "SCALE", "SUM"]:
        compute_time, bandwidth = stream_benchmark(n, op_type)
        
        print(f"Benchmarking GPU STREAM {op_type} in Double Precision")
        print(f"Number of Elements: {n}")
        print(f"Time taken: {compute_time:.5f} seconds")
        print(f"Bandwidth: {bandwidth:.2f} GB/s")
        print("\n" + "="*50 + "\n")  # Separator for clarity
