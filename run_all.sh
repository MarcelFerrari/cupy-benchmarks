# Run all benchmarks with reasonable values for the parameters.
python dgemm.py 20000 > results.txt
python dot.py 50000 >> results.txt
python eigh.py 10000 >> results.txt
python elementwise.py 40000 >> results.txt
python fft.py 800 >> results.txt
python lu.py 30000 >> results.txt
python stream.py 500 >> results.txt
python tensor_reduction.py 800 >> results.txt