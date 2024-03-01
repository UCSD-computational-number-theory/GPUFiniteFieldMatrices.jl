"""
Exploration of hsnf python package to solve Ax=b (mod q).
NOTE: You must disable assertions to run the file. 
That is, use "python3 -O HSNF_experiments.py" to run the file.

Findings:
5 trials with m = 100, n = 100, and q = 100:
1.3 seconds,  45.9 Megabytes
1.4 seconds,  45.5 Megabytes
1.3 seconds,  46.1 Megabytes
1.4 seconds,  45.8 Megabytes
1.3 seconds,  45.2 Megabytes

Conclusion:
Relatively fast, but as with most Python programs, does not scale well.
Memory usage is way too high.
"""

# Packages to compute
import numpy as np
from hsnf import integer_system

# Packages to benchmark
import time
import resource 
import sys
sys.setrecursionlimit(10000)

def genMatOverFp(p,m,n):
    """Generate random m x n matrix over Fp"""
    return np.random.randint(0,p,(m,n))

def genVecOverFp(p,m):
    """Generate random 1 x n vector over Fp"""
    return np.random.randint(0,p,(m,1))

def solveModQ(A,b,q):
    """Solve Ax = b (mod q)"""
    # https://hsnf.readthedocs.io/en/latest/_modules/hsnf/integer_system.html#solve_modular_integer_linear_system

    return integer_system.solve_modular_integer_linear_system(
        A, # numpy.ndarray[Any, numpy.dtype[numpy.int64]]
        b, # numpy.ndarray[Any, numpy.dtype[numpy.int64]]
        q # int
    )

# Create random matrice
q = 100
size = 220
A = genMatOverFp(q,size,size)
b = genVecOverFp(q,size)

# Benchmark computation time
time_start = time.perf_counter()
solveModQ(A,b,size)
time_elapsed = (time.perf_counter() - time_start)

# Print time and memory to console
memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("%5.1f seconds, %5.1f Megabytes" % (time_elapsed, memMb))