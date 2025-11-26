# GPU Matrices

This repository contains implementations for a variety of matrix types on the GPU.
Currently, the repository contains:

1. CuModMatrix, a CuArray wrapper which represents a matrix mod N. CuModMatrix pads elements to 32, and has various matrix operations implemented as GPU kernels. This includes:
   * matrix multiplication (for N < ~2^26)
   * if N is prime, RREF, LU decomposition, and PLUQ decomposition.
2. KaratsubaMatrices, a matrix type implementing Karatsuba (matrix) multiplication. This also supports addition, subtraction, and other basic operations.
