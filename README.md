# GPU Matrices

This repository contains implementations for a variety of matrix types on the GPU.
Currently, the repository contains:

1. CuModMatrix, a CuArray wrapper that pads to 32, has mod N built in, and has various matrix operations implemented as GPU kernels.
2. KaratsubaMatrices, a matrix type implementing Karatsuba matrix multiplication.
