"""
CUDA implementation to multiply two sparse matrices on the GPU

Workflow:
Given matrices A,B in SparseMatModP, to compute C=A*B
Convert A to CompressedSparseRow (CSR)
Convert B to CompressedSparseColumn (CSC)
Move CSR and CSC into GPU
Allocate space for C as a CompressedSparse (CS)
Define blocks on gpu_A,gpu_B
Call GPU Kernel to mutiply these blocks mod p
Save results into gpu_C
Convert gpu_C back into a SparseMatModP
"""

using CUDA
using Base.Threads


function matrixMul(a::CuArray{Int64, 2}, b::CuArray{Int64, 2})
    m, n = size(a)
    n, k = size(b)
    c = CUDA.zeros(Int64, m, k)
    threads = (16, 16)
    blocks = ((m + threads[1] - 1) ÷ threads[1], (k + threads[2] - 1) ÷ threads[2])
    
    @cuda threads=threads blocks=blocks mul_kernel(c, a, b, m, n, k)
    
    return collect(c)
end

function mul_kernel(c, a, b, m, n, k)
    idx, idy = threadIdx().x, threadIdx().y
    bidx, bidy = (blockIdx().x - 1) * blockDim().x, (blockIdx().y - 1) * blockDim().y
    
    if bidx + idx <= m && bidy + idy <= k
        val = 0.0
        for i = 1:n
            val += a[bidx + idx, i] * b[i, bidy + idy]
        end
        c[bidx + idx, bidy + idy] = val
    end
    
    return
end

# Example usage
A = CUDA.rand(Int64, 3, 4)
B = CUDA.rand(Int64, 4, 5)

C = matrixMul(A, B)
print(dump(C))