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

function gpu_matrix_multiply(A::SparseMatModP, B::SparseMatModP)


    # Allocate memory on the GPU for matrices A, B, and the result C
    d_A_indptr = CuArray(A)
    d_B_indptr = CuArray(B)
    d_C = CUDA.zeros(Int64, A.ncols, B.nrows)

    # Define the number of blocks and threads per block
    threads_per_block = 32
    blocks_per_grid = ceil(Int, m / threads_per_block)

    # Define the kernel function for matrix multiplication
    function kernel_multiply(d_A, d_B, d_C, m, n, q)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

        if i <= m && j <= q
            temp = 0.0f0
            for k = 1:n
                temp += d_A[i, k] * d_B[k, j]
            end
            d_C[i, j] = temp
        end
        return
    end

    # Launch kernel
    @cuda threads = (threads_per_block, threads_per_block) blocks = (blocks_per_grid, blocks_per_grid) kernel_multiply(d_A, d_B, d_C, m, n, q)

    # Copy the result back from the GPU to the CPU
    C = Array(d_C)

    return C
end

# function sparse_to_csr_P(mat::SparseMatModP)
#     """ Helper function to find CSR of input sparse matrix """
#     nvals = length(mat.vals) # Number of values

#     indptr = zeros(Int, nrows + 1)
#     indices = zeros(Int, nnz)
#     data = zeros(Float64, nnz)

#     # Count the number of non-zero elements in each row
#     @threads for i in 1:nnz
#         @inbounds indptr[row[i] + 1] += 1
#     end

#     # Cumulative sum to obtain indptr
#     cumsum = 0
#     for i in 1:nrows
#         tmp = indptr[i]
#         indptr[i] = cumsum
#         cumsum += tmp
#     end
#     indptr[nrows + 1] = nnz

#     # Fill indices and data arrays
#     @threads for i in 1:nnz
#         @inbounds r = row[i]
#         idx = atomic_add!(pointer(indptr, r + 1), 1) # atomic operation to avoid race condition
#         indices[idx] = col[i]
#         data[idx] = var[i]
#     end

#     # Shift indptr to the right by one
#     for i in nrows:-1:1
#         indptr[i + 1] = indptr[i]
#     end
#     indptr[1] = 0

#     return indptr, indices, data
# end

# # Example usage
# A = rand(Float32, 1000, 1000)
# B = rand(Float32, 1000, 1000)

# C = gpu_matrix_multiply(A, B)
