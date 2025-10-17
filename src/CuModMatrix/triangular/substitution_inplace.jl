"""
    forward_sub_kernel_32(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs forward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function forward_sub_kernel_32(
    A::CuDeviceMatrix{T1}, 
    A_inv::CuDeviceMatrix{T2}, 
    N::Int,
    row_shift::Int,
    col_shift::Int
) where {T1, T2}

    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    for row in 1:TILE_WIDTH
        sum = 0
        for k in 1:(row-1)
            sum = mod(sum + A[row + row_shift, k + col_shift] * A_inv[k, col], N)
        end 
        
        rhs = (row == col) ? 1 : 0
        diag_inv = mod_inv(A[row + row_shift, row + row_shift], N)
        A_inv[row, col] = mod(diag_inv * (rhs - sum + N), N)

        # CUDA.sync_threads()
    end

    return
end

"""
    forward_sub_gpu_type(A::CuModMatrix)

Performs forward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function forward_sub_gpu_type_32(A::CuModMatrix, row_shift::Int, col_shift::Int)

    d_A_inv = CUDA.zeros(eltype(A.data), TILE_WIDTH + size(A, 2), TILE_WIDTH + size(A, 1))

    @cuda threads=(TILE_WIDTH) blocks=(ceil(Int, size(A, 2) / TILE_WIDTH)) forward_sub_kernel_32(A.data, d_A_inv, A.N, row_shift, col_shift)
    return CuModMatrix(d_A_inv, A.N, new_size=(size(A, 2), size(A, 1)))
end

"""
    backward_sub_gpu_type_32(A::CuModMatrix)

Performs backward substitution on a CuModMatrix.

"""
function backward_sub_gpu_type_32(A::CuModMatrix, row_shift::Int, col_shift::Int)
    d_A_inv = CUDA.zeros(eltype(A.data), TILE_WIDTH + size(A, 2), TILE_WIDTH + size(A, 1))

    @cuda threads=(TILE_WIDTH) blocks=(ceil(Int, size(A, 1) / TILE_WIDTH)) backward_sub_kernel_32(A.data, d_A_inv, A.N, row_shift, col_shift)
    return CuModMatrix(d_A_inv, A.N, new_size=(size(A, 2), size(A, 1)))
end

"""
    backward_sub_kernel_32(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs backward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function backward_sub_kernel_32(
    A::CuDeviceMatrix{T1}, 
    A_inv::CuDeviceMatrix{T2}, 
    N::Int,
    row_shift::Int,
    col_shift::Int
) where {T1, T2}
    bid = blockIdx().x
    tid = threadIdx().x

    tid = (bid - 1) * blockDim().x + tid

    for row in TILE_WIDTH:-1:1
        CUDA.sync_threads()

        sum = 0
        for j in row+1:TILE_WIDTH
            sum += A[row + row_shift, j + col_shift] * A_inv[j, tid]
        end
        sum = mod(sum, N)

        rhs = (tid == row ? 1 : 0)
        diag = A[row + row_shift, row + row_shift]
        A_inv[row, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
    end

end