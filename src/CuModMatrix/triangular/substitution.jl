"""
    forward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs forward substitution on a CuModMatrix.

"""
function forward_sub_kernel(
    A::CuDeviceMatrix{T1}, 
    A_inv::CuDeviceMatrix{T2}, 
    N::Int
) where {T1, T2}

    # bid = blockIdx().x
    # tid = threadIdx().x
    # n = size(A, 1)

    # tid = (bid - 1) * 32 + tid

    # for row in 1:n
    #     CUDA.sync_threads()
        
    #     sum = 0
    #     for j in 1:row-1
    #         sum += A[row, j] * A_inv[j, tid]
    #     end
    #     sum = mod(sum, N)

    #     diag = A[row, row]
    #     rhs = (tid == row ? 1 : 0)
    #     A_inv[row, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
    # end

    bid = blockIdx().x
    tid = threadIdx().x
    n = size(A, 2)

    tid = (bid - 1) * TILE_WIDTH + tid

    for col in 1:n
        CUDA.sync_threads()
        
        sum = 0
        for j in 1:col-1
            sum += A[col, j] * A_inv[j, tid]
        end
        sum = mod(sum, N)

        diag = A[col, col]
        rhs = (tid == col ? 1 : 0)
        A_inv[col, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
    end

    return
end

"""
    forward_sub_gpu_type(A::CuModMatrix)

Performs forward substitution on a CuModMatrix.

"""
function forward_sub_gpu_type(A::CuModMatrix)
    padded_rows = size(A.data, 1)
    padded_cols = size(A.data, 2)
    d_A_inv = CuArray{eltype(A.data)}(undef, padded_cols, padded_rows)

    @cuda threads=(32) blocks=(ceil(Int, size(A, 1) / 32)) forward_sub_kernel(A.data, d_A_inv, A.N)
    return CuModMatrix(d_A_inv, A.N, new_size=(cols(A), rows(A)))
end

"""
    backward_sub_gpu_type(A::CuModMatrix)

Performs backward substitution on a CuModMatrix.

"""
function backward_sub_gpu_type(A::CuModMatrix)
    padded_rows = size(A.data, 1)
    padded_cols = size(A.data, 2)
    d_A_inv = CuArray{eltype(A.data)}(undef, padded_cols, padded_rows)

    @cuda threads=(32) blocks=(ceil(Int, size(A, 1) / 32)) backward_sub_kernel(A.data, d_A_inv, A.N)
    return CuModMatrix(d_A_inv, A.N, new_size=(cols(A), rows(A)))
end

"""
    backward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs backward substitution on a CuModMatrix.

"""
function backward_sub_kernel(
    A::CuDeviceMatrix{T1}, 
    A_inv::CuDeviceMatrix{T2}, 
    N::Int
) where {T1, T2}
    bid = blockIdx().x
    tid = threadIdx().x
    n = size(A, 1)

    tid = (bid - 1) * 32 + tid

    for row in n:-1:1
        CUDA.sync_threads()

        sum = 0
        for j in row+1:n
            sum += A[row, j] * A_inv[j, tid]
        end
        sum = mod(sum, N)

        rhs = (tid == row ? 1 : 0)
        diag = A[row, row]
        A_inv[row, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
    end

end