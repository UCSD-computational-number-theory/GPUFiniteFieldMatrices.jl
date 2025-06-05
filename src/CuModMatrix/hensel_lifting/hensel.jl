#using GPUFiniteFieldMatrices

# """
#     henselLift(p, precision, A, T)
# Hensel lifts mod p solution T to the linear system AX-I=0 to mod p^precision

# INPUTS:
# * "p" -- integer, a prime number 
# * "precision" -- integer 
# * "A" -- matrix, integer coefficients
# * "T" -- matrix, integer coefficients, satisfies AT-I=0 mod p
# """
# function henselLift(p, precision, A, T)
#     i = 1
#     while i < precision
#         T = 2*T - T * (A*T)
#         #println("After step $i: $(julia_signed_mod.(T,p^(i+1)))")
#         i *= 2
#     end
#     R, pi = residue_ring(ZZ, p^precision)
#     stuff = [R(x) for x in Array(T)]
#     return matrix(R,stuff)
# end

"""
    hensel_pseudoinverse(p, precision, A, T)

Hensel lifts mod p solution T to the linear system AX-I=0 to mod p^precision
Uses the CuModMatrix type.

INPUTS:
* "p" -- integer, a prime number 
* "precision" -- integer 
* "A" -- matrix, integer coefficients
* "T" -- matrix, integer coefficients, satisfies AT-I=0 mod p
"""
function hensel_pseudoinverse(p, precision, A, T)
    i = 1
    while i < precision
        T = 2*T - T * (A*T)
        i *= 2
    end
    R, pi = residue_ring(ZZ, p^precision)
    return matrix(R, [R(x) for x in Array(T)])
end

function forward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int) where {T1, T2}
    j = blockIdx().x
    t = threadIdx().x
    n = size(A, 1)

    row = t
    while row <= n
        CUDA.sync_threads()
        
        sum = (row == j) ? 1 : 0
        for k = 1:row-1
            sum = mod(sum - A[row, k] * A_inv[k, j], N)
        end
        diag = A[row, row]
        diag_inv = mod_inv(diag, N)
        A_inv[row, j] = mod(sum * diag_inv, N)
        
        row += 32
    end

    return
end

function forward_sub_gpu_type(A::CuModMatrix)
    padded_rows = size(A.data, 1)
    d_A_inv = CuArray{Int}(undef, padded_rows, padded_rows)

    @cuda threads=(TILE_WIDTH) blocks=(padded_rows-1) forward_sub_kernel(A.data, d_A_inv, A.N)
    return CuModMatrix(d_A_inv, A.N, new_size=(rows(A), rows(A)))
end

function backward_sub_gpu_type(A::CuModMatrix)
    padded_rows = size(A.data, 1)
    d_A_inv = CuArray{Int}(undef, padded_rows, padded_rows)

    @cuda threads=(TILE_WIDTH) blocks=(padded_rows-1) backward_sub_kernel(A.data, d_A_inv, A.N)
    return CuModMatrix(d_A_inv, A.N, new_size=(rows(A), rows(A)))
end

function backward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int) where {T1, T2}
    j = blockIdx().x
    t = threadIdx().x
    n = size(A, 1)

    row = t
    while row <= n
        CUDA.sync_threads()
        
        sum = (row == j) ? 1 : 0
        for k = row:n
            sum = mod(sum - A[row, k] * A_inv[k, j], N)
        end
        diag = A[row, row]
        diag_inv = mod_inv(diag, N)
        A_inv[row, j] = mod(sum * diag_inv, N)
        
        row += 32
    end

    return
end