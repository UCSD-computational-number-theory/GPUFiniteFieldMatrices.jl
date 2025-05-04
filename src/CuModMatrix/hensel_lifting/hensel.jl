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

function backsubstitution_shared_kernel(U::CuArray{T, 2}, x, b, N) where T
    tid = threadIdx().x
    bid = blockIdx().x
    n = size(U, 1)
    
    shared_sum = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))
    
    for i = n:-1:1
        if tid == 1 && bid == 1
            sum = b[i]
            for j = i+1:n
                sum = (sum - (U[i, j] * x[j]) % N) % N
                if sum < 0
                    sum += N
                end
            end
            
            diag_inv = mod_inv(U[i, i], N)
            x[i] = (sum * diag_inv) % N
        end
        
        CUDA.sync_threads()
    end
    
    return nothing
end

function backsubstitution_shared(U::CuModMatrix{T}, b) where T
    n = U.rows
    N = U.N
    x = CUDA.zeros(Int, n)
    d_b = CuArray(b)
    
    @cuda threads=1 blocks=1 backsubstitution_shared_kernel(U.data, x, d_b, N)
    
    return Array(x)
end

function backsubstitution_kernel(U::CuArray{T, 2}, x, b, N) where T
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(U, 1)
    
    if tid <= n
        for i = n:-1:1
            CUDA.sync_blocks()
            
            if tid == i
                sum = b[i]
                for j = i+1:n
                    sum = (sum - (U[i, j] * x[j]) % N) % N
                    if sum < 0
                        sum += N
                    end
                end
                
                diag_inv = mod_inv(U[i, i], N)
                x[i] = (sum * diag_inv) % N
            end
        end
    end
    
    return nothing
end

function backsubstitution(U::CuModMatrix{T}, b) where T
    n = U.rows
    N = U.N
    x = CUDA.zeros(Int, n)
    d_b = CuArray(b)
    
    threads_per_block = min(256, n)
    num_blocks = ceil(Int, n / threads_per_block)
    
    @cuda threads=threads_per_block blocks=num_blocks backsubstitution_kernel(U.data, x, d_b, N)
    
    return Array(x)
end
