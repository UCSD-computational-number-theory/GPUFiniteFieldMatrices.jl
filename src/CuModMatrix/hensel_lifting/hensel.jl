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

function triangular_inverse_kernel(A::CuArray, A_inv::CuArray, N::Int, is_upper::Bool)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)
    
    if tid <= n
        for j = 1:n
            # Choose iteration order based on matrix type
            if is_upper
                iter_range = n:-1:1  # Bottom to top for upper triangular
            else
                iter_range = 1:n     # Top to bottom for lower triangular
            end
            
            for i in iter_range
                CUDA.sync_blocks()
                
                if tid == i
                    sum = i == j ? 1 : 0
                    
                    if is_upper
                        # For upper triangular, sum over elements to the right
                        dep_range = i+1:n
                    else
                        # For lower triangular, sum over elements to the left
                        dep_range = 1:i-1
                    end
                    
                    for k in dep_range
                        sum = (sum - (A[i, k] * A_inv[k, j]) % N) % N
                        if sum < 0
                            sum += N
                        end
                    end
                    
                    diag_inv = mod_inv(A[i, i], N)
                    A_inv[i, j] = (sum * diag_inv) % N
                end
            end
        end
    end
    
    return nothing
end

function backward_sub_gpu_type(A::CuModMatrix, is_upper::Bool)
    d_A = GPUFiniteFieldMatrices.zeros(Int, rows(A), rows(A))

    @cuda threads=(TILE_WIDTH) blocks=(div(rows(A),TILE_WIDTH)+1) triangular_inverse_kernel(A.data, d_A, A.N, is_upper)
    
    return CuModMatrix(d_A, A.N, new_size=(rows(A), rows(A)))
end