using CUDA, LinearAlgebra, IterTools
# include("../gpu_matrix_mod_N/gpu_mat.jl")

"""
    mat_mul_gpu_type(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, [mod_N])

Matrix multiplication that works directly with GpuMatrixModN objects.
"""
function mat_mul_gpu_type(A::GpuMatrixModN, B::GpuMatrixModN, C::GpuMatrixModN=nothing, mod_N::Integer=-1)

    N = mod_N > 0 ? mod_N : A.N

    if C !== nothing && (C.rows != A.rows || C.cols != B.cols)
        throw(MatrixSizeMismatchException(
            "Output matrix C has incorrect dimensions"
        ))
    end
    
    if A.cols != B.rows
        throw(MatrixSizeMismatchException(
            "Matrix dimensions do not match.
            A has $A.rows rows and $A.cols cols, 
            B has $B.rows rows and $B.cols cols."
        ))
    end
    
    type = eltype(A.data)
    MAX_OPS = find_max_ops(type, N)
    
    if MAX_OPS >= A.cols # equal to B.rows
        REGIME = "⊡"
    else
        REGIME = "⊞"
    end
    
    REGIME = "⊡"
    if REGIME == "⊡"
        if C === nothing
            d_C = CUDA.CuArray{type}(undef, (A.rows, B.cols))
            d_C = mod.(A.data * B.data, N)
            return GpuMatrixModN(d_C, N, new_rows = A.rows, new_cols = B.cols)
        else
            d_C = mod.(A.data * B.data, N)
            C.data = d_C
            return C
        end
    else
        error("Invalid regime: $REGIME")
    end

    return
end

function mat_mul_gpu_type(A::GpuMatrixModN, B::GpuMatrixModN, mod_N=-1)

    N = mod_N > 0 ? mod_N : A.N
    
    if A.cols != B.rows
        throw(MatrixSizeMismatchException(
            "Matrix dimensions do not match.
            A has $A.rows rows and $A.cols cols, 
            B has $B.rows rows and $B.cols cols."
        ))
    end
    
    type = eltype(A.data)
    MAX_OPS = find_max_ops(type, N)
    
    if MAX_OPS >= A.cols # equal to B.rows
        REGIME = "⊡"
    else
        REGIME = "⊞"
    end
    
    REGIME = "⊡"
    if REGIME == "⊡"
        d_C = CUDA.CuArray{type}(undef, (A.rows, B.cols))
        d_C = mod.(A.data * B.data, N)
        return GpuMatrixModN(d_C, N, new_rows = A.rows, new_cols = B.cols)
    else
        error("Invalid regime: $REGIME")
    end

    return
end

# """
#     mat_mul_type_inplace!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, [mod_N])

# In-place matrix multiplication that works directly with GpuMatrixModN objects.
# """
# function mat_mul_type_inplace!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, mod_N::Integer=-1, REGIME="⊠", type=nothing)
#     # Use the provided modulus if available, otherwise use A's modulus
#     N = mod_N > 0 ? mod_N : A.N
    
#     # Check if matrices have compatible dimensions
#     A_rows, A_cols = size(A.data)
#     B_rows, B_cols = size(B.data)
#     C_rows, C_cols = size(C.data)
    
#     if A_cols != B_rows
#         error(
#             "Matrix dimensions do not match.
#             A has $A_rows rows and $A_cols cols, 
#             B has $B_rows rows and $B_cols cols."
#         ) 
#     end
    
#     if C_rows != A_rows || C_cols != B_cols
#         error(
#             "Output matrix C has incorrect dimensions.
#             C has $C_rows rows and $C_cols cols, but needs $A_rows rows and $B_cols cols."
#         )
#     end
    
#     # Use the element type from the inputs if not specified
#     if type === nothing
#         type = eltype(A.data)
#     end
    
#     MAX_OPS = find_max_ops(type, N)
    
#     if REGIME == "⊠"
#         if MAX_OPS >= A_cols # equal to B_rows
#             REGIME = "⊡"
#         elseif MAX_OPS > TILE_WIDTH
#             REGIME = "⊟"
#         else
#             REGIME = "⊞"
#         end
#     end
    
#     d_A = A.data
#     d_B = B.data
#     d_C = C.data
    
#     if REGIME == "⊡"
#         # Simple matrix multiplication
#         d_C = d_A * d_B
#         d_C = mod.(d_C, N)
#     elseif REGIME == "⊟"
#         # Use the no_ops kernel
#         @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_cols, TILE_WIDTH), div(A_rows, TILE_WIDTH)) mat_mul_no_ops(d_A, d_B, d_C, N, A_rows, type)
#     elseif REGIME == "⊞"
#         # Use the ops kernel that handles overflow
#         @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_cols, TILE_WIDTH), div(A_rows, TILE_WIDTH)) mat_mul_ops(d_A, d_B, d_C, N, A_rows, type, MAX_OPS)
#     else
#         error("Invalid regime: $REGIME")
#     end
    
#     # Copy result directly to C's GPU memory
#     C_inds = CartesianIndices((1:A_rows, 1:B_cols))
#     d_C_inds = CartesianIndices((1:A_rows, 1:B_cols))
#     copyto!(C.data, C_inds, d_C, d_C_inds)
    
#     return C
# end 