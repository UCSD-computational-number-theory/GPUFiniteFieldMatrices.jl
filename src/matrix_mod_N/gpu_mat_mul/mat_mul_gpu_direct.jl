using CUDA, LinearAlgebra, IterTools
include("../gpu_mat_type/gpu_mat.jl")

"""
    mat_mul_gpu_type(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, [mod_N])

Matrix multiplication that works directly with GPUFiniteFieldMatrix objects.
"""
function mat_mul_gpu_type(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1, REGIME="⊠", type=nothing, C=nothing)
    # TODO: Put if statement to check size of C matches size of A x B.
    # TODO: Change error --> throw, error alwast stops but throw can be try-except-handled
    # Use the provided modulus if available, otherwise use A's modulus
    N = mod_N > 0 ? mod_N : A.N
    
    # Check if matrices have compatible dimensions
    A_rows, A_cols = size(A)
    B_rows, B_cols = size(B)
    
    if A_cols != B_rows
        error(
            "Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end
    
    # Use the element type from the input if not specified
    if type === nothing
        type = eltype(A.data)
    end
    
    MAX_OPS = find_max_ops(type, N)
    
    if REGIME == "⊠"
        if MAX_OPS >= A_cols # equal to B_rows
            REGIME = "⊡"
        elseif MAX_OPS > TILE_WIDTH
            REGIME = "⊟"
        else
            REGIME = "⊞"
        end
    end
    
    d_A = A.data
    d_B = B.data
    if C === nothing
        d_C = CUDA.CuArray{t}(undef, (A_rows, B_cols))
    else
        if C.rows != A_rows || C.cols != B_cols
            throw(MatrixSizeMismatchException(
                "Output matrix C has incorrect dimensions.
                C has $C_rows rows and $C_cols cols, but needs $A_rows rows and $B_cols cols."
            ))
        end
        d_C = C.data
    end
    
    if REGIME == "⊡"
        # Simple matrix multiplication
        if C === nothing
            d_C = d_A * d_B
            d_C = mod.(d_C, N)
        else
            C.data = mod.(d_A * d_B, N)
        end
    elseif REGIME == "⊟"
        # Use the no_ops kernel
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_padded_cols, TILE_WIDTH), div(A_padded_rows, TILE_WIDTH)) mat_mul_no_ops(d_A, d_B, d_C, N, A_padded_rows, type)
    elseif REGIME == "⊞"
        # Use the ops kernel that handles overflow
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_padded_cols, TILE_WIDTH), div(A_padded_rows, TILE_WIDTH)) mat_mul_ops(d_A, d_B, d_C, N, A_padded_rows, type, MAX_OPS)
    else
        error("Invalid regime: $REGIME")
    end
    
    if C === nothing
        return GPUFiniteFieldMatrix(d_C, A_rows, B_cols, N)
    else
        return C
    end
end

"""
    mat_mul_type_inplace!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, [mod_N])

In-place matrix multiplication that works directly with GPUFiniteFieldMatrix objects.
"""
function mat_mul_type_inplace!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1, REGIME="⊠", type=nothing)
    # Use the provided modulus if available, otherwise use A's modulus
    N = mod_N > 0 ? mod_N : A.N
    
    # Check if matrices have compatible dimensions
    A_rows, A_cols = size(A)
    B_rows, B_cols = size(B)
    C_rows, C_cols = size(C)
    
    if A_cols != B_rows
        error(
            "Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end
    
    if C_rows != A_rows || C_cols != B_cols
        error(
            "Output matrix C has incorrect dimensions.
            C has $C_rows rows and $C_cols cols, but needs $A_rows rows and $B_cols cols."
        )
    end
    
    # Use the element type from the inputs if not specified
    if type === nothing
        type = eltype(A.data)
    end
    
    MAX_OPS = find_max_ops(type, N)
    
    if REGIME == "⊠"
        if MAX_OPS >= A_cols # equal to B_rows
            REGIME = "⊡"
        elseif MAX_OPS > TILE_WIDTH
            REGIME = "⊟"
        else
            REGIME = "⊞"
        end
    end
    
    d_A = A.data
    d_B = B.data
    d_C = C.data
    
    if REGIME == "⊡"
        # Simple matrix multiplication
        d_C = d_A * d_B
        d_C = mod.(d_C, N)
    elseif REGIME == "⊟"
        # Use the no_ops kernel
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_padded_cols, TILE_WIDTH), div(A_padded_rows, TILE_WIDTH)) mat_mul_no_ops(d_A, d_B, d_C, N, A_padded_rows, type)
    elseif REGIME == "⊞"
        # Use the ops kernel that handles overflow
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(B_padded_cols, TILE_WIDTH), div(A_padded_rows, TILE_WIDTH)) mat_mul_ops(d_A, d_B, d_C, N, A_padded_rows, type, MAX_OPS)
    else
        error("Invalid regime: $REGIME")
    end
    
    # Copy result directly to C's GPU memory
    C_inds = CartesianIndices((1:A_rows, 1:B_cols))
    d_C_inds = CartesianIndices((1:A_rows, 1:B_cols))
    copyto!(C.data, C_inds, d_C, d_C_inds)
    
    return C
end 