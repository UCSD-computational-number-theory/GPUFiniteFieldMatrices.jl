

"""
    mat_mul_gpu_type(A::CuModMatrix, B::CuModMatrix, [mod_N])

Matrix multiplication that works directly with CuModMatrix objects.
"""
function mat_mul_gpu_type(A::CuModMatrix, B::CuModMatrix, mod_N::Integer=-1; REGIME="⊠", type=nothing)
    N = mod_N > 0 ? mod_N : A.N
    
    if cols(A) != rows(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions do not match.
            A has $rows(A) rows and $cols(A) cols, 
            B has $rows(B) rows and $cols(B) cols."
        ))
    end
    
    type = eltype(A.data)
    MAX_OPS = find_max_ops(type, N)
    
    if REGIME == "⊠"
        if MAX_OPS >= cols(A) # equal to B_rows
            REGIME = "⊡"
        elseif MAX_OPS > TILE_WIDTH
            REGIME = "⊟"
        else
            REGIME = "⊞"
        end
    end
    
    d_C = CUDA.CuArray{type}(undef, (size(A.data, 1), size(B.data, 2)))
    
    REGIME = "⊡"
    if REGIME == "⊡"
        mul!(d_C, A.data, B.data)
        d_C .%= N
    else
        error("Invalid regime: $REGIME")
    end
    return CuModMatrix(d_C, N, new_size=(rows(A), cols(B)))
end

"""
    mat_mul_type_inplace!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix, [mod_N])

In-place matrix multiplication that works directly with CuModMatrix objects.
"""
function mat_mul_type_inplace!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix, mod_N::Integer=-1, REGIME="⊠", type=nothing)
    # Use the provided modulus if available, otherwise use A's modulus
    N = mod_N > 0 ? mod_N : A.N

    if rows(C) != rows(A) || cols(C) != cols(B)
        throw(MatrixSizeMismatchException(
            "Output matrix C has incorrect dimensions"
        ))
    end
    
    if cols(A) != rows(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions do not match.
            A has $rows(A) rows and $cols(A) cols, 
            B has $rows(B) rows and $cols(B) cols."
        ))
    end
    
    type = eltype(A.data)
    MAX_OPS = find_max_ops(type, N)
    println("MAX_OPS: $MAX_OPS")
    
    if MAX_OPS >= cols(A) # equal to rows(B)
        REGIME = "⊡"
    else
        REGIME = "⊞"
    end
    
    REGIME = "⊡"
    if REGIME == "⊡"
        LinearAlgebra.mul!(C.data,A.data,B.data)
        C.data .%= N
    else
        error("Matrix multiplication of ($(rows(A)),$(cols(A))) x ($(rows(B)),$(cols(B))) not justified modulo $N for data type $type")
    end
    return C
end 
