using CUDA, LinearAlgebra

function mat_mul_plain(A, B, P)
    """
    Using default implementation of MatMul.
    """

    C = A * B
    C .%= P
    return C
end

function mat_mul_plain!(C, A, B, P)
    LinearAlgebra.mul!(C,A,B)
    C .%= P
    return C
end
