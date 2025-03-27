using CUDA, SparseArrays, CUDA.CUSPARSE, LinearAlgebra
using Profile
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul, AdjOrTrans

function generic_matmatmul!(C::CuSparseMatrixCSC{T}, tA, tB, A::CuSparseMatrixCSC{T}, B::CuSparseMatrixCSC{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    CUDA.CUSPARSE.gemm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end

function testMul()
    n = 10000
    A = sprand(Int, n, n, 0.001)
    B = sprand(Int, n, n, 0.001)
    C = spzeros(Int, n, n)

    d_A = CuSparseMatrixCSC(A)
    d_B = CuSparseMatrixCSC(B)
    d_C = CuSparseMatrixCSC(C)

    d_C = d_A * d_B
    # generic_matmatmul!(d_C,'N','N',d_A,d_B,1,0)
    # d_C = CUDA.CUSPARSE.gemm!('N','N',1,d_A,d_B,0,d_C)

    println(dump(d_C))
end

testMul()