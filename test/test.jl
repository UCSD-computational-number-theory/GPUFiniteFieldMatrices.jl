using GPUFiniteFieldMatrices
using NVTX

p = 11
i = 7000

NVTX.@range "Init A" begin
    A = rand(1:p, i, i)
end

NVTX.@range "Init d_A" begin
    d_A = CuModMatrix(A, p)
end

NVTX.@range "is_invertible_with_inverse" begin
    GPUFiniteFieldMatrices.is_invertible_with_inverse(d_A)
end

NVTX.@range "Init A" begin
    A = rand(1:p, i, i)
end

NVTX.@range "Init d_A" begin
    d_A = CuModMatrix(A, p)
end

NVTX.@range "is_invertible_with_inverse" begin
    GPUFiniteFieldMatrices.is_invertible_with_inverse(d_A)
end