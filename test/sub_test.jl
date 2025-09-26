using GPUFiniteFieldMatrices
using NVTX

p = 11
i = 1000

NVTX.@range "Init A" begin
    A = rand(1:p, i, i)
end

NVTX.@range "Init d_A" begin
    d_A = CuModMatrix(A, p)
end

NVTX.@range "Setup PLUQ" begin
    P, L, U, Q = GPUFiniteFieldMatrices._setup_PLUQ(d_A)
end

NVTX.@range "Lower triangular inverse" begin
    L_inv = GPUFiniteFieldMatrices.lower_triangular_inverse(L)
end

NVTX.@range "Upper triangular inverse" begin
    U_inv = GPUFiniteFieldMatrices.upper_triangular_inverse(U)
end

NVTX.@range "Apply col inv perm" begin
    GPUFiniteFieldMatrices.apply_col_inv_perm!(P, U_inv)
end

NVTX.@range "Apply row inv perm" begin
    GPUFiniteFieldMatrices.apply_row_inv_perm!(Q, L_inv)
end

NVTX.@range "Multiply" begin
    A_inv = U_inv * L_inv
end