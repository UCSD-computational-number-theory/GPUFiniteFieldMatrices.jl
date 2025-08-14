for i = 1000:1000:20000
    p = 11
    R = GF(p)
    MS = matrix_space(R, i, 1000)
    A_nemo = MS([R(x) for x in rand(1:p, i, 1000)])
    println("Testing matrix of size $(size(A_nemo))")
    @btime Nemo.is_invertible_with_inverse($A_nemo, side=:left)
end

for i = 1000:1000:10000
    p = 11
    A = rand(1:p, i, i)
    d_A = CuModMatrix(A, p)
    println("Testing matrix of size $(size(d_A))")
    @btime GPUFiniteFieldMatrices.is_invertible_with_inverse($d_A)
end