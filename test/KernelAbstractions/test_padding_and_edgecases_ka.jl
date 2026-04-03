function test_padding_and_edgecases_ka()
    p = 101
    for n in (31, 32, 33, 63, 64, 65, 96, 128)
        A = _random_invertible_ka(n, p, Float32)
        F = pluq_new_ka(A)
        @test F.rank == n
        X = inverse_new_ka(A)
        AX = mod.(round.(Int, Array(A * X)), p)
        @test AX == _id_int(n)
    end
    @test_throws GPUFiniteFieldMatrices.CuModMatrixNotSquareException inverse_new_ka(CuModMatrix(rand(0:100, 2, 3), 101))
end
