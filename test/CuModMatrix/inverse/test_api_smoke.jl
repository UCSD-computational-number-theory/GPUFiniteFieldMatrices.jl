function test_api_smoke()
    N = 101
    Ahost = [
        3 5 7
        2 4 9
        1 8 6
    ]
    A = CuModMatrix(Ahost, N)
    F = pluq_new(A)
    @test length(F.p) == 3
    @test length(F.q) == 3
    @test F.rank == 3
    @test pluq_check_identity(F, A)
    Ainv = inverse_new(A)
    I1 = mod.(Array(A * Ainv), N)
    expected = Matrix{eltype(I1)}([1 0 0; 0 1 0; 0 0 1])
    @test I1 == expected
end
