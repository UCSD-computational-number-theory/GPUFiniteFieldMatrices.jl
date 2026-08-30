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

    opts = PLUQOptions(lazy_q=true, nftb=8)
    F2 = pluq_new(A, options=opts)
    @test F2.rank == 3
    @test pluq_check_identity(F2, A)

    batch = [A, A]
    invs = inverse_new_batch(batch, options=opts)
    @test length(invs) == 2
    I2 = mod.(Array(A * invs[1]), N)
    @test I2 == expected
end
