function test_crosscheck_cuda_vs_ka()
    p = 101
    for n in (2, 3, 4, 8, 16, 24, 32, 48, 64)
        A = _random_invertible_ka(n, p, Float32)
        Xka = inverse_new_ka(A)
        Xcu = inverse_new(A)
        Ika = mod.(round.(Int, Array(A * Xka)), p)
        Icu = mod.(round.(Int, Array(A * Xcu)), p)
        @test Ika == _id_int(n)
        @test Icu == _id_int(n)
    end
end
