function _random_invertible_ka(n::Int, p::Int, T::DataType)
    for _ in 1:64
        A = CuModMatrix(rand(0:(p - 1), n, n), p; elem_type=T)
        if is_invertible_new_ka(A)
            return A
        end
    end
    error("failed to sample invertible matrix")
end

function test_pluq_square_ka()
    opts = PLUQOptionsKA()
    for p in (7, 11, 101)
        for T in (Float32, Float64)
            for n in (1, 2, 3, 4, 7, 8, 16, 31, 32, 33)
                A = _random_invertible_ka(n, p, T)
                F = pluq_new_ka(A, options=opts)
                @test F.rank == n
                @test sort(F.p) == collect(1:n)
                @test sort(F.q) == collect(1:n)
                @test pluq_check_identity_ka(F, A, options=opts)
            end
        end
    end
end
