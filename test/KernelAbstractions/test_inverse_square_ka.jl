function _id_int(n::Int)
    M = zeros(Int, n, n)
    for i in 1:n
        M[i, i] = 1
    end
    return M
end

function test_inverse_square_ka()
    for p in (7, 11, 101)
        for T in (Float32, Float64)
            for n in (1, 2, 3, 4, 8, 12, 16, 24, 31, 32, 33)
                A = _random_invertible_ka(n, p, T)
                X = inverse_new_ka(A, options=PLUQOptionsKA(core=PLUQOptions(inverse_strategy=:augmented)))
                AX = mod.(round.(Int, Array(A * X)), p)
                XA = mod.(round.(Int, Array(X * A)), p)
                I = _id_int(n)
                @test AX == I
                @test XA == I
            end
        end
    end
    S = CuModMatrix([1 2; 1 2], 101)
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse_new_ka(S)
end
