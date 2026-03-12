function _random_invertible_matrix(n::Int, N::Int; max_tries::Int=40)
    for _ in 1:max_tries
        Ahost = rand(0:(N - 1), n, n)
        A = CuModMatrix(Ahost, N)
        if is_invertible_new(A)
            return A
        end
    end
    error("failed to sample invertible matrix of size $n modulo $N")
end

function _id_matrix(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i in 1:n
        M[i, i] = one(T)
    end
    return M
end

function test_identity_range()
    N = 101
    for n in 1:64
        Ih = _id_matrix(Int, n)
        Icu = CuModMatrix(Ih, N)
        F = pluq_new(Icu)
        @test F.rank == n
        @test pluq_check_identity(F, Icu)
        Iinv = inverse_new(Icu)
        @test Array(Iinv) == Ih
    end
end

function test_random_invertible_batch()
    N = 101
    for n in (2, 3, 4, 5, 7, 8, 12, 16, 24, 31, 32, 33, 48, 64)
        A = _random_invertible_matrix(n, N)
        F = pluq_new(A)
        @test F.rank == n
        @test pluq_check_identity(F, A)
        Ainv = inverse_new(A)
        left = mod.(Array(A * Ainv), N)
        right = _id_matrix(eltype(left), n)
        @test left == right
    end
end

function test_random_singular_batch()
    N = 101
    for n in (2, 3, 4, 5, 8, 16, 32, 33, 64)
        Ahost = rand(0:(N - 1), n, n)
        if n > 1
            Ahost[2, :] = Ahost[1, :]
        end
        A = CuModMatrix(Ahost, N)
        @test !is_invertible_new(A)
    end
end

function test_padding_sensitive_sizes()
    N = 101
    for n in (1, 2, 3, 31, 32, 33, 63, 64)
        A = _random_invertible_matrix(n, N)
        F = pluq_new(A)
        @test pluq_check_identity(F, A)
    end
end
