function _full_row_rank_host(m::Int, n::Int, p::Int, T::DataType)
    A = Matrix{T}(rand(0:(p - 1), m, n))
    for j in 1:m
        A[:, j] .= 0
        A[j, j] = 1
    end
    return A
end

function _full_col_rank_host(m::Int, n::Int, p::Int, T::DataType)
    A = Matrix{T}(rand(0:(p - 1), m, n))
    for i in 1:n
        A[i, :] .= 0
        A[i, i] = 1
    end
    return A
end

function test_inverse_rectangular_ka()
    for p in (7, 11, 101)
        for T in (Float32, Float64)
            for (m, n) in ((2, 3), (3, 5), (8, 16), (16, 33), (31, 64))
                A = CuModMatrix(_full_row_rank_host(m, n, p, T), p; elem_type=T)
                X = right_inverse_new_ka(A)
                AX = mod.(round.(Int, Array(A * X)), p)
                @test AX == _id_int(m)
            end
            for (m, n) in ((3, 2), (5, 3), (16, 8), (33, 16), (64, 31))
                A = CuModMatrix(_full_col_rank_host(m, n, p, T), p; elem_type=T)
                X = left_inverse_new_ka(A)
                XA = mod.(round.(Int, Array(X * A)), p)
                @test XA == _id_int(n)
            end
        end
    end
end
