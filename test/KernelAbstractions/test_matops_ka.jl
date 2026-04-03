using LinearAlgebra

function test_matops_ka()
    cases = [
        (1, 1, 1),
        (2, 3, 4),
        (7, 8, 5),
        (16, 16, 16),
        (31, 32, 33),
        (63, 64, 65),
    ]
    for p in (2, 3, 5, 7, 11, 101, 257)
        for T in (Float32, Float64)
            for (m, k, n) in cases
                A = CuModMatrix(rand(0:(p - 1), m, k), p; elem_type=T)
                B = CuModMatrix(rand(0:(p - 1), k, n), p; elem_type=T)
                C = GPUFiniteFieldMatrices.zeros(T, m, n, p)
                mul_ka!(C, A, B)
                @test mod.(round.(Int, Array(C)), p) == mod.(round.(Int, Array(A)) * round.(Int, Array(B)), p)
            end
            for (m, _, _) in cases
                A = CuModMatrix(rand(0:(p - 1), m, m), p; elem_type=T)
                B = CuModMatrix(rand(0:(p - 1), m, m), p; elem_type=T)
                C = GPUFiniteFieldMatrices.zeros(T, m, m, p)
                D = GPUFiniteFieldMatrices.zeros(T, m, m, p)
                add_ka!(C, A, B)
                sub_ka!(D, C, B)
                @test mod.(round.(Int, Array(D)), p) == mod.(round.(Int, Array(A)), p)
                Z = GPUFiniteFieldMatrices.zeros(T, m, m, p)
                sub_ka!(Z, A, A)
                @test all(mod.(round.(Int, Array(Z)), p) .== 0)
                Id = CuModMatrix(Matrix{Int}(LinearAlgebra.I, m, m), p; elem_type=T)
                L = GPUFiniteFieldMatrices.zeros(T, m, m, p)
                R = GPUFiniteFieldMatrices.zeros(T, m, m, p)
                mul_ka!(L, A, Id)
                mul_ka!(R, Id, A)
                @test mod.(round.(Int, Array(L)), p) == mod.(round.(Int, Array(A)), p)
                @test mod.(round.(Int, Array(R)), p) == mod.(round.(Int, Array(A)), p)
            end
        end
    end
end
