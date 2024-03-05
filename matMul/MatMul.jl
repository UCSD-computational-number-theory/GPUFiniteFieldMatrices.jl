using CUDA, LinearAlgebra
using Random
using Test, Profile, BenchmarkTools

function naive_matmul()
    A, B = gen_sample_mat()
    m1, n1 = size(A)
    m2, n2 = size(B)

    if n1 != m2
        throw("Incorrect matrix dimensions")
    end

    C = zeros(UInt32, m1, n2)

    for i in 1:m1
        for j in 1:n2
            for k in 1:n1
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
end

function cpu_matmul()
    A, B = gen_sample_mat()
    m1, n1 = size(A)
    m2, n2 = size(B)

    if n1 != m2
        throw("Incorrect matrix dimensions")
    end

    C = zeros(UInt32, m1, n2)

    Threads.@threads for i in 0:m1
        for j in 0:n2
            @inbounds C[i, j] = dot(A[i, :], B[:, i])
        end
    end
end

function linalg_matmul()
    A, B = gen_sample_mat()
    A * B
end

function gpu_matmul()
    d_A = CUDA.rand(Int32, (100, 100))
    d_B = CUDA.rand(Int32, (100, 100))
    d_C = d_A * d_B
    return d_C
end

function gen_sample_mat()
    m, n, p = 100, 100, 100
    A = rand(UInt32(0):UInt32(p), (m, n))
    B = rand(UInt32(0):UInt32(p), (m, n))
    return A, B
end

function compare()
    println("Naive time: ")
    @time naive_matmul()
    println("CPU Parallelized time: ")
    @time cpu_matmul()
    println("GPU Parallelized time: ")
    @time gpu_matmul()
    println("DONE")
end

compare()