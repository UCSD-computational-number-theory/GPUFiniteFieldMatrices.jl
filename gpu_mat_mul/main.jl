using CUDA, BenchmarkTools, LinearAlgebra, Test
include("mat_mul_bench.jl")
include("mat_mul_hybrid.jl")

function main()
    P = 10

    benchmarks = mat_mul_benchmark_sizes(
        [[[1000,1000] [1000,1000]]]
    , P)

    DEFAULT_SIZE = 5000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    @benchmark CUDA.@sync mat_mul_gpu($A, $B, $P)
end

main()