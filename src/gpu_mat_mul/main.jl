using CUDA, BenchmarkTools, LinearAlgebra, Test
include("mat_mul_bench.jl")
include("mat_mul_hybrid.jl")
include("mat_mul_plain.jl")
include("mat_mul_no_ops.jl")
include("mat_mul_ops.jl")

const global TILE_WIDTH = 25

function main()
    benchmarks = mat_mul_benchmark_all(
        [
            "⊡", "⊟", "⊞"
        ],
        [
            Float64,
            Float32,
            Float16
        ],
        [
            [[5000,5000] [5000,5000]]
        ],
        [
            343
        ]
    )

    # benchmarks = mat_mul_benchmark_all(
    #     [
    #         "⊡", "⊟", "⊞"
    #     ],
    #     [
    #         Float32,
    #         Float64,
    #     ],
    #     [
    #         [[1000,1000] [1000,1000]]
    #     ],
    #     [
    #         9, 101
    #     ]
    # )

    # DEFAULT_SIZE = 100
    # A = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    # B = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    # N = 9
    # type = Float64

    # A_rows, A_cols = size(A)
    # B_rows,B_cols = size(B)

    # A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    # A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH
    # B_padded_cols = ceil(Int, B_cols / TILE_WIDTH) * TILE_WIDTH

    # Ainds = CartesianIndices(A)
    # d_Ainds = CartesianIndices((1:A_rows,1:A_cols))
    # Binds = CartesianIndices(B)
    # d_Binds = CartesianIndices((1:B_rows,1:B_cols))

    # d_A = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_cols))
    # d_B = CUDA.CuArray{Int}(undef, (A_padded_cols, B_padded_cols))
    # d_C = CUDA.CuArray{Int}(undef, (A_padded_rows, B_padded_cols))

    # copyto!(A, Ainds, d_A, d_Ainds)
    # copyto!(B, Binds, d_B, d_Binds)

    # MAX_OPS = find_max_ops(type, N)

    # @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(B_padded_cols,TILE_WIDTH),div(A_padded_rows,TILE_WIDTH)) mat_mul_ops(d_A,d_B,d_C,N,A_padded_rows,type,MAX_OPS)

    # C = A * B
    # C = mod.(C, N)
    # return all(C .== Array(d_C)[1:A_rows, 1:B_cols])

    print(benchmarks)

    return
end

main()