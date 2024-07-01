using CUDA, BenchmarkTools, Test

function experiment(A, B, C)

    # sharedA = CUDA.CuStaticSharedArray(A, (64, 64))
    # sharedB = CUDA.CuStaticSharedArray(B, (64, 64))

    # row = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # col = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # for i = 0:1
    #     sharedA[row][col] = A[row][col]
    #     sharedB[row][col] = B[row][col]
    # end

    # CUDA.sync_threads()

    return
end

N = 64
dims = 2

A = rand(1:10, N, N)
B = ones(Int64, N, N)
C = zeros(Int64, N, N)

@cuda threads=(isqrt(N),isqrt(N),dims) blocks=(isqrt(N),isqrt(N)) experiment(A,B,C)