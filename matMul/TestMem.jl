using CUDA, BenchmarkTools, Test

function experiment(A, B, C, p)

    sharedA = CUDA.CuStaticSharedArray(int64, [128, 128])
    copyto!(sharedA,A)
    sharedB = CUDA.CuStaticSharedArray(int64, [128, 128])
    copyto!(sharedB,B)

    # Calculate row and column indices of the element in C
    row = (blockIdx().x-1) * blockDim().x + threadIdx().x
    col = (blockIdx().y-1) * blockDim().y + threadIdx().y

    for i = 0:1
        # load
    end

    CUDA.sync_threads()

    return
end

N = 128
A = rand(1:10, N, N)
B = ones(Int64, N, N)
C = zeros(Int64, N, N)

experiment(A,B,C,5)