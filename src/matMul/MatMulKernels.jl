using CUDA, BenchmarkTools, Test

CUDA.JULIA_CUDA_MEMORY_POOL

function naive_matmul_kernel(CC, A, B)
    row = (blockIdx().x-1) * blockDim().x + (threadIdx().x)
    col = (blockIdx().y-1) * blockDim().y + (threadIdx().y)

    total = 0
    thread = (threadIdx().z)
    # @cuprintln("row $row col $col thread $thread")
    for i = (thread-1)*128+1:(thread)*128
        total += A[row,i] * B[i,col]
    end
    CC[row,col,thread] = total % mod

    return
end

function twod_matmul_kernel(C, A, B, N, mod)
    row = (blockIdx().x-1) * blockDim().x + (threadIdx().x)
    col = (blockIdx().y-1) * blockDim().y + (threadIdx().y)

    total = 0
    for i = 1:N
        total += A[row,i] * B[i,col]
    end
    C[row,col] = total % mod
end

function naive_flatten_kernel(C, CC, mod)
    row = (blockIdx().x-1) * blockDim().x + threadIdx().x
    col = (blockIdx().y-1) * blockDim().y + threadIdx().y

    total = 0
    for i = 1:2
        total += CC[row,col,i] % mod
    end
    C[row,col] = total % mod

    return
end

N = 256
mod = 10
dims = 2

A = rand(1:10, N, N)
B = ones(Int8, N, N)
C = zeros(Int8, N, N)

function gpu_matmul(A, B)
    d_A = CUDA.CuArray(A)
    d_B = CUDA.CuArray(B)
    d_CC = CUDA.CuArray(zeros(Int8, N, N, dims))
    d_C = CUDA.CuArray(zeros(Int8, N, N))
    @cuda threads=(isqrt(N),isqrt(N),dims) blocks=(isqrt(N),isqrt(N)) naive_matmul_kernel(d_CC, d_A, d_B)
    @cuda threads=(isqrt(N),isqrt(N)) blocks=(isqrt(N),isqrt(N)) naive_flatten_kernel(d_C, d_CC, mod)
    return Array(d_C)
end

function cpu_matmul(A, B, C)
    C = A * B
end

println("
Benchmark details:\n
Matrix size: $N x $N, 
Modulus: $mod,
Dimensions: $dims
")
@time cpu_matmul(A, B, C)
@time gpu_matmul(A, B)

# println(C)
# println(Array(d_C))
