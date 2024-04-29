using CUDA, BenchmarkTools, Test

function naive_matmul_kernel(CC, A, B, N)
    row = (blockIdx().x-1) * blockDim().x + (threadIdx().x)
    col = (blockIdx().y-1) * blockDim().y + (threadIdx().y)

    total = 0
    thread = (threadIdx().z)
    # @cuprintln("row $row col $col thread $thread")

    for i = (thread-1)*8+1:(thread)*8
        total += A[row,i] * B[i,col]
    end
    CC[row,col,thread] = total

    return
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

N = 64^2
mod = 3

A = rand(1:10, N, N)
B = ones(Int, N, N)
C = zeros(Int, N, N)

d_A = CUDA.CuArray(A)
d_B = CUDA.CuArray(B)
d_CC = CUDA.CuArray(zeros(Int, N, N, 2))
d_C = CUDA.CuArray(zeros(Int, N, N))

function gpu_matmul(d_A, d_B, d_C, d_CC)
    @cuda threads=(isqrt(N),isqrt(N),2) blocks=(isqrt(N),isqrt(N)) naive_matmul_kernel(d_CC, d_A, d_B, N)
    @cuda threads=(isqrt(N),isqrt(N)) blocks=(isqrt(N),isqrt(N)) naive_flatten_kernel(d_C, d_CC, mod)
end

function cpu_matmul(A, B, C)
    C = A * B
end

@time cpu_matmul(A, B, C)
@time gpu_matmul(d_A, d_B, d_C, d_CC)

# println(C)
# println(Array(d_C))
