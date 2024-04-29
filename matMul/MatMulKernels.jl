using CUDA, BenchmarkTools, Test

function naive_matmul_kernel(CC, A, B)
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
    for i = 1:8
        total += CC[row,col,i] % mod
    end
    C[row,col] = total % mod

    return
end

N = 64
mod = 10
dims = 8

A = rand(1:10, N, N)
B = ones(Int, N, N)
C = zeros(Int, N, N)

function gpu_matmul(A, B)
    d_A = CUDA.CuArray(A)
    d_B = CUDA.CuArray(B)
    d_CC = CUDA.CuArray(zeros(Int, N, N, dims))
    d_C = CUDA.CuArray(zeros(Int, N, N))
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
