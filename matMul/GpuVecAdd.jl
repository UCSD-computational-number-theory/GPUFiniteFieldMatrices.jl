using CUDA
using Test, BenchmarkTools

N = 1000000
x_d = CUDA.zeros(Int, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.zeros(Int, N)  # a vector stored on the GPU filled with 2.0

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
fill!(x_d, 1)

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

@btime bench_gpu1!($y_d, $x_d)