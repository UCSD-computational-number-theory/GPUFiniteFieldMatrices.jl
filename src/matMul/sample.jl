using CUDA, BenchmarkTools, Test
CUDA.allowscalar(false) # This is TRUE now

N = 100
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
result = @test all(Array(y_d) .== 3.0f0)
println(result)

# a = CuArray([1,2,3,4])
# println(a)

# a .+= 1
# println(a)