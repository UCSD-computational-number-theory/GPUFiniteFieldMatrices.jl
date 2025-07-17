
function test_allocations()

    println("Begin testing allocations.")

    n = 3003
    A_data = Base.rand(0:10,n,n)
    B_data = Base.rand(0:10,n,n)
    x_data = Base.rand(0:10,n)

    A = CuModMatrix(A_data,11)
    B = CuModMatrix(B_data,11)
    x = CuModVector(x_data,11)

    C = GPUFiniteFieldMatrices.zeros(Float64,n,n,11)
    z = GPUFiniteFieldMatrices.zeros(Float64,n,11)

    #Note: BenchmarkTools doesn't seem to support tracking gpu allocations
    #as of May 2025

    result = CUDA.@timed GPUFiniteFieldMatrices.add!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.sub!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed negate!(C,A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_add!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_sub!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mul!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed elementwise_multiply!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.copy!(C,A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mod_elements!(C,3)
    @test result[:gpu_bytes] == 0

    # matmul needs to allocate a few views
    
    result = CUDA.@timed mul!(C,A,B)
    @test result[:gpu_bytes] < 20

    result = CUDA.@timed mul!(z,A,x)
    @test result[:gpu_bytes] < 20
   
    CUDA.@sync println("Done testing allocations.")
end


