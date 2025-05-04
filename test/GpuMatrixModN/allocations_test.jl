
function test_allocations()

    n = 3003
    A_data = Base.rand(0:10,n,n)
    B_data = Base.rand(0:10,n,n)
    x_data = Base.rand(0:10,n)

    A = GpuMatrixModN(A_data,11)
    B = GpuMatrixModN(B_data,11)
    x = GpuVectorModN(x_data,11)

    C = GPUFiniteFieldMatrices.zeros(Float32,n,n,11)
    z = GPUFiniteFieldMatrices.zeros(Float32,n,11)

    #Note: BenchmarkTools doesn't seem to support tracking gpu allocations
    #as of May 2025

    result = CUDA.@timed add!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed sub!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed negate!(C,A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_add!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_subtract!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_multiply!(C,A,2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed elementwise_multiply!(C,A,B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.copy!(C,A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mod_elements!(C,3)
    @test result[:gpu_bytes] == 0

    # matmul needs to allocate a few views
    
    result = CUDA.@timed multiply!(C,A,B)
    @test result[:gpu_bytes] < 10

    result = CUDA.@timed multiply!(z,A,x)
    @test result[:gpu_bytes] < 10
   
end


