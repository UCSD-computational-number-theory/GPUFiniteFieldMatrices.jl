using Test

include("gpu_mat/basic_operations_test.jl")
include("gpu_mat/inplace_operations_test.jl")
include("rref/rref_operations_test.jl")
include("matmul/matmul_operations_test.jl")
include("performance/benchmark_test.jl")

@testset "GPUFiniteFieldMatrices.jl" begin
    # GPU Matrix Type Tests
    @testset "GPU Matrix Type" begin
        test_gpu_mat()
    end
    
    # In-place Operations Tests
    @testset "In-place Operations" begin
        test_inplace_operations()
    end
    
    # RREF Tests
    @testset "RREF Operations" begin
        test_rref()
    end
    
    # Matrix Multiplication Tests
    @testset "Matrix Multiplication" begin
        test_matmul()
    end
end 