using GPUFiniteFieldMatrices
using Test

# Include all test files
# GPU Matrix Type Tests
include("gpu_mat/basic_operations_test.jl")
include("gpu_mat/inplace_operations_test.jl")

# RREF Tests
include("rref/rref_operations_test.jl")

# Matrix Multiplication Tests
include("matmul/matmul_operations_test.jl")

# Performance Tests (optional)
# include("performance/benchmark_test.jl")

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
    
    # Performance Tests are optional and can be run separately
    # @testset "Performance" begin
    #     run_benchmarks()
    # end
end 