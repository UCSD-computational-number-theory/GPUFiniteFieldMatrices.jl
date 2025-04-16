using Test
using Pkg

# Add the current directory to Julia's load path
Pkg.develop(path=".")

include("gpu_mat/basic_operations_test.jl")
include("gpu_mat/inplace_operations_test.jl")
include("pluq/pluq_operations_test.jl")
include("matmul/matmul_operations_test.jl")
include("performance/benchmark_test.jl")

open("test_results.log", "w") do io
    redirect_stdout(io) do
        redirect_stderr(io) do
            @testset "GPUFiniteFieldMatrices.jl" begin
                # GPU Matrix Type Tests
                @testset "GPU Matrix Type" begin
                    test_gpu_mat()
                end
                
                # # In-place Operations Tests
                # @testset "In-place Operations" begin
                #     test_inplace_operations()
                # end
                
                # # PLUQ Tests
                # @testset "PLUQ Operations" begin
                #     test_pluq()
                # end
                
                # # Matrix Multiplication Tests
                # @testset "Matrix Multiplication" begin
                #     test_matmul()
                # end
            end 
        end
    end
end
