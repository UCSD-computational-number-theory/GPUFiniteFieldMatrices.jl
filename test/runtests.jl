using Test
using CUDA
using LinearAlgebra
using BenchmarkTools

using GPUFiniteFieldMatrices
#using Pkg

# Add the current directory to Julia's load path
#Pkg.develop(path=".")

include("GpuMatrixModN/basic_operations_test.jl")
include("GpuMatrixModN/inplace_operations_test.jl")
include("GpuMatrixModN/pluq_operations_test.jl")
include("GpuMatrixModN/matmul_operations_test.jl")
include("GpuMatrixModN/benchmark_test.jl")
include("GpuMatrixModN/stripe_mul_test.jl")


#open("test_results.log", "w") do io
#    redirect_stdout(io) do
#        redirect_stderr(io) do
            @testset "GpuMatrixModN.jl" begin
                # GPU Matrix Type Tests
                @testset "GPU Matrix Type" begin
                    test_gpu_mat()
                end
                
                # In-place Operations Tests
                @testset "In-place Operations" begin
                    test_inplace_operations()
                end
                
                # PLUQ Tests
                #@testset "PLUQ Operations" begin
                #    test_pluq()
                #end
                
                # Matrix Multiplication Tests
                @testset "Matrix Multiplication" begin
                    test_matmul()
                    test_stripe_mul()
                end
            end 
#        end
#    end
#end
