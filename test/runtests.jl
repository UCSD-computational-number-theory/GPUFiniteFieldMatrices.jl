using Test
using CUDA
using LinearAlgebra
using BenchmarkTools

using GPUFiniteFieldMatrices
#using Pkg

# Add the current directory to Julia's load path
#Pkg.develop(path=".")

include("CuModMatrix/basic_operations_test.jl")
include("CuModMatrix/inplace_operations_test.jl")
include("CuModMatrix/pluq_operations_test.jl")
include("CuModMatrix/matmul_operations_test.jl")
include("CuModMatrix/benchmark_test.jl")
include("CuModMatrix/stripe_mul_test.jl")
include("CuModMatrix/allocations_test.jl")
include("CuModMatrix/timing_test.jl")


#open("test_results.log", "w") do io
#    redirect_stdout(io) do
#        redirect_stderr(io) do
            @testset "CuModMatrix.jl" begin
                # GPU Matrix Type Tests
                @testset "GPU Matrix Type" begin
                    test_gpu_mat()
                end
                
                # In-place Operations Tests
                @testset "In-place Operations" begin
                    test_inplace()
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

                @testset "Allocations" begin
                    test_allocations()
                end

                @testset "Timings" begin
                    test_timings()
                end

            end 
#        end
#    end
#end
