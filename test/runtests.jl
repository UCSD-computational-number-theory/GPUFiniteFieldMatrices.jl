using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using Suppressor
using Unroll

using GPUFiniteFieldMatrices

include("CuModMatrix/basic_operations_test.jl")
include("CuModMatrix/inplace_operations_test.jl")
include("CuModMatrix/matmul_operations_test.jl")
include("CuModMatrix/benchmark_test.jl")
include("CuModMatrix/stripe_mul_test.jl")
include("CuModMatrix/allocations_test.jl")
include("CuModMatrix/timing_test.jl")
include("CuModMatrix/de_rham_test.jl")
include("CuModMatrix/permutation_test.jl")
include("CuModMatrix/triangular_test.jl")

function run_all_tests()
    @testset "CuModMatrix.jl" begin

        @testset "Triangular Inverse" begin
            test_upper_triangular_inverse()
            test_lower_triangular_inverse()
        end

        @testset "De Rham" begin
            test_de_rham()
        end

        @testset "Permutations" begin
            test_permutations()
        end
        
        @testset "GPU Matrix Type" begin
            test_gpu_mat()
        end
        
        @testset "In-place Operations" begin
            test_inplace()
        end
        
        @testset "Matrix Multiplication" begin
            test_matmul()
            test_stripe_mul()
        end

        @testset "Allocations" begin
            test_allocations()
        end

        @testset "Timings" begin
            test_timings()

            return
        end

    end 
end

# if isfile("tests.log")
#     open("tests.log", "w") do io
#         redirect_stdout(io) do
#             redirect_stderr(io) do
#                 run_all_tests()
#             end
#         end
#     end
# else
run_all_tests()
# end
