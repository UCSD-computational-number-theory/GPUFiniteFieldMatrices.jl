using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using Suppressor
using Unroll
using Aqua

using GPUFiniteFieldMatrices

# Aqua quality gate — CPU-runnable, so it runs unconditionally (outside the
# CUDA.functional() guard below). Limited form matching test/Quality/aqua.jl:
# stale_deps / deps_compat stay off here; the FULL Aqua step (those two enabled)
# lives in bead gfm-kvf.4.1.1 after the compat work lands.
@testset "Aqua" begin
    Aqua.test_all(GPUFiniteFieldMatrices; stale_deps=false, deps_compat=false)
end

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

if CUDA.functional()
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
    run_all_tests()
else
    @testset "CuModMatrix.jl" begin
        @info "CUDA is not functional on this machine; skipping GPU-dependent tests."
    end
end
