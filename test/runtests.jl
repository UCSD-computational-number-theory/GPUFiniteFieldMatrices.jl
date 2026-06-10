using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using Suppressor
using Unroll
using ExplicitImports

using GPUFiniteFieldMatrices

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

# ExplicitImports is a CPU-runnable quality gate: it runs UNCONDITIONALLY,
# outside the `CUDA.functional()` guard below, so the GPU-less CI is not vacuous.
# Each granular check returns `nothing` on success and throws otherwise.
@testset "ExplicitImports" begin
    @test check_no_implicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_no_stale_explicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_via_owners(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_are_public(GPUFiniteFieldMatrices) === nothing
    @test check_no_self_qualified_accesses(GPUFiniteFieldMatrices) === nothing

    # `CuRef` is owned by CUDACore but re-exported through CUDA's CUBLAS
    # submodule; accessing it via CUBLAS is a re-export false positive.
    @test check_all_qualified_accesses_via_owners(
        GPUFiniteFieldMatrices; ignore=(:CuRef,)) === nothing

    # The ignored names are non-public internals that the GPU code genuinely
    # needs and for which there is no public alternative:
    #   CHOLMOD                          – SparseArrays.CHOLMOD.Dense, used in a copyto! signature
    #   CuRef, gemm!, gemv!, gemv_batched! – CUDA.CUBLAS low-level BLAS entry points
    @test check_all_qualified_accesses_are_public(
        GPUFiniteFieldMatrices;
        ignore=(:CHOLMOD, :CuRef, :gemm!, :gemv!, :gemv_batched!)) === nothing
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
