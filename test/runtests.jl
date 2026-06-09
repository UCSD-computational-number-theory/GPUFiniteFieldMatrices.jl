# --- harden: DispatchDoctor return-type-stability gate for this test run ---
# Flip the package's `@stable` wrap (src/GPUFiniteFieldMatrices.jl) from its
# downstream-safe `disable` default to an enforcing mode for the test run only.
# This MUST run before `using GPUFiniteFieldMatrices` so the package compiles in
# with the gate active. `codegen_level="min"` keeps the `@stable` precompile
# overhead low. See bead gfm-kvf.4.5 / procedure test-augment.md §3,§5.
using Preferences: set_preferences!
set_preferences!(
    "GPUFiniteFieldMatrices",
    "dispatch_doctor_mode" => "error",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using Suppressor
using Unroll

using GPUFiniteFieldMatrices

# DispatchDoctor return-type-stability gate (bead gfm-kvf.4.5). The package-wide
# `@stable` net (flipped to the enforcing mode above) makes any wrapped function
# throw a `TypeInstabilityError` if it returns a non-concrete type. This runs
# unconditionally (outside the CUDA.functional() guard below) over the
# CPU-runnable entry points, so the gate bites on a GPU-less runner too; the
# GPU kernels are exercised by the gated suite below on a CUDA runner. Each
# assertion doubles as a stability check — an unstable wrapped call would throw
# here rather than return.
@testset "DispatchDoctor — return type stability" begin
    @test mod_inv(3, 7) == 5
    @test GPUFiniteFieldMatrices.inverse_permutation([2, 3, 1]) == [3, 1, 2]
    @test GPUFiniteFieldMatrices.gcd(12, 18) == 6
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
