include("test_mod_arith_ka.jl")
include("test_perm_vectors_ka.jl")
include("test_pluq_square_ka.jl")
include("test_inverse_square_ka.jl")
include("test_inverse_rectangular_ka.jl")
include("test_batch_ka.jl")
include("test_batched_tiny_ka.jl")
include("test_matops_ka.jl")
include("test_padding_and_edgecases_ka.jl")
include("test_crosscheck_cuda_vs_ka.jl")

function test_kernel_abstractions_suite()
    @testset "mod_arith_ka" begin
        test_mod_arith_ka()
    end
    @testset "perm_vectors_ka" begin
        test_perm_vectors_ka()
    end
    @testset "pluq_square_ka" begin
        test_pluq_square_ka()
    end
    @testset "inverse_square_ka" begin
        test_inverse_square_ka()
    end
    @testset "inverse_rectangular_ka" begin
        test_inverse_rectangular_ka()
    end
    @testset "batch_ka" begin
        test_batch_ka()
    end
    @testset "batched_tiny_ka" begin
        test_batched_tiny_ka()
    end
    @testset "matops_ka" begin
        test_matops_ka()
    end
    @testset "padding_edgecases_ka" begin
        test_padding_and_edgecases_ka()
    end
    @testset "crosscheck_ka" begin
        test_crosscheck_cuda_vs_ka()
    end
end
