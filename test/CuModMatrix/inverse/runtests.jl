include("test_mod_arith.jl")
include("test_perm_vectors.jl")
include("test_basecase_pluq.jl")
include("test_api_smoke.jl")
include("test_rank_and_q.jl")
include("test_extensive.jl")
include("test_rectangular.jl")
include("test_phase0_regimes.jl")

function test_inverse_rewrite()
    @testset "Mod Arithmetic" begin
        test_mod_arith()
    end
    @testset "Permutation Vectors" begin
        test_perm_vectors()
    end
    @testset "Basecase PLUQ" begin
        test_basecase_pluq()
    end
    @testset "API Smoke" begin
        test_api_smoke()
    end
    @testset "Rank And Q" begin
        test_rank_and_q()
    end
    @testset "Identity Range" begin
        test_identity_range()
    end
    @testset "Random Invertible Batch" begin
        test_random_invertible_batch()
    end
    @testset "Random Singular Batch" begin
        test_random_singular_batch()
    end
    @testset "Padding Sensitive Sizes" begin
        test_padding_sensitive_sizes()
    end
    @testset "Rectangular Inverses" begin
        test_rectangular_inverses()
    end
    @testset "Rectangular Rank And Failures" begin
        test_rectangular_rank_and_failures()
    end
    @testset "Phase0 Regime Matrix Grid" begin
        test_phase0_regime_matrix_grid()
    end
end
