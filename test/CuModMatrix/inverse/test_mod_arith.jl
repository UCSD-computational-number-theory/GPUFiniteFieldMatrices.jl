function test_mod_arith()
    N = 101
    @test GPUFiniteFieldMatrices.pluq_mod_add(100, 5, N) == 4
    @test GPUFiniteFieldMatrices.pluq_mod_sub(3, 7, N) == 97
    @test GPUFiniteFieldMatrices.pluq_mod_mul(25, 4, N) == 100
    @test GPUFiniteFieldMatrices.pluq_mod_inv(2, N) == 51
    @test GPUFiniteFieldMatrices.pluq_is_prime(101)
    @test !GPUFiniteFieldMatrices.pluq_is_prime(100)
end
