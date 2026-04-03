function test_mod_arith_ka()
    for p in (3, 5, 7, 11, 101, 257, 65521)
        @test GPUFiniteFieldMatrices.pluq_mod_reduce(-3, p) == mod(-3, p)
        @test GPUFiniteFieldMatrices.pluq_mod_mul(25, 4, p) == mod(25 * 4, p)
        x = GPUFiniteFieldMatrices.pluq_mod_inv(2, p)
        @test GPUFiniteFieldMatrices.pluq_mod_mul(x, 2, p) == mod(1, p)
    end
end
