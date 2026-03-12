function test_perm_vectors()
    p = GPUFiniteFieldMatrices.pluq_init_perm(6)
    @test p == [1, 2, 3, 4, 5, 6]
    pinv = GPUFiniteFieldMatrices.pluq_inverse_perm([3, 1, 2, 6, 4, 5])
    @test pinv == [2, 3, 1, 5, 6, 4]
    q = [1, 2, 3, 4, 5, 6]
    locperm = [2, 1, 3]
    GPUFiniteFieldMatrices.pluq_compose_segment!(q, 2, locperm)
    @test q == [1, 3, 2, 4, 5, 6]
end
