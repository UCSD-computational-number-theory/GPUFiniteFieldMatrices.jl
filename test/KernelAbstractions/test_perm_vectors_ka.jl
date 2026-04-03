function test_perm_vectors_ka()
    p = GPUFiniteFieldMatrices.pluq_init_perm_ka(5)
    @test p == [1, 2, 3, 4, 5]
    GPUFiniteFieldMatrices.pluq_compose_segment_ka!(p, 2, [2, 1, 3])
    @test p == [1, 3, 2, 4, 5]
    pinv = GPUFiniteFieldMatrices.pluq_inverse_perm_ka([3, 1, 2])
    @test pinv == [2, 3, 1]
end
