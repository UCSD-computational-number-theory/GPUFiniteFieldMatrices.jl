function test_batch_ka()
    p = 101
    mats = [_random_invertible_ka(8, p, Float32) for _ in 1:4]
    Fs = pluq_new_batch_ka(mats)
    @test length(Fs) == 4
    invs = inverse_new_batch_ka(mats)
    @test length(invs) == 4
    for i in eachindex(mats)
        AX = mod.(round.(Int, Array(mats[i] * invs[i])), p)
        @test AX == _id_int(8)
    end
end
