function test_batched_tiny_ka()
    p = 101
    for n in (4, 8, 16, 32)
        mats = [_random_invertible_ka(n, p, Float32) for _ in 1:3]
        Fs = if n == 4
            pluq_batched_4x4_ka!(mats)
        elseif n == 8
            pluq_batched_8x8_ka!(mats)
        elseif n == 16
            pluq_batched_16x16_ka!(mats)
        else
            pluq_batched_32x32_ka!(mats)
        end
        @test length(Fs) == 3
        invs = if n == 4
            inverse_batched_4x4_ka!(mats)
        elseif n == 8
            inverse_batched_8x8_ka!(mats)
        elseif n == 16
            inverse_batched_16x16_ka!(mats)
        else
            inverse_batched_32x32_ka!(mats)
        end
        @test length(invs) == 3
        for i in eachindex(mats)
            AX = mod.(round.(Int, Array(mats[i] * invs[i])), p)
            @test AX == _id_int(n)
        end
    end
end
