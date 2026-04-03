using GPUFiniteFieldMatrices

function run_ka_phase_regime_benchmark(;
    long::Bool=false,
)
    small_specs = [(16, 16, 101, Float32), (32, 32, 101, Float32), (64, 64, 101, Float32)]
    medium_specs = [(128, 128, 101, Float32), (256, 256, 101, Float32)]
    long_specs = [(384, 384, 101, Float32), (512, 512, 101, Float32), (1024, 1024, 101, Float32)]
    rows = Any[]
    append!(rows, run_ka_inverse_benchmark(specs=small_specs, trials=3))
    append!(rows, run_ka_inverse_benchmark(specs=medium_specs, trials=2))
    if long
        append!(rows, run_ka_inverse_benchmark(specs=long_specs, trials=1))
    end
    return rows
end
