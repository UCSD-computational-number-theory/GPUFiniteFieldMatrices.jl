using GPUFiniteFieldMatrices
using CUDA
using Statistics
using Printf
using DelimitedFiles

function _bench_random_invertible_ka(n::Int, p::Int, T::DataType)
    for _ in 1:64
        A = CuModMatrix(rand(0:(p - 1), n, n), p; elem_type=T)
        if is_invertible_new_ka(A)
            return A
        end
    end
    error("failed to sample invertible matrix")
end

function _bench_id(n::Int)
    M = zeros(Int, n, n)
    for i in 1:n
        M[i, i] = 1
    end
    return M
end

function run_ka_inverse_benchmark(;
    specs::Vector{Tuple{Int,Int,Int,DataType}}=[(16, 16, 101, Float32), (32, 32, 101, Float32), (64, 64, 101, Float32)],
    trials::Int=5,
    mode::Symbol=:latency,
    csv_path::String="test/Experiments/ka_inverse_benchmark.csv",
)
    rows = NamedTuple[]
    for (m, n, p, T) in specs
        batch = mode == :throughput ? 8 : 1
        mats = [_bench_random_invertible_ka(n, p, T) for _ in 1:batch]
        times = Float64[]
        oks = Bool[]
        for _ in 1:trials
            t = @elapsed begin
                invs = inverse_new_batch_ka(mats)
                CUDA.synchronize()
                ok = true
                for i in eachindex(mats)
                    AX = mod.(round.(Int, Array(mats[i] * invs[i])), p)
                    ok &= AX == _bench_id(n)
                end
                push!(oks, ok)
            end
            push!(times, t / batch)
        end
        push!(rows, (
            backend="ka_cuda",
            rows=m,
            cols=n,
            p=p,
            dtype=string(T),
            operation="inverse",
            trials=trials,
            mean=mean(times),
            median=median(times),
            p95=quantile(times, 0.95),
            std=std(times),
            correctness_flag=all(oks),
        ))
        @printf("ka_inverse n=%d p=%d %s mean=%.6f\n", n, p, string(T), mean(times))
    end
    header = ["backend", "rows", "cols", "p", "dtype", "operation", "trials", "mean", "median", "p95", "std", "correctness_flag"]
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] = header
    for (i, r) in enumerate(rows)
        table[i + 1, :] = [r.backend, r.rows, r.cols, r.p, r.dtype, r.operation, r.trials, r.mean, r.median, r.p95, r.std, r.correctness_flag]
    end
    writedlm(csv_path, table, ',')
    return rows
end
