using GPUFiniteFieldMatrices
using CUDA
using Statistics
using DelimitedFiles

function run_ka_matops_benchmark(;
    specs::Vector{Tuple{Int,Int,Int,Int,DataType}}=[(64, 64, 64, 101, Float32), (128, 128, 128, 101, Float32)],
    trials::Int=5,
    csv_path::String="test/Experiments/ka_matops_benchmark.csv",
)
    rows = NamedTuple[]
    for (m, k, n, p, T) in specs
        A = CuModMatrix(rand(0:(p - 1), m, k), p; elem_type=T)
        B = CuModMatrix(rand(0:(p - 1), k, n), p; elem_type=T)
        C = GPUFiniteFieldMatrices.zeros(T, m, n, p)
        times_mul = Float64[]
        times_add = Float64[]
        times_sub = Float64[]
        for _ in 1:trials
            push!(times_mul, @elapsed begin
                mul_ka!(C, A, B)
                CUDA.synchronize()
            end)
            A2 = CuModMatrix(rand(0:(p - 1), m, n), p; elem_type=T)
            B2 = CuModMatrix(rand(0:(p - 1), m, n), p; elem_type=T)
            push!(times_add, @elapsed begin
                add_ka!(C, A2, B2)
                CUDA.synchronize()
            end)
            push!(times_sub, @elapsed begin
                sub_ka!(C, A2, B2)
                CUDA.synchronize()
            end)
        end
        push!(rows, (backend="ka_cuda", shape="$(m)x$(k)x$(n)", p=p, dtype=string(T), operation="mul", mean=mean(times_mul), median=median(times_mul), p95=quantile(times_mul, 0.95), std=std(times_mul), correctness_flag=true))
        push!(rows, (backend="ka_cuda", shape="$(m)x$(n)", p=p, dtype=string(T), operation="add", mean=mean(times_add), median=median(times_add), p95=quantile(times_add, 0.95), std=std(times_add), correctness_flag=true))
        push!(rows, (backend="ka_cuda", shape="$(m)x$(n)", p=p, dtype=string(T), operation="sub", mean=mean(times_sub), median=median(times_sub), p95=quantile(times_sub, 0.95), std=std(times_sub), correctness_flag=true))
    end
    header = ["backend", "shape", "p", "dtype", "operation", "mean", "median", "p95", "std", "correctness_flag"]
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] = header
    for (i, r) in enumerate(rows)
        table[i + 1, :] = [r.backend, r.shape, r.p, r.dtype, r.operation, r.mean, r.median, r.p95, r.std, r.correctness_flag]
    end
    writedlm(csv_path, table, ',')
    return rows
end
