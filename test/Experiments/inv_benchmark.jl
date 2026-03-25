using GPUFiniteFieldMatrices
using CUDA
using Nemo
using Statistics
using Printf
using Random
using LinearAlgebra

struct InverseBenchSpec
    rows::Int
    cols::Int
    p::Int
    elem_type::DataType
end

"""
    make_spec(rows, cols, p, dtype)

Create an `InverseBenchSpec` from tuple fields where `dtype` is one of
`:FP32`, `:FP64`, `Float32`, or `Float64`.
"""
function make_spec(rows::Int, cols::Int, p::Int, dtype)
    et = dtype == :FP32 ? Float32 :
         dtype == :FP64 ? Float64 :
         dtype == Float32 ? Float32 :
         dtype == Float64 ? Float64 :
         throw(ArgumentError("dtype must be :FP32, :FP64, Float32, or Float64"))
    return InverseBenchSpec(rows, cols, p, et)
end

"""
    normalize_specs(spec_tuples)

Convert a vector of `(rows, cols, p, dtype)` tuples into benchmark specs.
"""
function normalize_specs(spec_tuples::Vector{<:Tuple})
    return [make_spec(Int(s[1]), Int(s[2]), Int(s[3]), s[4]) for s in spec_tuples]
end

"""
    nemo_inverse_matrix(A_cpu, p)

Compute exact inverse in Nemo over `GF(p)` and return a dense Julia matrix.
"""
function nemo_inverse_matrix(A_cpu::AbstractMatrix, p::Int)
    R = Nemo.GF(p)
    n, m = size(A_cpu)

    A_nemo = Nemo.matrix(R, [R(Int(mod(x, p))) for x in A_cpu])
    Ainv_nemo = inv(A_nemo)
    out = Matrix{Int}(undef, n, n)
    for j in 1:n
        for i in 1:n
            out[i, j] = Int(lift(ZZ, Ainv_nemo[i, j]))
        end
    end
    return out
end

"""
    random_invertible_cumod(spec; max_tries=64, rng=Random.default_rng())

Generate an invertible `CuModMatrix` matching `spec`.
"""
function random_invertible_cumod(spec::InverseBenchSpec; max_tries::Int=64, rng=Random.default_rng())
    for _ in 1:max_tries
        Ahost = rand(rng, 0:(spec.p - 1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        A = CuModMatrix(Atyped, spec.p; elem_type=spec.elem_type)
        if GPUFiniteFieldMatrices.is_invertible_new(A)
            return A
        end
    end
    throw(ErrorException("failed to sample invertible matrix for spec $(spec.rows)x$(spec.cols), p=$(spec.p)"))
end

function random_one_sided_invertible_cumod(
    spec::InverseBenchSpec;
    max_tries::Int=64,
    rng=Random.default_rng(),
    rectangular_requires_full_rank::Bool=false
)
    if spec.rows != spec.cols && !rectangular_requires_full_rank
        Ahost = rand(rng, 0:(spec.p - 1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        return CuModMatrix(Atyped, spec.p; elem_type=spec.elem_type)
    end
    for _ in 1:max_tries
        Ahost = rand(rng, 0:(spec.p - 1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        A = CuModMatrix(Atyped, spec.p; elem_type=spec.elem_type)
        ok = if spec.rows == spec.cols
            GPUFiniteFieldMatrices.is_invertible_new(A)
        elseif spec.rows < spec.cols
            try
                GPUFiniteFieldMatrices.right_inverse_new(A)
                true
            catch
                false
            end
        else
            try
                GPUFiniteFieldMatrices.left_inverse_new(A)
                true
            catch
                false
            end
        end
        ok && return A
    end
    throw(ErrorException("failed to sample one-sided invertible matrix for spec $(spec.rows)x$(spec.cols), p=$(spec.p)"))
end

"""
    prime_gpu!(; p=101, elem_type=Float32, n=100)

Warm up the GPU by running both current and old inverse once.
"""
function prime_gpu!(; p::Int=101, elem_type::DataType=Float32, n::Int=100)
    Ih = Matrix{elem_type}(I, n, n)
    A = CuModMatrix(Ih, p; elem_type=elem_type)
    CUDA.@sync GPUFiniteFieldMatrices.inverse_new(A)
    CUDA.@sync GPUFiniteFieldMatrices.inverse(A)
    return nothing
end

function _time_gpu_call(f)
    t = @elapsed begin
        CUDA.@sync f()
    end
    return t
end

function _time_gpu_batch(f, batch)
    t = @elapsed begin
        f(batch)
        CUDA.synchronize()
    end
    return t / length(batch)
end

function _time_cpu_call(f)
    t = @elapsed f()
    return t
end

function nemo_is_invertible_with_inverse(A_cpu::AbstractMatrix, p::Int)
    R = Nemo.GF(p)
    n, m = size(A_cpu)
    A_nemo = Nemo.matrix(R, [R(Int(mod(x, p))) for x in A_cpu])
    side = n <= m ? :right : :left
    elapsed = @elapsed begin
        Nemo.is_invertible_with_inverse(A_nemo, side=side)
    end
    invflag, invmat = Nemo.is_invertible_with_inverse(A_nemo, side=side)
    return (elapsed=elapsed, invertible=invflag, inverse=invmat, side=side)
end

"""
    benchmark_inverse_suite(spec_tuples; trials=5, warmup=true, check_against_nemo=false, make_plot=true, plot_path="test/Experiments/inv_benchmark.png", verbose=true)

Benchmark current GPU inverse (`inverse_new`), old GPU inverse (`inverse`), and Nemo inverse
over a list of tuple specs `(rows, cols, p, dtype)`.
"""
function benchmark_inverse_suite(
    spec_tuples::Vector{<:Tuple};
    trials::Int=5,
    warmup::Bool=true,
    check_against_nemo::Bool=false,
    make_plot::Bool=true,
    plot_path::String="test/Experiments/inv_benchmark.png",
    batch_count::Int=4,
    compare_old::Bool=false,
    verbose::Bool=true
)
    specs = normalize_specs(spec_tuples)
    if warmup
        prime_gpu!()
    end
    results = NamedTuple[]
    for spec in specs
        local_batch = (max(spec.rows, spec.cols) >= 1024) ? 1 : max(1, batch_count)
        batch = [random_one_sided_invertible_cumod(spec) for _ in 1:local_batch]
        A = batch[1]
        A_cpu = mod.(round.(Int, Array(A)), spec.p)
        nemo_inv = check_against_nemo && spec.rows == spec.cols ? nemo_inverse_matrix(A_cpu, spec.p) : nothing
        new_times = Float64[]
        old_times = Float64[]
        nemo_times = Float64[]
        for _ in 1:trials
            t_new = _time_gpu_batch(B -> GPUFiniteFieldMatrices.inverse_new_batch(B), batch)
            t_old = (compare_old && spec.rows == spec.cols) ? _time_gpu_call(() -> GPUFiniteFieldMatrices.inverse(A)) : NaN
            t_nemo = _time_cpu_call(() -> spec.rows == spec.cols ? nemo_inverse_matrix(A_cpu, spec.p) : nemo_is_invertible_with_inverse(A_cpu, spec.p))
            push!(new_times, t_new)
            push!(old_times, t_old)
            push!(nemo_times, t_nemo)
        end
        Ainv_new = GPUFiniteFieldMatrices.inverse_new_batch([A])[1]
        Ainv_old = (compare_old && spec.rows == spec.cols) ? GPUFiniteFieldMatrices.inverse(A) : nothing
        new_ok = true
        old_ok = compare_old && spec.rows == spec.cols
        if check_against_nemo
            if spec.rows == spec.cols
                Ainv_new_cpu = mod.(round.(Int, Array(Ainv_new)), spec.p)
                new_ok = Ainv_new_cpu == nemo_inv
                if compare_old
                    Ainv_old_cpu = mod.(round.(Int, Array(Ainv_old)), spec.p)
                    old_ok = Ainv_old_cpu == nemo_inv
                end
            else
                nres = nemo_is_invertible_with_inverse(A_cpu, spec.p)
                new_ok = nres.invertible
                old_ok = false
            end
        end
        row = (
            rows=spec.rows,
            cols=spec.cols,
            p=spec.p,
            elem_type=spec.elem_type,
            new_mean=mean(new_times),
            old_mean=(compare_old && spec.rows == spec.cols) ? mean(old_times) : NaN,
            nemo_mean=mean(nemo_times),
            new_median=median(new_times),
            old_median=(compare_old && spec.rows == spec.cols) ? median(old_times) : NaN,
            nemo_median=median(nemo_times),
            new_ok=new_ok,
            old_ok=old_ok
        )
        push!(results, row)
        if verbose
            @printf("spec=(%d,%d,p=%d,%s)  new=%.6f s  old=%.6f s  nemo=%.6f s\n",
                spec.rows, spec.cols, spec.p, string(spec.elem_type), row.new_mean, row.old_mean, row.nemo_mean)
        end
    end
    if make_plot
        try
            @eval using Plots
            x = [r.rows for r in results]
            y_new = [r.new_mean for r in results]
            y_old = [r.old_mean for r in results]
            y_nemo = [r.nemo_mean for r in results]
            plt = plot(
                x, y_new;
                label="inverse_new (GPU)",
                color=:blue,
                marker=:circle,
                linewidth=2,
                xlabel="matrix size (rows)",
                ylabel="time (s)",
                title="Inverse benchmark: new vs old vs Nemo"
            )
            if compare_old
                plot!(plt, x, y_old; label="inverse old (GPU square-only)", color=:red, marker=:square, linewidth=2)
            end
            plot!(plt, x, y_nemo; label="Nemo (CPU)", color=:green, marker=:diamond, linewidth=2)
            savefig(plt, plot_path)
            if verbose
                println("saved plot to $(plot_path)")
            end
        catch err
            println("plot skipped: $(err)")
        end
    end
    return results
end

"""
    run_default_inverse_benchmark(; check_against_nemo=false, trials=5, warmup=true, make_plot=true)

Run a default benchmark grid and return row-wise results.
"""
function run_default_inverse_benchmark(; check_against_nemo::Bool=false, trials::Int=5, warmup::Bool=true, make_plot::Bool=true)
    specs = [
        (8, 8, 101, :FP32),
        (8, 8, 101, :FP64),
        (16, 16, 101, :FP32),
        (16, 16, 101, :FP64),
        (24, 24, 101, :FP32),
        (24, 24, 101, :FP64),
        (32, 32, 101, :FP32),
        (32, 32, 101, :FP64),
        (48, 48, 101, :FP32),
        (48, 48, 101, :FP64),
        (64, 64, 101, :FP32),
        (64, 64, 101, :FP64),
    ]
    return benchmark_inverse_suite(
        specs;
        trials=trials,
        warmup=warmup,
        check_against_nemo=check_against_nemo,
        make_plot=make_plot,
        compare_old=false
    )
end

function run_fastest_vs_nemo_benchmark(; check_against_nemo::Bool=false, trials::Int=5, warmup::Bool=true, make_plot::Bool=true, batch_count::Int=4)
    specs = [
        (16, 16, 101, :FP32),
        (16, 16, 101, :FP64),
        (64, 64, 101, :FP32),
        (64, 64, 101, :FP64),
        (256, 256, 101, :FP32),
        (256, 256, 101, :FP64),
        (256, 640, 101, :FP32),
        (256, 640, 101, :FP64),
        (1024, 1024, 101, :FP32),
        (1024, 1024, 101, :FP64),
        (2048, 2048, 101, :FP32),
        (2048, 2048, 101, :FP64),
        (4096, 4096, 101, :FP32),
        (4096, 4096, 101, :FP64),
        (8192, 8192, 101, :FP32),
        (8192, 8192, 101, :FP64),
        (16384, 16384, 101, :FP32),
        (16384, 16384, 101, :FP64),
    ]
    return benchmark_inverse_suite(
        specs;
        trials=trials,
        warmup=warmup,
        check_against_nemo=check_against_nemo,
        make_plot=make_plot,
        batch_count=batch_count,
        compare_old=false
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fastest_vs_nemo_benchmark()
end
