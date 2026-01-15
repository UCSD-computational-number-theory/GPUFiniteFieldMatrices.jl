#!/usr/bin/env julia

using CUDA
using BenchmarkTools
using PrettyTables

CUDA.allowscalar(false)

function k_rem!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = rem(A[i], m)
    end
    return
end

function k_mod!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i], m)
    end
    return
end

function k_pct!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = A[i] % m
    end
    return
end

const DEFAULT_THREADS = 256

function launch_kernel!(kernel, out, A, m; threads::Int=DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks kernel(out, A, m)
    return out
end

launch_rem!(out, A, m; threads::Int=DEFAULT_THREADS) = launch_kernel!(k_rem!, out, A, m; threads=threads)
launch_mod!(out, A, m; threads::Int=DEFAULT_THREADS) = launch_kernel!(k_mod!, out, A, m; threads=threads)
launch_pct!(out, A, m; threads::Int=DEFAULT_THREADS) = launch_kernel!(k_pct!, out, A, m; threads=threads)

broadcast_mod!(B, A, m) = (B .= mod.(A, m); nothing)
kernel_mod!(B, A, m; threads=DEFAULT_THREADS) = (launch_mod!(B, A, m; threads=threads); nothing)
kernel_rem!(B, A, m; threads=DEFAULT_THREADS) = (launch_rem!(B, A, m; threads=threads); nothing)
kernel_pct!(B, A, m; threads=DEFAULT_THREADS) = (launch_pct!(B, A, m; threads=threads); nothing)

function bench_specs(specs; threads::Int=DEFAULT_THREADS, seconds::Float64=0.25, csv_path::AbstractString="mod_bench_results.csv")
    rows = Any[]
    for (nr, nc, mod_in, T) in specs
        println("\n=== Spec: $(nr)x$(nc), mod=$(mod_in), T=$(T) ===")
        bytes_needed = Int64(nr) * Int64(nc) * Int64(sizeof(T)) * 2
        avail = CUDA.available_memory()
        if bytes_needed > Int64(floor(0.80 * avail))
            println("SKIP: needs $(round(bytes_needed / 2.0^30, digits=2)) GiB, available $(round(avail / 2.0^30, digits=2)) GiB")
            push!(rows, (nr, nc, string(T), mod_in, NaN, NaN, NaN, NaN, "skipped (insufficient GPU memory)"))
            continue
        end

        if T <: Integer
            A_cpu = rand(T(0):T(10^6), nr, nc)
        else
            A_cpu = rand(T, nr, nc) .* T(10^6)
        end

        A = CuArray(A_cpu)
        B = similar(A)
        m = convert(T, mod_in)

        broadcast_mod!(B, A, m); CUDA.synchronize()
        kernel_mod!(B, A, m; threads=threads); CUDA.synchronize()
        kernel_rem!(B, A, m; threads=threads); CUDA.synchronize()
        kernel_pct!(B, A, m; threads=threads); CUDA.synchronize()

        t_broadcast = @belapsed begin
            broadcast_mod!($B, $A, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_kmod = @belapsed begin
            kernel_mod!($B, $A, $m; threads=$threads)
            CUDA.synchronize()
        end seconds=seconds

        t_krem = @belapsed begin
            kernel_rem!($B, $A, $m; threads=$threads)
            CUDA.synchronize()
        end seconds=seconds

        t_kpct = @belapsed begin
            kernel_pct!($B, $A, $m; threads=$threads)
            CUDA.synchronize()
        end seconds=seconds

        push!(rows, (nr, nc, string(T), mod_in, 1e3*t_broadcast, 1e3*t_kmod, 1e3*t_krem, 1e3*t_kpct, ""))
    end

    header = ["nrows", "ncols", "dtype", "modulus", "broadcast mod. (ms)", "kernel mod (ms)", "kernel rem (ms)", "kernel % (ms)", "note"]

    println("\n--- Results (pretty) ---\n")
    pretty_table(rows, header; formatters=(ft_printf("%d", 1), ft_printf("%d", 2), ft_printf("%s", 3), ft_printf("%d", 4), ft_printf("%.3f", 5), ft_printf("%.3f", 6), ft_printf("%.3f", 7), ft_printf("%.3f", 8), ft_printf("%s", 9)))

    println("\n--- Results (Markdown) ---\n")
    pretty_table(rows, header; backend=Val(:markdown), formatters=(ft_printf("%d", 1), ft_printf("%d", 2), ft_printf("%s", 3), ft_printf("%d", 4), ft_printf("%.3f", 5), ft_printf("%.3f", 6), ft_printf("%.3f", 7), ft_printf("%.3f", 8), ft_printf("%s", 9)))

    open(csv_path, "w") do io
        println(io, "nrows,ncols,dtype,modulus,broadcast_mod_ms,kernel_mod_ms,kernel_rem_ms,kernel_pct_ms,note")
        for r in rows
            noteq = replace(String(r[9]), "\""=>"\"\"")
            println(io, "$(r[1]),$(r[2]),$(r[3]),$(r[4]),$(r[5]),$(r[6]),$(r[7]),$(r[8]),\"$noteq\"")
        end
    end
    println("\nSaved CSV: $csv_path")
    return nothing
end

specs = [
    (3200, 3200, 13, Int32),
    (4096, 4096, 13, Int32),
    (3200, 3200, 13, Float32),
]

bench_specs(specs; threads=256, seconds=0.25, csv_path="mod_bench_results.csv")
nothing
