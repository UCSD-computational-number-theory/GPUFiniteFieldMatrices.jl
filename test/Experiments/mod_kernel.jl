#!/usr/bin/env julia

using CUDA
using BenchmarkTools

# -----------------------------
# Config
# -----------------------------
const N = 3200
const MOD = Int32(13)

# For reproducibility / consistent compilation behavior
CUDA.allowscalar(false)

# -----------------------------
# Data setup
# -----------------------------
println("Allocating CPU array $(N)x$(N) of Int32 ...")
A_cpu = rand(Int32(0):Int32(10^6), N, N)

println("Transferring to GPU ...")
A_gpu = CuArray(A_cpu)

# Optionally pre-allocate an output buffer to avoid timing allocations
B_gpu = similar(A_gpu)

# Warm-up: make sure GPU context + kernel compilation happens before timing
CUDA.@sync begin
    B_gpu .= mod.(A_gpu, MOD)
end

println("\n--- Benchmark: broadcast mod.(A, 13) on GPU ---")
# Use CUDA.@sync so btime measures device work, not just launch overhead.
@btime CUDA.@sync begin
    $B_gpu .= mod.($A_gpu, $MOD)
end

function mod13_kernel!(out, A)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)

    @inbounds for i = idx:stride:len
        out[i] = rem(A[i], MOD)  # rem for integers; for nonnegative A, rem==mod
    end
    return
end

function launch_mod13_linear!(out::CuArray{Int32}, A::CuArray{Int32})
    threads = 256
    blocks = cld(length(A), threads)
    @cuda threads=threads blocks=blocks mod13_kernel!(out, A)
    return out
end

# Warm-up compile
CUDA.@sync launch_mod13_linear!(B_gpu, A_gpu)

println("\n--- Benchmark: custom kernel (linear indexing) ---")
@btime CUDA.@sync begin
    launch_mod13_linear!($B_gpu, $A_gpu)
end

# -----------------------------
# Kernel 2 (optional): warp-style "32 threads then next tile"
# This matches your "32 threads per block move in row-major order" idea,
# but implemented as "warp loads contiguous linear memory then advances".
# NOTE: still linear memory order (good). "row-major" isn't meaningful in column-major.
# -----------------------------
function mod13_warp_tile_kernel!(out, A)
    tid = threadIdx().x                # 1..32
    bid = blockIdx().x - 1             # 0-based
    lane = tid - 1                     # 0..31

    # Each block is one warp (32 threads).
    # Each "tile" is 32 consecutive elements in linear memory.
    base = bid * 32 + lane + 1         # 1-based indexing
    stride = 32 * gridDim().x
    len = length(A)

    @inbounds for i = base:stride:len
        out[i] = rem(A[i], MOD)
    end
    return
end

function launch_mod13_warp_tiles!(out::CuArray{Int32}, A::CuArray{Int32})
    @assert length(out) == length(A)
    threads = 32
    # one warp per block; each block covers 32 elements per iteration
    blocks = cld(length(A), 32)
    blocks = min(blocks, 65535)
    @cuda threads=threads blocks=blocks mod13_warp_tile_kernel!(out, A)
    return out
end

CUDA.@sync launch_mod13_warp_tiles!(B_gpu, A_gpu)

println("\n--- Benchmark: custom kernel (warp tiles, 32 threads/block) ---")
@btime CUDA.@sync begin
    launch_mod13_warp_tiles!($B_gpu, $A_gpu)
end

println("\nDone.")
