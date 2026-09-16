# Inverse benchmark technical specification

## Implementations

The existing benchmark driver is
[`test/Experiments/inv_benchmark.jl`](test/Experiments/inv_benchmark.jl). It
contains GPU latency/throughput helpers, square and rectangular cases, fixed-size
batch comparisons, and a Nemo reference path. Its experiment project now
declares Nemo, Plots, and the local GPUFiniteFieldMatrices source explicitly.
The CPU measurements below were taken through Oscar's Nemo backend in the
adjacent DeRham environment, which was already warm for the integration check.

The compared square algorithms are:

| Name | Entry point | Work and storage model |
|---|---|---|
| Blocked PLUQ | `inverse_new(..., inverse_strategy=:pluq)` | Recursive panel PLUQ, two triangular inverses, modular matrix multiplication, then permutation application. Lower nominal factorization arithmetic, but several full-size temporaries and host-visible pivot decisions. |
| Augmented Gauss–Jordan | `inverse_new(..., inverse_strategy=:augmented)` | Pivot, scale, and eliminate over `[A I]`. Roughly twice the elimination arithmetic and an augmented allocation, but a direct inverse result and simpler kernels. |
| Oscar/Nemo CPU | `inv(M)` for an Oscar `GF(p)` matrix | Exact finite-field inverse through Oscar's Nemo/FLINT backend on the CPU. |

Rectangular `right_inverse_new` uses rank-revealing augmented elimination;
`left_inverse_new` applies the right-inverse algorithm to a transpose. Tiny
4/8/16/32 batches use their fixed-size batch kernels and are not mixed into the
large-square table.

## Method

- Date: 2026-09-15.
- GPU: NVIDIA GeForce RTX 3060, 12 GB, driver 581.80, CUDA runtime 13.2.
- Julia: 1.12.7; CUDA.jl 6.1.1.
- Timings synchronize the GPU and exclude construction/host-to-device transfer.
- Each GPU path was warmed at 1k–5k; the 10k entries are single synchronized
  passes after the kernels had already compiled.
- The scale family is an invertible upper-bidiagonal matrix. A dense random
  upper-triangular family with unit diagonal was also measured at 1k to ensure
  the sparse-looking input was not hiding kernel work.
- `autotune=true`, `check_prime=true`; Float32 for moduli 2 and 101, Float64 for
  65521 because the composed Float32 modular matmul correctly rejects that
  modulus as unsafe.
- These are engineering measurements from one machine, not portable peak claims.

## Results

### Upper-bidiagonal scale family, modulus 101

| Size | PLUQ GPU | Augmented GPU | Oscar/Nemo CPU |
|---:|---:|---:|---:|
| 1,000² | 0.324 s | 0.188 s | 0.199 s |
| 2,000² | 1.230 s | 1.122 s | 1.112 s |
| 5,000² | 12.875 s | 12.517 s | 9.890 s |
| 10,000² | 91.443 s | 91.513 s | not run (CPU cap requested at 5k) |

### Modulus and matrix-family sensitivity at 1,000²

| Family | Modulus / storage | PLUQ GPU | Augmented GPU | Oscar/Nemo CPU |
|---|---|---:|---:|---:|
| Upper-bidiagonal | 2 / Float32 | 0.319 s | 0.212 s | — |
| Upper-bidiagonal | 101 / Float32 | 0.324 s | 0.188 s | 0.199 s |
| Upper-bidiagonal | 65521 / Float64 | 0.577 s | 0.489 s | — |
| Dense upper-triangular | 2 / Float32 | 0.325 s | 0.187 s | — |
| Dense upper-triangular | 101 / Float32 | 0.331 s | 0.196 s | 0.291 s |
| Dense upper-triangular | 65521 / Float64 | 0.544 s | 0.498 s | — |

PLUQ's lower arithmetic count does not make it faster on this implementation at
1k–5k. Its composed inverse pays for triangular extraction/inversion, modular
matmul, permutations, and serialized host-visible pivots. The two GPU methods
converge at 10k, where dense arithmetic dominates launch and synchronization
overhead. This result supports keeping both strategies and treating the default
as an API policy rather than a universal performance theorem.

The benchmark also caught an autotune bug: a 32×32 Schur launch requested 1024
threads while the compiled kernel allowed 640 on this GPU. Dispatch now uses the
portable 16×16 geometry, and the 2k–10k results above use that fix.

## Large-size feasibility

| Size | One Float32 `n×n` array | Test disposition |
|---:|---:|---|
| 10,000² | 0.373 GiB | Both GPU algorithms completed. |
| 20,000² | 1.490 GiB | Abandoned as requested: PLUQ needs several full-size arrays and approaches the 12 GB device limit; cubic scaling from the measured 10k run predicts roughly 12 minutes for one pass even if allocation succeeds. |

No 20k time is reported because no 20k inverse completed. This avoids presenting
an extrapolation as a measurement.
