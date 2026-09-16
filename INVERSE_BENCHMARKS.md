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
| Blocked PLUQ | `inverse_new(..., inverse_strategy=:pluq)` | Fused device-side panel PLUQ, cooperative triangular solves, cuBLAS Schur updates, two recursive triangular inverses, exact chunked modular matrix multiplication, then permutation application. |
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
- Each GPU path was run once at the measured size before the timed 1k–5k
  sample. The final 10k validation used the already compiled 5k path and ran
  two 4096² cuBLAS matrix multiplications to raise GPU clocks before timing.
  Every timed call ended with a CUDA synchronization.
  Input construction, host-to-device transfer, context initialization, kernel
  compilation, GPU priming, and cleanup are outside the reported time.
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
| 1,000² | **0.086 s** | 0.188 s¹ | 0.330 s |
| 2,000² | **0.477 s** | 1.122 s¹ | 1.253 s |
| 5,000² | **6.513 s** | 12.517 s¹ | 10.101 s |
| 10,000² | **51.371 s** | 96.838 s¹ | not run (CPU cap requested at 5k) |

¹ The augmented values are the earlier primed baseline and were not rerun
because this optimization pass changes only PLUQ and its related modular
matrix-multiplication code.

PLUQ is 3.83×, 2.63×, and 1.55× faster than the contemporaneous Oscar/Nemo
measurements at 1k, 2k, and 5k respectively.

### Dense upper-triangular family, modulus 101

| Size | PLUQ GPU | Oscar/Nemo CPU | GPU speedup |
|---:|---:|---:|---:|
| 1,000² | **0.086 s** | 0.303 s | 3.52× |
| 2,000² | **0.478 s** | 1.192 s | 2.49× |
| 5,000² | **6.511 s** | 10.491 s | 1.61× |

### Modulus sensitivity, upper-bidiagonal family

| Modulus / storage | 1,000² | 2,000² | 5,000² |
|---|---:|---:|---:|
| 2 / Float32 | 0.081 s | 0.475 s | 6.518 s |
| 101 / Float32 | 0.086 s | 0.477 s | 6.513 s |
| 65521 / Float64 | 0.129 s | 0.707 s | 9.520 s |

The speedup comes from four changes. A cooperative block now completes an
entire panel with one host rank read; triangular solves use warp-shuffle dot
products and reuse device-resident diagonal inverses; safe Schur updates use
cuBLAS GEMM with one modular reduction per output; and larger modular products
split the inner dimension into exactly representable chunks. The last point is
also a correctness fix: an unchecked 5k Float32 GEMM modulo 101 can exceed the
integer range represented exactly by Float32.

The explicit 10k priming confirms that its 51.4-second measurement excludes
context initialization, compilation, allocation of the input, host-to-device
transfer, and clock ramp. The kernels still perform dense work for the
upper-bidiagonal input; the similar dense-upper timings confirm that sparsity is
not responsible for the improvement.

The CUDA choices follow the CUDA.jl kernel guidance for occupancy-selected
launches, shared memory, and warp intrinsics, plus NVIDIA's recommendations for
coalesced accesses, warp-multiple block sizes, and avoiding repeated small
launches. CUDA Graph replay was not used because panel shapes change and each
panel depends on the prior Schur update; there is no repeated fixed graph to
amortize capture and instantiation.

## Large-size feasibility

| Size | One Float32 `n×n` array | Test disposition |
|---:|---:|---|
| 10,000² | 0.373 GiB | Both GPU algorithms completed. |
| 20,000² | 1.490 GiB | Abandoned as requested: PLUQ needs several full-size arrays and approaches the 12 GB device limit; cubic scaling from the measured 10k run predicts roughly 12 minutes for one pass even if allocation succeeds. |

No 20k time is reported because no 20k inverse completed. This avoids presenting
an extrapolation as a measurement.
