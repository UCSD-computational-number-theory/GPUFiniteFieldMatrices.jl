using CUDA, LinearAlgebra, BenchmarkTools
using CSV, DelimitedFiles
include("rref_new_kernels.jl")
include("test_swap.jl")

function main()
    # A = [
    #     [0 1 2 3]
    #     [2 3 4 4]
    #     [0 0 1 2]
    #     [0 0 0 1]
    # ]
    # P = 5

    # DEFAULT_SIZE = 1000
    # A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # # println("ORIGINAL:")
    # # println(A)
    # println(@benchmark begin
    #     A_rref = rref_gpu($A,$P)
    # end)
    # # println("REDUCED:")
    # # println(A_rref)

    # DEFAULT_SIZE = 1000
    # A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # # println("ORIGINAL:")
    # # println(A)
    # println(@benchmark begin
    #     A_rref = rref_gpu($A,$P)
    # end)
    # # println("REDUCED:")
    # # println(A_rref)

    # return

    # Current rough times, all p=11
    # 64x64 = 8s
    # Host-side activity: calling CUDA APIs took 24.4 ms (0.29% of the trace)
    # ┌──────────┬────────────┬───────┬───────────────────────────────────────┬──────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                     │ Name        ⋯
    # ├──────────┼────────────┼───────┼───────────────────────────────────────┼──────────────
    # │    0.09% │    7.18 ms │   768 │   9.35 µs ± 34.77  (  3.58 ‥ 961.3)   │ cuLaunchKer ⋯
    # │    0.07% │    5.57 ms │     8 │ 695.65 µs ± 909.41 (141.14 ‥ 2921.58) │ cuModuleLoa ⋯
    # │    0.03% │     2.2 ms │   449 │   4.89 µs ± 2.98   (  1.91 ‥ 25.75)   │ cuMemAllocF ⋯
    # │    0.01% │   811.1 µs │     1 │                                       │ cuMemHostAl ⋯
    # │    0.00% │  157.12 µs │    64 │   2.45 µs ± 1.09   (  1.67 ‥ 10.49)   │ cuMemcpyHto ⋯
    # │    0.00% │  134.71 µs │     8 │  16.84 µs ± 3.36   ( 11.21 ‥ 20.27)   │ cuCtxSynchr ⋯
    # │    0.00% │   62.94 µs │     1 │                                       │ cuMemcpyDto ⋯
    # │    0.00% │   46.97 µs │     6 │   7.83 µs ± 8.26   (  1.91 ‥ 18.84)   │ cuMemFreeAs ⋯
    # │    0.00% │   34.09 µs │    66 │ 516.57 ns ± 439.79 (238.42 ‥ 3337.86) │ cuStreamSyn ⋯
    # │    0.00% │   22.41 µs │     8 │    2.8 µs ± 0.9    (  1.19 ‥ 4.29)    │ cuModuleGet ⋯
    # │    0.00% │    2.15 µs │     2 │   1.07 µs ± 0.17   (  0.95 ‥ 1.19)    │ cuCtxSetCur ⋯
    # │    0.00% │  953.67 ns │     2 │ 476.84 ns ± 0.0    (476.84 ‥ 476.84)  │ cuCtxGetDev ⋯
    # │    0.00% │  238.42 ns │     2 │ 119.21 ns ± 168.59 (   0.0 ‥ 238.42)  │ cuDeviceGet ⋯
    # └──────────┴────────────┴───────┴───────────────────────────────────────┴──────────────
    #                                                                        1 column omitted

    # Device-side activity: GPU was busy for 1.42 ms (0.02% of the trace)
    # ┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                    │ Name         ⋯
    # ├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────────
    # │    0.00% │   362.4 µs │   192 │   1.89 µs ± 0.16   (  1.67 ‥ 2.15)   │ _Z15setindex ⋯
    # │    0.00% │  349.04 µs │   192 │   1.82 µs ± 0.17   (  1.43 ‥ 3.1)    │ _Z15getindex ⋯
    # │    0.00% │  117.06 µs │    64 │   1.83 µs ± 0.13   (  1.43 ‥ 2.15)   │ _Z15getindex ⋯
    # │    0.00% │  116.59 µs │    64 │   1.82 µs ± 0.41   (  1.19 ‥ 2.38)   │ _Z21update_s ⋯
    # │    0.00% │  111.58 µs │    64 │   1.74 µs ± 0.16   (  1.43 ‥ 1.91)   │ _Z3_3415CuKe ⋯
    # │    0.00% │   111.1 µs │    64 │   1.74 µs ± 0.17   (  1.43 ‥ 1.91)   │ _Z3_3415CuKe ⋯
    # │    0.00% │  110.39 µs │    64 │   1.72 µs ± 0.18   (  1.43 ‥ 1.91)   │ _Z15setindex ⋯
    # │    0.00% │   103.0 µs │    64 │   1.61 µs ± 0.17   (  1.43 ‥ 1.91)   │ _Z3_3415CuKe ⋯
    # │    0.00% │   31.23 µs │    64 │ 488.01 ns ± 143.62 (238.42 ‥ 715.26) │ [copy pageab ⋯
    # │    0.00% │   11.68 µs │     1 │      
    
    # 1000x1000 = 8s ???
    # ┌──────────┬────────────┬───────┬───────────────────────────────────────┬──────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                     │ Name        ⋯
    # ├──────────┼────────────┼───────┼───────────────────────────────────────┼──────────────
    # │    0.50% │   43.07 ms │ 12000 │   3.59 µs ± 9.48   (  2.38 ‥ 981.81)  │ cuLaunchKer ⋯
    # │    0.19% │    16.1 ms │  7001 │    2.3 µs ± 5.75   (  1.19 ‥ 320.91)  │ cuMemAllocF ⋯
    # │    0.07% │    6.09 ms │     1 │                                       │ cuMemcpyDto ⋯
    # │    0.05% │    4.51 ms │     8 │ 563.41 µs ± 580.55 ( 96.08 ‥ 1942.63) │ cuModuleLoa ⋯
    # │    0.05% │    4.25 ms │  1000 │   4.25 µs ± 1.13   (   3.1 ‥ 20.27)   │ cuMemcpyHto ⋯
    # │    0.01% │  857.11 µs │     1 │                                       │ cuMemHostAl ⋯
    # │    0.00% │  430.58 µs │  1002 │ 429.72 ns ± 264.38 (238.42 ‥ 5245.21) │ cuStreamSyn ⋯
    # │    0.00% │  130.18 µs │     8 │  16.27 µs ± 5.05   (  9.78 ‥ 21.22)   │ cuCtxSynchr ⋯
    # │    0.00% │    48.4 µs │     4 │   12.1 µs ± 9.08   (  2.62 ‥ 23.13)   │ cuMemFreeAs ⋯
    # │    0.00% │   19.79 µs │     8 │   2.47 µs ± 0.91   (  1.19 ‥ 3.58)    │ cuModuleGet ⋯
    # │    0.00% │    2.15 µs │     2 │   1.07 µs ± 0.17   (  0.95 ‥ 1.19)    │ cuCtxSetCur ⋯
    # │    0.00% │  953.67 ns │     2 │ 476.84 ns ± 0.0    (476.84 ‥ 476.84)  │ cuCtxGetDev ⋯
    # │    0.00% │  953.67 ns │     4 │ 238.42 ns ± 0.0    (238.42 ‥ 238.42)  │ cuDeviceGet ⋯
    # └──────────┴────────────┴───────┴───────────────────────────────────────┴──────────────
    #                                                                        1 column omitted

    # Device-side activity: GPU was busy for 39.31 ms (0.46% of the trace)
    # ┌──────────┬────────────┬───────┬───────────────────────────────────────┬──────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                     │ Name        ⋯
    # ├──────────┼────────────┼───────┼───────────────────────────────────────┼──────────────
    # │    0.13% │   10.86 ms │  1000 │  10.86 µs ± 6.25   (  1.19 ‥ 24.8)    │ _Z21update_ ⋯
    # │    0.07% │    6.31 ms │  3000 │    2.1 µs ± 0.16   (  1.91 ‥ 2.38)    │ _Z15setinde ⋯
    # │    0.07% │     6.1 ms │  3000 │   2.03 µs ± 0.17   (  1.67 ‥ 3.1)     │ _Z15getinde ⋯
    # │    0.06% │    5.39 ms │     1 │                                       │ [copy devic ⋯
    # │    0.03% │    2.17 ms │  1000 │   2.17 µs ± 0.18   (  1.91 ‥ 2.38)    │ _Z3_3415CuK ⋯
    # │    0.03% │    2.17 ms │  1000 │   2.17 µs ± 0.17   (  1.91 ‥ 2.38)    │ _Z3_3415CuK ⋯
    # │    0.02% │    1.89 ms │  1000 │   1.89 µs ± 0.17   (  1.67 ‥ 2.15)    │ _Z3_3415CuK ⋯
    # │    0.02% │    1.85 ms │  1000 │   1.85 µs ± 0.15   (  1.67 ‥ 2.15)    │ _Z15getinde ⋯
    # │    0.02% │    1.74 ms │  1000 │   1.74 µs ± 0.16   (  1.43 ‥ 2.15)    │ _Z15setinde ⋯
    # │    0.01% │  828.98 µs │  1000 │ 828.98 ns ± 148.48 (715.26 ‥ 1192.09) │ [copy pagea 

    # 5000x5000 = 15.54s
    # ┌──────────┬────────────┬───────┬───────────────────────────────────────┬──────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                     │ Name        ⋯
    # ├──────────┼────────────┼───────┼───────────────────────────────────────┼──────────────
    # │    3.48% │  540.34 ms │ 60000 │   9.01 µs ± 44.82  (  2.38 ‥ 4167.32) │ cuLaunchKer ⋯
    # │    0.83% │  129.31 ms │     1 │                                       │ cuMemcpyDto ⋯
    # │    0.57% │   88.64 ms │  5000 │  17.73 µs ± 5.7    (  9.06 ‥ 272.75)  │ cuMemcpyHto ⋯
    # │    0.52% │   81.56 ms │ 35001 │   2.33 µs ± 9.08   (  1.19 ‥ 1161.58) │ cuMemAllocF ⋯
    # │    0.04% │    5.93 ms │     8 │ 741.48 µs ± 924.11 (128.98 ‥ 3005.03) │ cuModuleLoa ⋯
    # │    0.01% │    2.33 ms │  5002 │  465.4 ns ± 235.0  (238.42 ‥ 7867.81) │ cuStreamSyn ⋯
    # │    0.01% │  863.31 µs │     1 │                                       │ cuMemHostAl ⋯
    # │    0.00% │  140.43 µs │     8 │  17.55 µs ± 4.05   ( 10.25 ‥ 20.98)   │ cuCtxSynchr ⋯
    # │    0.00% │   62.23 µs │     1 │                                       │ cuMemGetInf ⋯
    # │    0.00% │   23.13 µs │     8 │   2.89 µs ± 0.69   (  1.91 ‥ 3.81)    │ cuModuleGet ⋯
    # │    0.00% │   19.55 µs │     1 │                                       │ cuMemFreeAs ⋯
    # │    0.00% │    3.34 µs │     2 │   1.67 µs ± 2.02   (  0.24 ‥ 3.1)     │ cuMemPoolGe ⋯
    # │    0.00% │  715.26 ns │     1 │                                       │ cuCtxSetCur ⋯
    # │    0.00% │  476.84 ns │     1 │                                       │ cuCtxGetDev ⋯
    # │    0.00% │  476.84 ns │     3 │ 158.95 ns ± 137.65 (   0.0 ‥ 238.42)  │ cuDeviceGet ⋯
    # └──────────┴────────────┴───────┴───────────────────────────────────────┴──────────────
    #                                                                        1 column omitted
    
    # Device-side activity: GPU was busy for 1.44 s (9.26% of the trace)
    # ┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                    │ Name         ⋯
    # ├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────────
    # │    7.66% │     1.19 s │  5000 │ 238.14 µs ± 151.73 (  0.95 ‥ 546.22) │ _Z21update_s ⋯
    # │    0.83% │  128.36 ms │     1 │                                      │ [copy device ⋯
    # │    0.24% │   38.06 ms │ 15000 │   2.54 µs ± 1.38   (  1.19 ‥ 5.72)   │ _Z15getindex ⋯
    # │    0.15% │    24.0 ms │ 15000 │    1.6 µs ± 0.22   (  1.19 ‥ 2.38)   │ _Z15setindex ⋯
    # │    0.13% │   20.68 ms │  5000 │   4.14 µs ± 0.22   (  3.81 ‥ 5.01)   │ [copy pageab ⋯
    # │    0.05% │    8.27 ms │  5000 │   1.65 µs ± 0.23   (  1.43 ‥ 2.38)   │ _Z3_3415CuKe ⋯
    # │    0.05% │    8.26 ms │  5000 │   1.65 µs ± 0.24   (  1.43 ‥ 2.38)   │ _Z3_3415CuKe ⋯
    # │    0.05% │    7.21 ms │  5000 │   1.44 µs ± 0.22   (  1.19 ‥ 2.15)   │ _Z15getindex ⋯
    # │    0.05% │    7.13 ms │  5000 │   1.43 µs ± 0.23   (  1.19 ‥ 2.15)   │ _Z3_3415CuKe ⋯
    # │    0.04% │    6.83 ms │  5000 │   1.37 µs ± 0.2    (  0.95 ‥ 2.15)   │ _Z15setindex 

    # 10000x10000 = 40.41s

    # ┌──────────┬────────────┬────────┬────────────────────────────────────────┬────────────
    # │ Time (%) │ Total time │  Calls │ Time distribution                      │ Name      ⋯
    # ├──────────┼────────────┼────────┼────────────────────────────────────────┼────────────
    # │   21.94% │     8.87 s │ 120000 │   73.9 µs ± 296.05 (  2.38 ‥ 4651.07)  │ cuLaunchK ⋯
    # │    1.24% │  500.22 ms │      1 │                                        │ cuMemcpyD ⋯
    # │    0.53% │  215.88 ms │  70001 │   3.08 µs ± 34.7   (  1.43 ‥ 5997.9)   │ cuMemAllo ⋯
    # │    0.46% │  185.26 ms │  10000 │  18.53 µs ± 8.51   ( 16.45 ‥ 833.51)   │ cuMemcpyH ⋯
    # │    0.01% │     6.0 ms │      8 │ 749.65 µs ± 910.07 (242.47 ‥ 2984.29)  │ cuModuleL ⋯
    # │    0.01% │    4.38 ms │  10002 │ 437.86 ns ± 222.27 (238.42 ‥ 10967.25) │ cuStreamS ⋯
    # │    0.00% │  909.57 µs │      1 │                                        │ cuMemHost ⋯
    # │    0.00% │  202.18 µs │      2 │ 101.09 µs ± 99.47  ( 30.76 ‥ 171.42)   │ cuMemGetI ⋯
    # │    0.00% │  147.58 µs │      8 │  18.45 µs ± 3.6    ( 10.73 ‥ 21.7)     │ cuCtxSync ⋯
    # │    0.00% │   24.32 µs │      8 │   3.04 µs ± 0.67   (  1.91 ‥ 4.05)     │ cuModuleG ⋯
    # │    0.00% │   24.08 µs │      1 │                                        │ cuMemFree ⋯
    # │    0.00% │    3.58 µs │      4 │ 894.07 ns ± 900.01 (238.42 ‥ 2145.77)  │ cuMemPool ⋯
    # │    0.00% │    1.67 µs │      1 │                                        │ cuCtxSetC ⋯
    # │    0.00% │  476.84 ns │      1 │                                        │ cuCtxGetD ⋯
    # │    0.00% │  238.42 ns │      3 │  79.47 ns ± 137.65 (   0.0 ‥ 238.42)   │ cuDeviceG ⋯
    # └──────────┴────────────┴────────┴────────────────────────────────────────┴────────────
    #                                                                        1 column omitted
    
    # Device-side activity: GPU was busy for 11.45 s (28.33% of the trace)
    # ┌──────────┬────────────┬───────┬─────────────────────────────────────┬────────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                   │ Name          ⋯
    # ├──────────┼────────────┼───────┼─────────────────────────────────────┼────────────────
    # │   26.27% │    10.62 s │ 10000 │   1.06 ms ± 0.64   (   0.0 ‥ 2.37)  │ _Z21update_su ⋯
    # │    1.24% │  499.43 ms │     1 │                                     │ [copy device  ⋯
    # │    0.26% │  104.65 ms │ 30000 │   3.49 µs ± 2.45   (  1.43 ‥ 8.11)  │ _Z15getindex_ ⋯
    # │    0.22% │   90.19 ms │ 10000 │   9.02 µs ± 0.36   (  8.58 ‥ 21.22) │ [copy pageabl ⋯
    # │    0.15% │   60.52 ms │ 30000 │   2.02 µs ± 0.34   (  1.43 ‥ 3.58)  │ _Z15setindex_ ⋯
    # │    0.04% │   16.45 ms │ 10000 │   1.65 µs ± 0.18   (  1.43 ‥ 2.38)  │ _Z3_3415CuKer ⋯
    # │    0.04% │   16.34 ms │ 10000 │   1.63 µs ± 0.18   (  1.43 ‥ 2.38)  │ _Z3_3415CuKer ⋯
    # │    0.04% │   14.76 ms │ 10000 │   1.48 µs ± 0.18   (  1.19 ‥ 2.38)  │ _Z15getindex_ ⋯
    # │    0.03% │   14.04 ms │ 10000 │    1.4 µs ± 0.18   (  1.19 ‥ 2.15)  │ _Z3_3415CuKer ⋯
    # │    0.03% │   13.59 ms │ 10000 │   1.36 µs ± 0.17   (  1.19 ‥ 2.15)  │ _Z15setindex_ 

    # 20000x20000 = 191.39s

    # Host-side activity: calling CUDA APIs took 90.87 s (47.48% of the trace)
    # ┌──────────┬────────────┬────────┬──────────────────────────────────────────┬──────────
    # │ Time (%) │ Total time │  Calls │ Time distribution                        │ Name    ⋯
    # ├──────────┼────────────┼────────┼──────────────────────────────────────────┼──────────
    # │   45.19% │    86.49 s │ 240000 │ 360.38 µs ± 1459.09 (  2.15 ‥ 20972.73)  │ cuLaunc ⋯
    # │    1.03% │     1.96 s │      1 │                                          │ cuMemcp ⋯
    # │    0.38% │   735.9 ms │  20000 │  36.79 µs ± 10.09  ( 31.23 ‥ 164.75)     │ cuMemcp ⋯
    # │    0.26% │  488.33 ms │  20005 │  24.41 µs ± 2654.98 (  0.24 ‥ 354344.13) │ cuStrea ⋯
    # │    0.20% │  377.07 ms │ 140004 │   2.69 µs ± 39.31  (  1.19 ‥ 13452.05)   │ cuMemAl ⋯
    # │    0.11% │  211.08 ms │ 120884 │   1.75 µs ± 22.52  (  0.72 ‥ 3987.55)    │ cuMemFr ⋯
    # │    0.00% │    5.46 ms │      8 │ 682.65 µs ± 950.17 (207.66 ‥ 3019.33)    │ cuModul ⋯
    # │    0.00% │  794.41 µs │      1 │                                          │ cuMemHo ⋯
    # │    0.00% │  583.89 µs │     13 │  44.91 µs ± 15.23  ( 25.75 ‥ 78.92)      │ cuMemGe ⋯
    # │    0.00% │  133.99 µs │      8 │  16.75 µs ± 5.78   ( 10.01 ‥ 25.99)      │ cuCtxSy ⋯
    # │    0.00% │   21.93 µs │      8 │   2.74 µs ± 0.74   (  1.67 ‥ 3.58)       │ cuModul ⋯
    # │    0.00% │   20.98 µs │     26 │ 806.96 ns ± 2108.98 (   0.0 ‥ 10967.25)  │ cuMemPo ⋯
    # │    0.00% │    7.87 µs │      5 │   1.57 µs ± 1.07   (  0.72 ‥ 2.86)       │ cuCtxSe ⋯
    # │    0.00% │    6.91 µs │     19 │  363.9 ns ± 184.14 (   0.0 ‥ 715.26)     │ cuDevic ⋯
    # │    0.00% │    1.91 µs │      5 │ 381.47 ns ± 130.59 (238.42 ‥ 476.84)     │ cuCtxGe ⋯
    # └──────────┴────────────┴────────┴──────────────────────────────────────────┴──────────
    #                                                                        1 column omitted
    
    # Device-side activity: GPU was busy for 97.81 s (51.10% of the trace)
    # ┌──────────┬────────────┬───────┬─────────────────────────────────────┬────────────────
    # │ Time (%) │ Total time │ Calls │ Time distribution                   │ Name          ⋯
    # ├──────────┼────────────┼───────┼─────────────────────────────────────┼────────────────
    # │   49.55% │    94.83 s │ 20000 │   4.74 ms ± 2.8    (   0.0 ‥ 10.46) │ _Z21update_su ⋯
    # │    1.03% │     1.96 s │     1 │                                     │ [copy device  ⋯
    # │    0.17% │  327.78 ms │ 60000 │   5.46 µs ± 3.79   (  1.91 ‥ 13.83) │ _Z15getindex_ ⋯
    # │    0.16% │  311.01 ms │ 20000 │  15.55 µs ± 0.3    ( 15.02 ‥ 27.89) │ [copy pageabl ⋯
    # │    0.12% │  221.32 ms │ 60000 │   3.69 µs ± 1.33   (  1.91 ‥ 7.15)  │ _Z15setindex_ ⋯
    # │    0.02% │   33.86 ms │ 20000 │   1.69 µs ± 0.18   (  1.19 ‥ 2.38)  │ _Z15getindex_ ⋯
    # │    0.02% │   33.34 ms │ 20000 │   1.67 µs ± 0.18   (  1.43 ‥ 2.38)  │ _Z3_3415CuKer ⋯
    # │    0.02% │   33.19 ms │ 20000 │   1.66 µs ± 0.18   (  1.43 ‥ 2.62)  │ _Z3_3415CuKer ⋯
    # │    0.02% │   29.49 ms │ 20000 │   1.47 µs ± 0.16   (  1.19 ‥ 2.15)  │ _Z15setindex_ ⋯
    # │    0.01% │   28.48 ms │ 20000 │   1.42 µs ± 0.17   (  1.19 ‥ 2.15)  │ _Z3_3415CuKer 

    size = 4
    TILE_WIDTH = 2
    A = rand(1:11, (size,size))
    d_A = CUDA.CuArray(A)
    Perm = CUDA.CuArray(collect(1:size))
    d_L = CUDA.zeros(size,size)

    res_A, res_L, res_Perm = 0,0,0
    println(CUDA.@profile begin
        (res_A, res_L, res_Perm) = lu_gpu(A, 11)
    end)
    writedlm("plu_A.csv", res_A, ',')
    writedlm("plu_L.csv", res_L, ',')
    writedlm("plu_Perm.csv", res_Perm, ',')

    # println(CUDA.@profile begin
    #     CUDA.@sync @cuda threads=(100) blocks=(10) normalize(d_A,5000,1,7,11)
    # end)

    # println(CUDA.@profile begin
    #     CUDA.@sync @cuda threads=(100) blocks=(10) normalize_lu(d_A,5000,1,7,11,d_L)
    # end)

    # println(CUDA.@profile begin 
    #     CUDA.@sync normalize_broadcast(d_A,1,7,11)
    # end)

    # println(CUDA.@profile begin 
    #     CUDA.@sync normalize_lu_broadcast(d_A,1,7,11,d_L)
    # end)

end

main()