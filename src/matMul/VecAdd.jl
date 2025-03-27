using CUDA
using Test, Profile, BenchmarkTools

"""
Exploratory program to test different CPU implementations for vector addition.

**Usage**
Julia by default starts with 1 thread. To use more call 
julia --threads 8 vec_add_cpu.jl
Do NOT use more threads than your machine has, 
this dramatically increases allocation time.
  
**Summary of Findings:**
Parallelized vector addition is only faster for very large vectors 
with around 2^30 entires. Otherwise, naive implementations (e.g. .+=) are
faster due to far less allocation calls. \
However, memory usage remains constant regardless of the method used.

**Raw Results:**
For N = 2^10:
Naive time: 
  0.000013 seconds (2 allocations: 4.250 KiB)
CPU sequential time: 
  0.000006 seconds (2 allocations: 4.250 KiB)
CPU parallel time: 
  0.006870 seconds (15.89 k allocations: 849.683 KiB, 99.15% compilation time)

For N = 2^20:
Naive time: 
  0.000425 seconds (4 allocations: 4.000 MiB)
CPU sequential time: 
  0.000403 seconds (4 allocations: 4.000 MiB)
CPU parallel time: 
  0.008954 seconds (15.90 k allocations: 4.826 MiB, 94.67% compilation time)

For N = 2^30:
Naive time: 
  0.960206 seconds (4 allocations: 4.000 GiB, 0.83% gc time)
CPU sequential time: 
  0.680675 seconds (4 allocations: 4.000 GiB, 18.65% gc time)
CPU parallel time: 
  0.529868 seconds (16.11 k allocations: 4.001 GiB, 19.77% gc time, 1.63% compilation time)
"""

function naive_vector_add!()
    x, y = gen_sample_vec()
    x .+= y
end

function cpu_seq_vector_add!()
    x, y = gen_sample_vec()
    for i in eachindex(x, y)
        @inbounds x[i] += y[i]
    end
end

function cpu_par_vector_add!()
    x, y = gen_sample_vec()
    Threads.@threads for i in eachindex(x, y)
        @inbounds x[i] += y[i]
    end
end

function gen_sample_vec()
    N = 2^30
    x = fill(UInt16(1), N)
    y = fill(UInt16(2), N)
    return x, y
end

function compare()
    println("Naive time: ")
    @time naive_vector_add!()
    println("CPU sequential time: ")
    @time cpu_seq_vector_add!()
    println("CPU parallel time: ")
    @time cpu_par_vector_add!()
end

compare()