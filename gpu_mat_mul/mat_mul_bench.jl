using CUDA, BenchmarkTools, LinearAlgebra, Test
include("mat_mul_hybrid.jl")

"""
Function to format benchmark
"""
function mat_mul_benchmark_sizes(sizes, P)

    # Print out GPU information
    gpu_info()
    
    # Primer and sanity check
    DEFAULT_SIZE = 5000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, P)

    # Define group of benchmarks
    suite = BenchmarkGroup()

    for size in sizes

        println(@benchmark begin
            A = rand(1:($P-1), $size[1,1], $size[1,2])
        end)

        println(@benchmark begin
            B = rand(1:($P-1), $size[2,1], $size[2,2])
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_gpu($A, $B, $P)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_cpu($A, $B, $P)
        end)
    end

    return
end

function mat_mul_benchmark_types(types, P)

    # Print out GPU information
    gpu_info()
    
    # Primer and sanity check
    DEFAULT_SIZE = 5000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, P)

    # Define group of benchmarks
    suite = BenchmarkGroup()

    for type in types

        println(@benchmark begin
            A = rand(1:($P-1), DEFAULT_SIZE, DEFAULT_SIZE)
        end)

        println(@benchmark begin
            B = rand(1:($P-1), DEFAULT_SIZE, DEFAULT_SIZE)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_gpu($A, $B, $P, -1, $type)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_cpu($A, $B, $P)
        end)
    end

    return
end

function mat_mul_benchmark_regimes(regimes, P)

    # Print out GPU information
    gpu_info()
    
    # Primer and sanity check
    DEFAULT_SIZE = 5000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, P)

    # Define group of benchmarks
    suite = BenchmarkGroup()

    for regime in regimes

        println(@benchmark begin
            A = rand(1:($P-1), DEFAULT_SIZE, DEFAULT_SIZE)
        end)

        println(@benchmark begin
            B = rand(1:($P-1), DEFAULT_SIZE, DEFAULT_SIZE)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_gpu($A, $B, $P, $regime)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_cpu($A, $B, $P)
        end)
    end

    return
end

function mat_mul_benchmark_all(regimes, types, sizes, P)

    # Print out GPU information
    gpu_info()
    
    # Primer and sanity check
    DEFAULT_SIZE = 5000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, P)

    # Define group of benchmarks
    suite = BenchmarkGroup()

    for (regime, type, size) in IterTools.product(regimes, types, sizes)

        println(@benchmark begin
            A = rand(1:($P-1), $size[1,1], $size[1,2])
        end)

        println(@benchmark begin
            B = rand(1:($P-1), $size[2,1], $size[2,2])
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_gpu($A, $B, $P, $regime, $type)
        end)

        println(@benchmark begin
            C = CUDA.@sync mat_mul_cpu($A, $B, $P)
        end)
    end

    return
end

"""
Run program once to remove compilation time
"""
function mat_mul_primer(A, B, P)
    
    C = mat_mul_gpu(A, B, P)
    C_ref = A * B
    C_ref = mod.(C_ref, P)
    @test all(C_ref .== C)

    return
end

"""
Single threaded CPU multiplication for comparison
"""
function mat_mul_cpu(A, B, P)
    C = A * B
    return mod.(C, P)
end

"""
Prints out GPU information
"""
function gpu_info()
    # Get the device
    dev = device()

    # Get device properties
    name = CUDA.name(dev)
    memory = CUDA.totalmem(dev)
    capability = CUDA.capability(dev)
    warpSize = CUDA.warpsize(dev)

    # Print device properties
    println("Name: $name")
    println("Total Memory: $memory bytes")
    println("Capability: $capability")
    println("Warp Size: $warpSize threads")

end