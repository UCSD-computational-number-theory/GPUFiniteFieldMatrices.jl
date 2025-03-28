using CUDA, BenchmarkTools, LinearAlgebra, Test, Serialization
include("mat_mul_hybrid.jl")
include("mat_mul_flops.jl")

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
    DEFAULT_SIZE = 1000
    A = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, 9)

    # Define group of benchmarks
    suite = BenchmarkGroup()

    for (regime, type, size, P) in IterTools.product(regimes, types, sizes, P)

        println(
"""
Beginning Test for:
    Regime: $regime
    Type $type
    Size: $size
    N: $P
"""
        )

        #try

        suite[regime, type, size, P] = BenchmarkGroup()

        # suite[regime, type, size, P]["allocA"] = @benchmark begin
        #     A = rand(1:($P-1), $size[1,1], $size[1,2])
        # end

        # println("Alloc A done")

        # suite[regime, type, size, P]["allocB"] = @benchmark begin
        #     B = rand(1:($P-1), $size[2,1], $size[2,2])
        # end

        # println("Alloc B done")

        A = rand(1:(P-1), size[1,1], size[1,2])
        B = rand(1:(P-1), size[2,1], size[2,2])

        println("Allocs done")

        suite[regime, type, size, P]["gpu"] = @benchmark begin
            C = CUDA.@sync mat_mul_gpu($A, $B, $P, $regime, $type)
        end

        println("GPU done")

        suite[regime, type, size, P]["cpu"] = @benchmark begin
            C = CUDA.@sync mat_mul_cpu($A, $B, $P)
        end

        println("CPU done")

        #catch
        #    
        #println("An error had occurred!")

        #end

        println("")

        println(suite[regime, type, size, P])

        println("")

        # catch
        #     println("Something went wrong with: $regime, $type, $size")
        # end
    end

    f = serialize("suite.dat", suite)

    return suite
end 

using Oscar

function mat_mul_benchmark_naive(regime, type, sizes, P, nTries)

    # Print out GPU information
    gpu_info()
    
    # Primer and sanity check
    DEFAULT_SIZE = 1000
    A = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    B = rand(1:(9-1), DEFAULT_SIZE, DEFAULT_SIZE)
    mat_mul_primer(A, B, 9)

    results = []

    for (size, P) in IterTools.product(sizes, P)

        println(
"""
Beginning Test for:
    Regime: $regime
    Type $type
    Size: $size
    N: $P
"""
        )


        avg_time = 0

        if regime == "cpu" 

            zz = zero(type)
            A = rand(zz:(P-1), size[1,1], size[1,2])
            B = rand(zz:(P-1), size[2,1], size[2,2])
            C = zeros(type,size[1,1],size[2,2])

            println("Allocs done")

            times = zeros(nTries)
            for i = 1:nTries
                CC = @timed mat_mul_cpu(C,A,B,P)
                times[i] = CC.time 
            end

            avg_time = sum(times) / nTries

            push!(results,(regime,type,size,P,avg_time))

            println("CPU done: $avg_time")

        elseif regime == "oscar"

            R = GF(P)
            s = size[1,1] # don't bother with making two matrix spaces, we only need to test square matrices right now

            #TODO: this needs to be much faster 
            AA = rand(0:(P-1), s, s)
            BB = rand(0:(P-1), s, s)
            A = matrix(R,AA)
            B = matrix(R,BB)
            C = parent(A)()

            println("Allocs done")

            times = zeros(nTries)
            for i = 1:nTries
                CC = @timed mat_mul_oscar(C,A,B)
                times[i] = CC.time 
            end

            avg_time = sum(times) / nTries

            push!(results,(regime,type,size,P,avg_time))

            println("Oscar done: $avg_time")
        else 

            A = CUDA.rand(size[1,1], size[1,2])
            B = CUDA.rand(size[2,1], size[2,2])
            C = CUDA.zeros(type,size[1,1],size[2,2])
            
            @. A = floor(A * P)
            @. B = floor(B * P)

            println("Allocs done")

            
            CC = @btimed begin 
                CUDA.@sync mat_mul_gpu!($C,$A,$B,$P,$regime,$type) 
            end #samples=nTries

            avg_time = CC.time 

            push!(results,(regime,type,size,P,avg_time))

            println("GPU $regime done: $avg_time")

        end

        if 180 < avg_time
            println("Reached time limit, ending tests")
            break
        end

        println("")

        # catch
        #     println("Something went wrong with: $regime, $type, $size")
        # end
    end

    f = serialize("results_$(regime)_$P.dat", results)

    return results
end 

function mat_mul_benchmark_allocs(type,size,p,nTries)

    A = CUDA.rand(type, size, size)
    B = CUDA.rand(type, size, size)
    C = CUDA.zeros(type, size, size)
    
    @. A = floor(A * p)
    @. B = floor(B * p)

    # MARK - in place matmul
    res_inplace = @btimed begin
        mul!($C,$A,$B)
        CUDA.@sync $C .%= $p
    end samples=nTries

    println("In place multiplication: $(res_inplace.time)")
    
    # MARK - matmul with allocation of product

    res_alloc = @btimed begin
        D = $A*$B
        CUDA.@sync D .%= $p
    end samples=nTries

    println("Multiplication with allocation of result matrix: $(res_alloc.time)")
    
    # MARK - matmul with allocation of product and moving to the GPU

    AA = rand(type,size,size)
    BB = rand(type,size,size)

    @. AA = floor(AA * p)
    @. BB = floor(BB * p)

    res_alloc_and_copy = @btimed begin
        AAA = CuArray($AA)
        BBB = CuArray($BB)
        DD = AAA*BBB
        DD .%= $p
        Array(DD)
    end samples=nTries

    println("Multiplication with copying to GPU and allocating result matrix: $(res_alloc_and_copy.time)")

    (res_inplace,res_alloc,res_alloc_and_copy)

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

function mat_mul_oscar(C,A,B)
    Oscar.Nemo.mul!(C,A,B)
    return C
end

"""
Single threaded CPU multiplication for comparison
"""
function mat_mul_cpu(C, A, B, P)
    mul!(C,A,B)
    C .%= 11
    return C
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
    println("Capability (CUDA Version): $capability")
    println("Warp Size: $warpSize threads")

end
