using CUDA, LinearAlgebra, BenchmarkTools

"""
Swaps row i with row j, and then mods the new row i (old row j) by p.
Does it by broadcasting... though this doesn't actually work
"""
function swap_rows_broadcast(A,startcol,i,j,p)
    s = startcol + 1

    tmp = copy(A[j,s:end])
    #tmp .%= p
    A[j,s:end] .= A[i,s:end]

    @. A[i,s:end] = tmp % p

    nothing
end

function swap_rows_swapsyntax(A,startcol,i,j,p)
    s = startcol + 1

    A[j,s:end], A[i,s:end] = A[i,s:end], A[j,s:end]

    A[i,s:end] .%= p

    nothing
end

const NUM_THREADS_PER_BLOCK = 32

"""
    swap_rows(A,startcol,i,j,p)
Swaps the rows i and j in A, then mods row i by p
Note: only swaps entries that are to the left of
startcol. I.e. startcol and everthing to the
right will remain unchanged.
Note: 
"""
function swap_rows(A,startcol,i,j,p)
    # clal the kernel

    t = NUM_THREADS_PER_BLOCK
    nCols = size(A,2)
    b = div(nCols,t) # will miss the last few entries, be sure to pad

    @cuda threads=t blocks=b swap_rows_kernel(A,startcol,i,j,p)

    nothing
end

function swap_rows_kernel(A,startcol,i,j,p)
    offset = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    c = startcol + offset 

    tmp = A[j,c]
    A[j,c] = A[i,c]
    tmp = tmp % p
    A[i,c] = tmp

    nothing
end

function test_swap_rows()
    n = 1_000_000 
    NUM_ROWS = 32 

    A = CUDA.rand(NUM_ROWS,n)
    @. A = floor(A * 100)

    B = rand(NUM_ROWS,n)
    @. B = floor(B * 100)

    benchres1 = @benchmark CUDA.@sync swap_rows($A,0,2,3,11)
    benchres2 = @benchmark CUDA.@sync swap_rows_broadcast($A,0,3,4,11)
    benchres3 = @benchmark CUDA.@sync swap_rows_swapsyntax($A,0,4,5,11)
    benchres4 = @benchmark swap_rows_broadcast($B,0,2,3,11)
    benchres5 = @benchmark swap_rows_swapsyntax($B,0,3,4,11)

    println("CUDA Kernel: ")
    display(benchres1)
    println("CUDA broadcasting: ")
    display(benchres2)
    println("CUDA swap syntax: ")
    display(benchres3)
    println("CPU broadcasting: ")
    display(benchres4)
    println("CPU swap syntax: ")
    display(benchres5)

    (benchres1,benchres2,benchres3,benchres4,benchres5)
end
