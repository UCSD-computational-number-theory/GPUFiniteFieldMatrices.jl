using CUDA, LinearAlgebra

"""
Main rref function
"""
function rref_gpu(A, N)

    m, n = size(A)

    p = Vector{Int}(undef, m)

    for k = 1:m-1
        # Can't we combine these two?
        p[k] = find_pivot(A,k)
        swap(A[k,:],A[p[k],:])

        normalize(A(:,k+1))
        updateSubMatrix(A[k+1:end,k+1:end])
    end

    return p
end

function find_pivot(A,k)

    row, col = size(A)

    # Technically we do not need to find the largest in column
    # Since we do not care about numerical stability
    max_row = argmax(A[k:rows, k])[1] + k - 1

    if max_row != k
        A[k,:], A[max_row,:] = A[max_row,:], A[k,:]
    end
    
    return A[max_rows,k]

end

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

    A = CUDA.rand(1000,1000)
    @. A = floor(A * 100)

    benchres1 = @benchmark CUDA.@sync swap_rows(A,0,2,3,11)
    benchres2 = @benchmark CUDA.@sync swap_rows_broadcast(A,0,3,4,11)

    (benchres1,benchres2)
end
