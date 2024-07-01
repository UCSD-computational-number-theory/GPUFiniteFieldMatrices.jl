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