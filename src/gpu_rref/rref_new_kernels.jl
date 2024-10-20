using CUDA, LinearAlgebra

const global TILE_WIDTH = 2
const global TYPE = Float32

function rref_gpu(A, P)

    A_rows, A_cols = size(A)
    A_padded_rows = (ceil(Int, A_rows / TILE_WIDTH)+1) * TILE_WIDTH
    A_padded_cols = (ceil(Int, A_cols / TILE_WIDTH)+1) * TILE_WIDTH 

    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))
    
    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows,1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1

    while row <= A_rows && col <= A_cols

        # println("$row $col")

        k = find_pivot(d_A, A_rows, row, col)
        p = 1
        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod(d_A, k, row, p_inv, P)

        normalize_broadcast(d_A, col, p_inv, P)

        # println("A: $A")
        # println("row: $row")
        # println("col: $col")
        # println("A_padded_rows: $A_padded_rows")

        @cuda threads=(TILE_WIDTH) blocks=(div(A_rows,TILE_WIDTH)) update_sub_matrix_row(d_A, row, col, div(A_cols-col,TILE_WIDTH), P)

        row += 1
        col += 1

    end

    # A = Array(d_A)

    return Array(d_A)
end

function lu_gpu(A, P)

    # Find padded dimensions
    A_rows, A_cols = size(A)
    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH 

    # Deifne gpu matrices
    # We add another TILE_WDITH for the reduction of the last few rows
    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))
    d_L = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))
    Perm = Array(1:A_padded_rows)
    
    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows,1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1

    while row <= A_rows && col <= A_cols

        k = find_pivot(d_A, A_rows, row, col)
        p = 1
        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod_lu(d_A, k, row, p_inv, P, Perm)

        # @cuda threads=(TILE_WIDTH) blocks=(div(A_padded_rows-row,TILE_WDITH)) normalize_lu(d_A, A_rows, col, p_inv, P, d_L)

        normalize_lu_broadcast(d_A, d_L, row, col, p_inv, P)

        @cuda threads=(TILE_WIDTH) blocks=(div(A_rows,TILE_WIDTH)) update_sub_matrix_row(d_A, row, col, div(A_cols-col,TILE_WIDTH), P)

        row += 1
        col += 1

    end

    # return Array(d_A)[1:A_rows,1:A_cols], Array(d_L)[1:A_cols,1:A_rows], P
    return (Array(d_A), Array(d_L), Perm)
end

function find_pivot(d_A, A_rows, row, col)
    # return argmax(d_A[row:A_rows,col])
    #TODO move back to original after testing
    return row
end

function find_pivot_new(d_A, A_rows, row, col)
    CUDA.allowscalar() do 
        return argmax(Array(d_A[row:A_rows,col]))
    end
end

function find_pivot_custom(d_A, A_rows, row, col, res)
    tidx = threadIdx().x
    bidx = blockIdx().x
    idx = tid + (bid - 1) * blockDim().x

    if arr[idx,col] != 0
        res[1] = idx
        return
    end

    return
end

function mod_inv(p, P)

    # Remark: Since p is prime, we are in a field
    # And thus an inverse is guaranteed to exist
    inv, new_inv = 0, 1
    rem, new_rem = P, p

    while new_rem != 0
        quotient = div(rem, new_rem)
        inv, new_inv = new_inv, inv - quotient * new_inv
        rem, new_rem = new_rem, rem - quotient * new_rem
    end

    if inv < 0
        inv += P
    end

    return inv
end

function mod_inv_no_if(p, P)

    # Remark: Since p is prime, we are in a field
    # And thus an inverse is guaranteed to exist
    inv, new_inv = 0, 1
    rem, new_rem = P, p

    while new_rem != 0
        quotient = div(rem, new_rem)
        inv, new_inv = new_inv, inv - quotient * new_inv
        rem, new_rem = new_rem, rem - quotient * new_rem
    end

    return mod(inv, P)
end

function swap_and_mod_threaded(d_A, k, p_row, inv)
    
    col = (blockIdx().x-1)*TILE_WIDTH + threadIdx().x

    # swap k and p_row
    d_A[k,col], d_A[p_row,col] = d_A[p_row,col], d_A[k,col]
    
    # normalize p_row
    d_A[p_row,col] = inv * d_A[p_row,col] % P

    return
end

function swap_and_mod(d_A, k, p_row, inv, P)

    # swap k and p_row
    d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]

    # normalize p_row
    d_A[p_row,:] = (d_A[p_row,:] .* inv) .% P

    return
end

function swap_and_mod_lu(d_A, k, p_row, inv, P, Perm)

    d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]
    Perm[k], Perm[p_row] = Perm[p_row], Perm[k]

    d_A[p_row,:] = (d_A[p_row,:] .* inv) .% P 
    
    return
end

function normalize(d_A, A_rows, col, p_inv, P)

    idx = threadIdx().x
    bx = blockIdx().x
    
    i = (bx-1)*TILE_WIDTH + col + idx
    while i <= A_rows
        d_A[i,col] = (d_A[i,col] * p_inv) % P
        i += 1
    end
    
    return
end

function normalize_lu(d_A, A_rows, col, p_inv, P, d_L)

    idx = threadIdx().x
    bx = blockIdx().x
    
    i = (bx-1)*TILE_WIDTH + col + idx
    while i <= A_rows
        res = (d_A[i,col] * p_inv) % P
        d_A[i,col] = res
        d_L[i,col] = res
        i += 1
    end
    
    return
end

function normalize_broadcast(d_A, col, p_inv, P)

    d_A[:,col] = (d_A[:,col] * p_inv) .% P

    return
end

function normalize_lu_broadcast(d_A, d_L, row, col, p_inv, P)
    CUDA.allowscalar() do
        d_L[row:end,col] .= d_A[row:end,col] .% P
    end
    res = (d_A[:,col] * p_inv) .% P
    d_A[:,col] = res

    return
end

function update_sub_matrix_row(d_A, p_row, p_col, bound, P)

    # Sample computation
    # A = 8x8, thread = 2,1,1, block = 1,1,1, p_row=8, p_col=8
    # tid = 2, bid = 1, idx = 2
    # row_inv = d_A[10,8]
    # shared_row = Arr[2]
    # m = 0
    # m = 1
    # m = 2
    # m = 3; shared[2] = d_A[8,8+2]

    tid = threadIdx().x
    bid = blockIdx().x
    idx = tid + (bid - 1) * blockDim().x

    row_inv = d_A[p_row+idx,p_col]
    shared_row = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))

    m = 0
    while m < bound
        shared_row[tid] = d_A[p_row,p_col + tid + m*TILE_WIDTH]
        d_A[p_row + idx,p_col + tid + m*TILE_WIDTH] = (d_A[p_row + idx,p_col + tid + m*TILE_WIDTH] + shared_row[tid] * row_inv) % P
        m += 1
    end
    d_A[p_row+idx,p_col] = 0

    return
end

function update_sub_matrix_col(d_A, p_row, p_col, bound, P)
    
    tid = threadIdx().x
    bid = blockIdx().x
    idx = tid + (bid - 1) * blockDim().x

    row_val = d_A[p_row,p_col+idx]
    shared_col = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))

    m = 0
    while m < bound
        shared_col[tid] = d_A[p_row + tid + m*TILE_WIDTH,p_col]
        d_A[p_row + tid + m*TILE_WIDTH,p_col + idx] = (d_A[p_row + tid + m*TILE_WIDTH,p_col + idx] + shared_col[tid] * row_val) % P
        m += 1
    end
    d_A[p_row+idx,p_col] = 0

    return
end

function update_sub_matrix_elem(d_A, p_row, p_col, P)

    tidx = threadIdx().x
    bidx = blockIdx().x
    tidy = threadIdx().y
    bidy = blockIdx().y
    idx = tidx + (bidx - 1) * blockDim().x
    idy = tidy + (bidy - 1) * blockDim().y

    d_A[p_row + idy, p_col + idx] = (d_A[p_row + idy, p_col + idx] + d_A[p_row, p_col + idx] * d_A[p_row + idy, p_col]) % P

    return
end

function update_sub_matrix_square(d_A, p_row, p_col, P)

    idx = threadIdx().x
    idy = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y

    row = (by-1)*TILE_WIDTH + idy
    col = (bx-1)*TILE_WIDTH + idx

    d_A[row,col] = (d_A[row,col] - d_A[row,p_col] * d_A[p_row,col]) % P
    # CUDA.sync_threads()

    return
end

function update_sub_matrix_broadcast(d_A, A_rows, p_row, p_col, P)

    for row=p_row:A_rows
        CUDA.@allowscalar d_A[row,:] = (d_A[row,:] + d_A[p_row,:] .* d_A[row, p_col]) .% P
    end

    return
end

function update_sub_matrix_broadcast2(d_A, A_rows, p_row, p_col, P)

    temp = zeros(A_rows)
    temp .= Array(d_A[:, p_col])

    for row=p_row:A_rows
        CUDA.@allowscalar d_A[row,:] = (d_A[row,:] + d_A[p_row,:] .* temp[row]) .% P
    end

    return
end

function update_sub_matrix_broadcast4(d_A, A_rows, p_row, p_col, P)

    temp = zeros(A_rows-p_row+1,1)
    temp_inds = CartesianIndices((A_rows,1))
    d_A_inds = CartesianIndices((p_row:A_rows,p_col))
    CUDA.@allowscalar copyto!(d_A, d_A_inds, temp, temp_inds)

    for row=p_row:A_rows
        d_A[row,:] = (d_A[row,:] + d_A[p_row,:] .* temp[row-p_row+1]) .% P
    end

    return
end