using CUDA, LinearAlgebra

const global TILE_WIDTH = 4
const global TYPE = Float32

function rref_gpu(A, P)

    A_rows, A_cols = size(A)
    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH 

    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))
    
    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows,1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1

    while row <= A_rows && col <= A_cols

        p = find_pivot(d_A, A_rows, row, col)
        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod(d_A, row, inv)

        @cuda threads=(TILE_WIDTH) blocks=(div(A_padded_rows-row,TILE_WDITH)) normalize(d_A, A_rows, col, p_inv, P)

        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_padded_rows-rows,TILE_WIDTH),div(A_padded_cols-cols,TILE_WIDTH)) update_sub_matrix_col(d_A, row, col, P)

        row += 1
        col += 1

    end

    return Array(d_A)[1:A_rows,1:A_cols]
end

function lu_gpu(A, P)

    A_rows, A_cols = size(A)
    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH 

    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))
    d_L = CUDA.CuArray{Int}(Matrix{Int}(I,3,3))
    Perm = 1:A_padded_rows
    
    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows,1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1

    while row <= A_rows && col 

        p = find_pivot(d_A, A_rows, row, col)
        if p == 0
            k += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod_lu(d_A, row, inv, Perm)

        @cuda threads=(TILE_WIDTH) blocks=(div(A_padded_rows-row,TILE_WDITH)) normalize_lu(d_A, A_rows, col, p_inv, P, Perm)

        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_padded_rows-row,TILE_WIDTH),div(A_padded_cols-col,TILE_WIDTH)) update_sub_matrix_col(d_A, row, col, P)

        row += 1
        col += 1

    end

    return Array(d_A)[1:A_rows,1:A_cols], Array(d_L)[1:A_cols,1:A_rows], P
end

function find_pivot(d_A, A_rows, row, col)
    return argmax(d_A[row:A_rows,col])
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

function swap_and_mod(d_A, p_row, inv)

    # swap k and p_row
    d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]

    # normalize p_row
    d_A[p_row,:] = (inv * d_A[p_row]) % P

    return
end

function swap_and_mod_lu(d_A, p_row, inv, Perm)

    d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]
    Perm[k], Perm[p_row] = Perm[p_row], Perm[k]

    d_A[p_row, :] = (inv * d_A[p_row]) % P 
    
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
    CUDA.sync_threads()
    
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
    CUDA.sync_threads()
    
    return
end

function update_sub_matrix_row(d_A, p_row, p_col, P)

    p_row_shared = CUDA.CuStaticSharedArray(TYPE, TILE_WIDTH)

    idx = threadIdx().x
    row = p_row + (blockIdx().x * TILE_WIDTH) + idx

    inv = d_A[row,p_col]
    d_A[row,p_col] = 0

    p_row_shared[idx] = d_A[p_row, p_col+idx]
    CUDA.sync_threads()
    
    i = p_col + 1
    while i <= A_cols
        d_A[row,i] = (d_A[row,i] - p_row_shared[i] * inv) % P
        i += 1
    end
    CUDA.sync_threads()

    return
end

function update_sub_matrix_col(d_A, p_row, p_col, P)

    p_col_shared = CUDA.CuStaticSharedArray(TYPE, TILE_WIDTH)

    idx = threadIdx().x
    col = p_row + (blockIdx().x * TILE_WIDTH) + idx

    p_val = d_A[p_row, col]

    p_col_shared[idx] = d_A[p_row+idx, p_col]
    d_A[p_row+idx, p_col] = 0
    CUDA.sync_threads()
    
    i = p_row + 1
    while i <= A_cols
        d_A[i,col] = (d_A[i,col] - p_col_shared[i] * p_val) % P
        i += 1
    end
    CUDA.sync_threads()

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
    CUDA.sync_threads()

    return
end