using CUDA, LinearAlgebra
include("rref_new_kernels.jl")
include("../gpu_mat_type/gpu_mat.jl")

"""
    rref_gpu_type(A::GpuMatrixModN, [mod_N])

Row reduction (RREF) that works directly with GpuMatrixModN objects.
Returns a new GpuMatrixModN in row reduced echelon form.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function rref_gpu_type(A::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    A_rows, A_cols = size(A)
    TILE_WIDTH = 32
    
    A_padded_rows = (ceil(Int, A_rows / TILE_WIDTH)+1) * TILE_WIDTH
    A_padded_cols = (ceil(Int, A_cols / TILE_WIDTH)+1) * TILE_WIDTH 

    d_A = copy(A.data)

    row = 1
    col = 1

    while row <= A_rows && col <= A_cols
        k = find_pivot(d_A, A_rows, row, col)
        p = 1
        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, N)
        swap_and_mod(d_A, k, row, p_inv, N)

        normalize_broadcast(d_A, col, p_inv, N)

        @cuda threads=(TILE_WIDTH) blocks=(div(A_rows,TILE_WIDTH)) update_sub_matrix_row(d_A, row, col, div(A_cols-col,TILE_WIDTH), N)

        row += 1
        col += 1
    end

    result = GpuMatrixModN(d_A, A_rows, A_cols, N)
    
    return result
end

"""
    lu_gpu_type(A::GpuMatrixModN, [mod_N])

LU decomposition that works directly with GpuMatrixModN objects.
Returns matrices in GpuMatrixModN format.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function lu_gpu_type(A::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    A_rows, A_cols = size(A)
    TILE_WIDTH = 32

    d_A = copy(A.data)
    d_L = CUDA.CuArray{Int}(undef, (A_rows, A_rows))
    Perm = Array(1:A_rows)

    row = 1
    L_col = 1
    col = 1

    while row <= A_rows && col <= A_cols
        k = find_pivot_idx(d_A, A_rows, row, col)
        p = find_pivot_val(d_A, A_rows, row, col)
        
        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, N)
        swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, N, Perm)
        
        normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, N)
        
        if row == A_rows || col == A_cols
            break
        end

        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_rows-row,TILE_WIDTH)+1,1) update_sub_matrix_row(d_A, row, col, div(A_cols-col,TILE_WIDTH), N)
        
        row += 1
        L_col += 1
        col += 1
    end
    
    # Create GpuMatrixModN objects for the results using the new constructor
    U = GpuMatrixModN(d_A, A_rows, A_cols, N)
    L = GpuMatrixModN(d_L, A_rows, A_rows, N)
    
    return (U, L, Perm)
end

"""
    plup_gpu_type(A::GpuMatrixModN, [mod_N])

PLUP decomposition that works directly with GpuMatrixModN objects.
Returns U and L matrices in GpuMatrixModN format.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function plup_gpu_type(A::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    TILE_WIDTH = 32
    
    A_padded_rows = size(A.data, 1)
    A_padded_cols = size(A.data, 2)

    d_A = copy(A.data)
    d_L = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_rows))
    Perm_rows = Array(1:A_padded_rows)
    Perm_cols = Array(1:A_padded_cols)

    row = 1
    col = 1
    Perm_col_idx = A_padded_cols

    while row <= A_padded_rows && col <= A_padded_cols
        while find_zero_col_and_swap(d_A, A_padded_rows, row, col, Perm_cols, Perm_col_idx)
            Perm_col_idx -= 1
        end

        k = find_pivot_idx(d_A, A.rows, row, col)
        p = find_pivot_val(d_A, A.rows, row, col)
        
        if p == 0
            d_L[row:end,col] .= 1
            d_L[row+1:end,col] .= 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, N)
        swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, N, Perm_rows)
        
        normalize_lu_broadcast(d_A, d_L, A.rows, row, col, p, N)
        
        if row == A_padded_rows || col == A_padded_cols
            break
        end

        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_padded_rows-row,TILE_WIDTH)+1,1) update_sub_matrix_row(d_A, row, col, div(A_padded_cols-col,TILE_WIDTH), N)
        
        row += 1
        col += 1
    end
    
    # Create GpuMatrixModN objects for the results using the new constructor
    U = GpuMatrixModN(d_A, A.rows, A.cols, N)
    L = GpuMatrixModN(d_L, A.rows, A.rows, N)
    
    return (U, L, Perm_rows, Perm_cols)
end 
