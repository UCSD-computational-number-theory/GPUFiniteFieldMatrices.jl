"""
    rref_gpu_type(A::CuModMatrix, [mod_N])

Row reduction (RREF) that works directly with CuModMatrix objects.
Returns a new CuModMatrix in row reduced echelon form.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function rref_gpu_type(A::CuModMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    A_rows, A_cols = size(A.data)

    d_A = CUDA.zeros(Int, (A_rows, A_cols))
    copyto!(d_A, A.data)

    row = 1
    col = 1

    while row <= rows(A) && col <= cols(A)

        p = find_pivot_val(d_A, rows(A), row, col)
        
        if p == 0
            col += 1
            continue
        end

        # Find pivot row
        k = find_pivot_idx(d_A, rows(A), row, col) + row - 1
        
        # Only swap if needed
        p_inv = mod_inv(p, N)
        if k != row   
            swap_and_mod(d_A, k, row, p_inv, N)
        end

        # Only normalize if needed
        if p != 1
            normalize_broadcast(d_A, col, p_inv, N)
        end

        if row < rows(A) && col < cols(A)
            @cuda threads=(TILE_WIDTH) blocks=(div(rows(A)-row,TILE_WIDTH)+1) update_sub_matrix_row(d_A, row, col, N)
        end

        row += 1
        col += 1
    end

    return CuModMatrix(d_A, N; new_size=(rows(A),cols(A)))
end

"""
    lu_gpu_type(A::CuModMatrix, [mod_N])

LU decomposition that works directly with CuModMatrix objects.
Returns matrices in CuModMatrix format.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function lu_gpu_type(A::CuModMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    A_rows, A_cols = size(A.data)

    d_A = CUDA.zeros(Int, (A_rows, A_cols))
    copyto!(d_A, A.data)
    d_L = CUDA.zeros(Int, (A_rows, A_rows))
    Perm = Array(1:A_rows)

    row = 1
    L_col = 1
    col = 1

    while row <= rows(A) && col <= cols(A)
        p = find_pivot_val(d_A, rows(A), row, col)
        
        if p == 0
            col += 1
            continue
        end

        k = find_pivot_idx(d_A, rows(A), row, col)

        p_inv = mod_inv(p, N)
        swap_and_mod_lu(d_A, d_L, k, row, p_inv, N, Perm)
        
        normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, p, N)
        
        if row == rows(A) || col == cols(A)
            break
        end

        @cuda threads=(TILE_WIDTH) blocks=(div(rows(A)-row,TILE_WIDTH)+1) update_sub_matrix_row(d_A, row, col, N)

        row += 1
        L_col += 1
        col += 1
    end
    
    U = CuModMatrix(d_A, N; new_size=(rows(A), cols(A)))
    L = CuModMatrix(d_L, N; new_size=(rows(A), rows(A)))
    return (U, L, Perm)
end

"""
    plup_gpu_type(A::CuModMatrix, [mod_N])

PLUP decomposition that works directly with CuModMatrix objects.
Returns U and L matrices in CuModMatrix format.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function plup_gpu_type(A::CuModMatrix)
    N = A.N
    
    A_padded_rows = size(A.data, 1)
    A_padded_cols = size(A.data, 2)

    d_A = copy(A.data)
    d_L = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_rows))
    Perm_rows = Array(1:A_padded_rows)
    Perm_cols = Array(1:A_padded_cols)

    row = 1
    col = 1
    Perm_col_idx = A_padded_cols

    while row <= rows(A) && col <= cols(A)
        # println("New iteration")
        # println("Perm_col_idx: ", Perm_col_idx)
        # println("Perm_cols: ", Perm_cols)
        # println("row: ", row)
        # println("col: ", col)
        # println("d_A: ", @view d_A[1:rows(A),1:cols(A)])
        # println("d_L: ", @view d_L[1:rows(A),1:rows(A)])
        # println("--------------------------------")
        # while find_zero_col_and_swap(d_A, rows(A), row, col, Perm_cols, Perm_col_idx)

        #     Perm_col_idx -= 1
        #     println("ZEROED")
        # end

        k = find_pivot_idx(d_A, rows(A), row, col)
        p = find_pivot_val(d_A, rows(A), row, col)
        # println("k: ", k)
        # println("p: ", p)
        
        if p == 0
            d_L[row:end,col] .= 1
            d_L[row+1:end,col] .= 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, N)
        # println("p_inv: ", p_inv)
        swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, N, Perm_rows)
        
        normalize_lu_broadcast(d_A, d_L, rows(A), row, col, p_inv, p, N)

        # println("Normalized")
        # println(Perm_col_idx)
        # println(Perm_cols)
        # println(row)
        # println(col)
        # println(@view d_A[1:rows(A),1:cols(A)])
        # println(@view d_L[1:rows(A),1:rows(A)])
        # println("--------------------------------")
        
        if row == rows(A) || col == cols(A)
            break
        end

        @cuda threads=(TILE_WIDTH) blocks=(div(rows(A)-row,TILE_WIDTH)+1) update_sub_matrix_row(d_A, row, col, N)

        # println("Updated Sub Matrix")
        # println(Perm_col_idx)
        # println(Perm_cols)
        # println(row)
        # println(col)
        # println(@view d_A[1:rows(A),1:cols(A)])
        # println(@view d_L[1:rows(A),1:rows(A)])
        # println("--------------------------------")

        row += 1
        col += 1
    end
    
    # Create CuModMatrix objects for the results using the new constructor
    U = CuModMatrix(d_A, N; new_size=(rows(A),cols(A)))
    L = CuModMatrix(d_L, N; new_size=(rows(A),rows(A)))
    
    return (U, L, Perm_rows, Perm_cols)
end 

"""
    perm_array_to_matrix(perm::Vector{Int}, N::Integer=11)

Convert a permutation array to a permutation matrix.
Returns a CuModMatrix where the column with 1 in the ith column
is at the position of i in the input array.

# Arguments
- `perm`: A permutation array (values from 1 to n)
- `N`: The modulus for the CuModMatrix (default: 11)

# Returns
- A CuModMatrix representation of the permutation
"""
function perm_array_to_matrix(perm::Vector{Int}, N::Integer; new_size::Tuple{Int,Int}=(length(perm),length(perm)))
    n = length(perm)
    
    # Create a CPU array for the permutation matrix
    P = Base.zeros(Int, n, n)
    
    # Set the 1s according to the permutation
    for i in 1:n
        P[perm[i], i] = 1
    end
    
    return CuModMatrix(P, N; new_size=new_size)
end