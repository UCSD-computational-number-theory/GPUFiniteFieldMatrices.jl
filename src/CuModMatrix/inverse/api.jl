@inline _to_i32(x::Integer) = Int32(x)

"""
    pluq_new!(A; options=PLUQOptions())

Compute a blocked PLUQ factorization in place on GPU-resident data and return
`PLUQFactorization` with packed LU and permutation vectors.
"""
function pluq_new!(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    m = rows(A)
    n = cols(A)
    p, q, rank = if m == n
        pluq_blocked_gpu!(A.data, A.N, options, n)
    else
        pluq_rectangular_rank_gpu!(A.data, A.N, m, n)
    end
    return PLUQFactorization(A, p, q, rank)
end

"""
    pluq_new(A; options=PLUQOptions())

Compute a blocked PLUQ factorization on a copy of `A`.
"""
function pluq_new(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    Adata = copy(A.data)
    Awork = CuModMatrix(Adata, A.N; new_size=size(A))
    return pluq_new!(Awork, options=options)
end

"""
    is_invertible_new(A; options=PLUQOptions())

Return `true` iff `A` has full rank under the new GPU PLUQ path.
"""
function is_invertible_new(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    m = rows(A)
    n = cols(A)
    F = pluq_new(A, options=options)
    return m == n && F.rank == m
end

"""
    pluq_init_aug_kernel!(aug, Adata, n)

Initialize augmented matrix `[A | I]` in `aug` for Gauss-Jordan inversion.
Internal kernel used by `inverse_new`.
"""
function pluq_init_aug_kernel!(aug, Adata, n::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= 2n
        if j <= n
            aug[i, j] = Adata[i, j]
        else
            aug[i, j] = (j - n == i) ? one(eltype(aug)) : zero(eltype(aug))
        end
    end
    return
end

"""
    pluq_aug_find_pivot_kernel!(aug, pivot_slot, k, n, N)

Find first nonzero pivot row in column `k` of augmented matrix.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_find_pivot_kernel!(aug, pivot_slot, k::Int32, n::Int32, N::Int32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    span = n - k + 1
    total = span
    stride = blockDim().x * gridDim().x
    while idx <= total
        i = k + idx - 1
        v = _pluq_mod_t(aug[i, k], N)
        if v != zero(eltype(aug))
            CUDA.@atomic pivot_slot[1] = min(pivot_slot[1], i)
        end
        idx += stride
    end
    return
end

"""
    pluq_aug_scale_row_kernel!(aug, row, jstart, n2, invpivot, N)

Scale one augmented row by the modular inverse of the pivot.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_scale_row_kernel!(aug, row::Int32, jstart::Int32, n2::Int32, invpivot_slot, N::Int32)
    invpivot = invpivot_slot[1]
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + jstart - 1
    stride = blockDim().x * gridDim().x
    while j <= n2
        aug[row, j] = _pluq_mod_mul_t(aug[row, j], invpivot, N)
        j += stride
    end
    return
end

"""
    pluq_aug_elim_kernel!(aug, k, n, n2, N)

Eliminate column `k` from all rows except the pivot row in augmented matrix.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_elim_kernel!(aug, k::Int32, n::Int32, n2::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k - 1
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n2 && i != k
        f = _pluq_mod_t(aug[i, k], N)
        if f != zero(eltype(aug))
            aug[i, j] = _pluq_mod_t(aug[i, j] - _pluq_mod_mul_t(f, aug[k, j], N), N)
        end
    end
    return
end

"""
    pluq_copy_block_kernel!(dest, src, n)

Copy the top-left `n x n` block from `src` to `dest`.
Internal utility used by `inverse_new`.
"""
function pluq_copy_block_kernel!(dest, src, n::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        dest[i, j] = src[i, j]
    end
    return
end

"""
    inverse_new(A; options=PLUQOptions())

Compute the inverse over `mod N` using a fully GPU Gauss-Jordan elimination
path on an augmented matrix, without converting `A` to a CPU `Array`.
"""
function inverse_new(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    if rows(A) != cols(A)
        throw(CuModMatrixNotSquareException("matrix must be square"))
    end
    n = rows(A)
    N = A.N
    n2 = 2 * n
    aug = CUDA.zeros(eltype(A.data), size(A.data, 1), size(A.data, 2) + n + TILE_WIDTH)
    tx = 16
    ty = 16
    bx = max(1, cld(n2, tx))
    by = max(1, cld(n, ty))
    n32 = _to_i32(n)
    n232 = _to_i32(n2)
    N32 = _to_i32(N)
    @cuda threads=(tx, ty) blocks=(bx, by) pluq_init_aug_kernel!(aug, A.data, n32)
    threads = 256
    pivot_slot = CUDA.fill(_to_i32(n + 1), 1)
    invpivot_slot = CUDA.zeros(eltype(aug), 1)
    for k in 1:n
        k32 = _to_i32(k)
        CUDA.fill!(pivot_slot, _to_i32(n + 1))
        @cuda threads=threads blocks=max(1, cld(n - k + 1, threads)) pluq_aug_find_pivot_kernel!(aug, pivot_slot, k32, n32, N32)
        prow = Int(Array(@view pivot_slot[1:1])[1])
        if prow > n
            throw(InverseNotDefinedException("matrix is singular modulo $(A.N)"))
        end
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n2, threads)) pluq_swap_rows_kernel!(aug, k32, _to_i32(prow), n232)
        end
        @cuda threads=1 blocks=1 pluq_compute_invpivot_kernel!(invpivot_slot, aug, k32, N32)
        @cuda threads=threads blocks=max(1, cld(n2 - k + 1, threads)) pluq_aug_scale_row_kernel!(aug, k32, k32, n232, invpivot_slot, N32)
        bx2 = max(1, cld(n2 - k + 1, tx))
        by2 = max(1, cld(n, ty))
        @cuda threads=(tx, ty) blocks=(bx2, by2) pluq_aug_elim_kernel!(aug, k32, n32, n232, N32)
    end
    invdata = @view aug[1:n, (n + 1):n2]
    out = CUDA.zeros(eltype(A.data), size(A.data, 1), size(A.data, 2))
    bx3 = max(1, cld(n, tx))
    by3 = max(1, cld(n, ty))
    @cuda threads=(tx, ty) blocks=(bx3, by3) pluq_copy_block_kernel!(out, invdata, n32)
    return CuModMatrix(out, N; new_size=(n, n))
end

function pluq_init_rect_aug_kernel!(aug, Adata, m::Int32, n::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= m && j <= (n + m)
        if j <= n
            aug[i, j] = Adata[i, j]
        else
            aug[i, j] = (j - n == i) ? one(eltype(aug)) : zero(eltype(aug))
        end
    end
    return
end

function pluq_scale_row_rect_aug_kernel!(aug, row::Int32, jstart::Int32, w::Int32, invpivot_slot, N::Int32)
    invpivot = invpivot_slot[1]
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + jstart - 1
    stride = blockDim().x * gridDim().x
    while j <= w
        aug[row, j] = _pluq_mod_mul_t(aug[row, j], invpivot, N)
        j += stride
    end
    return
end

function pluq_elim_rect_aug_kernel!(aug, k::Int32, m::Int32, w::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k - 1
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= m && j <= w && i != k
        f = _pluq_mod_t(aug[i, k], N)
        if f != zero(eltype(aug))
            aug[i, j] = _pluq_mod_t(aug[i, j] - _pluq_mod_mul_t(f, aug[k, j], N), N)
        end
    end
    return
end

function pluq_load_z_kernel!(Z, Y, m::Int32, n::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= m
        if i <= m
            Z[i, j] = Y[i, j]
        else
            Z[i, j] = zero(eltype(Z))
        end
    end
    return
end

function pluq_scatter_solution_kernel!(X, Z, qdev, n::Int32, m::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= n && j <= m
        dst = qdev[i]
        X[dst, j] = Z[i, j]
    end
    return
end

"""
    right_inverse_new(A; options=PLUQOptions())

Compute a right inverse `X` such that `A*X = I` for full row-rank rectangular
`A` with `rows(A) <= cols(A)` over `GF(N)`.

This uses Gauss-Jordan elimination on `[A | I_m]` with row and column pivoting.
If rank is smaller than `m`, no right inverse exists and an exception is thrown.

Example:
```julia
A = CuModMatrix([1 2 3; 0 1 4], 101)
X = right_inverse_new(A)
Array(A * X)
```
"""
function right_inverse_new(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    m = rows(A)
    n = cols(A)
    if m > n
        throw(CuModArraySizeMismatchException("right inverse requires rows(A) <= cols(A)"))
    end
    N = A.N
    w = n + m
    aug = CUDA.zeros(eltype(A.data), size(A.data, 1), w + TILE_WIDTH)
    tx = 16
    ty = 16
    m32 = _to_i32(m)
    n32 = _to_i32(n)
    w32 = _to_i32(w)
    N32 = _to_i32(N)
    @cuda threads=(tx, ty) blocks=(max(1, cld(w, tx)), max(1, cld(m, ty))) pluq_init_rect_aug_kernel!(aug, A.data, m32, n32)
    q = collect(1:n)
    threads = 256
    pivot_slot = CUDA.fill(_to_i32(m * n + 1), 1)
    invpivot_slot = CUDA.zeros(eltype(aug), 1)
    rank = 0
    for k in 1:m
        span_r = m - k + 1
        span_c = n - k + 1
        if span_r <= 0 || span_c <= 0
            break
        end
        total = span_r * span_c
        k32 = _to_i32(k)
        CUDA.fill!(pivot_slot, _to_i32(total + 1))
        @cuda threads=threads blocks=max(1, cld(total, threads)) pluq_find_pivot_rect_kernel!(aug, pivot_slot, k32, m32, n32, N32)
        pivlin = Int(Array(@view pivot_slot[1:1])[1])
        if pivlin > total
            break
        end
        joff = (pivlin - 1) ÷ span_r
        ioff = (pivlin - 1) % span_r
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(w, threads)) pluq_swap_rows_kernel!(aug, k32, _to_i32(prow), w32)
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_swap_cols_kernel!(aug, k32, _to_i32(pcol), m32)
            q[k], q[pcol] = q[pcol], q[k]
        end
        @cuda threads=1 blocks=1 pluq_compute_invpivot_kernel!(invpivot_slot, aug, k32, N32)
        @cuda threads=threads blocks=max(1, cld(w - k + 1, threads)) pluq_scale_row_rect_aug_kernel!(aug, k32, k32, w32, invpivot_slot, N32)
        @cuda threads=(tx, ty) blocks=(max(1, cld(w - k + 1, tx)), max(1, cld(m, ty))) pluq_elim_rect_aug_kernel!(aug, k32, m32, w32, N32)
        rank += 1
    end
    if rank != m
        throw(InverseNotDefinedException("matrix is not full row-rank modulo $(A.N); right inverse undefined"))
    end
    Y = @view aug[1:m, (n + 1):(n + m)]
    Z = CUDA.zeros(eltype(A.data), n + TILE_WIDTH, m + TILE_WIDTH)
    X = CUDA.zeros(eltype(A.data), n + TILE_WIDTH, m + TILE_WIDTH)
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(m, ty))) pluq_load_z_kernel!(Z, Y, m32, n32)
    qdev = CuArray(Int32.(q))
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(m, ty))) pluq_scatter_solution_kernel!(X, Z, qdev, n32, m32)
    return CuModMatrix(X, N; new_size=(n, m))
end

"""
    left_inverse_new(A; options=PLUQOptions())

Compute a left inverse `X` such that `X*A = I` for full column-rank
rectangular `A` with `rows(A) >= cols(A)` over `GF(N)`.

Example:
```julia
A = CuModMatrix([1 2; 0 1; 3 4], 101)
X = left_inverse_new(A)
Array(X * A)
```
"""
function left_inverse_new(A::CuModMatrix; options::PLUQOptions=PLUQOptions())
    m = rows(A)
    n = cols(A)
    if m < n
        throw(CuModArraySizeMismatchException("left inverse requires rows(A) >= cols(A)"))
    end
    ATdata = permutedims(A.data, (2, 1))
    AT = CuModMatrix(ATdata, A.N; new_size=(n, m))
    R = right_inverse_new(AT, options=options)
    Ldata = permutedims(R.data, (2, 1))
    return CuModMatrix(Ldata, A.N; new_size=(n, m))
end
