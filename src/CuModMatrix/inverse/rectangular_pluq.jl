"""
    pluq_find_pivot_rect_kernel!(A, pivot_slot, k, m, n, N)

Find the first nonzero pivot in the active rectangular submatrix
`A[k:m, k:n]`, encoded as a linear position in `pivot_slot[1]`.
"""
function pluq_find_pivot_rect_kernel!(A, pivot_slot, k::Int32, m::Int32, n::Int32, N::Int32)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    span_r = m - k + 1
    span_c = n - k + 1
    total = span_r * span_c
    stride = blockDim().x * gridDim().x
    idx = tid
    while idx <= total
        joff = (idx - 1) ÷ span_r
        ioff = (idx - 1) % span_r
        i = k + ioff
        j = k + joff
        v = _pluq_mod_t(A[i, j], N)
        if v != zero(eltype(A))
            CUDA.@atomic pivot_slot[1] = min(pivot_slot[1], idx)
        end
        idx += stride
    end
    return
end

"""
    pluq_scale_column_rect_kernel!(A, k, m, invpivot, N)

Scale entries below the pivot in column `k`:
`A[k+1:m, k] *= invpivot (mod N)`.
"""
function pluq_scale_column_rect_kernel!(A, k::Int32, m::Int32, invpivot, N::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    stride = blockDim().x * gridDim().x
    while i <= m
        A[i, k] = _pluq_mod_mul_t(A[i, k], invpivot, N)
        i += stride
    end
    return
end

"""
    pluq_rank1_update_rect_kernel!(A, k, m, n, N)

Apply rank-1 elimination update on rectangular trailing block:
`A[k+1:m, k+1:n] -= A[k+1:m,k] * A[k,k+1:n]`.
"""
function pluq_rank1_update_rect_kernel!(A, k::Int32, m::Int32, n::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y + k
    if i <= m && j <= n
        A[i, j] = _pluq_mod_t(A[i, j] - _pluq_mod_mul_t(A[i, k], A[k, j], N), N)
    end
    return
end

"""
    pluq_rectangular_rank_gpu!(Adata, N, m, n)

In-place rank-revealing PLUQ-style elimination for rectangular matrices.
Returns `(p, q, rank)` where:
- `p` is row permutation vector (length `m`)
- `q` is column permutation vector (length `n`)
- `rank` is computed rank over `GF(N)`.
"""
function pluq_rectangular_rank_gpu!(Adata::CuArray{T,2}, N::Int, m::Int, n::Int) where {T}
    rmax = min(m, n)
    p = collect(1:m)
    q = collect(1:n)
    rank = 0
    threads = 256
    N32 = Int32(N)
    maxspan = max(1, (m - 1 + 1) * (n - 1 + 1))
    pivot_slot = CUDA.fill(Int32(maxspan + 1), 1)
    for k in 1:rmax
        span_r = m - k + 1
        span_c = n - k + 1
        if span_r <= 0 || span_c <= 0
            break
        end
        total = span_r * span_c
        CUDA.fill!(pivot_slot, Int32(total + 1))
        blocks = max(1, cld(total, threads))
        @cuda threads=threads blocks=blocks pluq_find_pivot_rect_kernel!(Adata, pivot_slot, Int32(k), Int32(m), Int32(n), N32)
        pivlin = Int(Array(@view pivot_slot[1:1])[1])
        if pivlin > total
            break
        end
        joff = (pivlin - 1) ÷ span_r
        ioff = (pivlin - 1) % span_r
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_rows_kernel!(Adata, Int32(k), Int32(prow), Int32(n))
            p[k], p[prow] = p[prow], p[k]
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_swap_cols_kernel!(Adata, Int32(k), Int32(pcol), Int32(m))
            q[k], q[pcol] = q[pcol], q[k]
        end
        piv = Int(Array(@view Adata[k:k, k:k])[1])
        invpiv = convert(T, pluq_mod_inv(piv, N))
        if k < m
            @cuda threads=threads blocks=max(1, cld(m - k, threads)) pluq_scale_column_rect_kernel!(Adata, Int32(k), Int32(m), invpiv, N32)
        end
        if k < m && k < n
            tx = 16
            ty = 16
            bx = max(1, cld(n - k, tx))
            by = max(1, cld(m - k, ty))
            @cuda threads=(tx, ty) blocks=(bx, by) pluq_rank1_update_rect_kernel!(Adata, Int32(k), Int32(m), Int32(n), N32)
        end
        rank += 1
    end
    return p, q, rank
end
