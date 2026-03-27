"""
    pluq_find_pivot_rect_kernel!(A, pivot_slot, k, m, n, N)

Find the first nonzero pivot in the active rectangular submatrix
`A[k:m, k:n]`, encoded as a linear position in `pivot_slot[1]`.
"""
function pluq_find_pivot_rect_kernel!(A, pivot_slot, k::Int32, m::Int32, n::Int32, N::Int32)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ltid = Int(threadIdx().x)
    span_r = m - k + 1
    span_c = n - k + 1
    total = span_r * span_c
    stride = blockDim().x * gridDim().x
    idx = gtid
    local_min = Int32(total + 1)
    while idx <= total
        joff = (idx - 1) ÷ span_r
        ioff = (idx - 1) % span_r
        i = k + ioff
        j = k + joff
        v = _pluq_mod_t(A[i, j], N)
        if v != zero(eltype(A))
            local_min = min(local_min, idx)
        end
        idx += stride
    end
    smins = CuStaticSharedArray(Int32, 256)
    smins[ltid] = local_min
    sync_threads()
    step = Int(blockDim().x) >>> 1
    while step >= 1
        if ltid <= step
            smins[ltid] = min(smins[ltid], smins[ltid + step])
        end
        sync_threads()
        step >>>= 1
    end
    if ltid == 1
        CUDA.@atomic pivot_slot[1] = min(pivot_slot[1], smins[1])
    end
    return
end

function pluq_find_pivot_rect_warp_kernel!(A, pivot_slot, k::Int32, m::Int32, n::Int32, N::Int32)
    lane = Int(threadIdx().x)
    if lane > 32
        return
    end
    span_r = m - k + 1
    span_c = n - k + 1
    joff = Int32(0)
    while joff < span_c
        row = k + Int32(lane - 1)
        pred = Int32(lane) <= span_r && _pluq_mod_t(A[row, k + joff], N) != zero(eltype(A))
        bits = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pred)
        if bits != UInt32(0)
            if lane == 1
                first_lane = Int32(trailing_zeros(bits) + 1)
                pivot_slot[1] = joff * span_r + first_lane
            end
            return
        end
        joff += 1
    end
    return
end

function pluq_find_pivot_rect_warp_shfl_kernel!(A, pivot_slot, k::Int32, m::Int32, n::Int32, N::Int32)
    lane = Int32(threadIdx().x)
    if lane > Int32(32)
        return
    end
    span_r = m - k + Int32(1)
    span_c = n - k + Int32(1)
    local_min = span_r * span_c + Int32(1)
    if lane <= span_r
        row = k + lane - Int32(1)
        joff = Int32(0)
        while joff < span_c
            if _pluq_mod_t(A[row, k + joff], N) != zero(eltype(A))
                cand = joff * span_r + lane
                local_min = min(local_min, cand)
            end
            joff += Int32(1)
        end
    end
    wmin = _pluq_warp_min_shfl_i32(local_min)
    if lane == Int32(1)
        pivot_slot[1] = wmin
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

function pluq_scale_column_rect_from_diag_kernel!(A, k::Int32, m::Int32, N::Int32)
    invslot = CuStaticSharedArray(eltype(A), 1)
    if threadIdx().x == 1
        invslot[1] = _pluq_mod_inv_t(A[k, k], N)
    end
    sync_threads()
    invpivot = invslot[1]
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
function pluq_rectangular_rank_gpu!(Adata::CuArray{T,2}, N::Int, m::Int, n::Int; options::PLUQOptions=PLUQOptions()) where {T}
    rmax = min(m, n)
    p = collect(1:m)
    q = collect(1:n)
    rank = 0
    threads = 256
    N32 = Int32(N)
    m32 = Int32(m)
    n32 = Int32(n)
    pivot_slot = CUDA.fill(Int32(max(1, m * n + 1)), 1)
    for k in 1:rmax
        span_r = m - k + 1
        span_c = n - k + 1
        total = span_r * span_c
        CUDA.fill!(pivot_slot, Int32(total + 1))
        blocks = max(1, cld(total, threads))
        k32 = Int32(k)
        if span_r <= 32
            if options.pivot_warp_kernel == :shfl
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_shfl_kernel!(Adata, pivot_slot, k32, m32, n32, N32)
            else
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_kernel!(Adata, pivot_slot, k32, m32, n32, N32)
            end
        else
            @cuda threads=threads blocks=blocks pluq_find_pivot_rect_kernel!(Adata, pivot_slot, k32, m32, n32, N32)
        end
        pivlin = Int(Array(@view pivot_slot[1:1])[1])
        if pivlin > total
            break
        end
        joff = (pivlin - 1) ÷ span_r
        ioff = (pivlin - 1) % span_r
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_rows_kernel!(Adata, k32, Int32(prow), n32)
            p[k], p[prow] = p[prow], p[k]
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_swap_cols_kernel!(Adata, k32, Int32(pcol), m32)
            q[k], q[pcol] = q[pcol], q[k]
        end
        if k < m
            @cuda threads=threads blocks=max(1, cld(m - k, threads)) pluq_scale_column_rect_from_diag_kernel!(Adata, k32, m32, N32)
        end
        if k < n
            tx = 16
            ty = 16
            bx = max(1, cld(n - k, tx))
            by = max(1, cld(m - k, ty))
            @cuda threads=(tx, ty) blocks=(bx, by) pluq_rank1_update_rect_kernel!(Adata, k32, m32, n32, N32)
        end
        rank += 1
    end
    return p, q, rank
end
