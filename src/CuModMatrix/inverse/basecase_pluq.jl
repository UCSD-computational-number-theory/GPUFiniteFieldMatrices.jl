"""
    _pluq_mod_t(x, N)

Device-friendly modulo reduction returning the same element type as `x`.

`Int64` intermediates are used to avoid overflow in intermediate integer
products/sums before reducing modulo `N`, then values are converted back to
the matrix element type (`Float32`/`Float64` in this project).
"""
@inline function _pluq_mod_t(x::T, N::Int32) where {T}
    v = Int64(x)
    r = rem(v, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

"""
    _pluq_mod_mul_t(a, b, N)

Device-friendly modular multiplication returning the same element type.
"""
@inline function _pluq_mod_mul_t(a::T, b::T, N::Int32) where {T}
    av = Int64(a)
    bv = Int64(b)
    r = rem(av * bv, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

@inline function _pluq_mod_inv_t(a::T, N::Int32) where {T}
    aa = Int32(rem(Int64(a), Int64(N)))
    if aa < 0
        aa += N
    end
    if aa == 0
        return zero(T)
    end
    t = Int32(0)
    newt = Int32(1)
    r = N
    newr = aa
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if t < 0
        t += N
    end
    return T(t)
end

"""
    pluq_find_pivot_kernel!(A, pivot_slot, k, kend, N)

Search for the first nonzero pivot in the active block `[k:kend, k:kend]`.
Writes the earliest linearized position into `pivot_slot[1]`.
"""
function pluq_find_pivot_kernel!(A, pivot_slot, k::Int32, kend::Int32, N::Int32)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ltid = Int(threadIdx().x)
    span = kend - k + 1
    total = span * span
    stride = blockDim().x * gridDim().x
    idx = gtid
    local_min = Int32(total + 1)
    while idx <= total
        joff = (idx - 1) ÷ span
        ioff = (idx - 1) % span
        i = k + ioff
        j = k + joff
        val = _pluq_mod_t(A[i, j], N)
        if val != zero(eltype(A))
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

function pluq_find_pivot_warp_kernel!(A, pivot_slot, k::Int32, kend::Int32, N::Int32)
    lane = Int(threadIdx().x)
    span = kend - k + 1
    if lane > 32
        return
    end
    for joff in 0:(span - 1)
        row = k + lane - 1
        pred = lane <= span && _pluq_mod_t(A[row, k + joff], N) != zero(eltype(A))
        bits = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pred)
        if bits != UInt32(0)
            if lane == 1
                first_lane = trailing_zeros(bits) + 1
                pivot_slot[1] = Int32(joff * span + first_lane)
            end
            return
        end
    end
    return
end

"""
    pluq_swap_rows_kernel!(A, r1, r2, ncols)

Swap two rows in a dense device matrix.
"""
function pluq_swap_rows_kernel!(A, r1::Int32, r2::Int32, ncols::Int32)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while col <= ncols
        tmp = A[r1, col]
        A[r1, col] = A[r2, col]
        A[r2, col] = tmp
        col += stride
    end
    return
end

"""
    pluq_swap_cols_kernel!(A, c1, c2, nrows)

Swap two columns in a dense device matrix.
"""
function pluq_swap_cols_kernel!(A, c1::Int32, c2::Int32, nrows::Int32)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while row <= nrows
        tmp = A[row, c1]
        A[row, c1] = A[row, c2]
        A[row, c2] = tmp
        row += stride
    end
    return
end

"""
    pluq_scale_column_kernel!(A, k, kend, invpivot, N)

Scale `A[k+1:kend, k]` by `invpivot` modulo `N`.
"""
function pluq_scale_column_kernel!(A, k::Int32, kend::Int32, invpivot_slot, N::Int32)
    invpivot = invpivot_slot[1]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    stride = blockDim().x * gridDim().x
    while i <= kend
        A[i, k] = _pluq_mod_mul_t(A[i, k], invpivot, N)
        i += stride
    end
    return
end

function pluq_compute_invpivot_kernel!(invpivot_slot, A, k::Int32, N::Int32)
    if blockIdx().x == 1 && threadIdx().x == 1
        invpivot_slot[1] = _pluq_mod_inv_t(A[k, k], N)
    end
    return
end

"""
    pluq_rank1_update_kernel!(A, k, kend, N)

Apply one elimination update step to the trailing active block.
"""
function pluq_rank1_update_kernel!(A, k::Int32, kend::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y + k
    if i <= kend && j <= kend
        aik = A[i, k]
        akj = A[k, j]
        A[i, j] = _pluq_mod_t(A[i, j] - _pluq_mod_mul_t(aik, akj, N), N)
    end
    return
end

"""
    pluq_basecase_gpu!(Adata, N, p, q, k0, kend, n)

Perform in-place PLUQ elimination on a block `[k0:kend, k0:kend]` of `Adata`
using CUDA kernels for pivot search, swaps, scaling, and rank-1 updates.
Returns the rank contributed by this block.
"""
function pluq_basecase_gpu!(Adata::CuArray{T,2}, N::Int, p::Vector{Int}, q::Vector{Int}, k0::Int, kend::Int, n::Int) where {T}
    rank = 0
    n32 = Int32(n)
    N32 = Int32(N)
    threads = 256
    maxspan = kend - k0 + 1
    pivot_slot = CUDA.fill(Int32(maxspan * maxspan + 1), 1)
    invpivot_slot = CUDA.zeros(T, 1)
    for k in k0:kend
        kk = Int32(k)
        kend32 = Int32(kend)
        span = kend - k + 1
        total = span * span
        CUDA.fill!(pivot_slot, Int32(total + 1))
        if span <= 32
            @cuda threads=32 blocks=1 pluq_find_pivot_warp_kernel!(Adata, pivot_slot, kk, kend32, N32)
        else
            blocks = max(1, cld(total, threads))
            @cuda threads=threads blocks=blocks pluq_find_pivot_kernel!(Adata, pivot_slot, kk, kend32, N32)
        end
        pivot_lin = Int(Array(@view pivot_slot[1:1])[1])
        if pivot_lin > total
            break
        end
        # Decode pivot index from linearized active-submatrix coordinates.
        joff = (pivot_lin - 1) ÷ span
        ioff = (pivot_lin - 1) % span
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_rows_kernel!(Adata, Int32(k), Int32(prow), n32)
            p[k], p[prow] = p[prow], p[k]
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_cols_kernel!(Adata, Int32(k), Int32(pcol), n32)
            q[k], q[pcol] = q[pcol], q[k]
        end
        # Normalize pivot row/column step in packed LU form.
        @cuda threads=1 blocks=1 pluq_compute_invpivot_kernel!(invpivot_slot, Adata, Int32(k), N32)
        if k < kend
            @cuda threads=threads blocks=max(1, cld(kend - k, threads)) pluq_scale_column_kernel!(Adata, Int32(k), Int32(kend), invpivot_slot, N32)
            tx = 16
            ty = 16
            bx = max(1, cld(kend - k, tx))
            by = max(1, cld(kend - k, ty))
            @cuda threads=(tx, ty) blocks=(bx, by) pluq_rank1_update_kernel!(Adata, Int32(k), Int32(kend), N32)
        end
        rank += 1
    end
    return rank
end
