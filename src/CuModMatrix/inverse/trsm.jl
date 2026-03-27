"""
    pluq_trsm_left_panel_kernel!(A, k0, kend, n, N)

Kernel for left lower-unit triangular solve on trailing columns:
`L11 * U12 = A12`, writing `U12` in place.

Each thread processes one trailing column and solves all panel rows in order.
"""
function pluq_trsm_left_panel_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while j <= n
        i = k0
        while i <= kend
            acc = _pluq_mod_t(A[i, j], N)
            t = k0
            while t < i
                acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
                t += 1
            end
            A[i, j] = acc
            i += 1
        end
        j += stride
    end
    return
end

function pluq_trsm_left_panel_warp_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    j = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while j <= n
        i = k0
        while i <= kend
            psum = Int64(0)
            t = k0 + lane - 1
            while t < i
                psum += Int64(_pluq_mod_mul_t(A[i, t], A[t, j], N))
                t += 32
            end
            v = Int32(rem(psum, Int64(N)))
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 16, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 8, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 4, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 2, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 1, 32)
            v = Int32(rem(Int64(v), Int64(N)))
            if lane == 1
                acc = _pluq_mod_t(A[i, j] - eltype(A)(v), N)
                A[i, j] = acc
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            i += 1
        end
        j += stride
    end
    return
end

"""
    pluq_trsm_right_panel_kernel!(A, k0, kend, n, N)

Kernel for right upper-triangular solve on trailing rows:
`L21 * U11 = A21`, writing `L21` in place.

Each thread processes one trailing row and solves panel columns backward.
"""
function pluq_trsm_right_panel_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while i <= n
        j = kend
        while j >= k0
            acc = _pluq_mod_t(A[i, j], N)
            t = j + 1
            while t <= kend
                acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
                t += 1
            end
            invdiag = _pluq_mod_inv_t(A[j, j], N)
            A[i, j] = _pluq_mod_mul_t(acc, invdiag, N)
            j -= 1
        end
        i += stride
    end
    return
end

function pluq_trsm_right_panel_warp_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    i = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while i <= n
        j = kend
        while j >= k0
            psum = Int64(0)
            t = j + lane
            while t <= kend
                psum += Int64(_pluq_mod_mul_t(A[i, t], A[t, j], N))
                t += 32
            end
            v = Int32(rem(psum, Int64(N)))
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 16, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 8, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 4, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 2, 32)
            v += CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 1, 32)
            v = Int32(rem(Int64(v), Int64(N)))
            if lane == 1
                acc = _pluq_mod_t(A[i, j] - eltype(A)(v), N)
                invdiag = _pluq_mod_inv_t(A[j, j], N)
                A[i, j] = _pluq_mod_mul_t(acc, invdiag, N)
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            j -= 1
        end
        i += stride
    end
    return
end

"""
    pluq_trsm_left_lower_unit_gpu!(Adata, N, k0, kend, n)

Compute the left solve on trailing block columns:
`L11 * U12 = A12`, writing `U12` in place in `Adata`.

`L11` is interpreted as unit-lower from the packed LU panel.
Rows `k0:kend` are solved independently over columns `kend+1:n`.

Example:
```julia
pluq_trsm_left_lower_unit_gpu!(A.data, A.N, 1, 16, rows(A))
```
"""
function pluq_trsm_left_lower_unit_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptions=PLUQOptions()) where {T}
    if kend >= n
        return
    end
    panel_depth = kend - k0 + 1
    use_warp = options.trsm_mode == :warp || (options.trsm_mode == :auto && panel_depth <= options.trsm_warp_threshold)
    threads = use_warp ? 128 : 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    if use_warp
        nwarps = max(1, threads >>> 5)
        @cuda threads=threads blocks=max(1, cld(n - kend, nwarps)) pluq_trsm_left_panel_warp_kernel!(Adata, k032, kend32, n32, N32)
    else
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_left_panel_kernel!(Adata, k032, kend32, n32, N32)
    end
    return
end

"""
    pluq_trsm_right_upper_gpu!(Adata, N, k0, kend, n)

Compute the right solve on trailing block rows:
`L21 * U11 = A21`, writing `L21` in place in `Adata`.

`U11` is interpreted as upper-triangular from packed LU panel.
Columns `kend:-1:k0` are solved backward over rows `kend+1:n`.

Example:
```julia
pluq_trsm_right_upper_gpu!(A.data, A.N, 1, 16, rows(A))
```
"""
function pluq_trsm_right_upper_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptions=PLUQOptions()) where {T}
    if kend >= n
        return
    end
    panel_depth = kend - k0 + 1
    use_warp = options.trsm_mode == :warp || (options.trsm_mode == :auto && panel_depth <= options.trsm_warp_threshold)
    threads = use_warp ? 128 : 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    if use_warp
        nwarps = max(1, threads >>> 5)
        @cuda threads=threads blocks=max(1, cld(n - kend, nwarps)) pluq_trsm_right_panel_warp_kernel!(Adata, k032, kend32, n32, N32)
    else
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_right_panel_kernel!(Adata, k032, kend32, n32, N32)
    end
    return
end
