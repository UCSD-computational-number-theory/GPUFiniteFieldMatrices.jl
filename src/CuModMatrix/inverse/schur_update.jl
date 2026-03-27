"""
    pluq_schur_update_tiled_kernel!(A, k0, kend, n, N)

In-place tiled Schur update on trailing block:
`A22 -= L21*U12` where
- `L21 = A[kend+1:n, k0:kend]`
- `U12 = A[k0:kend, kend+1:n]`
- `A22 = A[kend+1:n, kend+1:n]`
"""
function pluq_schur_update_tiled_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32, transpose_u::Bool)
    tx = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + kend
    sL = CuStaticSharedArray(eltype(A), (16, 16))
    sU = CuStaticSharedArray(eltype(A), (16, 16))
    acc = zero(eltype(A))
    kspan = kend - k0 + 1
    tilek = Int32(0)
    while tilek < kspan
        kkL = k0 + tilek + Int32(ty) - 1
        kkU = k0 + tilek + Int32(tx) - 1
        if i <= n && kkL <= kend
            sL[tx, ty] = A[i, kkL]
        else
            sL[tx, ty] = zero(eltype(A))
        end
        if kkU <= kend && j <= n
            if transpose_u
                sU[ty, tx] = A[kkU, j]
            else
                sU[tx, ty] = A[kkU, j]
            end
        else
            if transpose_u
                sU[ty, tx] = zero(eltype(A))
            else
                sU[tx, ty] = zero(eltype(A))
            end
        end
        sync_threads()
        t = 1
        while t <= 16
            if transpose_u
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[ty, t], N), N)
            else
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[t, ty], N), N)
            end
            t += 1
        end
        sync_threads()
        tilek += Int32(16)
    end
    if i <= n && j <= n
        A[i, j] = _pluq_mod_t(A[i, j] - acc, N)
    end
    return
end

function pluq_schur_update_tiled_kernel_8!(A, k0::Int32, kend::Int32, n::Int32, N::Int32, transpose_u::Bool)
    tx = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + kend
    sL = CuStaticSharedArray(eltype(A), (8, 8))
    sU = CuStaticSharedArray(eltype(A), (8, 8))
    acc = zero(eltype(A))
    kspan = kend - k0 + 1
    tilek = Int32(0)
    while tilek < kspan
        kkL = k0 + tilek + Int32(ty) - 1
        kkU = k0 + tilek + Int32(tx) - 1
        if i <= n && kkL <= kend
            sL[tx, ty] = A[i, kkL]
        else
            sL[tx, ty] = zero(eltype(A))
        end
        if kkU <= kend && j <= n
            if transpose_u
                sU[ty, tx] = A[kkU, j]
            else
                sU[tx, ty] = A[kkU, j]
            end
        else
            if transpose_u
                sU[ty, tx] = zero(eltype(A))
            else
                sU[tx, ty] = zero(eltype(A))
            end
        end
        sync_threads()
        t = 1
        while t <= 8
            if transpose_u
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[ty, t], N), N)
            else
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[t, ty], N), N)
            end
            t += 1
        end
        sync_threads()
        tilek += Int32(8)
    end
    if i <= n && j <= n
        A[i, j] = _pluq_mod_t(A[i, j] - acc, N)
    end
    return
end

function pluq_schur_update_tiled_kernel_32!(A, k0::Int32, kend::Int32, n::Int32, N::Int32, transpose_u::Bool)
    tx = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + kend
    sL = CuStaticSharedArray(eltype(A), (32, 32))
    sU = CuStaticSharedArray(eltype(A), (32, 32))
    acc = zero(eltype(A))
    kspan = kend - k0 + 1
    tilek = Int32(0)
    while tilek < kspan
        kkL = k0 + tilek + Int32(ty) - 1
        kkU = k0 + tilek + Int32(tx) - 1
        if i <= n && kkL <= kend
            sL[tx, ty] = A[i, kkL]
        else
            sL[tx, ty] = zero(eltype(A))
        end
        if kkU <= kend && j <= n
            if transpose_u
                sU[ty, tx] = A[kkU, j]
            else
                sU[tx, ty] = A[kkU, j]
            end
        else
            if transpose_u
                sU[ty, tx] = zero(eltype(A))
            else
                sU[tx, ty] = zero(eltype(A))
            end
        end
        sync_threads()
        t = 1
        while t <= 32
            if transpose_u
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[ty, t], N), N)
            else
                acc = _pluq_mod_t(acc + _pluq_mod_mul_t(sL[tx, t], sU[t, ty], N), N)
            end
            t += 1
        end
        sync_threads()
        tilek += Int32(32)
    end
    if i <= n && j <= n
        A[i, j] = _pluq_mod_t(A[i, j] - acc, N)
    end
    return
end

"""
    pluq_schur_update_gpu!(Adata, N, k0, kend, n)

Apply the Schur complement update `A22 -= L21 * U12` in place on the trailing block.
"""
function pluq_schur_update_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptions=PLUQOptions()) where {T}
    if kend >= n
        return
    end
    trailing = n - kend
    tile = options.schur_tile
    if trailing <= 64
        tile = 8
    elseif trailing >= 1024 && T == Float32
        tile = max(tile, 32)
    end
    tx = tile
    ty = tile
    blocks = (max(1, cld(trailing, tx)), max(1, cld(trailing, ty)))
    if tile == 8
        @cuda threads=(tx, ty) blocks=blocks pluq_schur_update_tiled_kernel_8!(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N), options.schur_transpose_u)
    elseif tile == 32
        @cuda threads=(tx, ty) blocks=blocks pluq_schur_update_tiled_kernel_32!(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N), options.schur_transpose_u)
    else
        @cuda threads=(16, 16) blocks=blocks pluq_schur_update_tiled_kernel!(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N), options.schur_transpose_u)
    end
    return
end
