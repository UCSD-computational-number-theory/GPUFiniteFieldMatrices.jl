"""
    pluq_copy_block_kernel!(dest, src, row0, col0, nr, nc)

Copy a rectangular block from `src[row0:row0+nr-1, col0:col0+nc-1]`
into `dest[1:nr, 1:nc]`.
"""
function pluq_copy_block_kernel!(dest, src, row0::Int32, col0::Int32, nr::Int32, nc::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nr && j <= nc
        dest[i, j] = src[row0 + i - 1, col0 + j - 1]
    end
    return
end

"""
    pluq_write_block_kernel!(dest, src, row0, col0, nr, nc)

Write a rectangular block from `src[1:nr, 1:nc]`
into `dest[row0:row0+nr-1, col0:col0+nc-1]`.
"""
function pluq_write_block_kernel!(dest, src, row0::Int32, col0::Int32, nr::Int32, nc::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nr && j <= nc
        dest[row0 + i - 1, col0 + j - 1] = src[i, j]
    end
    return
end

"""
    pluq_schur_update_gpu!(Adata, N, k0, kend, n)

Apply the Schur complement update `A22 -= L21 * U12` in place on the trailing block.

This routine materializes the three coupled blocks (`L21`, `U12`, `A22`),
uses existing `CuModMatrix` `mul!`/`sub!`, and writes the updated `A22` back.
"""
function pluq_schur_update_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int) where {T}
    if kend >= n
        return
    end
    m = n - kend
    k = kend - k0 + 1
    L21 = GPUFiniteFieldMatrices.zeros(eltype(Adata), m, k, N)
    U12 = GPUFiniteFieldMatrices.zeros(eltype(Adata), k, m, N)
    A22 = GPUFiniteFieldMatrices.zeros(eltype(Adata), m, m, N)
    prod = GPUFiniteFieldMatrices.zeros(eltype(Adata), m, m, N)
    tx = 16
    ty = 16
    row0 = Int32(kend + 1)
    col0 = Int32(k0)
    kendp1 = Int32(kend + 1)
    m32 = Int32(m)
    k32 = Int32(k)
    @cuda threads=(tx, ty) blocks=(max(1, cld(k, tx)), max(1, cld(m, ty))) pluq_copy_block_kernel!(L21.data, Adata, row0, col0, m32, k32)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(k, ty))) pluq_copy_block_kernel!(U12.data, Adata, Int32(k0), kendp1, k32, m32)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_copy_block_kernel!(A22.data, Adata, row0, kendp1, m32, m32)
    mul!(prod, L21, U12)
    sub!(A22, A22, prod)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_write_block_kernel!(Adata, A22.data, row0, kendp1, m32, m32)
    return
end
