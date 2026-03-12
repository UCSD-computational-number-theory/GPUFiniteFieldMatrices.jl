"""
    pluq_trsm_left_kernel!(A, i, jstart, n, k0, N)

Kernel for one row `i` of left lower-unit triangular solve on trailing columns.
"""
function pluq_trsm_left_kernel!(A, i::Int32, jstart::Int32, n::Int32, k0::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + jstart - 1
    stride = blockDim().x * gridDim().x
    while j <= n
        acc = _pluq_mod_t(A[i, j], N)
        t = k0
        while t < i
            acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
            t += 1
        end
        A[i, j] = acc
        j += stride
    end
    return
end

"""
    pluq_trsm_right_kernel!(A, j, istart, n, kend, N, invdiag)

Kernel for one column `j` of right upper-triangular solve on trailing rows.
"""
function pluq_trsm_right_kernel!(A, j::Int32, istart::Int32, n::Int32, kend::Int32, N::Int32, invdiag)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + istart - 1
    stride = blockDim().x * gridDim().x
    while i <= n
        acc = _pluq_mod_t(A[i, j], N)
        t = j + 1
        while t <= kend
            acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
            t += 1
        end
        A[i, j] = _pluq_mod_mul_t(acc, invdiag, N)
        i += stride
    end
    return
end

"""
    pluq_trsm_left_lower_unit_gpu!(Adata, N, k0, kend, n)

Compute the left solve on trailing block columns:
`L11 * U12 = A12`, writing `U12` in place in `Adata`.
"""
function pluq_trsm_left_lower_unit_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int) where {T}
    if kend >= n
        return
    end
    threads = 256
    N32 = Int32(N)
    for i in k0:kend
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_left_kernel!(Adata, Int32(i), Int32(kend + 1), Int32(n), Int32(k0), N32)
    end
    return
end

"""
    pluq_trsm_right_upper_gpu!(Adata, N, k0, kend, n)

Compute the right solve on trailing block rows:
`L21 * U11 = A21`, writing `L21` in place in `Adata`.
"""
function pluq_trsm_right_upper_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int) where {T}
    if kend >= n
        return
    end
    threads = 256
    N32 = Int32(N)
    for j in kend:-1:k0
        diag = Int(Array(@view Adata[j:j, j:j])[1])
        invdiag = convert(T, pluq_mod_inv(diag, N))
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_right_kernel!(Adata, Int32(j), Int32(kend + 1), Int32(n), Int32(kend), N32, invdiag)
    end
    return
end
