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
function pluq_trsm_left_lower_unit_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int) where {T}
    if kend >= n
        return
    end
    threads = 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_left_panel_kernel!(Adata, k032, kend32, n32, N32)
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
function pluq_trsm_right_upper_gpu!(Adata::CuArray{T,2}, N::Int, k0::Int, kend::Int, n::Int) where {T}
    if kend >= n
        return
    end
    threads = 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_right_panel_kernel!(Adata, k032, kend32, n32, N32)
    return
end
