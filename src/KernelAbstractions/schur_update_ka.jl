@kernel function pluq_schur_update_kernel_ka!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1]) + kend
    j = Int32(I[2]) + kend
    if i <= n && j <= n
        acc = zero(eltype(A))
        t = k0
        while t <= kend
            acc = _pluq_mod_t_ka(acc + _pluq_mod_mul_t_ka(A[i, t], A[t, j], N), N)
            t += 1
        end
        A[i, j] = _pluq_mod_t_ka(A[i, j] - acc, N)
    end
end

function pluq_schur_update_ka!(Adata, backend, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    if kend >= n
        return
    end
    trailing = n - kend
    pluq_schur_update_kernel_ka!(backend, options.workgroupsize_2d)(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N); ndrange=(trailing, trailing))
    _ka_sync(backend)
    return
end
