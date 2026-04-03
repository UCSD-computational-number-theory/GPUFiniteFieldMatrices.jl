@kernel function pluq_trsm_left_panel_kernel_ka!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    joff = @index(Global, Linear)
    j = joff + kend
    if j <= n
        i = k0
        while i <= kend
            acc = _pluq_mod_t_ka(A[i, j], N)
            t = k0
            while t < i
                acc = _pluq_mod_t_ka(acc - _pluq_mod_mul_t_ka(A[i, t], A[t, j], N), N)
                t += 1
            end
            A[i, j] = acc
            i += 1
        end
    end
end

@kernel function pluq_trsm_right_panel_kernel_ka!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    ioff = @index(Global, Linear)
    i = ioff + kend
    if i <= n
        j = kend
        while j >= k0
            acc = _pluq_mod_t_ka(A[i, j], N)
            t = j + 1
            while t <= kend
                acc = _pluq_mod_t_ka(acc - _pluq_mod_mul_t_ka(A[i, t], A[t, j], N), N)
                t += 1
            end
            invdiag = _pluq_mod_inv_t_ka(A[j, j], N)
            A[i, j] = _pluq_mod_mul_t_ka(acc, invdiag, N)
            j -= 1
        end
    end
end

function pluq_trsm_left_lower_unit_ka!(Adata, backend, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    if kend >= n
        return
    end
    pluq_trsm_left_panel_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N); ndrange=(n - kend))
    _ka_sync(backend)
    return
end

function pluq_trsm_right_upper_ka!(Adata, backend, N::Int, k0::Int, kend::Int, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    if kend >= n
        return
    end
    pluq_trsm_right_panel_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k0), Int32(kend), Int32(n), Int32(N); ndrange=(n - kend))
    _ka_sync(backend)
    return
end
