@kernel function pluq_pivot_rect_flags_kernel_ka!(flags, A, k::Int32, m::Int32, n::Int32, N::Int32)
    idx = @index(Global, Linear)
    span_r = m - k + 1
    span_c = n - k + 1
    total = span_r * span_c
    if idx <= total
        joff = (idx - 1) ÷ span_r
        ioff = (idx - 1) % span_r
        i = k + ioff
        j = k + joff
        flags[idx] = _pluq_mod_t_ka(A[i, j], N) != zero(eltype(A)) ? Int32(1) : Int32(0)
    end
end

@kernel function pluq_rank1_update_rect_kernel_ka!(A, k::Int32, m::Int32, n::Int32, N::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1]) + k
    j = Int32(I[2]) + k
    if i <= m && j <= n
        A[i, j] = _pluq_mod_t_ka(A[i, j] - _pluq_mod_mul_t_ka(A[i, k], A[k, j], N), N)
    end
end

function _pluq_find_rect_pivot_lin_ka(A, backend, k::Int, m::Int, n::Int, N::Int, options::PLUQOptionsKA)
    span_r = m - k + 1
    span_c = n - k + 1
    total = span_r * span_c
    flags = KA.zeros(backend, Int32, total)
    pluq_pivot_rect_flags_kernel_ka!(backend, options.workgroupsize_1d)(flags, A, Int32(k), Int32(m), Int32(n), Int32(N); ndrange=total)
    _ka_sync(backend)
    hflags = flags isa Array ? flags : Array(flags)
    for i in 1:total
        if hflags[i] == 1
            return i
        end
    end
    return total + 1
end

function pluq_rectangular_rank_ka!(Adata, backend, N::Int, m::Int, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    rmax = min(m, n)
    p = collect(1:m)
    q = collect(1:n)
    lp = options.core.lazy_q ? collect(1:m) : Int[]
    lq = options.core.lazy_q ? collect(1:n) : Int[]
    rank = 0
    for k in 1:rmax
        span_r = m - k + 1
        span_c = n - k + 1
        total = span_r * span_c
        pivlin = _pluq_find_rect_pivot_lin_ka(Adata, backend, k, m, n, N, options)
        if pivlin > total
            break
        end
        joff = (pivlin - 1) ÷ span_r
        ioff = (pivlin - 1) % span_r
        prow = k + ioff
        pcol = k + joff
        if prow != k
            pluq_swap_rows_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(prow), Int32(n); ndrange=n)
            _ka_sync(backend)
            if options.core.lazy_q
                lp[k], lp[prow] = lp[prow], lp[k]
            else
                p[k], p[prow] = p[prow], p[k]
            end
        end
        if pcol != k
            pluq_swap_cols_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(pcol), Int32(m); ndrange=m)
            _ka_sync(backend)
            if options.core.lazy_q
                lq[k], lq[pcol] = lq[pcol], lq[k]
            else
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        if k < m
            invpivot = _pluq_mod_inv_t_ka(_get_scalar_ka(Adata, k, k), Int32(N))
            pluq_scale_column_from_diag_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(m), invpivot, Int32(N); ndrange=(m - k))
            _ka_sync(backend)
        end
        if k < n
            pluq_rank1_update_rect_kernel_ka!(backend, options.workgroupsize_2d)(Adata, Int32(k), Int32(m), Int32(n), Int32(N); ndrange=(m - k, n - k))
            _ka_sync(backend)
        end
        rank += 1
    end
    if options.core.lazy_q
        p = lp
        q = lq
    end
    return p, q, rank
end
