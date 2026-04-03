@kernel function pluq_pivot_flags_kernel_ka!(flags, A, k::Int32, kend::Int32, N::Int32)
    idx = @index(Global, Linear)
    span = kend - k + 1
    total = span * span
    if idx <= total
        joff = (idx - 1) ÷ span
        ioff = (idx - 1) % span
        i = k + ioff
        j = k + joff
        flags[idx] = _pluq_mod_t_ka(A[i, j], N) != zero(eltype(A)) ? Int32(1) : Int32(0)
    end
end

@kernel function pluq_swap_rows_kernel_ka!(A, r1::Int32, r2::Int32, ncols::Int32)
    col = @index(Global, Linear)
    if col <= ncols
        tmp = A[r1, col]
        A[r1, col] = A[r2, col]
        A[r2, col] = tmp
    end
end

@kernel function pluq_swap_cols_kernel_ka!(A, c1::Int32, c2::Int32, nrows::Int32)
    row = @index(Global, Linear)
    if row <= nrows
        tmp = A[row, c1]
        A[row, c1] = A[row, c2]
        A[row, c2] = tmp
    end
end

@kernel function pluq_scale_column_from_diag_kernel_ka!(A, k::Int32, kend::Int32, invpivot, N::Int32)
    ioff = @index(Global, Linear)
    i = ioff + k
    if i <= kend
        A[i, k] = _pluq_mod_mul_t_ka(A[i, k], invpivot, N)
    end
end

@kernel function pluq_rank1_update_kernel_ka!(A, k::Int32, kend::Int32, N::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1]) + k
    j = Int32(I[2]) + k
    if i <= kend && j <= kend
        aik = A[i, k]
        akj = A[k, j]
        A[i, j] = _pluq_mod_t_ka(A[i, j] - _pluq_mod_mul_t_ka(aik, akj, N), N)
    end
end

function _pluq_find_pivot_lin_ka(A, backend, k::Int, kend::Int, N::Int, options::PLUQOptionsKA)
    span = kend - k + 1
    total = span * span
    flags = KA.zeros(backend, Int32, total)
    pluq_pivot_flags_kernel_ka!(backend, options.workgroupsize_1d)(flags, A, Int32(k), Int32(kend), Int32(N); ndrange=total)
    _ka_sync(backend)
    hflags = flags isa Array ? flags : Array(flags)
    for i in 1:total
        if hflags[i] == 1
            return i
        end
    end
    return total + 1
end

function pluq_basecase_ka!(Adata, backend, N::Int, p::Vector{Int}, q::Vector{Int}, k0::Int, kend::Int, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    rank = 0
    locp = options.core.lazy_q ? collect(1:(kend - k0 + 1)) : Int[]
    locq = options.core.lazy_q ? collect(1:(kend - k0 + 1)) : Int[]
    for k in k0:kend
        span = kend - k + 1
        total = span * span
        pivot_lin = _pluq_find_pivot_lin_ka(Adata, backend, k, kend, N, options)
        if pivot_lin > total
            break
        end
        joff = (pivot_lin - 1) ÷ span
        ioff = (pivot_lin - 1) % span
        prow = k + ioff
        pcol = k + joff
        if prow != k
            pluq_swap_rows_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(prow), Int32(n); ndrange=n)
            _ka_sync(backend)
            if options.core.lazy_q
                lock = k - k0 + 1
                locprow = prow - k0 + 1
                locp[lock], locp[locprow] = locp[locprow], locp[lock]
            else
                p[k], p[prow] = p[prow], p[k]
            end
        end
        if pcol != k
            pluq_swap_cols_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(pcol), Int32(n); ndrange=n)
            _ka_sync(backend)
            if options.core.lazy_q
                lock = k - k0 + 1
                locpcol = pcol - k0 + 1
                locq[lock], locq[locpcol] = locq[locpcol], locq[lock]
            else
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        if k < kend
            invpivot = _pluq_mod_inv_t_ka(_get_scalar_ka(Adata, k, k), Int32(N))
            pluq_scale_column_from_diag_kernel_ka!(backend, options.workgroupsize_1d)(Adata, Int32(k), Int32(kend), invpivot, Int32(N); ndrange=(kend - k))
            pluq_rank1_update_kernel_ka!(backend, options.workgroupsize_2d)(Adata, Int32(k), Int32(kend), Int32(N); ndrange=(kend - k, kend - k))
            _ka_sync(backend)
        end
        rank += 1
    end
    if options.core.lazy_q
        pluq_compose_segment_ka!(p, k0, locp)
        pluq_compose_segment_ka!(q, k0, locq)
    end
    return rank
end
