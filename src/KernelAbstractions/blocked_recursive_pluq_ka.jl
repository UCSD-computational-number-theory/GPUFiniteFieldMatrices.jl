function pluq_blocked_recursive_ka!(Adata, backend, N::Int, opts::PLUQOptionsKA, p::Vector{Int}, q::Vector{Int}, start::Int, stop::Int, n::Int)
    if start > stop
        return 0
    end
    seglen = stop - start + 1
    if seglen <= opts.core.basecase
        return pluq_basecase_ka!(Adata, backend, N, p, q, start, stop, n; options=opts)
    end
    b = min(opts.core.blocksize, seglen)
    kend = min(start + b - 1, stop)
    rank = pluq_basecase_ka!(Adata, backend, N, p, q, start, kend, n; options=opts)
    pluq_trsm_left_lower_unit_ka!(Adata, backend, N, start, kend, stop; options=opts)
    pluq_trsm_right_upper_ka!(Adata, backend, N, start, kend, stop; options=opts)
    pluq_schur_update_ka!(Adata, backend, N, start, kend, stop; options=opts)
    rank += pluq_blocked_recursive_ka!(Adata, backend, N, opts, p, q, kend + 1, stop, n)
    return rank
end

function pluq_blocked_ka!(Adata, backend, N::Int, opts::PLUQOptionsKA, n::Int)
    p = pluq_init_perm_ka(n)
    q = pluq_init_perm_ka(n)
    rank = pluq_blocked_recursive_ka!(Adata, backend, N, opts, p, q, 1, n, n)
    return p, q, rank
end
