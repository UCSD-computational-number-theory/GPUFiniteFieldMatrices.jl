"""
    pluq_blocked_recursive_gpu!(Adata, N, opts, p, q, start, stop, n)

Recursively apply blocked PLUQ on active index interval `[start, stop]`.
Uses basecase elimination, triangular solves, and Schur updates on GPU.
"""
function pluq_blocked_recursive_gpu!(Adata::CuArray{T,2}, N::Int, opts::PLUQOptions, p::Vector{Int}, q::Vector{Int}, start::Int, stop::Int, n::Int) where {T}
    if start > stop
        return 0
    end
    seglen = stop - start + 1
    if seglen <= opts.basecase
        return pluq_basecase_gpu!(Adata, N, p, q, start, stop, n)
    end
    b = min(opts.blocksize, seglen)
    kend = min(start + b - 1, stop)
    rank = pluq_basecase_gpu!(Adata, N, p, q, start, kend, n)
    pluq_trsm_left_lower_unit_gpu!(Adata, N, start, kend, stop)
    pluq_trsm_right_upper_gpu!(Adata, N, start, kend, stop)
    pluq_schur_update_gpu!(Adata, N, start, kend, stop)
    rank += pluq_blocked_recursive_gpu!(Adata, N, opts, p, q, kend + 1, stop, n)
    return rank
end

"""
    pluq_blocked_gpu!(Adata, N, opts, n)

Run recursive blocked PLUQ on `Adata` and return `(p, q, rank)`.
"""
function pluq_blocked_gpu!(Adata::CuArray{T,2}, N::Int, opts::PLUQOptions, n::Int) where {T}
    p = pluq_init_perm(n)
    q = pluq_init_perm(n)
    rank = pluq_blocked_recursive_gpu!(Adata, N, opts, p, q, 1, n, n)
    return p, q, rank
end
