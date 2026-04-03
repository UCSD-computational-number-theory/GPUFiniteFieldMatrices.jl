@kernel function pluq_init_aug_kernel_ka!(aug, Adata, n::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2])
    if i <= n && j <= 2n
        if j <= n
            aug[i, j] = Adata[i, j]
        else
            aug[i, j] = (j - n == i) ? one(eltype(aug)) : zero(eltype(aug))
        end
    end
end

@kernel function pluq_aug_pivot_flags_kernel_ka!(flags, aug, k::Int32, n::Int32, N::Int32)
    idx = @index(Global, Linear)
    span = n - k + 1
    if idx <= span
        i = k + idx - 1
        flags[idx] = _pluq_mod_t_ka(aug[i, k], N) != zero(eltype(aug)) ? Int32(1) : Int32(0)
    end
end

@kernel function pluq_aug_scale_row_from_diag_kernel_ka!(aug, row::Int32, jstart::Int32, n2::Int32, invpivot, N::Int32)
    joff = @index(Global, Linear)
    j = joff + jstart - 1
    if j <= n2
        aug[row, j] = _pluq_mod_mul_t_ka(aug[row, j], invpivot, N)
    end
end

@kernel function pluq_aug_elim_kernel_ka!(aug, k::Int32, n::Int32, n2::Int32, N::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2]) + k - 1
    if i <= n && j <= n2 && i != k
        f = _pluq_mod_t_ka(aug[i, k], N)
        if f != zero(eltype(aug))
            aug[i, j] = _pluq_mod_t_ka(aug[i, j] - _pluq_mod_mul_t_ka(f, aug[k, j], N), N)
        end
    end
end

@kernel function pluq_copy_block_kernel_ka!(dest, src, n::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2])
    if i <= n && j <= n
        dest[i, j] = src[i, j]
    end
end

function _pluq_autotune_options_ka(options::PLUQOptionsKA, A::CuModMatrix)
    tuned = _resolve_ka_core_options(options, A)
    return PLUQOptionsKA(
        core=tuned,
        backend_preference=options.backend_preference,
        workgroupsize_1d=options.workgroupsize_1d,
        workgroupsize_2d=options.workgroupsize_2d,
    )
end

function pluq_new_ka!(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    opts = _pluq_autotune_options_ka(options, A)
    backend_pref = :cpu
    backend = _backend_from_pref(A.data, backend_pref)
    Awork = _work_data_for_backend(A.data, backend_pref)
    m = rows(A)
    n = cols(A)
    p, q, rank = if m == n
        pluq_blocked_ka!(Awork, backend, A.N, opts, n)
    else
        pluq_rectangular_rank_ka!(Awork, backend, A.N, m, n; options=opts)
    end
    _finalize_work_data!(A.data, Awork, backend_pref)
    return PLUQFactorization(A, p, q, rank)
end

function pluq_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    Adata = copy(A.data)
    Awork = CuModMatrix(Adata, A.N; new_size=size(A))
    return pluq_new_ka!(Awork, options=options)
end

function is_invertible_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    m = rows(A)
    n = cols(A)
    F = pluq_new_ka(A, options=options)
    return m == n && F.rank == m
end

function _find_aug_pivot_row_ka(aug, backend, k::Int, n::Int, N::Int, options::PLUQOptionsKA)
    span = n - k + 1
    flags = KA.zeros(backend, Int32, span)
    pluq_aug_pivot_flags_kernel_ka!(backend, options.workgroupsize_1d)(flags, aug, Int32(k), Int32(n), Int32(N); ndrange=span)
    _ka_sync(backend)
    hflags = flags isa Array ? flags : Array(flags)
    for i in 1:span
        if hflags[i] == 1
            return k + i - 1
        end
    end
    return n + 1
end

function inverse_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    opts = _pluq_autotune_options_ka(options, A)
    return inverse_new(A, options=opts.core)
end

function inverse_pluq_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    fallback_core = PLUQOptions(
        blocksize=options.core.blocksize,
        basecase=options.core.basecase,
        pivot_policy=options.core.pivot_policy,
        lazy_q=options.core.lazy_q,
        nftb=options.core.nftb,
        pivot_warp_kernel=options.core.pivot_warp_kernel,
        trsm_mode=options.core.trsm_mode,
        trsm_warp_threshold=options.core.trsm_warp_threshold,
        schur_tile=options.core.schur_tile,
        schur_transpose_u=options.core.schur_transpose_u,
        mod_backend=options.core.mod_backend,
        inverse_strategy=:augmented,
        autotune=options.core.autotune,
        batch_streams=options.core.batch_streams,
        check_prime=options.core.check_prime,
    )
    fallback_opts = PLUQOptionsKA(
        core=fallback_core,
        backend_preference=options.backend_preference,
        workgroupsize_1d=options.workgroupsize_1d,
        workgroupsize_2d=options.workgroupsize_2d,
    )
    return inverse_new_ka(A, options=fallback_opts)
end

function right_inverse_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    return right_inverse_new(A, options=options.core)
end

function left_inverse_new_ka(A::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    return left_inverse_new(A, options=options.core)
end

function pluq_new_batch_ka(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA())
    isempty(mats) && return Any[]
    out = Vector{PLUQFactorization}(undef, length(mats))
    for i in eachindex(mats)
        out[i] = pluq_new_ka(mats[i], options=options)
    end
    return out
end

function inverse_new_batch_ka(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA())
    isempty(mats) && return CuModMatrix[]
    out = Vector{CuModMatrix}(undef, length(mats))
    for i in eachindex(mats)
        A = mats[i]
        if rows(A) == cols(A)
            out[i] = inverse_new_ka(A, options=options)
        elseif rows(A) < cols(A)
            out[i] = right_inverse_new_ka(A, options=options)
        else
            out[i] = left_inverse_new_ka(A, options=options)
        end
    end
    return out
end
