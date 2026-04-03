@kernel function pluq_extract_l_kernel_ka!(L, LU, n::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2])
    if i <= n && j <= n
        if i == j
            L[i, j] = one(eltype(L))
        elseif i > j
            L[i, j] = LU[i, j]
        else
            L[i, j] = zero(eltype(L))
        end
    end
end

@kernel function pluq_extract_u_kernel_ka!(U, LU, n::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2])
    if i <= n && j <= n
        if i <= j
            U[i, j] = LU[i, j]
        else
            U[i, j] = zero(eltype(U))
        end
    end
end

@kernel function pluq_apply_paq_kernel_ka!(PAQ, A, p, q, n::Int32)
    I = @index(Global, NTuple)
    i = Int32(I[1])
    j = Int32(I[2])
    if i <= n && j <= n
        PAQ[i, j] = A[p[i], q[j]]
    end
end

function pluq_extract_L_ka(F::PLUQFactorization; options::PLUQOptionsKA=PLUQOptionsKA())
    n = rows(F.LU)
    L = GPUFiniteFieldMatrices.zeros(eltype(F.LU.data), n, n, F.LU.N)
    backend = _backend_from_pref(F.LU.data, options.backend_preference)
    Lw = _work_data_for_backend(L.data, options.backend_preference)
    LUw = _work_data_for_backend(F.LU.data, options.backend_preference)
    pluq_extract_l_kernel_ka!(backend, options.workgroupsize_2d)(Lw, LUw, Int32(n); ndrange=(n, n))
    _ka_sync(backend)
    _finalize_work_data!(L.data, Lw, options.backend_preference)
    return L
end

function pluq_extract_U_ka(F::PLUQFactorization; options::PLUQOptionsKA=PLUQOptionsKA())
    n = rows(F.LU)
    U = GPUFiniteFieldMatrices.zeros(eltype(F.LU.data), n, n, F.LU.N)
    backend = _backend_from_pref(F.LU.data, options.backend_preference)
    Uw = _work_data_for_backend(U.data, options.backend_preference)
    LUw = _work_data_for_backend(F.LU.data, options.backend_preference)
    pluq_extract_u_kernel_ka!(backend, options.workgroupsize_2d)(Uw, LUw, Int32(n); ndrange=(n, n))
    _ka_sync(backend)
    _finalize_work_data!(U.data, Uw, options.backend_preference)
    return U
end
