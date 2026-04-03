@kernel function pluq_nonzero_mod_kernel_ka!(flag, D, n::Int32, N::Int32)
    idx = @index(Global, Linear)
    total = n * n
    if idx <= total
        j = (idx - 1) ÷ n + 1
        i = (idx - 1) % n + 1
        if _pluq_mod_t_ka(D[i, j], N) != zero(eltype(D))
            flag[1] = Int32(1)
        end
    end
end

function pluq_check_identity_ka(F::PLUQFactorization, Aorig::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    n = rows(Aorig)
    N = Aorig.N
    pdev = Int32.(F.p)
    qdev = Int32.(F.q)
    PAQ = GPUFiniteFieldMatrices.zeros(eltype(Aorig.data), n, n, N)
    backend = _backend_from_pref(Aorig.data, options.backend_preference)
    PAQw = _work_data_for_backend(PAQ.data, options.backend_preference)
    Aw = _work_data_for_backend(Aorig.data, options.backend_preference)
    pwork = options.backend_preference == :cpu ? pdev : CuArray(pdev)
    qwork = options.backend_preference == :cpu ? qdev : CuArray(qdev)
    pluq_apply_paq_kernel_ka!(backend, options.workgroupsize_2d)(PAQw, Aw, pwork, qwork, Int32(n); ndrange=(n, n))
    _ka_sync(backend)
    _finalize_work_data!(PAQ.data, PAQw, options.backend_preference)

    L = pluq_extract_L_ka(F, options=options)
    U = pluq_extract_U_ka(F, options=options)
    LU = GPUFiniteFieldMatrices.zeros(eltype(Aorig.data), n, n, N)
    mul_ka!(LU, L, U, options=options)
    D = GPUFiniteFieldMatrices.zeros(eltype(Aorig.data), n, n, N)
    sub_ka!(D, PAQ, LU, options=options)
    flag = options.backend_preference == :cpu ? zeros(Int32, 1) : CUDA.zeros(Int32, 1)
    pluq_nonzero_mod_kernel_ka!(backend, options.workgroupsize_1d)(flag, _work_data_for_backend(D.data, options.backend_preference), Int32(n), Int32(N); ndrange=n * n)
    _ka_sync(backend)
    v = flag isa Array ? flag[1] : Array(flag)[1]
    return v == 0
end
