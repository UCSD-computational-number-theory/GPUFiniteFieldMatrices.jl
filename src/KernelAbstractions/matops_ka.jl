@kernel function _add_mod_kernel_ka!(C, A, B, nrows::Int32, ncols::Int32, N::Int32)
    idx = @index(Global, Linear)
    total = nrows * ncols
    if idx <= total
        j = (idx - 1) ÷ nrows + 1
        i = (idx - 1) % nrows + 1
        C[i, j] = _pluq_mod_t_ka(A[i, j] + B[i, j], N)
    end
end

@kernel function _sub_mod_kernel_ka!(C, A, B, nrows::Int32, ncols::Int32, N::Int32)
    idx = @index(Global, Linear)
    total = nrows * ncols
    if idx <= total
        j = (idx - 1) ÷ nrows + 1
        i = (idx - 1) % nrows + 1
        C[i, j] = _pluq_mod_t_ka(A[i, j] - B[i, j], N)
    end
end

function add_ka!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    if A.N != B.N || A.N != C.N
        throw(CuModArrayModulusMismatchException("all matrices must have same modulus"))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException("matrix sizes must match"))
    end
    nrows = Int32(rows(A))
    ncols = Int32(cols(A))
    backend = _backend_from_pref(C.data, options.backend_preference)
    Cw = _work_data_for_backend(C.data, options.backend_preference)
    Aw = _work_data_for_backend(A.data, options.backend_preference)
    Bw = _work_data_for_backend(B.data, options.backend_preference)
    total = rows(A) * cols(A)
    _add_mod_kernel_ka!(backend, options.workgroupsize_1d)(Cw, Aw, Bw, nrows, ncols, Int32(A.N); ndrange=total)
    _ka_sync(backend)
    _finalize_work_data!(C.data, Cw, options.backend_preference)
    return C
end

function sub_ka!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA())
    if A.N != B.N || A.N != C.N
        throw(CuModArrayModulusMismatchException("all matrices must have same modulus"))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException("matrix sizes must match"))
    end
    nrows = Int32(rows(A))
    ncols = Int32(cols(A))
    backend = _backend_from_pref(C.data, options.backend_preference)
    Cw = _work_data_for_backend(C.data, options.backend_preference)
    Aw = _work_data_for_backend(A.data, options.backend_preference)
    Bw = _work_data_for_backend(B.data, options.backend_preference)
    total = rows(A) * cols(A)
    _sub_mod_kernel_ka!(backend, options.workgroupsize_1d)(Cw, Aw, Bw, nrows, ncols, Int32(A.N); ndrange=total)
    _ka_sync(backend)
    _finalize_work_data!(C.data, Cw, options.backend_preference)
    return C
end

function _gemm_gemmkernels_try!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix)
    if !(@isdefined _HAS_GEMMKERNELS) || !_HAS_GEMMKERNELS
        return false
    end
    try
        GemmKernels.mul!(C.data, A.data, B.data)
        return true
    catch
        return false
    end
end

function mul_ka!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix; options::PLUQOptionsKA=PLUQOptionsKA(), method::Symbol=:gemmkernels)
    if A.N != B.N || A.N != C.N
        throw(CuModArrayModulusMismatchException("all matrices must have same modulus"))
    end
    if cols(A) != rows(B) || rows(C) != rows(A) || cols(C) != cols(B)
        throw(CuModArraySizeMismatchException("invalid matrix dimensions for multiplication"))
    end
    used_gemmkernels = false
    if method == :gemmkernels && options.backend_preference != :cpu
        used_gemmkernels = _gemm_gemmkernels_try!(C, A, B)
    end
    if !used_gemmkernels
        LinearAlgebra.mul!(C, A, B)
    end
    C.data .= mod.(C.data, C.N)
    return C
end
