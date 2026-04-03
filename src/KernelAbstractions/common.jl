@inline _to_i32_ka(x::Integer) = Int32(x)

@inline function _backend_from_pref(data, pref::Symbol)
    if pref == :cpu
        return KA.CPU()
    end
    return KA.get_backend(data)
end

@inline function _is_cpu_backend(backend)
    return backend isa KA.CPU
end

@inline function _ensure_backend_match(backend, data)
    b = KA.get_backend(data)
    if !_is_cpu_backend(backend) && b != backend
        throw(ArgumentError("backend does not match input array backend"))
    end
    return nothing
end

function _ka_sync(backend)
    KA.synchronize(backend)
    return nothing
end

function _work_data_for_backend(data::CuArray, backend_pref::Symbol)
    if backend_pref == :cpu
        return Array(data)
    end
    return data
end

function _finalize_work_data!(dest::CuArray, src, backend_pref::Symbol)
    if backend_pref == :cpu
        copyto!(dest, src)
    end
    return nothing
end

@inline function _get_scalar_ka(A, i::Int, j::Int)
    return A[i, j]
end

function _get_scalar_ka(A::CuArray, i::Int, j::Int)
    return Array(@view A[i:i, j:j])[1]
end
