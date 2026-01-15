const DEFAULT_THREADS = 256

function _alloc_like(A::CuModArray, N::Integer)
    T = eltype(A.data)
    D = ndims(A.data)
    data = similar(A.data)
    return CuModArray{T,D}(data, N; mod=false, new_size=size(A))
end

