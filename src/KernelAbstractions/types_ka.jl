struct PLUQOptionsKA
    core::PLUQOptions
    backend_preference::Symbol
    workgroupsize_1d::Int
    workgroupsize_2d::Tuple{Int,Int}
end

function PLUQOptionsKA(;
    core::PLUQOptions=PLUQOptions(),
    backend_preference::Symbol=:auto,
    workgroupsize_1d::Int=256,
    workgroupsize_2d::Tuple{Int,Int}=(16, 16),
)
    if !(backend_preference in (:auto, :cuda, :cpu))
        throw(ArgumentError("backend_preference must be :auto, :cuda, or :cpu"))
    end
    if workgroupsize_1d < 1 || workgroupsize_2d[1] < 1 || workgroupsize_2d[2] < 1
        throw(ArgumentError("workgroupsizes must be positive"))
    end
    return PLUQOptionsKA(core, backend_preference, workgroupsize_1d, workgroupsize_2d)
end

@inline function _resolve_ka_core_options(options::PLUQOptionsKA, A::CuModMatrix)
    return _resolve_options(options.core, A)
end
