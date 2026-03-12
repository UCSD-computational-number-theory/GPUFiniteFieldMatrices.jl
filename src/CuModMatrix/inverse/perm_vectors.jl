"""
    pluq_init_perm(n)

Create the identity permutation vector `1:n`.
"""
function pluq_init_perm(n::Int)
    return collect(1:n)
end

"""
    pluq_inverse_perm(p)

Compute the inverse permutation of `p`.
"""
function pluq_inverse_perm(p::Vector{Int})
    pinv = Vector{Int}(undef, length(p))
    for i in eachindex(p)
        pinv[p[i]] = i
    end
    return pinv
end

"""
    pluq_compose_segment!(perm, offset, locperm)

Compose a local permutation `locperm` into `perm[offset:offset+length(locperm)-1]`.
"""
function pluq_compose_segment!(perm::Vector{Int}, offset::Int, locperm::Vector{Int})
    lastidx = offset + length(locperm) - 1
    tmp = copy(perm[offset:lastidx])
    for i in eachindex(locperm)
        perm[offset + i - 1] = tmp[locperm[i]]
    end
    return perm
end
