function _pluq_batched_tiny_ka!(mats::AbstractVector{<:CuModMatrix}, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    for m in mats
        if rows(m) != n || cols(m) != n
            throw(CuModArraySizeMismatchException("all matrices must be $(n)x$(n)"))
        end
    end
    out = Vector{PLUQFactorization}(undef, length(mats))
    for i in eachindex(mats)
        out[i] = pluq_new_ka(mats[i], options=options)
    end
    return out
end

function _inverse_batched_tiny_ka(mats::AbstractVector{<:CuModMatrix}, n::Int; options::PLUQOptionsKA=PLUQOptionsKA())
    for m in mats
        if rows(m) != n || cols(m) != n
            throw(CuModArraySizeMismatchException("all matrices must be $(n)x$(n)"))
        end
    end
    out = Vector{CuModMatrix}(undef, length(mats))
    for i in eachindex(mats)
        out[i] = inverse_new_ka(mats[i], options=options)
    end
    return out
end

pluq_batched_4x4_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _pluq_batched_tiny_ka!(mats, 4, options=options)
pluq_batched_8x8_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _pluq_batched_tiny_ka!(mats, 8, options=options)
pluq_batched_16x16_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _pluq_batched_tiny_ka!(mats, 16, options=options)
pluq_batched_32x32_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _pluq_batched_tiny_ka!(mats, 32, options=options)

inverse_batched_4x4_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _inverse_batched_tiny_ka(mats, 4, options=options)
inverse_batched_8x8_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _inverse_batched_tiny_ka(mats, 8, options=options)
inverse_batched_16x16_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _inverse_batched_tiny_ka(mats, 16, options=options)
inverse_batched_32x32_ka!(mats::AbstractVector{<:CuModMatrix}; options::PLUQOptionsKA=PLUQOptionsKA()) = _inverse_batched_tiny_ka(mats, 32, options=options)
