@inline function _pluq_mod_t_ka(x::T, N::Int32) where {T}
    v = Int64(x)
    r = rem(v, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

@inline function _pluq_mod_mul_t_ka(a::T, b::T, N::Int32) where {T}
    av = Int64(a)
    bv = Int64(b)
    r = rem(av * bv, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

@inline function _pluq_mod_inv_t_ka(a::T, N::Int32) where {T}
    aa = Int32(rem(Int64(a), Int64(N)))
    if aa < 0
        aa += N
    end
    if aa == 0
        return zero(T)
    end
    t = Int32(0)
    newt = Int32(1)
    r = N
    newr = aa
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if r != 1
        return zero(T)
    end
    if t < 0
        t += N
    end
    return T(t)
end
