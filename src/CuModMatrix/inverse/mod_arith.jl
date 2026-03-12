"""
    pluq_mod_reduce(x, N)

Reduce `x` modulo `N`.
"""
function pluq_mod_reduce(x, N::Int)
    return mod(x, N)
end

"""
    pluq_mod_add(a, b, N)

Return `(a + b) mod N`.
"""
function pluq_mod_add(a, b, N::Int)
    return mod(a + b, N)
end

"""
    pluq_mod_sub(a, b, N)

Return `(a - b) mod N`.
"""
function pluq_mod_sub(a, b, N::Int)
    return mod(a - b, N)
end

"""
    pluq_mod_mul(a, b, N)

Return `(a * b) mod N`.
"""
function pluq_mod_mul(a, b, N::Int)
    return mod(a * b, N)
end

"""
    pluq_mod_inv(a, N)

Compute the modular inverse of `a` modulo `N` using extended Euclid.
Throws if no inverse exists.
"""
function pluq_mod_inv(a::Number, N::Int)
    aa = Int(mod(round(Int, a), N))
    if aa == 0
        throw(DomainError(a, "no inverse for zero modulo N"))
    end
    t = 0
    newt = 1
    r = N
    newr = aa
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if r != 1
        throw(DomainError(a, "value is not invertible modulo N"))
    end
    if t < 0
        t += N
    end
    return t
end

"""
    pluq_is_prime(n)

Return `true` if `n` is prime.
"""
function pluq_is_prime(n::Int)
    if n <= 1
        return false
    end
    if n <= 3
        return true
    end
    if n % 2 == 0 || n % 3 == 0
        return false
    end
    i = 5
    while i * i <= n
        if n % i == 0 || n % (i + 2) == 0
            return false
        end
        i += 6
    end
    return true
end
