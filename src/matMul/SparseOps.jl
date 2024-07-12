"""
File containing operator overloads for elements in mod p and mod pn
"""

function Base.:+(a::IntModP, b::IntModP)
    """ Overloads addition for elements mod p """
    result = a.val + b.val
    return IntModP{typeof(a.p)}(result - a.p_mults[div(result, p)], a.p, a.p_mults)
end

function Base.:+(a::IntModPn, b::IntModPn)
    """ Overloads addition for elements mod p^n """
    result = a.val + b.val
    return IntModPn{typeof(a.p)}(result - a.p_mults[div(result, p^n)], a.p, a.n, a.p_mults)
end

function Base.:-(a::IntModP, b::IntModP)
    """ Overloads subtraction for elements mod p """
    result = a.val - b.val
    return IntModP{typeof(a.p)}(result + a.p_mults[div(result, p)], a.p, a.p_mults)
end

function Base.:-(a::IntModPn, b::IntModPn)
    """ Overloads subtraction for elements mod p^n """
    result = a.val - b.val
    return IntModPn{typeof(a.p)}(result + a.p_mults[div(result, p^n)], a.p, a.n, a.p_mults)
end

function Base.:*(a::IntModP, b::IntModP)
    """ Overloads multiplication for elements mod p """
    result = a.val * b.val
    return IntModP{typeof(a.p)}(result - a.p_mults[div(result, p)], a.p, a.p_mults)
end

function Base.:*(a::IntModP, b::IntModP)
    """ Overloads multiplication for elements mod p^n """
    result = a.val * b.val
    return IntModPn{typeof(a.p)}(result - a.p_mults[div(result, p^n)], a.p, a.n, a.p_mults)
end

# # Start of example
# p = 5
# n = 3
# a1 = IntModP{Int64}(4, p, find_P_Mults(p))
# a2 = IntModP{Int64}(2, p, find_P_Mults(p))

# a3 = a1 + a2
# println(dump(a3))

# b1 = IntModPn{Int64}(12, p, n, find_Pn_Mults(p, n))
# b2 = IntModPn{Int64}(124, p, n, find_Pn_Mults(p, n))

# b3 = b1 + b2
# println(dump(b3))
# # End of example