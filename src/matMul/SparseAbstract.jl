"""
File containing immutable structs for mod p and mod p n matrices.

# Data Structure Explanation
Immutable -> faster;
Separate structs for elements and matrices -> See rant below:

It is only sensible to overload operators on element-to-element basis;
For example, if we wanted to overload + on sparse matrices mod p to do addition
mod p, we would need to first overload + for the elements.

However, if we were to store the information on how to overload + for
element-to-element in the element struct, we would waste a lot of memory, since
we would effectively store many, many copies of p,n. So we store in the matrix.

Therefore, we have to create a wrapper struct for elements in case we wanted 
to do computations with elements. But that also means that we cannot immediately
pass an element mod p into an a matrix. Technically we can, but it is better to
extract the values of val,p,n,pmults and move these into the matrix instead.
"""

struct SparseMatModP{T} <: AbstractMatrix
    """ Custom data structure for sparse matrices"""
    rows::Array{T,1}
    cols::Array{T,1}
    vals::Array{T,1}
    nrows::T
    ncols::T
    p::T
    p_mults::Array{T,1}
end

struct SparseMatModPn{T}
    """ Custom data structure for sparse matrices"""
    rows::Array{T,1}
    cols::Array{T,1}
    vals::Array{T,1}
    nrows::T
    ncols::T
    p::T
    n::T
    p_mults::Array{T,1}
end

struct IntModP{T}
    """ Custom data structure for a single element mod p """
    val::T
    p::T
    p_mults::Array{T,1}
end

struct IntModPn{T}
    """ Custom data structure for a single element mod p """
    val::T
    p::T
    n::T
    p_mults::Array{T,1}
end

struct ModTypeError <: Exception
    message::String
end

function find_P_Mults(p)
    """ Find p_mults, all possible multiples of p needed for reduction """
    return [p * i for i in 1:div((p - 1)^2, p)]
end

function find_Pn_Mults(p, n)
    """ Find p_mults, all possible multiples of p needed for reduction """
    return [p^n * i for i in 1:div((p^n - 1)^2, p^n)]
end

# # Start of example
# rows = [1, 2, 3, 4, 5]
# cols = [1, 2, 3, 4, 5]
# vals = [1, 1, 1, 1, 1]
# p = 5
# n = 2

# A = SparseMatModP{Int64}(rows, cols, vals, p, find_P_Mults(p))
# println(A.p_mults)
# B = SparseMatModPn{Int64}(rows, cols, vals, p, n, find_Pn_mults(p, n))
# println(B.p_mults)
# # End of example