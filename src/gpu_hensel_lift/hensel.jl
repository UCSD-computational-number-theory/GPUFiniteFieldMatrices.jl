using GPUFiniteFieldMatrices

# """
#     henselLift(p, precision, A, T)
# Hensel lifts mod p solution T to the linear system AX-I=0 to mod p^precision

# INPUTS:
# * "p" -- integer, a prime number 
# * "precision" -- integer 
# * "A" -- matrix, integer coefficients
# * "T" -- matrix, integer coefficients, satisfies AT-I=0 mod p
# """
# function henselLift(p, precision, A, T)
#     i = 1
#     while i < precision
#         T = 2*T - T * (A*T)
#         #println("After step $i: $(julia_signed_mod.(T,p^(i+1)))")
#         i *= 2
#     end
#     R, pi = residue_ring(ZZ, p^precision)
#     stuff = [R(x) for x in Array(T)]
#     return matrix(R,stuff)
# end

"""
    hensel_pseudoinverse(p, precision, A, T)

Hensel lifts mod p solution T to the linear system AX-I=0 to mod p^precision
Uses the GPUFiniteFieldMatrix type.

INPUTS:
* "p" -- integer, a prime number 
* "precision" -- integer 
* "A" -- matrix, integer coefficients
* "T" -- matrix, integer coefficients, satisfies AT-I=0 mod p
"""
function hensel_pseudoinverse(p, precision, A, T)
    i = 1
    while i < precision
        T = 2*T - T * (A*T)
        i *= 2
    end
    # TODO Move from GPU to Oscar
    R, pi = residue_ring(ZZ, p^precision)
    return matrix(R, [R(x) for x in Array(T)])
end

function hensel_pseudoinverse!(p ,precision, A, T)
    # TODO Make inplace version
