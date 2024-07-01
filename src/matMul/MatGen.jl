"""
File containing basic generators for creating matrices over Fp.

Notes:
Call "include("generators/MatrixGenerators.jl")" in repl,
and use display(MatrixGenerators.genMatOverFq(4,4,4)) to display.
"""
module MatGen

using LinearAlgebra
using BandedMatrices

"""
See documentation below for other matrix generators.
https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Standard-functions
"""

"""
Returns a random mxn matrix with entries from Fp (1-p).
"""
function genMatOverFq(m,n,p)
    return rand(1:p,m,n)
end

"""
Returns banded mxn matrix with entries from Fp (1-p), 
l lower bands, and u upper bands.
"""
function genBandedMatOverFq(m,n,p,l,u)
    return BandedMatrix(genMatOverFq(m,n,p), (l,u))
end

"""
Returns a random nxn diagonal matrix with entries from Fp (1-p).
"""
function genDiagMatOverFq(n,p)
    return Diagonal(rand(1:p,n))
end

end