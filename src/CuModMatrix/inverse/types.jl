"""
    PLUQOptions

Configuration for the new GPU PLUQ path.
"""
struct PLUQOptions
    blocksize::Int
    basecase::Int
    pivot_policy::Symbol
    lazy_q::Bool
    check_prime::Bool
end

"""
    PLUQOptions(; blocksize=64, basecase=32, pivot_policy=:first_nonzero, lazy_q=true, check_prime=true)

Construct `PLUQOptions` with validated positive block sizes.
"""
function PLUQOptions(;
    blocksize::Int=64,
    basecase::Int=32,
    pivot_policy::Symbol=:first_nonzero,
    lazy_q::Bool=true,
    check_prime::Bool=true
)
    if blocksize < 1 || basecase < 1
        throw(ArgumentError("blocksize and basecase must be positive"))
    end
    return PLUQOptions(blocksize, basecase, pivot_policy, lazy_q, check_prime)
end

"""
    PLUQFactorization{T}

Result container for GPU PLUQ:
- `LU`: packed in-place factor storage
- `p`: row permutation vector
- `q`: column permutation vector
- `rank`: computed rank over `mod N`
"""
struct PLUQFactorization{T}
    LU::CuModMatrix{T}
    p::Vector{Int}
    q::Vector{Int}
    rank::Int
end
