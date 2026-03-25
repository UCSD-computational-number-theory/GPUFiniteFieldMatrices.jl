"""
    PLUQOptions

Configuration for the GPU PLUQ path.

Fields:
- `blocksize`: recursive panel width.
- `basecase`: switch to basecase kernel when segment length is small.
- `pivot_policy`: pivot policy tag (`:first_nonzero` currently).
- `lazy_q`: enable lazy permutation-vector composition inside basecase.
- `nftb`: tunable tiny-kernel thread grouping factor (32 * `nftb`, clamped).
"""
struct PLUQOptions
    blocksize::Int
    basecase::Int
    pivot_policy::Symbol
    lazy_q::Bool
    nftb::Int
    check_prime::Bool
end

"""
    PLUQOptions(; blocksize=64, basecase=32, pivot_policy=:first_nonzero, lazy_q=true, nftb=8, check_prime=false)

Construct `PLUQOptions` with validated positive block sizes.
"""
function PLUQOptions(;
    blocksize::Int=64,
    basecase::Int=32,
    pivot_policy::Symbol=:first_nonzero,
    lazy_q::Bool=true,
    nftb::Int=8,
    check_prime::Bool=false
)
    if blocksize < 1 || basecase < 1 || nftb < 1
        throw(ArgumentError("blocksize, basecase, and nftb must be positive"))
    end
    return PLUQOptions(blocksize, basecase, pivot_policy, lazy_q, nftb, check_prime)
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
