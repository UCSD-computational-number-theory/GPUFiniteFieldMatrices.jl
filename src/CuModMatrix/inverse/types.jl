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
    pivot_warp_kernel::Symbol
    trsm_mode::Symbol
    trsm_warp_threshold::Int
    schur_tile::Int
    schur_transpose_u::Bool
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
    pivot_warp_kernel::Symbol=:ballot,
    trsm_mode::Symbol=:auto,
    trsm_warp_threshold::Int=32,
    schur_tile::Int=16,
    schur_transpose_u::Bool=false,
    check_prime::Bool=false
)
    if blocksize < 1 || basecase < 1 || nftb < 1 || trsm_warp_threshold < 1
        throw(ArgumentError("blocksize, basecase, nftb, and trsm_warp_threshold must be positive"))
    end
    if !(pivot_warp_kernel in (:ballot, :shfl))
        throw(ArgumentError("pivot_warp_kernel must be :ballot or :shfl"))
    end
    if !(trsm_mode in (:auto, :panel, :warp))
        throw(ArgumentError("trsm_mode must be :auto, :panel, or :warp"))
    end
    if !(schur_tile in (8, 16, 32))
        throw(ArgumentError("schur_tile must be one of 8, 16, or 32"))
    end
    return PLUQOptions(
        blocksize,
        basecase,
        pivot_policy,
        lazy_q,
        nftb,
        pivot_warp_kernel,
        trsm_mode,
        trsm_warp_threshold,
        schur_tile,
        schur_transpose_u,
        check_prime,
    )
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
