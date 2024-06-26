using CUDA, LinearAlgebra
include("mat_mul_plain.jl")
include("mat_mul_no_ops.jl")
include("mat_mul_ops.jl")

const global TILE_WIDTH = 25

function mat_mul_gpu(A, B, P, REGIME=-1, type=Float64, tile_width=25)
    """
    Hybrid matmul algorithm that incorporates three different regimes:

    if MAX_OPS >= A_cols and MAX_OPS >= B_cols,
        do normal matrix multiplication and broadcast mod
    else if MAX_OPS >= TILE_WIDTH
        call the custom kernel without counting operations
    else
        call the custom kernel with counting operations

    The regime choice can be overidden using the optional argument regime,
    where 1 is the first case, and 3 is the last case.

    The optional argument type determines the datatype used.
    By default, type is Float64 or Int52. 
    For reference, Float32 is Int23; Float16 is Int10.
    """

    # Define rows and cols of matrices
    A_rows, A_cols = size(A)
    B_rows,B_cols = size(B)

    # Check for proper dimensions
    if A_cols != B_rows
        error(
            "Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end

    # Calculate number of tiles for each dimensions
    # Note that A_padded_cols = B_padded_rows by matrix multiplication
    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH
    B_padded_cols = ceil(Int, B_cols / TILE_WIDTH) * TILE_WIDTH

    # Define indices for moving to CUDA Arrays
    Ainds = CartesianIndices(A)
    d_Ainds = CartesianIndices((1:A_rows,1:A_cols))
    Binds = CartesianIndices(B)
    d_Binds = CartesianIndices((1:B_rows,1:B_cols))

    # Define CUDA arrays of appropriate size
    # Note that undef makes all values default to 0
    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_cols))
    d_B = CUDA.CuArray{Int}(undef, (A_padded_cols, B_padded_cols))
    d_C = CUDA.CuArray{Int}(undef, (A_padded_rows, B_padded_cols))

    # Move the matrices from CPU to GPU CUDA Arrays
    copyto!(A, Ainds, d_A, d_Ainds)
    copyto!(B, Binds, d_B, d_Binds)

    # Hardcode tile width unles inputted
    if tile_width < 1
        error("Invalid tile width")
    end

    # Compute the MAX_OPS
    if occursin("Float", string(type))
        bits = match(r"\d+", string(type))
        d = Dict("64"=>51, "32"=>22, "16"=>9)
        bits = get(d, bits.match, -1)

    elseif occursin("UInt", string(type))
        bits = int(match(r"\d+", string(type)).match) - 1

    elseif occursin("Int", string(type))
        bits = int(match(r"\d+", string(type)).match)
    
    else
        error("The input type is neither Int, UInt, nor Float.")
    end

    if bits == -1
        error("Input type is not recognized.")
    end

    MAX_OPS = floor((2^bits-1)/P^2) - 1

    # Determine regiment to use if not hardcoded
    if REGIME == -1
        if MAX_OPS >= A_rows && MAX_OPS >= B_cols
            REGIME = 1
        elseif MAX_OPS > TILE_WIDTH
            REGIME = 2
        else
            REGIME = 3
        end
    end

    # Compute based on regime
    if REGIME == 1
        return Array(mat_mul_plain(d_A,d_B,P))[1:A_rows, 1:B_cols]

    elseif REGIME == 2
        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(B_padded_cols,TILE_WIDTH),div(A_padded_rows,TILE_WIDTH)) mat_mul_ops(d_A,d_B,d_C,P,TILE_WIDTH,type,MAX_OPS)
        return Array(d_C)[1:A_rows, 1:B_cols]

    elseif REGIME == 3
        @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(B_padded_cols,TILE_WIDTH),div(A_padded_rows,TILE_WIDTH)) mat_mul_ops(d_A,d_B,d_C,P,TILE_WIDTH,type)
        return Array(d_C)[1:A_rows, 1:B_cols]

    else
        error("Input regime is invalid.")
    end

    return 
end