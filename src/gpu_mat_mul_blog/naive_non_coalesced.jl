using CUDA, LinearAlgebra, IterTools

const global TILE_WIDTH = 32

function naive_non_coalesced_kernel(
    d_A, d_B, d_C, P, 
    bound, MAX_OPS
)
    row = (blockIdx().x-1) * blockDim().x + (threadIdx().x)
    col = (blockIdx().y-1) * blockDim().x + (threadIdx().y)

    m = 0
    o = 0
    total = 0
    while m < bound
        if o == MAX_OPS
            total = total % P
        end
        total += d_A[row,col + m*TILE_WIDTH] * d_B[col,row + m*TILE_WIDTH]
        m += 1
        o += 1
    end
    d_C[row,col] = total % P

    return
end

function naive_non_coalesced(
    A, B, P,
    time,
)
    A_rows, A_cols = size(A)
    B_rows,B_cols = size(B)

    if A_cols != B_rows
        error(
            "Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end

    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH
    B_padded_cols = ceil(Int, B_cols / TILE_WIDTH) * TILE_WIDTH

    Ainds = CartesianIndices(A)
    d_Ainds = CartesianIndices((1:A_rows,1:A_cols))
    Binds = CartesianIndices(B)
    d_Binds = CartesianIndices((1:B_rows,1:B_cols))

    TYPE = Int
    d_A = CUDA.CuArray{TYPE}(undef, (A_padded_rows, A_padded_cols))
    d_B = CUDA.CuArray{TYPE}(undef, (A_padded_cols, B_padded_cols))
    d_C = CUDA.CuArray{TYPE}(undef, (A_padded_rows, B_padded_cols))

    copyto!(d_A, d_Ainds, A, Ainds)
    copyto!(d_B, d_Binds, B, Binds)

    MAX_OPS = find_max_ops(TYPE, P)
    bound = div(A_padded_cols,TILE_WIDTH)

    @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_padded_rows,TILE_WIDTH),div(B_padded_cols,TILE_WIDTH)) naive_non_coalesced_kernel(d_A,d_B,d_C,P,bound,MAX_OPS)
    return d_C[1:A_rows, 1:B_cols]
end

function find_max_ops(type, N)

    if occursin("Float", string(type))
        bits = match(r"\d+", string(type))
        d = Dict("64"=>51, "32"=>22, "16"=>9)
        bits = get(d, bits.match, -1)

    elseif occursin("UInt", string(type))
        bits = int(match(r"\d+", string(type)).match) - 1

    elseif occursin("Int", string(type))
        bits = parse(Int, match(r"\d+", string(type)).match)
    
    else
        error("The input type is neither Int, UInt, nor Float.")
    end

    if bits == -1
        error("Input type is not recognized.")
    end

    return floor((2^bits-1)/N^2) - 1
end