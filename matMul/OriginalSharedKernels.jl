using CUDA, BenchmarkTools, Test, LinearAlgebra

const global TILE_WIDTH = 16
const global MAX_OPS = 16

# __global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, width)
function SharedMatMul(d_A, d_B, d_C, P, width)

    # __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    # __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    sharedA = CUDA.CuStaticSharedArray(Float64, (TILE_WIDTH, TILE_WIDTH))
    sharedB = CUDA.CuStaticSharedArray(Float64, (TILE_WIDTH, TILE_WIDTH))

    # int bx = blockIdx.x; int by = blockIdx.y;
    # int tx = threadIdx.x; int ty = threadIdx.y;
    
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x
    ty = threadIdx().y

    # int Row = by * TILE_WIDTH + ty;
    # int Col = bx * TILE_WIDTH + tx;

    row = (by-1) * TILE_WIDTH + ty
    col = (bx-1) * TILE_WIDTH + tx

    # float Pvalue = 0;
    total = 0

    # for (int m = 0; m < Width/TILE_WIDTH; ++m) {
    bound = div(width,TILE_WIDTH)
    m = 0
    while m < bound

        # Debugging print
        # @cuprintln("Row: $row Col: $col m: $m")

        # Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
        sharedA[ty, tx] = d_A[row, m*TILE_WIDTH + tx]

        # Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
        sharedB[ty, tx] = d_B[m*TILE_WIDTH + ty, col]

        # __syncthreads();
        CUDA.sync_threads()

        # for (int k = 0; k < TILE_WIDTH; ++k) {
        #   Pvalue += Mds[ty][k] * Nds[k][tx];
        # }
        counter = 0
        k = 1
        while k <= TILE_WIDTH
            if counter >= MAX_OPS
                counter = 0
                total = total % P
            end
            total += sharedA[ty, k] * sharedB[k, tx]
            counter += 1
            k += 1
        end

        # __syncthreads();
        CUDA.sync_threads()

        m += 1
    # }
    end

    # d_P[Row*Width + Col] = Pvalue; 
    d_C[row, col] = total % P

    return

# }
end

M = 500
N = 400
K = 400
MOD = 10

function matmul(A, B, p)

    A_rows,A_cols = size(A)
    B_rows,B_cols = size(B)

    if A_cols != B_rows
        error(
            "Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end

    padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, B_cols / TILE_WIDTH) * TILE_WIDTH
    max_size = max(padded_rows, padded_cols)

    A_padded = zeros(eltype(A), max_size, max_size)
    B_padded = zeros(eltype(A), max_size, max_size)

    A_padded[1:A_rows, 1:A_cols] .= A
    B_padded[1:B_rows, 1:B_cols] .= B
    C_padded = zeros(eltype(A), max_size, max_size)

    # println(A_padded)
    # println(B_padded)
    println(size(C_padded))

    d_A = CUDA.CuArray(A_padded)
    d_B = CUDA.CuArray(B_padded)
    d_C = CUDA.CuArray(C_padded)

    println("RAW COMPUTE TIME")
    @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(max_size,TILE_WIDTH),div(max_size,TILE_WIDTH)) SharedMatMul(d_A,d_B,d_C,p,padded_rows)
    println("FULL SETUP TIME")

    return Array(d_C)[1:A_rows, 1:B_cols]
end

A = rand(1:(MOD-1), M, N)
B = Matrix{Int64}(I, N, K)
@time C = matmul(A, B, 10)
# println(C)

A = rand(1:(MOD-1), M, N)
B = Matrix{Int64}(I, N, K)
@time C = matmul(A, B, 10)

println("CPU TIME")
@time C_ref = A*B
@test all(C_ref .== C)


# println("GPU for $N x $N")

# @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C,p)
# @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C,p)

# println("CPU for $N x $N")
# @time C = A*B
# @time C = A*B
# @time C = A*B