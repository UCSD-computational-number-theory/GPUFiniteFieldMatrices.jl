using CUDA, BenchmarkTools, Test, LinearAlgebra

const global TILE_WIDTH = 2
const global MAX_OPS = 16

# __global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, width)
function SharedMatMul(d_A, d_B, d_C, P)

    width = blockDim().x

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

    row = (by-1) * width + ty
    col = (bx-1) * width + tx

    # float Pvalue = 0;
    total = 0

    # for (int m = 0; m < Width/TILE_WIDTH; ++m) {
    for m = 0:(width-1)

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
        for k = 1:width
            if counter >= MAX_OPS
                counter = 0
                total = total % P
            end
            total += sharedA[ty, k] * sharedB[k, tx]
            counter += 1
        end

        # __syncthreads();
        CUDA.sync_threads()

    # }
    end

    # d_P[Row*Width + Col] = Pvalue; 
    d_C[row, col] = total % P

    return

# }
end

N = 4
MOD = 10

A = rand(1:(MOD-1), N, N)
B = ones(Int64, N, N)
B = Matrix{Int64}(I, N, N)
C = zeros(Int64, N, N)


function matmul(A, B, p)

    A_rows,A_cols = size(A)
    B_rows,B_cols = size(B)

    if A_rows != B_cols
        error(
            "Matrix dimensions do not match.
            A has $rows rows and $cols cols, 
            B has $B_rows rows and $B_cols cols."
        ) 
    end

    padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH

    A_padded = zeros(eltype(A), padded_rows, padded_cols)
    B_padded = zeros(eltype(A), padded_cols, padded_rows)

    A_padded[1:A_rows, 1:A_cols] .= A
    B_padded[1:B_cols, 1:B_rows] .= B
    C = zeros(eltype(A), A_rows, B_cols)

    println(A_padded)
    println(B_padded)

    d_A = CUDA.CuArray(A_padded)
    d_B = CUDA.CuArray(B_padded)
    d_C = CUDA.CuArray(C)

    @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(div(A_rows,TILE_WIDTH),div(B_cols,TILE_WIDTH)) SharedMatMul(d_A,d_B,d_C,p)
    
    return Array(d_C)
end

C = matmul(A, B, 10)
println(C)


# println("GPU for $N x $N")

# @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C,p)
# @time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C,p)

# println("CPU for $N x $N")
# @time C = A*B
# @time C = A*B
# @time C = A*B