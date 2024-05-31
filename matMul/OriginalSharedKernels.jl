using CUDA, BenchmarkTools, Test

const global TILE_WIDTH = 32

# __global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, width)
function SharedMatMul(d_A, d_B, d_C)

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
    for m = 0:127

        # Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
        sharedA[ty, tx] = d_A[row, m*TILE_WIDTH + tx]

        # Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
        sharedB[ty, tx] = d_B[m*TILE_WIDTH + ty, col]

        # __syncthreads();
        CUDA.sync_threads()

        # for (int k = 0; k < TILE_WIDTH; ++k) {
        #   Pvalue += Mds[ty][k] * Nds[k][tx];
        # }
        for k = 1:width
            total += sharedA[ty, k] * sharedB[k, tx]
        end

        # __syncthreads();
        CUDA.sync_threads()

    # }
    end

    # d_P[Row*Width + Col] = Pvalue; 
    d_C[row, col] = total

    return

# }
end

N = 4096
MOD = 10

A = rand(1:MOD, N, N)
B = ones(Int64, N, N)
C = zeros(Int64, N, N)

d_A = CUDA.CuArray(A)
d_B = CUDA.CuArray(B)
d_C = CUDA.CuArray(C)

println("GPU for $N x $N")
@time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C)
@time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C)
@time @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(TILE_WIDTH,TILE_WIDTH) SharedMatMul(d_A,d_B,d_C)

println("CPU for $N x $N")
@time C = A*B
@time C = A*B
@time C = A*B