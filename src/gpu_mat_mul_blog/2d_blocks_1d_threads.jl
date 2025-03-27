using CUDA, LinearAlgebra, IterTools

const global TILE_WIDTH = 32

function _2d_blocks_1d_threads_kernel()
    blockX

    num_threads = blockDim().x * blockDim().y / TILE_WIDTH
    thread_linear_idx = threadIdx().y * blockDim().x + threadIdx().x

    A_thread_block_tile = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH,TILE_WIDTH))
    B_thread_block_tile = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH,TILE_WIDTH))

    num_thread_block_tiles = (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K

    C_thread_results = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))

    thread_block_tile_idx = 0
    while thread_block_tile_idx < num_thread_block_tiles
        load_to_shared_memory()
        CUDA.sync_threads()
        thread_block_tile_idx += 1

        # unroll
        k_i = 0
        while k_i < BLOCK_TILE_SIZE_K
            B_thread_block_tile[B_thread_block_tile_row_idx]
        end
    end

    return
end