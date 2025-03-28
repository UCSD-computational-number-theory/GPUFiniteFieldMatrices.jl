
function mat_mul_no_ops(d_A, d_B, d_C, P, width, TYPE)
    """
    GPU MatMul without counting number of operations,
    so MAX_OPS > TILE_WIDTH.
    Note that TILE_WDITH must be a constant variable.
    """

    # Define shared matrices for storage
    sharedA = CUDA.CuStaticSharedArray(
        TYPE, (TILE_WIDTH, TILE_WIDTH)
    )
    sharedB = CUDA.CuStaticSharedArray(
        TYPE, (TILE_WIDTH, TILE_WIDTH)
    )

    # Define common values as variables in cache
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x
    ty = threadIdx().y

    # Identify row and col
    row = (by-1) * TILE_WIDTH + ty
    col = (bx-1) * TILE_WIDTH + tx

    # Loop setup
    total = 0
    bound = div(width,TILE_WIDTH)
    m = 0

    # Load and compute one entry
    while m < bound

        # Load the rows and cols into shared matrices
        sharedA[ty, tx] = d_A[row, m*TILE_WIDTH + tx]
        sharedB[ty, tx] = d_B[m*TILE_WIDTH + ty, col]

        CUDA.sync_threads()

        # Loop through the row and column and find total
        k = 1
        while k <= TILE_WIDTH
            total += sharedA[ty, k] * sharedB[k, tx]
            k += 1
        end

        CUDA.sync_threads()

        m += 1
    end

    # Assign value
    d_C[row, col] = total % P

    return
end


function mat_mul_no_ops_everyreduce(d_A, d_B, d_C, P, width, TYPE)
    """
    GPU MatMul without counting number of operations,
    so MAX_OPS > TILE_WIDTH.
    Note that TILE_WDITH must be a constant variable.
    """

    # Define shared matrices for storage
    sharedA = CUDA.CuStaticSharedArray(
        TYPE, (TILE_WIDTH, TILE_WIDTH)
    )
    sharedB = CUDA.CuStaticSharedArray(
        TYPE, (TILE_WIDTH, TILE_WIDTH)
    )

    # Define common values as variables in cache
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x
    ty = threadIdx().y

    # Identify row and col
    row = (by-1) * TILE_WIDTH + ty
    col = (bx-1) * TILE_WIDTH + tx

    # Loop setup
    total = 0
    bound = div(width,TILE_WIDTH)
    m = 0

    # Load and compute one entry
    while m < bound

        # Load the rows and cols into shared matrices
        sharedA[ty, tx] = d_A[row, m*TILE_WIDTH + tx]
        sharedB[ty, tx] = d_B[m*TILE_WIDTH + ty, col]

        CUDA.sync_threads()

        # Loop through the row and column and find total
        #k = 1
        #while k <= TILE_WIDTH
        #    total += sharedA[ty, k] * sharedB[k, tx]
        #    #total %= P
        #    k += 1

        #end

        k = 1
        while k <= TILE_WIDTH
            total += sharedA[ty, k] * sharedB[k, tx]
            k += 1
            #total += sharedA[ty, k] * sharedB[k, tx]
            #k += 1
            #total += sharedA[ty, k] * sharedB[k, tx]
            #k += 1
            #total += sharedA[ty, k] * sharedB[k, tx]
            #k += 1
            

        end

        #for k = 1:TILE_WIDTH
        #    total += sharedA[ty, k] * sharedB[k, tx]
        #end

#        k=1
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 4
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 8
#        
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 12
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 16
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 20
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 24
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 28
#
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        total += sharedA[ty, k] * sharedB[k, tx]
#        k += 1
#        # 32

        CUDA.sync_threads()

        m += 1
    end

    # Assign value
    d_C[row, col] = total % P

    return
end
