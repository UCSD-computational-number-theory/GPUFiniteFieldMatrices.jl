using GpuFiniteFieldMatrices

function main()
    
    TILE_WIDTH = 32
    P = 7
    # A = rand(0:P-1, (A_rows,A_cols))
    # A = [2 0 1 2 2; 
    # 1 0 2 0 1; 
    # 2 0 1 1 1; 
    # 2 0 2 2 1]

    A = [0 0 0 0 0 0 0 0 0 0 0 0 340 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 340 0 0 0 0;
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 340 0 0 0;
    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    341 0 0 0 0 0 0 0 0 0 0 0 0 0 0 340 0 0;
    0 341 0 0 0 0 2 0 0 0 0 0 0 0 0 0 340 0;
    0 0 341 1 0 0 0 2 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0 2 0 0 0 0 0 0 0 0 0;
    340 0 0 341 0 0 0 0 0 0 0 0 342 0 0 0 0 340;
    0 340 0 0 341 0 0 0 0 2 0 0 0 342 0 0 0 0;
    0 0 340 0 0 1 0 0 0 0 2 0 0 0 342 0 0 0;
    0 0 0 340 0 341 0 0 0 0 0 0 0 0 0 342 0 0;
    0 0 0 0 340 0 0 0 0 0 0 2 0 0 0 0 342 0;
    0 0 0 0 0 340 0 0 0 0 0 0 0 0 0 0 0 342]

    A_rows, A_cols = size(A)

    res_U, res_L, res_Perm_rows, res_Perm_cols = 0,0,0,0
    println(CUDA.@profile begin
        (res_U, res_L, res_Perm_rows, res_Perm_cols) = plup_gpu(A, P)
    end)

    writedlm("plu_A.csv", A, ',')
    writedlm("plu_U.csv", res_U, ',')
    writedlm("plu_L.csv", res_L, ',')
    writedlm("plu_Perm_rows.csv", res_Perm_rows, ',')
    writedlm("plu_Perm_cols.csv", res_Perm_cols, ',')

    Perm_rows = zeros(Int, A_rows, A_rows)
    for i in 1:A_rows
        Perm_rows[i, res_Perm_rows[i]] = 1
    end

    Perm_cols = zeros(Int, A_cols, A_cols)
    for i in 1:A_cols
        Perm_cols[i, res_Perm_cols[i]] = 1
    end

    @assert Perm_rows * A .% P == (res_L * res_U) * Perm_cols .% P "Result is not equal: $((res_L * res_U) * Perm_cols .% P)"

end

main()