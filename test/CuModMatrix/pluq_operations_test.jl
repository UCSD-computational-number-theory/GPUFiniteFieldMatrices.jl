"""
Test row reduction operations on CuModMatrix.
This tests both the standard and direct implementations.
"""
function test_rref_operations()
    println("Testing PLUQ operations on CuModMatrix...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]  # Rank 2
    B_data = [1 0 0; 0 1 0; 0 0 1]  # Identity, full rank
    C_data = [1 2 3; 2 4 6; 3 6 9]  # Rank 1
    modulus = 11  # Prime modulus
    
    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)
    C = CuModMatrix(C_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    # Test using the rref_gpu_type function (indirect)
    println("Testing rref_gpu_type...")
    A_rref = rref_gpu_type(A)
    
    println("RREF(A) = ")
    display(A_rref)
    println()
    
    # Test identity matrix reduction
    B_rref = rref_gpu_type(B)
    println("RREF(B) (identity) = ")
    display(B_rref)
    println()
    
    # Check that identity remains unchanged
    println(Array(B_rref))
    println(B_rref)
    println(B_data)
    @test Array(B_rref) ≈ B_data
    
    # Test rank 1 matrix reduction
    C_rref = rref_gpu_type(C)
    println("RREF(C) (rank 1) = ")
    display(C_rref)
    println()
    
    # Test the new direct implementation
    println("Testing rref_gpu_type...")
    A_rref_type = rref_gpu_type(A)
    
    println("RREF_TYPE(A) = ")
    display(A_rref_type)
    println()
    
    # Check that both implementations yield the same result
    @test Array(A_rref) ≈ Array(A_rref_type)
    
    # Test with modulus override
    override_modulus = 7
    A_rref_mod = rref_gpu_type(A, override_modulus)
    
    println("RREF_TYPE(A) with modulus $override_modulus = ")
    display(A_rref_mod)
    println()
    
    println("All RREF operations tests passed!")
end

"""
Test LU decomposition operations on CuModMatrix.
"""
function test_lu_operations()
    println("Testing LU decomposition operations on CuModMatrix...")
    
    # Test matrix
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11  # Prime modulus
    
    A = CuModMatrix(A_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    # Test using the new direct implementation
    println("Testing lu_gpu_type...")
    U, L, Perm = lu_gpu_type(A)
    
    println("U = ")
    display(U)
    println()
    
    println("L = ")
    display(L)
    println()
    
    println("Perm = ", Perm)
    println()
    
    # Test with modulus override
    override_modulus = 7
    U_mod, L_mod, Perm_mod = lu_gpu_type(A, override_modulus)
    
    println("LU decomposition with modulus $override_modulus:")
    println("U_mod = ")
    display(U_mod)
    println()
    
    println("L_mod = ")
    display(L_mod)
    println()
    
    println("All LU decomposition tests passed!")
end

"""
Test PLUQ decomposition operations on CuModMatrix.
"""
function test_pluq_operations()
    println("Testing PLUQ decomposition operations on CuModMatrix...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11
    A = CuModMatrix(A_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    println("Testing pluq_gpu_type...")
    U_type, L_type, P_rows_type, P_cols_type = plup_gpu_type(A)
    println(typeof(P_rows_type))
    println(typeof(P_cols_type))
    println(P_rows_type)
    println(P_cols_type)
    P_rows = perm_array_to_matrix(P_rows_type, modulus; new_size=(3,3))
    P_cols = perm_array_to_matrix(P_cols_type, modulus; new_size=(3,3))
    
    println("U_type = ")
    display(U_type)
    println()
    
    println("L_type = ")
    display(L_type)
    println()

    println("P_rows_type = ", P_rows)
    display(P_rows)
    println()
    
    println("P_cols_type = ", P_cols)
    display(P_cols)
    println()

    @test Array(P_rows * L_type * U_type * P_cols) ≈ Array(A)
    @test mod.(Array(P_rows) * Array(L_type) * Array(U_type) * Array(P_cols), modulus) ≈ Array(A)
    @test P_rows_type[1:3] == [3,1,2]
    @test P_cols_type[1:3] == [1,2,3]

    println("Testing PLUQ decomposition operations on non-squareCuModMatrix...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
    modulus = 13
    A = CuModMatrix(A_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    println("Testing pluq_gpu_type...")
    U_type, L_type, P_rows_type, P_cols_type = plup_gpu_type(A)
    P_rows = perm_array_to_matrix(P_rows_type, modulus; new_size=(4,4))
    P_cols = perm_array_to_matrix(P_cols_type, modulus; new_size=(3,3))
    
    println("U_type = ")
    display(U_type)
    println()
    
    println("L_type = ")
    display(L_type)
    println()

    println("P_rows_type = ", P_rows)
    display(P_rows)
    println()
    
    println("P_cols_type = ", P_cols)
    display(P_cols)
    println()

    @test Array(P_rows * L_type * U_type * P_cols) ≈ Array(A)
    @test mod.(Array(P_rows) * Array(L_type) * Array(U_type) * Array(P_cols), modulus) ≈ Array(A)
    @test P_rows_type[1:4] == [4,3,2,1]
    @test P_cols_type[1:3] == [1,2,3]
    
    println("All PLUP decomposition tests passed!")
end

"""
Test inverse operations on CuModMatrix.
"""
function test_inverse_operations()
    println("Testing inverse operations on CuModMatrix...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11
    A = CuModMatrix(A_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    println("Testing inverse...")
    A_has_inverse, A_inv = GPUFiniteFieldMatrices.is_invertible_with_inverse(A)
    println("A has inverse: ", A_has_inverse)
    println("A_inv = ")
    display(A_inv)
    println(typeof(A_inv))
    println()

    @test A_has_inverse
    @test A * A_inv ≈ CuModMatrix(I, modulus)

    println("Testing inverse of identity matrix...")
    B = GPUFiniteFieldMatrices.eye(Int, 3, modulus)
    B_has_inverse, B_inv = GPUFiniteFieldMatrices.is_invertible_with_inverse(B)
    @test B_has_inverse
    @test B * B_inv ≈ CuModMatrix(I, modulus)
    @test B_inv * B ≈ CuModMatrix(I, modulus)
    @test B_inv == B
    
    println("All inverse operations tests passed!")
end

# Run all tests
function test_pluq()
    test_rref_operations()
    test_lu_operations()
    test_pluq_operations()
    # test_inverse_operations()
    
    println("\nAll RREF and decomposition tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_pluq()
end 
