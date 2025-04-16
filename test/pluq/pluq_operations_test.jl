using GPUFiniteFieldMatrices
using Test
using CUDA
using LinearAlgebra

"""
Test row reduction operations on GPUFiniteFieldMatrix.
This tests both the standard and direct implementations.
"""
function test_pluq_operations()
    println("Testing PLUQ operations on GPUFiniteFieldMatrix...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]  # Rank 2
    B_data = [1 0 0; 0 1 0; 0 0 1]  # Identity, full rank
    C_data = [1 2 3; 2 4 6; 3 6 9]  # Rank 1
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrix(B_data, modulus)
    C = GPUFiniteFieldMatrix(C_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    # Test using the rref_gpu_direct function (indirect)
    println("Testing rref_gpu_direct...")
    A_rref = rref_gpu_direct(A)
    
    println("RREF(A) = ")
    display(A_rref)
    println()
    
    # Test identity matrix reduction
    B_rref = rref_gpu_direct(B)
    println("RREF(B) (identity) = ")
    display(B_rref)
    println()
    
    # Check that identity remains unchanged
    @test Array(B_rref) ≈ B_data
    
    # Test rank 1 matrix reduction
    C_rref = rref_gpu_direct(C)
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
Test LU decomposition operations on GPUFiniteFieldMatrix.
"""
function test_lu_operations()
    println("Testing LU decomposition operations on GPUFiniteFieldMatrix...")
    
    # Test matrix
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    
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
Test PLUP decomposition operations on GPUFiniteFieldMatrix.
"""
function test_plup_operations()
    println("Testing PLUP decomposition operations on GPUFiniteFieldMatrix...")
    
    # Test matrix
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    # Test using the plup_gpu_direct function
    println("Testing plup_gpu_direct...")
    U, L, P_rows, P_cols = plup_gpu_direct(A)
    
    println("U = ")
    display(U)
    println()
    
    println("L = ")
    display(L)
    println()
    
    println("P_rows = ", P_rows)
    println("P_cols = ", P_cols)
    println()
    
    # Test using the new direct implementation
    println("Testing plup_gpu_type...")
    U_type, L_type, P_rows_type, P_cols_type = plup_gpu_type(A)
    
    println("U_type = ")
    display(U_type)
    println()
    
    println("L_type = ")
    display(L_type)
    println()
    
    # Check that both implementations yield the same result
    @test Array(U) ≈ Array(U_type)
    @test Array(L) ≈ Array(L_type)
    @test P_rows == P_rows_type
    @test P_cols == P_cols_type
    
    # Test with modulus override
    override_modulus = 7
    U_mod, L_mod, P_rows_mod, P_cols_mod = plup_gpu_type(A, override_modulus)
    
    println("PLUP decomposition with modulus $override_modulus:")
    println("U_mod = ")
    display(U_mod)
    println()
    
    println("L_mod = ")
    display(L_mod)
    println()
    
    println("All PLUP decomposition tests passed!")
end

# Run all tests
function test_pluq()
    test_pluq_operations()
    test_lu_operations()
    test_plup_operations()
    
    println("\nAll RREF and decomposition tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_pluq()
end 
