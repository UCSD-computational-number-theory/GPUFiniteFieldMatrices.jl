using GPUFiniteFieldMatrices
using Test
using CUDA
using LinearAlgebra

"""
Test matrix multiplication operations on GPUFiniteFieldMatrix.
This tests both the standard and direct implementations.
"""
function test_matmul_operations()
    println("Testing matrix multiplication operations on GPUFiniteFieldMatrix...")
    
    # Test matrices
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    C_data = [58 64; 139 154]  # Expected result A*B mod 11 = [3 9; 7 0]
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrix(B_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    println("Matrix B = ")
    display(B)
    println()
    
    # Test using the standard matrix multiplication
    println("Testing standard matrix multiplication...")
    C = A * B
    
    println("A * B = ")
    display(C)
    println()
    
    # Verify the result matches expected calculation (mod 11)
    expected_C = GPUFiniteFieldMatrix(C_data .% modulus, modulus)
    @test Array(C) ≈ Array(expected_C)
    
    # Test using the direct multiplication implementation
    println("Testing mat_mul_gpu_type direct implementation...")
    C_direct = mat_mul_gpu_type(A, B)
    
    println("mat_mul_gpu_type(A, B) = ")
    display(C_direct)
    println()
    
    # Check that both implementations yield the same result
    @test Array(C) ≈ Array(C_direct)
    
    # Test multiplication with modulus override
    override_modulus = 7
    C_mod = mat_mul_gpu_type(A, B, override_modulus)
    
    println("mat_mul_gpu_type(A, B) with modulus $override_modulus = ")
    display(C_mod)
    println()
    
    # Verify the result matches expected calculation (mod override_modulus)
    expected_C_mod = (A_data * B_data) .% override_modulus
    @test Array(C_mod) ≈ expected_C_mod
    
    println("All matrix multiplication operations tests passed!")
end

"""
Test in-place matrix multiplication operations on GPUFiniteFieldMatrix.
"""
function test_inplace_matmul_operations()
    println("Testing in-place matrix multiplication operations on GPUFiniteFieldMatrix...")
    
    # Test matrices
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    C_data = zeros(Int, 2, 2)  # Destination matrix
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrix(B_data, modulus)
    C = GPUFiniteFieldMatrix(C_data, modulus)
    
    println("Matrix A = ")
    display(A)
    println()
    
    println("Matrix B = ")
    display(B)
    println()
    
    println("Initial Matrix C = ")
    display(C)
    println()
    
    # Test using the in-place multiplication implementation
    println("Testing mat_mul_type_inplace!...")
    mat_mul_type_inplace!(C, A, B)
    
    println("After mat_mul_type_inplace!(C, A, B):")
    display(C)
    println()
    
    # Verify the result matches expected calculation (mod 11)
    expected_C = (A_data * B_data) .% modulus
    @test Array(C) ≈ expected_C
    
    # Test in-place multiplication with modulus override
    override_modulus = 7
    C2 = GPUFiniteFieldMatrix(zeros(Int, 2, 2), modulus)
    
    mat_mul_type_inplace!(C2, A, B, override_modulus)
    
    println("In-place multiplication with modulus $override_modulus:")
    display(C2)
    println()
    
    # Verify the result matches expected calculation (mod override_modulus)
    expected_C2 = (A_data * B_data) .% override_modulus
    @test Array(C2) ≈ expected_C2
    
    println("All in-place matrix multiplication operations tests passed!")
end

"""
Test matrix multiplication with different regimes.
"""
function test_matmul_regimes()
    println("Testing matrix multiplication with different regimes...")
    
    # Test matrices
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    modulus = 11  # Prime modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrix(B_data, modulus)
    
    # Test standard regime
    println("Testing standard regime (⊠)...")
    C1 = mat_mul_gpu_type(A, B, -1, "⊠")
    
    println("mat_mul_gpu_type(A, B, regime='⊠') = ")
    display(C1)
    println()
    
    # Test hybrid regime
    println("Testing hybrid regime (hybrid)...")
    C2 = mat_mul_gpu_type(A, B, -1, "hybrid")
    
    println("mat_mul_gpu_type(A, B, regime='hybrid') = ")
    display(C2)
    println()
    
    # Verify both regimes yield the same result
    @test Array(C1) ≈ Array(C2)
    
    println("All matrix multiplication regime tests passed!")
end

# Run all tests
function test_matmul()
    test_matmul_operations()
    test_inplace_matmul_operations()
    test_matmul_regimes()
    
    println("\nAll matrix multiplication tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_matmul()
end 