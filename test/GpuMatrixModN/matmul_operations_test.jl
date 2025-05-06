using GPUFiniteFieldMatrices
using Test
using CUDA
using LinearAlgebra

"""
Test matrix multiplication operations on GpuMatrixModN.
This tests both the standard and direct implementations.
"""
function test_matmul_operations()
    println("Testing matrix multiplication operations on GpuMatrixModN...")
    
    # Test matrices
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    C_data = [58 64; 139 154]  # Expected result A*B mod 11 = [3 9; 7 0]
    modulus = 11  # Prime modulus
    
    A = GpuMatrixModN(A_data, modulus)
    B = GpuMatrixModN(B_data, modulus)
    
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
    # TODO: Note we cast just to check the construtor works (though I think this should be removed)
    expected_C = GpuMatrixModN(C_data .% modulus, modulus)
    @test Array(C) ≈ Array(expected_C)
    
    # Test using the direct multiplication implementation
    println("Testing mat_mul_gpu_type direct implementation...")
    C_direct = mat_mul_gpu_type(A, B)
    
    println("mat_mul_gpu_type(A, B) = ")
    display(C_direct)
    println()
    
    @test Array(C) ≈ Array(C_direct)
    
    # Test multiplication with modulus override
    override_modulus = 7
    C_mod = mat_mul_gpu_type(A, B, override_modulus)
    
    println("mat_mul_gpu_type(A, B) with modulus $override_modulus = ")
    display(C_mod)
    println()
    
    # Verify the result matches expected calculation (mod override_modulus)
    # See previous TODO
    expected_C_mod = (A_data * B_data) .% override_modulus
    @test Array(C_mod) ≈ expected_C_mod
    
    println("All matrix multiplication operations tests passed!")
end

"""
Test in-place matrix multiplication operations on GpuMatrixModN.
"""
function test_inplace_matmul_operations()
    println("Testing in-place matrix multiplication operations on GpuMatrixModN...")
    
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    C = GPUFiniteFieldMatrices.zeros(Int, 2, 2, 11) 
    C_data = C.data
    modulus = 11  # Prime modulus
    
    A = GpuMatrixModN(A_data, modulus)
    B = GpuMatrixModN(B_data, modulus)
    
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
    println("Testing mat_mul_gpu_type...")
    mat_mul_gpu_type(A, B, C)
    
    println("After mat_mul_gpu_type(A, B, C):")
    display(C)
    println()
    
    # Verify the result matches expected calculation (mod 11)
    # See previous TODO
    expected_C = (A_data * B_data) .% modulus
    @test Array(C) ≈ expected_C
    
    # Test in-place multiplication with modulus override
    override_modulus = 7
    C2 = GPUFiniteFieldMatrices.zeros(Int, 2, 2, override_modulus)
    
    mat_mul_gpu_type(A, B, C2, override_modulus)
    
    println("In-place multiplication with modulus $override_modulus:")
    display(C2)
    println()
    
    # Verify the result matches expected calculation (mod override_modulus)
    # see previous TODO
    expected_C2 = mod.(A_data * B_data, override_modulus)
    @test Array(C2) ≈ expected_C2
    
    println("All in-place matrix multiplication operations tests passed!")
end

function test_matmul()
    test_matmul_operations()
    test_inplace_matmul_operations()
    
    println("\nAll matrix multiplication tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_matmul()
end 
