#using GPUFiniteFieldMatrices
#using Test
#using CUDA
#using LinearAlgebra

"""
Test in-place operations of the GpuMatrixModN type.
This includes:
- Basic in-place arithmetic operations
- In-place scalar operations
- In-place operations with modulus override
"""
function test_inplace_operations()
    println("Testing in-place operations of GpuMatrixModN...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    B_data = [9 8 7; 6 5 4; 3 2 1]
    modulus = 11  # Use a prime modulus
    
    A = GpuMatrixModN(A_data, modulus)
    B = GpuMatrixModN(B_data, modulus)
    
    C = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)
    D = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)
    
    println("A = ")
    display(A)
    println()
    
    println("B = ")
    display(B)
    println()
    
    # Test add!
    add!(C, A, B)
    println("C = A + B (in-place) = ")
    display(C)
    println()
    
    expected = mod.(A_data + B_data, modulus)
    @test Array(C) == expected
    
    # Test sub!
    sub!(D, A, B)
    println("D = A - B (in-place) = ")
    display(D)
    println()
    
    expected = mod.(A_data - B_data, modulus)
    @test Array(D) == expected
    
    # Test elementwise_multiply!
    elementwise_multiply!(C, A, B)
    println("C = A .* B (in-place) = ")
    display(C)
    println()
    
    expected = mod.(A_data .* B_data, modulus)
    @test Array(C) == expected
    
    # Test negate!
    negate!(D, A)
    println("D = -A (in-place) = ")
    display(D)
    println()
    
    expected = mod.(-A_data, modulus)
    @test Array(D) == expected
    
    # Test scalar operations
    scalar = 3
    
    # Test scalar_add!
    scalar_add!(C, A, scalar)
    println("C = A + $scalar (in-place) = ")
    display(C)
    println()
    
    expected = mod.(A_data .+ scalar, modulus)
    @test Array(C) == expected
    
    # Test scalar_sub!
    scalar_subtract!(D, A, scalar)
    println("D = A - $scalar (in-place) = ")
    display(D)
    println()
    
    expected = mod.(A_data .- scalar, modulus)
    @test Array(D) == expected
    
    # Test scalar_multiply!
    scalar_multiply!(C, A, scalar)
    println("C = A * $scalar (in-place) = ")
    display(C)
    println()
    
    expected = mod.(A_data .* scalar, modulus)
    @test Array(C) == expected
    
    # Test matrix multiplication
    E = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)
    multiply!(E, A, B)
    println("E = A * B (in-place) = ")
    display(E)
    println()
    
    expected = mod.(A_data * B_data, modulus)
    @test Array(E) == expected
    
    println("All in-place operations tests passed!")
end

"""
Test in-place operations with modulus override.
"""
function test_inplace_modulus_override()
    println("Testing in-place operations with modulus override...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus1 = 11
    modulus2 = 7
    
    A = GpuMatrixModN(A_data, modulus1)
    B = GpuMatrixModN(A_data, modulus2)
    C = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus1) 
    
    # Test add! with modulus override
    override_modulus = 5
    add!(C, A, B, override_modulus)
    
    println("C = A + B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data + A_data, override_modulus)
    @test Array(C) == expected
    
    # Test sub! with modulus override
    sub!(C, A, B, override_modulus)
    
    println("C = A - B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data - A_data, override_modulus)
    @test Array(C) == expected
    
    # Test elementwise_multiply! with modulus override
    elementwise_multiply!(C, A, B, override_modulus)
    
    println("C = A .* B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data .* A_data, override_modulus)
    @test Array(C) == expected
    
    scalar = 3
    
    # Test scalar_add! with modulus override
    scalar_add!(C, A, scalar, override_modulus)
    println("C = A + $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data .+ scalar, override_modulus)
    @test Array(C) == expected
    
    # Test scalar_subtract! with modulus override
    scalar_subtract!(C, A, scalar, override_modulus)
    println("C = A - $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data .- scalar, override_modulus)
    @test Array(C) == expected
    
    # Test scalar_multiply! with modulus override
    scalar_multiply!(C, A, scalar, override_modulus)
    println("C = A * $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()
    
    expected = mod.(A_data .* scalar, override_modulus)
    @test Array(C) == expected
    
    # Test matrix multiplication with modulus override
    D = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus1)
    multiply!(D, A, B, override_modulus)
    println("D = A * B (in-place with modulus override $override_modulus) = ")
    display(D)
    println()
    
    expected = mod.(A_data * A_data, override_modulus)
    @test Array(D) == expected
    
    println("All in-place operations with modulus override tests passed!")
end

function test_inplace_copy_and_mod()
    println("Testing in-place copy and modulus operations...")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11
    
    A = GpuMatrixModN(A_data, modulus)
    B = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)
    
    # Test copy!
    copy!(B, A)
    println("B = copy of A (in-place) = ")
    display(B)
    println()
    
    @test Array(B) == A_data
    
    # Test mod_elements!
    F = GpuMatrixModN(A_data * 2, modulus)
    mod_elements!(F)
    println("F with modulus applied in-place = ")
    display(F)
    println()
    
    expected = mod.(A_data * 2, modulus)
    @test Array(F) == expected
    
    # Test mod_elements! with modulus override
    override_modulus = 5
    mod_elements!(F, override_modulus)
    println("F with modulus $override_modulus applied in-place = ")
    display(F)
    println()
    
    expected = mod.(expected, override_modulus)
    @test Array(F) == expected
    
    println("All in-place copy and modulus operations tests passed!")
end

function test_inplace()
    test_inplace_operations()
    test_inplace_modulus_override()
    test_inplace_copy_and_mod()
    
    println("\nAll in-place operation tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_inplace()
end 
