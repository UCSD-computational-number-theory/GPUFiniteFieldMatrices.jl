using GPUFiniteFieldMatrices
using Test
using CUDA
using LinearAlgebra
using BenchmarkTools

function test_inplace_operations()
    println("Testing In-place operations for GPUFiniteFieldMatrix")
    
    A_data = [1 2 3; 4 5 6; 7 8 9]
    B_data = [9 8 7; 6 5 4; 3 2 1]
    modulus = 11
    
    A = GPUFiniteFieldMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrix(B_data, modulus)
    
    C = zeros(Int, 3, 3, modulus)
    D = zeros(Int, 3, 3, modulus)
    
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
    
    # Test subtract!
    subtract!(D, A, B)
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
    
    # Test scalar_add!
    scalar = 3
    scalar_add!(C, A, scalar)
    println("C = A + $scalar (in-place) = ")
    display(C)
    println()
    
    expected = mod.(A_data .+ scalar, modulus)
    @test Array(C) == expected
    
    # Test scalar_subtract!
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
    
    # Test copy!
    copy!(D, A)
    println("D = copy of A (in-place) = ")
    display(D)
    println()
    
    @test Array(D) == A_data
    
    # Test multiply!
    E = zeros(Int, 3, 3, modulus)
    multiply!(E, A, B)
    println("E = A * B (in-place) = ")
    display(E)
    println()
    
    expected = mod.(A_data * B_data, modulus)
    @test Array(E) == expected
    
    # Test mod_elements!
    F = GPUFiniteFieldMatrix(A_data * 2, modulus)
    mod_elements!(F)
    println("F with modulus applied in-place = ")
    display(F)
    println()
    
    expected = mod.(A_data * 2, modulus)
    @test Array(F) == expected
    
    println("\nPerformance Comparison: Regular vs In-place")
    println("--------------------------------------------")
    
    # Regular operations
    println("Regular addition:")
    @btime C = $A + $B;
    
    # In-place operations
    println("In-place addition:")
    @btime add!($C, $A, $B);
    
    println("\nAll in-place operation tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_inplace_operations()
end 