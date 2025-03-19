using GPUFiniteFieldMatrices
using Test
using CUDA
using LinearAlgebra
using BenchmarkTools

function test_modulus_changes_and_direct_operations()
    println("Testing modulus-changing operations and direct GPU functions")
    
    # Test initialization
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus1 = 11  # First modulus
    modulus2 = 7   # Second modulus
    
    A = GPUFiniteFieldMatrix(A_data, modulus1)
    
    println("A (mod $modulus1) = ")
    display(A)
    println()
    
    #---- PART 1: Test modulus changing functions ----
    
    # Test change_modulus (creates new matrix)
    B = change_modulus(A, modulus2)
    
    println("B = change_modulus(A, $modulus2) = ")
    display(B)
    println()
    
    # Check properties and values
    @test B.N == modulus2
    @test size(B) == size(A)
    @test Array(B) == mod.(A_data, modulus2)
    
    # Test that A is unchanged
    @test A.N == modulus1
    
    # Test change_modulus! (in-place)
    C = GPUFiniteFieldMatrix(A_data, modulus1)
    change_modulus!(C, modulus2)
    
    println("C after change_modulus!(C, $modulus2) = ")
    display(C)
    println()
    
    # Check properties and values
    @test C.N == modulus2
    @test Array(C) == mod.(A_data, modulus2)
    
    #---- PART 2: Test in-place operations with modulus override ----
    
    # Create matrices with different moduli
    D = GPUFiniteFieldMatrix(A_data, modulus1)
    E = GPUFiniteFieldMatrix(A_data, modulus2)
    F = zeros(Int, 3, 3, modulus1)  # Result matrix with modulus1
    
    # Test add! with modulus override
    override_modulus = 5
    add!(F, D, E, override_modulus)
    
    println("F = D + E (in-place with modulus override $override_modulus) = ")
    display(F)
    println()
    
    # Verify result uses the override modulus
    expected = mod.(A_data + A_data, override_modulus)
    @test Array(F) == expected
    
    # Test subtract! with modulus override
    subtract!(F, D, E, override_modulus)
    
    println("F = D - E (in-place with modulus override $override_modulus) = ")
    display(F)
    println()
    
    # Verify result
    expected = mod.(A_data - A_data, override_modulus)
    @test Array(F) == expected
    
    # Test elementwise_multiply! with modulus override
    elementwise_multiply!(F, D, E, override_modulus)
    
    println("F = D .* E (in-place with modulus override $override_modulus) = ")
    display(F)
    println()
    
    # Verify result
    expected = mod.(A_data .* A_data, override_modulus)
    @test Array(F) == expected
    
    #---- PART 3: Test direct GPU operations ----
    
    # Test matmul_gpu_direct
    D = GPUFiniteFieldMatrix([1 2; 3 4], modulus1)
    E = GPUFiniteFieldMatrix([5 6; 7 8], modulus1)
    
    println("D = ")
    display(D)
    println()
    
    println("E = ")
    display(E)
    println()
    
    # Matrix multiplication
    F = matmul_gpu_direct(D, E)
    
    println("F = D * E (using matmul_gpu_direct) = ")
    display(F)
    println()
    
    # Verify result
    expected = mod.([1 2; 3 4] * [5 6; 7 8], modulus1)
    @test Array(F) == expected
    @test F.N == modulus1
    
    # Test rref_gpu_direct
    G = GPUFiniteFieldMatrix([1 2 3; 2 4 6; 3 6 9], modulus1)
    
    println("G = ")
    display(G)
    println()
    
    # Row reduction
    H = rref_gpu_direct(G)
    
    println("H = rref_gpu_direct(G) = ")
    display(H)
    println()
    
    # Test plup_gpu_direct
    println("Testing plup_gpu_direct...")
    
    I = GPUFiniteFieldMatrix([1 2 3; 4 5 6; 7 8 9], modulus1)
    
    println("I = ")
    display(I)
    println()
    
    # PLUP decomposition
    U, L, P_rows, P_cols = plup_gpu_direct(I)
    
    println("U = ")
    display(U)
    println()
    
    println("L = ")
    display(L)
    println()
    
    @test U isa GPUFiniteFieldMatrix
    @test L isa GPUFiniteFieldMatrix
    @test U.N == modulus1
    @test L.N == modulus1
    
    #---- PART 4: Performance comparison ----
    
    println("\nPerformance Comparison: Standard vs Direct Operations")
    println("----------------------------------------------------")
    
    large_A = rand(Int, 100, 100, modulus1)
    large_B = rand(Int, 100, 100, modulus1)
    
    # Standard operations (with conversion to/from Array)
    println("Standard matrix multiplication (with Array conversion):")
    @btime C = $large_A * $large_B;
    
    # Direct operations
    println("Direct matrix multiplication:")
    @btime C = matmul_gpu_direct($large_A, $large_B);
    
    println("\nAll tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_modulus_changes_and_direct_operations()
end 