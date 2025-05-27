using Test
using LinearAlgebra
using Oscar
using CUDA

function test_basic_operations()
    
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    a = 9
    AK = MatToKMat(A)
    BK = MatToKMat(B)
    CK = KMat([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9])
    DK = KMat([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19])
    C = Array(CK)
    D = Array(DK)
    display(C)
    display(D)

    @testset "Cpu Matrices" begin
        @test A + B == Array(AK + BK)
        @test A - B == Array(AK - BK)
        @test a*A == Array(a*AK)
        #@test A*B == Array(AK*BK)
        @test C + D == Array(CK + DK)
        @test C - D == Array(CK - DK)
        @test a*C == Array(a*CK)
        #@test C*D == Array(CK*DK)
    end
    
    A = CuArray([1 2; 3 4])
    B = CuArray([5 6; 7 8])
    a = 9
    AK = MatToKMat(A)
    BK = MatToKMat(B)
    CK = KMat([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9])
    DK = KMat([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19])
    C = CuArray(Array(CK))
    D = CuArray(Array(DK))
    display(C)
    display(D)

    @testset "Gpu Matrices" begin
        @test A + B == CuArray(Array(AK + BK))
        @test A - B == CuArray(Array(AK - BK))
        @test a*A == CuArray(Array(a*AK))
        #@test A*B == CuArray(Array(AK*BK))
        @test C + D == CuArray(Array(CK + DK))
        @test C - D == CuArray(Array(CK - DK))
        @test a*C == CuArray(Array(a*CK))
        #@test C*D == CuArray(Array(CK*DK))
    end
end