using Test
using LinearAlgebra
using Oscar

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

    @testset begin
        @test A + B == Array(AK + BK)
        @test A - B == Array(AK - BK)
        @test a*A == Array(a*AK)
        #@test A*B == KMatToMat(AK*BK)
        @test C + D == Array(CK + DK)
        @test C - D == Array(CK - DK)
        @test a*C == Array(a*CK)
        #@test C*D == KMatToMat(CK*DK)
    end
end