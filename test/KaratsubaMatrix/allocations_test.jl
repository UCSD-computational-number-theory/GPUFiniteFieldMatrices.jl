
function test_allocations()

    A = GPUFiniteFieldMatrices.CuModMatrix([1 2; 3 4],100)
    B = GPUFiniteFieldMatrices.CuModMatrix([5 6; 7 8],100)
    C = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0],100)
    AK = MatToKMat(A, 100)
    BK = MatToKMat(B, 100)
    CK = MatToKMat(C, 100)

    @testset "Allocation tests" begin
        result = CUDA.@timed add!(CK,AK,BK)
        @test result[:gpu_bytes] == 0

        result = CUDA.@timed sub!(CK,AK,BK)
        @test result[:gpu_bytes] == 0
    end
end