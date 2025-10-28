
function test_stripe_mul()
    A_data = Base.rand(1:100,100,100)
    B_data = Base.rand(1:100,100,100)
    A = CuModMatrix(A_data,2^11,elem_type=Float32)
    B = CuModMatrix(B_data,2^11,elem_type=Float32)
    C = GPUFiniteFieldMatrices.zeros(Float32,100,100,2^11)
    
    GPUFiniteFieldMatrices.stripe_mul!(C,A,B)

    C_cpu = A_data * B_data .% 2^11

    @test convert.(Float32,C_cpu) == Array(C)

    A_data = Base.rand(1:100,100,100)
    B_data = Base.rand(1:100,100,100)
    A_data[5,5] = 11^7 - 5
    B_data[95,95] = 11^7 - 23

    A = CuModMatrix(A_data,11^7,elem_type=Float32)
    B = CuModMatrix(B_data,11^7,elem_type=Float32)
    C = GPUFiniteFieldMatrices.zeros(Float32,100,100,11^7)
    
    GPUFiniteFieldMatrices.stripe_mul!(C,A,B)

    C_cpu = A_data * B_data .% 11^7

    @test C_cpu == Array(C)
    
    println("Starting bigger example")
    s = 3003
    # bigger example 
    println("Allocating memory...")
    A_data = Base.rand(1:11^7,s,s)
    B_data = Base.rand(1:11^7,s,s)
    A_data[5,5] = 11^7 - 5
    B_data[95,95] = 11^7 - 23

    A = CuModMatrix(A_data,11^7,elem_type=Float32)
    B = CuModMatrix(B_data,11^7,elem_type=Float32)
    C = GPUFiniteFieldMatrices.zeros(Float32,s,s,11^7)
    
    println("Multiplying on the GPU...")
    CUDA.@time GPUFiniteFieldMatrices.stripe_mul!(C,A,B)
    CUDA.@sync println("Done with GPU")

    println("Multiplying on the CPU...")
    C_cpu = A_data * B_data .% 11^7

    @test C_cpu == Array(C)
    println("Done testing stripe mat-mat mul")

    println("Testing matrix-vector multiplication")

    A_data = Base.rand(1:100,100,100)
    A_data[5,5] = 11^7 - 5
    A_data[95,95] = 11^7 - 23
    x_data = Base.rand(1:100,100)

    A = CuModMatrix(A_data,11^7,elem_type=Float32)
    x = CuModVector(x_data,11^7,elem_type=Float32)
    z = GPUFiniteFieldMatrices.zeros(Float32,100,11^7)
    
    GPUFiniteFieldMatrices.stripe_mul!(z,A,x)

    z_cpu = A_data * x_data .% 11^7

    @test C_cpu == Array(C)

    println("Starting bigger mat-vec-mul example")

    s = 3003
    # bigger example 
    println("Allocating memory...")
    A_data = Base.rand(1:7^8,s,s)
    x_data = Base.rand(1:7^8,s)
    A_data[5,5] = 7^8 - 5
    A_data[95,95] = 7^8 - 23

    A = CuModMatrix(A_data,7^8,elem_type=Float32)
    x = CuModVector(x_data,7^8,elem_type=Float32)
    z = GPUFiniteFieldMatrices.zeros(Float32,s,7^8)
    
    println("Multiplying on the GPU...")
    CUDA.@time GPUFiniteFieldMatrices.stripe_mul!(z,A,x)
    CUDA.@sync println("Done with GPU")

    println("Multiplying on the CPU...")
    z_cpu = A_data * x_data .% 11^7

    @test C_cpu == Array(C)
    println("Done testing stripe mat-vec-mul")
end
