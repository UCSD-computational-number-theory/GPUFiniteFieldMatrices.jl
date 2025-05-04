
#function gemv!(transpose::Bool,alpha::Integer,A::GpuMatrixModN,x::GpuVectorModN,beta::Integer,y::GpuVectorModN)
#
#end

function find_max_stripe_ops(type,N)
    if occursin("Float", string(type))
        bits_dict = Dict("64" => 53, "32" => 24, "16" => 11)
        bits_match = match(r"\d+", string(type))
        bits = get(bits_dict, bits_match.match, -1)
    else
        throw(ArgumentError("The input type is not supported for CUBLAS gemm"))
    end

    if bits == -1
        throw(ArgumentError("Input type is not recognized."))
    end

    floor(Int, (2^bits - 1) / (N-1)^2) - 1
end

"""
Light wrapper over the provided interface in CUDA.jl. 

Does not keep track of overflow, so errors could happen if the matrices are too big.

Also does not check to see if the modulus of the varios matrices agree

Since we are working mod N, `alpha` and `beta` must be integers
"""
function unsafe_gemm!(transposeA::Bool,transposeB::Bool,alpha::Integer,A::GpuMatrixModN,B::GpuMatrixModN,beta::Integer,C::GpuMatrixModN)

    tAchar = transposeA ? 'Y' : 'N'
    tBchar = transposeB ? 'Y' : 'N'

    #TODO: convert alpha and beta to floats?

    CUDA.CUBLAS.gemm!(tAchar,tBchar,alpha,A.data,B.data,beta,C.data)

    C.data .%= C.N
end


"""
Matrix-vector multiplication based on stripes
"""
#TODO: replayce CuVector{Float64} with padded custom type. 
#the ".data" stuff won't work until then
function stripe_mul!(z::GpuVectorModN,A::GpuMatrixModN,x::GpuVectorModN)

    if A.N != z.N || z.N != z.N 
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != length(z)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by CUBLAS"))
    elseif eltype(A.data) != eltype(z.data) || eltype(z.data) != eltype(x.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same

    M = find_max_stripe_ops(eltype(A.data),A.N)

    if M < 1
        throw(ArgumentError("cannot perform a single multiplication for modulus $(A.N) with datatype $(eltype(A.data))"))
    end

    summed_size = cols(A)#size(A,2)

    num_stripes = div(summed_size,M) + 1

    if num_stripes == 1
        mul!(z.data,A.data,x.data)
        return
    end

    i = 1

    range = 1:M
    A_temp = @view A.data[:,range]
    x_temp = @view x.data[range]
    CUDA.CUBLAS.gemv!('N',1,A_temp,x_temp,0,z.data)
    z.data .%= z.N

    i += 1

    while i < num_stripes
        range = M*(i-1)+1:M*i 
        A_temp = @view A.data[:,range]
        x_temp = @view x.data[range]
        CUDA.CUBLAS.gemv!('N',1,A_temp,x_temp,1,C.data)
        z.data .%= z.N

        i += 1
    end
    # i == num_stripes

    range = (M*(i-1)+1):cols(A)
    A_temp = @view A.data[:,range]
    x_temp = @view x.data[range]
    CUDA.CUBLAS.gemm!('N',1,A_temp,x_temp,1,z.data)
    z.data .%= z.N

end

"""
Matrix multiplication mod N based on stripes.

"""
function stripe_mul!(C::GpuMatrixModN,A::GpuMatrixModN,B::GpuMatrixModN)

    if A.N != B.N || B.N != C.N 
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != rows(B)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by CUBLAS"))
    elseif eltype(A.data) != eltype(B.data) || eltype(B.data) != eltype(C.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same


    M = find_max_stripe_ops(eltype(A.data),A.N)

    if M < 1
        throw(ArgumentError("cannot perform a single multiplication for modulus $(A.N) with datatype $(eltype(A.data))"))
    end
                       
    summed_size = cols(A)#size(A,2)

    num_stripes = div(summed_size,M) + 1

    if num_stripes == 1
        mul!(C.data,A.data,B.data)
        C.data .%= C.N
        return
    end

    i = 1

    range = 1:M
    A_temp = @view A.data[:,range]
    B_temp = @view B.data[range,:]
    CUDA.CUBLAS.gemm!('N','N',1,A_temp,B_temp,0,C.data)
    C.data .%= C.N

    i += 1

    while i < num_stripes
        range = M*(i-1)+1:M*i 
        A_temp = @view A.data[:,range]
        B_temp = @view B.data[range,:]
        CUDA.CUBLAS.gemm!('N','N',1,A_temp,B_temp,1,C.data)
        C.data .%= C.N

        i += 1
    end
    # i == num_stripes

    range = (M*(i-1)+1):cols(A)
    A_temp = @view A.data[:,range]
    B_temp = @view B.data[range,:]
    CUDA.CUBLAS.gemm!('N','N',1,A_temp,B_temp,1,C.data)
    C.data .%= C.N
end

