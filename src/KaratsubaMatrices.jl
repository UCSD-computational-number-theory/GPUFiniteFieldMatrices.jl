struct KaratsubaArray{T,D} <: AbstractArray{T,D}
    data1::AbstractArray{T,D}
    data2::AbstractArray{T,D}
    N1::Integer
    N2::Integer
    M::Integer

    function KaratsubaArray{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D}) where {T,D}
        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end

        M = find_max_ops(eltype(A),max(size(A)...,size(B)...))
        return new{T,D}(A,B,M,M,M)
    end

    function KaratsubaArray{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D},N1::Integer,N2::Integer,M::Integer) where {T,D}
        type = eltype(A)

        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end
        #=
        if !(all(x->x<N,A) && all(x->x<N,B))
            error("Cannot have entries larger than modulus in matrices")
        end
        =#
        if occursin("Float", string(type))
            bits_dict = Dict("64" => 51, "32" => 22, "16" => 9)
            bits_match = match(r"\d+", string(type))
            bits = get(bits_dict, bits_match.match, -1)
        elseif occursin("UInt", string(type))
            bits_match = match(r"\d+", string(type))
            bits = parse(Int, bits_match.match) - 1
        elseif occursin("Int", string(type))
            bits_match = match(r"\d+", string(type))
            bits = parse(Int, bits_match.match)
        else
            error("The input type is neither Int, UInt, nor Float.")
        end
    
        if bits == -1
            error("Input type is not recognized.")
        end
        if M >= BigInt(2)^bits
            error("Modulus too large")
        end

        return new{T,D}(A,B,N1,N2,M)
    end
end

const KaratsubaMatrix{T} = KaratsubaArray{T,2}
const KaratsubaVector{T} = KaratsubaArray{T,1}

function KaratsubaMatrix(A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    KaratsubaArray{T,2}(A,B)
end

function KaratsubaMatrix(A::AbstractMatrix{T},B::AbstractMatrix{T},M::Integer) where T
    KaratsubaArray{T,2}(A,B,M,M,M)
end

function KaratsubaMatrix(A::AbstractMatrix{T},B::AbstractMatrix{T},N1::Integer,N2::Integer,M::Integer) where T
    KaratsubaArray{T,2}(A,B,N1,N2,M)
end

function KaratsubaVector(A::AbstractVector{T},B::AbstractVector{T}) where T
    KaratsubaArray{T,1}(A,B)
end

function KaratsubaVector(A::AbstractVector{T},B::AbstractVector{T},M::Integer) where T
    KaratsubaArray{T,1}(A,B,M,M,M)
end

struct KMatMulPlan{T,D}
    temp1::AbstractArray{T,D}
    temp2::AbstractArray{T,D}
    temp1mod::Integer
    temp2mod::Integer

    function KMatMulPlan{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D},N1::Integer,N2::Integer) where {T,D}
        new{T,D}(A,B,N1,N2)
    end
end

function zerosplan(T::Type,rows::Integer,cols::Integer,N1::Integer,N2::Integer,use_gpu=false)
    if use_gpu == true
        return KMatMulPlan{T,2}(GPUFiniteFieldMatrices.zeros(T,rows,cols,N1),GPUFiniteFieldMatrices.zeros(T,rows,cols,N2),N1,N2)
    else
        return KMatMulPlan{T,2}(zeros(T,rows,cols),zeros(T,rows,cols),N1,N2)
    end
end

function find_max_ops(type, N)

    if occursin("Float", string(type))
        bits_dict = Dict("64" => 51, "32" => 22, "16" => 9)
        bits_match = match(r"\d+", string(type))
        bits = get(bits_dict, bits_match.match, -1)
    elseif occursin("UInt", string(type))
        bits_match = match(r"\d+", string(type))
        bits = parse(Int, bits_match.match) - 1
    elseif occursin("Int", string(type))
        bits_match = match(r"\d+", string(type))
        bits = parse(Int, bits_match.match)
    else
        error("The input type is neither Int, UInt, nor Float.")
    end

    if bits == -1
        error("Input type is not recognized.")
    end

    if 64 ≤ bits
        floor(BigInt, (BigInt(2)^bits - 1) / N^2) - 1
    else
        floor(Int, (2^bits - 1) / N^2) - 1    
    end

    
end

function KMatMul!(C::KaratsubaArray,A::KaratsubaArray,B::KaratsubaArray,plan::KaratsubaArray)
    if (A.M != B.M) || (A.M != C.M)
        error("Matrices must have the same modulus m")
    end
    if size(A.data1)[2] != size(B.data1)[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (size(A.data1)[1] != size(C.data1)[1]) || (size(B.data1)[2] != size(C.data1)[2])
        error("Output matrix has wrong dimensions")
    end
    #=
    max_ops = find_max_ops(eltype(A.data1),max(size(A.data1)...,size(B.data1)...))

    if A.M > max_ops
        error("Modulus is larger than max_ops")
    end
    =#
    #=
    plan.data1 .= A.data1 + A.data2
    plan.data2 .= B.data1 + B.data2
    C.data1 .= A.data1*B.data1
    C.data2 .= plan.data1*plan.data2
    C.data2 .= C.data2 - C.data1
    C.data2 .= C.data2 - A.data2*B.data2
    =#
    GPUFiniteFieldMatrices.add!(plan.data1,A.data1,A.data2,A.M^2)
    GPUFiniteFieldMatrices.add!(plan.data2,B.data1,B.data2,B.M^2)
    LinearAlgebra.mul!(C.data1,A.data1,B.data1)
    LinearAlgebra.mul!(C.data2,plan.data1,plan.data2)
    GPUFiniteFieldMatrices.sub!(C.data2,C.data2,B.data1,B.M^2)
    LinearAlgebra.mul!(plan.data2,A.data2,B.data2)
    GPUFiniteFieldMatrices.sub!(C.data2,C.data2,plan.data2,B.M^2)
    C
end

#Converts KMat to Mat
function Base.Array(K::KaratsubaArray)
    A = zeros(eltype(K.data1),size(K.data1)...)
    A = K.data1 + K.M*K.data2
    A
end

function KMatToMat(T::Type,K::KaratsubaArray)
    A = zeros(T,size(K.data1)...)
    A = K.data1 + K.M*K.data2
    A
end

function MatToKMat(A::AbstractArray,M::Integer)
    MatToKMat(eltype(A),A,M)
end

function MatToKMat(A::AbstractArray)
    M = find_max_ops(eltype(A),max(size(A)...))
    MatToKMat(eltype(A),A,M)
end

function MatToKMat(T::Type,A::AbstractArray,M::Integer)
    KaratsubaMatrix(T,A,M,M,M)
end

function KaratsubaMatrix(T::Type,A::AbstractArray,N1::Integer,N2::Integer,M::Integer)
    if occursin("CuMod", string(typeof(A)))
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A),size(A)...,M),GPUFiniteFieldMatrices.zeros(eltype(A),size(A)...,M),N1,N2,M)
    elseif occursin("Cu", string(typeof(A)))
        K = KaratsubaMatrix(CUDA.zeros(T,size(A)...),CUDA.zeros(T,size(A)...),N1,N2,M)
        M = Int(M)
    else
        K = KaratsubaMatrix(zeros(T,size(A)...),zeros(T,size(A)...),N1,N2,M)
    end
    #=
    K.data2 .= mod.(trunc.(A./M),N2)
    K.data1 .= mod.(A - Int(K.M)*K.data2,N1)
    =#
    LinearAlgebra.mul!(K.data2,A,1/M)
    GPUFiniteFieldMatrices.trunc_elements!(K.data2)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,N2)
    LinearAlgebra.mul!(K.data1,K.data2,Int(K.M))
    display(K.data1)
    display(A)
    GPUFiniteFieldMatrices.sub!(K.data1,A,K.data1,M)
    GPUFiniteFieldMatrices.mod_elements!(K.data1,N1)
    K
end

function MatToKMat(T::Type,A::AbstractArray)
    M = find_max_ops(T,max(size(A)...))
    MatToKMat(T,A,M)
end

function KaratsubaZeros(T,rows,cols,N1,N2,M,use_gpu)
    if use_gpu == true
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(T,rows,cols,M),GPUFiniteFieldMatrices.zeros(T,rows,cols,M),N1,N2,M)
    else
        K = KaratsubaMatrix(zeros(T,rows,cols),zeros(T,rows,cols),N1,N2,M)
    end
    K
end

function KaratsubaZeros(T,length,N1,N2,M,use_gpu)
    if use_gpu == true
        K = KaratsubaVector(GPUFiniteFieldMatrices.zeros(T,length,M),GPUFiniteFieldMatrices.zeros(T,length,M),N1,N2,M)
    else
        K = KaratsubaVector(zeros(T,length),zeros(T,length),N1,N2,M)
    end
    K
end

import Base: +, -, *

function +(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),A.N1,A.N2,A.M)
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.data1),size(A.data1)...),CUDA.zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    else
        K = KaratsubaMatrix(zeros(eltype(A.data1),size(A.data1)...),zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    end
    add!(K,A,B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),A.N1,A.N2,A.M)
        temp = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),A.N1,A.N2,A.M)
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.data1),size(A.data1)...),CUDA.zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
        temp = KaratsubaMatrix(CUDA.zeros(eltype(A.data1),size(A.data1)...),CUDA.zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    else
        K = KaratsubaMatrix(zeros(eltype(A.data1),size(A.data1)...),zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
        temp = KaratsubaMatrix(zeros(eltype(A.data1),size(A.data1)...),zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    end
    sub!(K,A,B,temp)
end

function *(a::Number, A::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.data1.N),A.N1,A.N2,A.M)
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.data1),size(A.data1)...),CUDA.zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    else
        K = KaratsubaMatrix(zeros(eltype(A.data1),size(A.data1)...),zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    end
    scalar_multiply!(K,A,a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function *(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2],A.M),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2],A.M),A.N1,A.N2,A.M)
        plan = KaratsubaMatrix(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2],A.M),GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2],A.M),A.N1,A.N2,A.M)
        #plan = KMatMulPlan(GPUFiniteFieldMatrices.zeros(eltype(A.data1),size(A.data1)...,A.M),GPUFiniteFieldMatrices.zeros(eltype(B.data1),size(B.data1)...,A.M),A.N1,A.N2)
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.data1),size(A.data1)...),CUDA.zeros(eltype(A.data1),size(A.data1)...),A.N1,A.N2,A.M)
    else
        K = KaratsubaMatrix(zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2]),zeros(eltype(A.data1),size(A.data1)[1],size(B.data1)[2]),A.N1,A.N2,A.M)
        plan = KMatMulPlan{Matrix{eltype(A.data1)}}(zeros(eltype(A.data1),size(A.data1)...),zeros(eltype(B.data1),size(B.data1)...),A.N1,A.N2)
    end
    KMatMul!(K,A,B,plan)
    K
end

function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end
    #=
    K.data2 .= mod.(trunc.((A.data1+B.data1)./A.M),A.N2)
    K.data1 .= mod.(A.data1 + B.data1 - A.M*K.data2,A.N1)
    K.data2 .= mod.(K.data2 + A.data2 + B.data2,A.N2)
    =#
    
    GPUFiniteFieldMatrices.add!(K.data2,A.data1,B.data1,2*A.M)
    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
    GPUFiniteFieldMatrices.trunc_elements!(K.data2)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,A.N2)
    LinearAlgebra.mul!(K.data1,K.data2,A.M,A.M^2)
    GPUFiniteFieldMatrices.sub!(K.data1,B.data1,K.data1,B.M^2)
    GPUFiniteFieldMatrices.add!(K.data1,A.data1,K.data1,A.M^2)
    GPUFiniteFieldMatrices.mod_elements!(K.data1,A.N1)
    GPUFiniteFieldMatrices.add!(K.data2,K.data2,A.data2)
    GPUFiniteFieldMatrices.add!(K.data2,K.data2,B.data2)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,A.N2)
    #=
    GPUFiniteFieldMatrices.add!(K.data1,A.data1,B.data1,2*A.M)
    GPUFiniteFieldMatrices.add!(K.data2,A.data2,B.data2)
    GPUFiniteFieldMatrices.add!(K.data2,K.data2,GPUFiniteFieldMatrices.divides(K.data1,K.M),K.N2)
    GPUFiniteFieldMatrices.mod_elements!(K.data1,K.N1)
    =#

    K
end

function sub!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray,temp::KaratsubaArray)
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end

    #=
    C.data2 .= mod.(trunc.((A.data1-B.data1)./A.M),A.N2)
    C.data1 .= mod.(A.data1 - B.data1 - A.M*C.data2,A.N1)
    C.data2 .= mod.(C.data2 + A.data2 - B.data2 - (A.data1.<B.data1),A.N2)
    
    GPUFiniteFieldMatrices.sub!(K.data2,A.data1,B.data1)
    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
    GPUFiniteFieldMatrices.trunc_elements!(K.data2)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,A.N2)
    LinearAlgebra.mul!(K.data1,K.data1,A.M,A.M^2)
    GPUFiniteFieldMatrices.sub!(K.data1,B.data1,K.data1)
    GPUFiniteFieldMatrices.sub!(K.data1,A.data1,K.data1)
    GPUFiniteFieldMatrices.mod_elements!(K.data1,A.N1)
    GPUFiniteFieldMatrices.add!(K.data2,K.data2,A.data2)
    GPUFiniteFieldMatrices.sub!(K.data2,K.data2,B.data2)
    GPUFiniteFieldMatrices.sub!(K.data2,K.data2,A.data1.<B.data1)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,A.N2)
    =#
    negate!(temp,B)
    add!(K,A,temp)
    K
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if A.M != B.M
        error("Matrices must have the same modulus m")
    end
    if size(A.data1) != size(B.data1)
        error("Matrix dimensions must match")
    end

    B.data2 .= mod.(trunc.((s*A.data1)/A.M),A.N2)
    B.data1 .= mod.(s*A.data1 - B.M*B.data2,A.N1)
    B.data2 .= mod.(B.data2 + s*A.data2,A.N2)
end

function negate!(K::KaratsubaArray,A::KaratsubaArray)
    #=
    K.data2 .= trunc.((A.M .- A.data1)/A.M)
    K.data1 .= mod.(A.M - A.data1 - A.M*K.data2,A.N1)
    K.data2 .= mod.(mod.(K.data2 + A.M - A.data2 - 1,A.M),A.N2)
    =#
    GPUFiniteFieldMatrices.negate!(K.data2,A.data1,A.M)
    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
    GPUFiniteFieldMatrices.trunc_elements!(K.data2)
    LinearAlgebra.mul!(K.data1,K.data2,A.M,A.M^2)
    GPUFiniteFieldMatrices.add!(K.data1,K.data1,A.data1,A.M^2)
    GPUFiniteFieldMatrices.negate!(K.data1,K.data1,A.M)
    GPUFiniteFieldMatrices.mod_elements!(K.data1,A.N1)
    GPUFiniteFieldMatrices.scalar_add!(K.data2,K.data2,A.M,A.M^2)
    GPUFiniteFieldMatrices.scalar_sub!(K.data2,K.data2,1,A.M^2)
    GPUFiniteFieldMatrices.sub!(K.data2,K.data2,A.data2,A.M)
    GPUFiniteFieldMatrices.mod_elements!(K.data2,A.N2)
    K
end