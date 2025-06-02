struct KaratsubaArray{T,D} <: AbstractArray{T,D}
    l::AbstractArray{T,D}
    h::AbstractArray{T,D}
    m::Integer

    function KaratsubaArray{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D}) where {T,D}
        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end

        M = find_max_ops(eltype(A),max(size(A)...,size(B)...))
        return new{T,D}(A,B,M)
    end

    function KaratsubaArray{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D},N::Integer) where {T,D}
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
        if N >= BigInt(2)^bits
            error("Modulus too large")
        end

        return new{T,D}(A,B,N)
    end
end

const KaratsubaMatrix{T} = KaratsubaArray{T,2}
const KaratsubaVector{T} = KaratsubaArray{T,1}

function KaratsubaMatrix(A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    KaratsubaArray{T,2}(A,B)
end

function KaratsubaMatrix(A::AbstractMatrix{T},B::AbstractMatrix{T},M::Integer) where T
    KaratsubaArray{T,2}(A,B,M)
end

function KaratsubaVector(A::AbstractMatrix{T},B::AbstractMatrix{T}) where T
    KaratsubaArray{T,1}(A,B)
end

function KaratsubaVector(A::AbstractMatrix{T},B::AbstractMatrix{T},M::Integer) where T
    KaratsubaArray{T,1}(A,B,M)
end

struct KMatMulPlan{AbstractArray}
    temp1::AbstractArray
    temp2::AbstractArray
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

function KMatMul!(C::KaratsubaArray,A::KaratsubaArray,B::KaratsubaArray,plan::KMatMulPlan)
    if (A.m != B.m) || (A.m != C.m)
        error("Matrices must have the same modulus m")
    end
    if ncols(A.l) != nrows(B.l)
        error("Matrix dimensions don't work for multiplication")
    end
    if (nrows(A.l) != nrows(C.l)) || (ncols(B.l) != ncols(C.l))
        error("Output matrix has wrong dimensions")
    end

    max_ops = find_max_ops(eltype(A.l),max(size(A.l)...,size(B.l)...))

    if A.m > max_ops
        error("Modulus is larger than max_ops")
    end

    plan.temp1 .= A.l + A.h
    plan.temp2 .= B.l + B.h
    C.l .= A.l*B.l
    C.h .= plan.temp1*plan.temp2
    C.h .= C.h - C.l
    C.h .= C.h - A.h*B.h
end

#Converts KMat to Mat
function Base.Array(K::KaratsubaArray)
    A = zeros(eltype(K.l),size(K.l)...)
    A = K.l + K.m*K.h
    println(typeof(A))
    A
end

function KMatToMat(T::Type,K::KaratsubaArray)
    A = zeros(T,size(K.l)...)
    A = K.l + K.m*K.h
    A
end

function MatToKMat(A::AbstractArray,M)
    MatToKMat(eltype(A),A,M)
end

function MatToKMat(A::AbstractArray)
    M = find_max_ops(eltype(A),max(size(A)...))
    MatToKMat(eltype(A),A,M)
end

function MatToKMat(T::Type,A::AbstractArray,M::Integer)
    if occursin("Cu", string(typeof(A)))
        K = KaratsubaMatrix(CUDA.zeros(T,size(A)...),CUDA.zeros(T,size(A)...),M)
        M = Int(M)
    else
        K = KaratsubaMatrix(zeros(T,size(A)...),zeros(T,size(A)...),M)
    end
    K.h .= trunc.(A./M)
    K.l .= A - Int(K.m)*K.h
    K
end

function MatToKMat(T::Type,A::AbstractArray)
    M = find_max_ops(T,max(size(A)...))
    MatToKMat(T,A,M)
end

function KaratsubaZeros(T,rows,cols,M)
    K = KaratsubaMatrix(zeros(T,rows,cols),zeros(T,rows,cols),M)
end

function KaratsubaZeros(T,length,M)
    K = KaratsubaVector(zeros(T,length),zeros(T,length),M)
end

import Base: +, -, *

function +(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("Cu", string(typeof(A)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.l),size(A.l)...),CUDA.zeros(eltype(A.l),size(A.l)...),A.m)
    else
        K = KaratsubaMatrix(zeros(eltype(A.l),size(A.l)...),zeros(eltype(A.l),size(A.l)...),A.m)
    end
    add!(K,A,B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("Cu", string(typeof(A)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.l),size(A.l)...),CUDA.zeros(eltype(A.l),size(A.l)...),A.m)
    else
        K = KaratsubaMatrix(zeros(eltype(A.l),size(A.l)...),zeros(eltype(A.l),size(A.l)...),A.m)
    end
    sub!(K,A,B)
end

function *(a::Number, A::KaratsubaArray)
    if occursin("Cu", string(typeof(A)))
        K = KaratsubaMatrix(CUDA.zeros(eltype(A.l),size(A.l)...),CUDA.zeros(eltype(A.l),size(A.l)...),A.m)
    else
        K = KaratsubaMatrix(zeros(eltype(A.l),size(A.l)...),zeros(eltype(A.l),size(A.l)...),A.m)
    end
    scalar_multiply!(K,A,a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function *(A::KaratsubaArray, B::KaratsubaArray)
    K = KaratsubaMatrix(zeros(eltype(A.l),size(A.l)[1],size(B.l)[2]),zeros(eltype(A.l),size(A.l)[1],size(B.l)[2]),A.m)
    plan = KMatMulPlan{Matrix{eltype(A.l)}}(zeros(eltype(A.l),size(A.l)...),zeros(eltype(B.l),size(B.l)...))
    KMatMul!(K,A,B,plan)
    K
end

function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.m != B.m) || (A.m != K.m)
        error("Matrices must have the same modulus m")
    end
    if (size(A.l) != size(B.l)) || (size(A.l) != (size(K.l)))
        error("Matrix dimensions must match")
    end

    K.h .= trunc.((A.l+B.l)./A.m)
    K.l .= A.l + B.l - A.m*K.h
    K.h .= K.h + A.h + B.h
    K
end

function sub!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.m != B.m) || (A.m != C.m)
        error("Matrices must have the same modulus m")
    end
    if (size(A.l) != size(B.l)) || (size(A.l) != (size(C.l)))
        error("Matrix dimensions must match")
    end

    C.h .= trunc.((A.l-B.l)./A.m)
    C.l .= A.l - B.l - A.m*C.h
    C.h .= C.h + A.h - B.h
    C
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if A.m != B.m
        error("Matrices must have the same modulus m")
    end
    if size(A.l) != size(B.l)
        error("Matrix dimensions must match")
    end

    B.h .= trunc.((s*A.l)/A.m)
    B.l .= s*A.l - B.m*B.h
    B.h .= B.h + s*A.h
end