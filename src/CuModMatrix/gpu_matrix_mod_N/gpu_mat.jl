
const TILE_WIDTH = 32
const DEFAULT_TYPE = Float32

struct CuModArraySizeMismatchException <: Exception
    message::String
end

struct CuModArrayModulusMismatchException <: Exception
    message::String
end

struct CuModMatrixTooLargeException <: Exception
    message::String
end

struct CuModMatrixNotSquareException <: Exception
    message::String
end

struct CuModMatrixModulusNotPrimeException <: Exception
    message::String
end

function mod_kernel!(data, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(data)
        data[i] = data[i] % N
    end
    return nothing
end

"""
    CuModArray{T}

A matrix over a finite field mod N, implemented using CuArray. The internal representation
pads to multiples of TILE_WIDTH (32) for efficient GPU computation.

Note matrices over the permitted size for efficient matmul may be created, but matmul 
will throw an exception if these matrices are multiplied.
"""
struct CuModArray{T,D} <: AbstractArray{T,D}
    data::CuArray{T, D}  # Padded CuArray data
    size::Dims{D}        # True dimensions
    N::Int               # The modulus N
    
    """
        CuModArray{T,D}(A::AbstractArray{T}, N)
    
    General contructor from abstract CPU array.
    Pads data on the GPU to fit a multiple of 32.
    """
    function CuModArray{T,D}(A::AbstractArray{S,D}, N::Int; mod=true, new_size=nothing) where {T,D,S}

        if 2^52 < N
            throw(CuModArrayModulusMismatchException(
                 "Modulus $N is bigger than 2^52 = $(2^52), the largest supported modulus"
                 ))
        end

        pad(d) = ceil(Int, d / TILE_WIDTH) * TILE_WIDTH

        padded_size = pad.(size(A))
        
        # Initialize the padded areas to zero
        data = CUDA.fill(zero(T),padded_size...) 
        
        A_inds = CartesianIndices(A)
        rangesize = map(x -> 1:x,size(A))
        data_inds = CartesianIndices(rangesize)

        # The desired behavior is for this to error if the conversion is
        # impossible
        converted = convert.(T,A)

        #TODO: We would like to use an in-place version:
        #`copyto!(data, data_inds, converted, A_inds)`
        #but this doesn't seem to be implemented in CUDA.jl.
        #For now, we can deal with a little extra (cpu) allocation 
        #in creation of matrices.
        data[data_inds] = converted[A_inds]
        
        if mod
            threads = 32
            blocks = cld(length(data), threads)
            @cuda threads=threads blocks=blocks mod_kernel!(data, N)
        end

        if new_size != nothing 
            new{T,D}(data, new_size, N)
        else
            new{T,D}(data, size(A), N)
        end
    end

    """
        CuModArray{T,D}(data::CuArray, N)

    Wrapper constructor for results already on the GPU.
    Does not pad.

    Thus, for some functionality to work properly, you must manage
    the padding yourself.

    This constructor is useful for creating a new CuModArray while 
    keeping the same data as a previous CuModarray.
    """
    function CuModArray{T,D}(A::CuArray{T, D}, N::Int; mod=false, new_size=nothing) where {T,D}
        data = A

        if new_size != nothing
            new{T,D}(data, new_size, N)
        else
            new{T,D}(data, size(data), N)
        end
    end
end

"""
    CuModArray(A::AbstractArray{T}, N)

General contructor from abstract CPU array.
Pads data on the GPU to fit a multiple of 32.
"""
function CuModArray(A::AbstractArray{T,D}, N::Int; mod=true, new_size=nothing, elem_type=DEFAULT_TYPE) where {T,D}
    CuModArray{elem_type,D}(A,N; mod=mod,new_size=new_size)
end

const CuModMatrix{T} = CuModArray{T,2}
const CuModVector{T} = CuModArray{T,1}

"""
    CuModMatrix(A::AbstractMatrix{T}, N)

General contructor from abstract CPU matrix.
Pads data on the GPU to fit a multiple of 32.
"""
function CuModMatrix(A::AbstractMatrix{T}, N::Int; mod=true, new_size=nothing, elem_type=DEFAULT_TYPE) where T
    CuModArray{elem_type,2}(A,N; mod=mod,new_size=new_size)
end

"""
    CuModMatrix(data::CuArray, N)

Wrapper constructor for results already on the GPU.
Does not pad.

Thus, for some functionality to work properly, you must manage
the padding yourself.

This constructor is useful for creating a new CuModMatrix while 
keeping the same data as a previous CuModMatrix.
"""
function CuModMatrix(A::CuArray{T, 2}, N::Int; mod=false, new_size=nothing) where {T}
    CuModArray{T,2}(A,N,mod=mod,new_size=new_size)
end

"""
    CuModVector(A::AbstractMatrix{T}, N)

General contructor from abstract CPU matrix.
Pads data on the GPU to fit a multiple of 32.
"""
function CuModVector(A::AbstractVector{T}, N::Int; mod=true, new_size=nothing,elem_type=DEFAULT_TYPE) where T
    CuModArray{elem_type,1}(A,N; mod=mod,new_size=new_size)
end

"""
    CuModVector(data::CuArray, N)

Wrapper constructor for results already on the GPU.
Does not pad.
"""
function CuModVector(A::CuArray{T, 1}, N::Int; mod=false, new_size=nothing) where {T}
    CuModArray{T,1}(A,N,mod=mod,new_size=new_size)
end

"""
    find_max_ops(type, N)

Returns the maximum number of operations before a N is necessary given a datatype and N N.
"""
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

Base.size(A::CuModArray) = A.size
Base.size(A::CuModArray,i::Int) = A.size[i]

rows(A::CuModArray{T,2}) where T = size(A)[1]
cols(A::CuModArray{T,2}) where T = size(A)[2]

Base.length(A::CuModArray{T,1}) where T = size(A)[1]
Base.length(A::CuModArray{T,2}) where T = rows(A)*cols(A)
Base.length(A::CuModArray) = prod(size(A))

# User needs to do CUDA.@allowscalar to use these
#
# Also, 3D indexing is currently not supported because we have no
# use cases.
Base.getindex(A::CuModArray, i::Int) = A.data[i]
Base.getindex(A::CuModArray, i::Int, j::Int) = A.data[i, j]

Base.setindex!(A::CuModArray, v, i::Int, j::Int) = A.data[i, j] = mod(v, A.N)
Base.setindex!(A::CuModArray, v, i::Int) = A.data[i] = mod(v, A.N)

Base.getindex(A::CuModArray, I::AbstractArray{Int}, J::AbstractArray{Int}) = 
    CuModArray(Array(A.data[I, J]), A.N)
Base.getindex(A::CuModArray, I::AbstractArray{Int}, j::Int) = 
    CuModArray(reshape(Array(A.data[I, j]), length(I), 1), A.N)
Base.getindex(A::CuModArray, i::Int, J::AbstractArray{Int}) = 
    CuModArray(reshape(Array(A.data[i, J]), 1, length(J)), A.N)

# Convert back to CPU array, with padding
# Unsafe because there is no guarantee that the padding is zero
function unsafe_Array(A::CuModArray)
    Array(A.data)
end

# Convert back to CPU array, without padding
function Base.Array(A::CuModArray) 
     
    rangesize = map(x -> 1:x,size(A))
    inds = CartesianIndices(rangesize)
    Array(A.data)[inds]
end

# Display function
function Base.show(io::IO, A::CuModArray)
    println("$(join(string.(size(A)),"×")) CuModArray modulo $(A.N):")

    # Note this does not display the padding
    println(Array(A))
end

function Base.show(io::IO, ::MIME"text/plain", A::CuModArray)
    println("$(join(string.(size(A)),"×")) CuModArray modulo $(A.N):")
    # Base.println_matrix(Array(A))
    remove_first_line(s) = join(split(s,"\n")[2:end], "\n")
    println(remove_first_line(repr("text/plain", Array(A))))
end

function Base.show(io::IO, A::CuModVector)
    println("$(join(string.(size(A)),"×")) CuModVector modulo $(A.N):")

    # Note this does not display the padding
    println(Array(A))
end

function Base.show(io::IO, ::MIME"text/plain", A::CuModVector)
    println("$(join(string.(size(A)),"×")) CuModVector modulo $(A.N):")
    # Base.println_matrix(Array(A))
    remove_first_line(s) = join(split(s,"\n")[2:end], "\n")
    println(remove_first_line(repr("text/plain", Array(A))))
end

function Base.show(io::IO, A::CuModMatrix)
    println("$(join(string.(size(A)),"×")) CuModMatrix modulo $(A.N):")

    # Note this does not display the padding
    println(Array(A))
end

function Base.show(io::IO, ::MIME"text/plain", A::CuModMatrix)
    println("$(join(string.(size(A)),"×")) CuModMatrix modulo $(A.N):")
    # Base.println_matrix(Array(A))
    remove_first_line(s) = join(split(s,"\n")[2:end], "\n")
    println(remove_first_line(repr("text/plain", Array(A))))
end

# Basic arithmetic operations with greedy reduction
import Base: +, -, *

function +(A::CuModArray, B::CuModArray)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data + B.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

function -(A::CuModArray, B::CuModArray)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data - B.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

# Scalar operations
function +(a::Number, A::CuModArray)
    result = mod.(a .+ A.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

function +(A::CuModArray, a::Number)
    a + A
end

function -(a::Number, A::CuModArray)
    result = mod.(a .- A.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

function -(A::CuModArray, a::Number)
    result = mod.(A.data .- a, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

function *(a::Number, A::CuModArray)
    result = mod.(a .* A.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

function *(A::CuModArray, a::Number)
    a * A
end

function *(A::CuModMatrix, B::CuModMatrix)
    if A.N != B.N
        throw(CuModArraySizeMismatchException(
            "Matrices must have the same modulus N"
        ))
    end
    # Note size checks etc. are done in mat_mul_gpu
    
    mat_mul_gpu_type(A, B)
end

function Base.broadcasted(::typeof(*), A::CuModArray, B::CuModArray)
    if A.N != B.N
        throw(CuModArrayModulusMismatchException(
            "Matrices must have the same modulus N."
        ))
    end

    # Note that this size check is done with the unpadded sizes
    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match for .*."
        ))
    end
    
    result = mod.(A.data .* B.data, A.N)
    
    CuModArray(result, A.N, new_size=size(A))
end

function -(A::CuModArray)
    result = mod.(-A.data, A.N)
    CuModArray(result, A.N, new_size=size(A))
end

# Recursive binary exponentiation algorithm
function Base.:^(A::CuModMatrix, n::Integer)
    if rows(A) != cols(A) 
        throw(CuModMatrixNotSquareException(
            "Matrix must be square for power operation"
        ))
    end
    
    if n == 0
        I_mat = Matrix{eltype(A.data)}(I, rows(A), cols(A))
        return CuModMatrix(I_mat, A.N)
    elseif n < 0
        return inverse(A)^(-n)
    else
        if n == 1
            return A
        elseif n % 2 == 0
            half_pow = A^(n ÷ 2)
            return half_pow * half_pow
        else
            return A * (A^(n-1))
        end
    end
end

"""
    is_invertible_with_inverse(A::CuModMatrix)

Checks if a matrix is invertible. If not, returns
false and nothing. If it is, returns true and the
inverse matrix.
"""
function is_invertible_with_inverse(A::CuModMatrix)
    if !is_invertible(A)
        return false, nothing
    end

    return true, inverse(A)
end

"""
    is_invertible(A::CuModMatrix)

Checks if a matrix is invertible mod N.
"""
function is_invertible(A::CuModMatrix)
    P, L, U, Q = plup_gpu_type(A)
    P = perm_array_to_matrix(P, A.N; new_size=(rows(A), rows(A)))
    Q = perm_array_to_matrix(Q, A.N; new_size=(cols(A), cols(A)))

    println("The GCD Matrix")
    println(Array(P*U))
    println(Array(U))
    println(Array(L))
    println(Array(P))
    println(Array(Q))
    println(Array(P*L*U*Q))
    for diag_elem in diag(Array(U))
        if gcd(diag_elem, A.N) > 1
            return false
        end
    end

    return true
end

function gcd(a,b)
    a, b = abs(a), abs(b)
    while b != 0
        a, b = b, a % b
    end
    return a
end

"""
    inverse(A::CuModMatrix)

Computes the inverse of a matrix mod N.
"""
function inverse(A::CuModMatrix)
    if !is_invertible(A)
        throw(MatrixNotInvertibleException(
            "Matrix is not invertible mod $(A.N)"
        ))
    end
    
    U, L, P, Q = plup_gpu_type(A)

    U_diag = diag(Array(U))
    rank = count(U_diag .!= 0)

    if r == 0
        return zeros(eltype(A.data), rows(A), cols(A))
    end

    L_new = @view L[:, 1:r]
    U_new = @view U[1:r, :]

    U_inv = backward_sub_gpu_type(U_new, true)
    L_inv = backward_sub_gpu_type(L_new, false)

    A_inv = Q * U_inv * L_inv * P

    return A_inv
end

# Additional utility functions
"""
    eye(T, n, N)

Creates an n×n identity matrix in the finite field.
"""
function eye(::Type{T}, n::Integer, N::Integer) where T
    padded_size = ceil(Int, n / TILE_WIDTH) * TILE_WIDTH
    CuModMatrix(Matrix{T}(I, n, n), N, new_size=(n, n))
end

"""
    zeros(T, length, N)

Creates a vector of zeros of the mod N ring. 
"""
function zeros(::Type{T}, length::Integer, N::Integer) where T
    padded_entries = ceil(Int, length / TILE_WIDTH) * TILE_WIDTH
    CuModVector(CUDA.zeros(T,padded_entries), N, new_size=(length,))
end

"""
    zeros(T, rows, cols, N)

Creates a matrix of zeros in the mod N ring.
"""
function zeros(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
    CuModMatrix(CUDA.zeros(T, padded_rows, padded_cols), N, new_size=(rows,cols))
end

"""
    rand(T, length, N)

Creates a vector of zeros of the mod N ring. 
"""
function rand(::Type{T}, length::Integer, N::Integer) where T
    padded_entries = ceil(Int, length / TILE_WIDTH) * TILE_WIDTH
    CuModVector(CUDA.rand(T,padded_entries), N, new_size=(length,))
end

"""
    rand(T, rows, cols, N)

Creates a random matrix with elements in the finite field.
"""
function rand(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
    # TODO: Here even the padding is nonzero.
    CuModMatrix(mod.(CUDA.rand(T, padded_rows, padded_cols), N), N, new_size=(rows,cols))
end

# In-place operations with optional modulus
"""
    add!(C, A, B, [mod_N])

In-place addition: C = A + B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function add!(C::CuModArray, A::CuModArray, B::CuModArray, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(CuModArrayModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException(
            "All matrix dimensions must match"
        ))
    end

    # TODO: Time them
    # TODO: figure out how to get the dots to fuse
    C.data .= mod.(A.data .+ B.data, N)
    #mod.(add!(C.data, A.data, B.data), N)

    return C
end

"""
    sub!(C, A, B, [mod_N])

In-place subtraction: C = A - B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.

"""
function sub!(C::CuModArray, A::CuModArray, B::CuModArray, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(CuModArrayModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException(
            "All matrix dimensions must match"
        ))
    end
    
    #TODO: fuse the dots
    C.data .= mod.(A.data .- B.data, N)
    #mod.(sub!(C.data, A.data, B.data), N)
    
    return C
end

"""
    elementwise_multiply!(C, A, B, [mod_N])

In-place element-wise multiplication: C = A .* B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function elementwise_multiply!(C::CuModArray, A::CuModArray, B::CuModArray, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(CuModArrayModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException(
            "All matrix dimensions must match"
        ))
    end
    
    @. C.data = mod(A.data * B.data, N)
    return C
end

"""
    negate!(B, A, [mod_N])

In-place negation: B = -A mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function negate!(B::CuModArray, A::CuModArray, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(CuModArrayModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end

    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match"
        ))
    end

    B.data .= mod.(.-A.data .+ N, N)
    return B
end

"""
    scalar_add!(B, A, s, [mod_N])

In-place scalar addition: B = A + s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_add!(B::CuModArray, A::CuModArray, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(CuModArrayModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match"
        ))
    end

    @. B.data = mod(A.data + s, N)
    return B
end

"""
    scalar_sub!(B, A, s, [mod_N])

In-place scalar subtraction: B = A - s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_sub!(B::CuModArray, A::CuModArray, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(CuModArrayModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    @. B.data = mod(A.data - s, N)
    return B
end

"""
    mul!(B, A, s, [mod_N])

In-place scalar multiplication: B = A * s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function LinearAlgebra.mul!(B::CuModArray, A::CuModArray, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(CuModArrayModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    @. B.data = mod(A.data * s, N)
    return B
end

"""
    rmul!(A,s)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!

TODO: make this and lmul! extend base
"""
rmul!(A::CuModArray, s::Number) = mul!(A,A,s)

"""
    lmul!(s,A)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!
"""
lmul!(s::Number,A::CuModArray) = mul!(A,A,s)

"""
    copy!(B, A)

In-place copy: B = A. No allocation is performed.
B is updated with the contents of A.

This does not test the modulus or normalize the entries.
"""
function Base.copy!(B::CuModArray, A::CuModArray)
    if size(A) != size(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    CUDA.copy!(B.data, A.data)
    return B
end

"""
    fill!(A::CuModArray{T,D}, s::T) where {T,D}

fills (in place) A with the value s (mod N)
"""
function Base.fill!(A::CuModArray, s::Number)
    smod = mod(s, A.N)

    @. A.data = smod
end

"""
    zero!(A::CuModArray)

Sets all of the entries of A to zero, in place.

"""
zero!(A::CuModArray) = CUDA.fill!(A.data, zero(eltype(A.data)))

"""
    mod_elements!(A, [mod_N])

Apply modulus to all elements of A in-place.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function mod_elements!(A::CuModArray, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    @. A.data = mod(A.data, N)
    return A
end

# Utility functions to change modulus
"""
    change_modulus(A, new_N)

Creates a new CuModArray with the same values as A but with a different modulus.
All elements are reduced modulo new_N.
"""
function change_modulus(A::CuModArray, new_N::Integer)
    (dataRows,dataCols) = size(A.data)
    result = GPUFiniteFieldMatrices.zeros(eltype(A.data), dataRows, dataCols, new_N)
    
    if new_N < A.N
        @. result.data = mod(A.data, new_N)
    else
        @. result.data = A.data
    end
     
    return CuModArray(result.data, new_N, new_size=size(A))
end

"""
    change_modulus!(A, new_N)

Changes the modulus of A in-place to new_N.
All elements are reduced modulo new_N.
"""
#TODO: we are not allowed to actually change an immutable struct.
#This returns a new struct, but modifies the underlying data
#
#This will allocate a litte bit the way the constructor is written right now.
function change_modulus_no_alloc!(A::CuModArray, new_N::Integer)
    if new_N < A.N
        @. A.data = mod(A.data, new_N)
    end

    return CuModArray(A.data,new_N,new_size=size(A))
end

"""
    mul!(C, A, B)

In-place matrix multiplication: C = A * B mod N.
"""
function LinearAlgebra.mul!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix)
    
    if (A.N != B.N || A.N != C.N)
        throw(CuModArrayModulusMismatchException(
            "All matrices must have the same modulus N"
        ))
    end
    if cols(A) != rows(B)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions do not match for multiplication"
        ))
    end
    if rows(C) != rows(A) || cols(C) != cols(B)
        throw(CuModArraySizeMismatchException(
            "Output matrix C has incorrect dimensions"
        ))
    end
    
    stripe_mul!(C, A, B)
    return C
end

"""
    mul!(z, A, x)

In-place matrix-vector multiplication: z = A * x mod N.
"""
function LinearAlgebra.mul!(z::CuModVector, A::CuModMatrix, x::CuModVector)
    
    if (A.N != z.N || A.N != x.N)
        throw(CuModArrayModulusMismatchException(
            "All matrices must have the same modulus N"
        ))
    end
    if cols(A) != length(x)
        throw(CuModArraySizeMismatchException(
            "Matrix dimensions do not match for multiplication"
        ))
    end
    if length(z) != rows(A) 
        throw(CuModArraySizeMismatchException(
            "Output matrix C has incorrect dimensions"
        ))
    end
    
    stripe_mul!(z, A, x)
    return z
end

#TODO: addmul! (add and scalar multiply), gemv!, gemm!
#
#Note: gemv! and gemm! could be more efficient by incorporating the add into the 
#CUBLAS gemm/gemv that happens in stripe_mul!
