#using CUDA, LinearAlgebra

const TILE_WIDTH = 32
const DEFAULT_TYPE = Float32

struct MatrixTooLargeException <: Exception
    message::String
end

struct MatrixSizeMismatchException <: Exception
    message::String
end

struct MatrixNotSquareException <: Exception
    message::String
end

struct MatrixModulusMismatchException <: Exception
    message::String
end

struct MatrixModulusNotPrimeException <: Exception
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
    GpuMatrixModN{T}

A matrix over a finite field mod N, implemented using CuArray. The internal representation
pads to multiples of TILE_WIDTH (32) for efficient GPU computation.

Note matrices over the permitted size for efficient matmul may be created, but matmul 
will throw an exception if these matrices are multiplied.
"""
struct GpuMatrixModN{T}
    data::CuArray{T, 2}  # Padded CuArray data
    rows::Int            # Actual row count
    cols::Int            # Actual column count
    N::Int               # The modulus N
    
    """
        GpuMatrixModN(A::AbstractMatrix{T}, N)
    
    General contructor from abstract CPU matrix.
    """
    function GpuMatrixModN(A::AbstractMatrix{T}, N::Int; elem_type::DataType=DEFAULT_TYPE, mod=true, new_rows=nothing, new_cols=nothing) where T
        rows, cols = size(A)
        
        padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
        padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
        
        data = CUDA.CuArray{T}(undef, (padded_rows, padded_cols))
        
        # Initialize the padded areas to zero
        data .= T(0)
        
        A_inds = CartesianIndices(A)
        data_inds = CartesianIndices((1:rows, 1:cols))
        copyto!(data, data_inds, A, A_inds)
        
        if mod
            threads = 32
            blocks = cld(length(data), threads)
            @cuda threads=threads blocks=blocks mod_kernel!(data, N)
        end

        if new_rows != nothing && new_cols != nothing
            new{T}(data, new_rows, new_cols, N)
        else
            new{T}(data, rows, cols, N)
        end
    end

    """
        GpuMatrixModN(data::CuArray, N)

    Wrapper constructor for results already on the GPU.
    """
    function GpuMatrixModN(A::CuArray{T, 2}, N::Int; elem_type::DataType=DEFAULT_TYPE, mod=false, new_rows=nothing, new_cols=nothing) where T
        data = A
        rows, cols = size(data)
    
        if new_rows != nothing && new_cols != nothing
            new{T}(data, new_rows, new_cols, N)
        else
            new{T}(data, rows, cols, N)
        end
    end
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

Base.size(A::GpuMatrixModN) = (A.rows, A.cols)

# User needs to do CUDA.@allowscalar to use these
Base.getindex(A::GpuMatrixModN, i::Int, j::Int) = A.data[i, j]
Base.setindex!(A::GpuMatrixModN, v, i::Int, j::Int) = A.data[i, j] = mod(v, A.N)
Base.getindex(A::GpuMatrixModN, I::AbstractArray{Int}, J::AbstractArray{Int}) = 
    GpuMatrixModN(Array(A.data[I, J]), A.N)
Base.getindex(A::GpuMatrixModN, I::AbstractArray{Int}, j::Int) = 
    GpuMatrixModN(reshape(Array(A.data[I, j]), length(I), 1), A.N)
Base.getindex(A::GpuMatrixModN, i::Int, J::AbstractArray{Int}) = 
    GpuMatrixModN(reshape(Array(A.data[i, J]), 1, length(J)), A.N)

# Convert back to CPU array, with padding
# Unsafe because there is no guarantee that the padding is zero
function unsafe_Array(A::GpuMatrixModN)
    Array(A.data)
end

# Convert back to CPU array, without padding
Base.Array(A::GpuMatrixModN) = Array(A.data)[1:A.rows, 1:A.cols]

# Display function
function Base.show(io::IO, A::GpuMatrixModN)
    println(io, "$(A.rows)×$(A.cols) GpuMatrixModN modulo $(A.N):")

    # Note this does not display the padding
    show(io, Array(A))
end

function Base.display(A::GpuMatrixModN)
    println("$(A.rows)×$(A.cols) GpuMatrixModN modulo $(A.N):")
    println(Array(A))
end

# Basic arithmetic operations with delayed reduction
import Base: +, -, *

function +(A::GpuMatrixModN, B::GpuMatrixModN)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data + B.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function -(A::GpuMatrixModN, B::GpuMatrixModN)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data - B.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

# Scalar operations
function +(a::Number, A::GpuMatrixModN)
    result = mod.(a .+ A.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function +(A::GpuMatrixModN, a::Number)
    a + A
end

function -(a::Number, A::GpuMatrixModN)
    result = mod.(a .- A.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function -(A::GpuMatrixModN, a::Number)
    result = mod.(A.data .- a, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function *(a::Number, A::GpuMatrixModN)
    result = mod.(a .* A.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function *(A::GpuMatrixModN, a::Number)
    a * A
end

function *(A::GpuMatrixModN, B::GpuMatrixModN)
    if A.N != B.N
        throw(MatrixSizeMismatchException(
            "Matrices must have the same modulus N"
        ))
    end
    # Note size checks etc. are done in mat_mul_gpu
    
    mat_mul_gpu_type(A, B)
end

function Base.broadcasted(::typeof(*), A::GpuMatrixModN, B::GpuMatrixModN)
    if A.N != B.N
        throw(MatrixModulusMismatchException(
            "Matrices must have the same modulus N."
        ))
    end

    # Note that this size check is done with the unpadded sizes
    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match for .*."
        ))
    end
    
    result = mod.(A.data .* B.data, A.N)
    
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

function -(A::GpuMatrixModN)
    result = mod.(-A.data, A.N)
    GpuMatrixModN(result, A.N, new_rows = A.rows, new_cols = A.cols)
end

# Recursive binary exponentiation algorithm
function Base.:^(A::GpuMatrixModN, n::Integer)
    if A.rows != A.cols
        throw(MatrixNotSquareException(
            "Matrix must be square for power operation"
        ))
    end
    
    if n == 0
        I_mat = Matrix{eltype(A.data)}(I, A.rows, A.cols)
        return GpuMatrixModN(I_mat, A.N)
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
    is_invertible_with_inverse(A::GpuMatrixModN)

Checks if a matrix is invertible. If not, returns
false and nothing. If it is, returns true and the
inverse matrix.
"""
function is_invertible_with_inverse(A::GpuMatrixModN)
    if !is_invertible(A)
        return false, nothing
    end
    return true, inverse(A)
end

"""
    is_invertible(A::GpuMatrixModN)

Checks if a matrix is invertible mod N.
"""
function is_invertible(A::GpuMatrixModN{T}) where T
    # assuming isprime(A.N)
    if A.rows != A.cols
        return false
    end
    return true
end

"""
    inverse(A::GpuMatrixModN)

Computes the inverse of a matrix mod N.
"""
function inverse(A::GpuMatrixModN)
    if !is_invertible(A)
        throw(MatrixNotInvertibleException(
            "Matrix is not invertible mod $(A.N)"
        ))
    end
    
    # Find PLUQ decomposition and find inverse of each component
    U, L, P_rows, P_cols = plup_gpu_type(A)
    n = A.rows

    # Convert permutation vectors to matrices
    P = zeros(Int, n, n)
    Q = zeros(Int, n, n)
    for i in 1:n
        P[P_rows[i], i] = 1
        Q[i, P_cols[i]] = 1
    end

    # Calculate L⁻¹
    L_inv = CUDA.ones(Int, n, n)
    L_inv = CUDA.tril(L_inv) 
    
    # Using iterative approach for L⁻¹
    for k in 2:n
        L_sub = L_gpu[k, 1:k-1]
        L_inv_sub = L_inv[1:k-1, 1:k-1]
        new_row = -(L_sub * L_inv_sub)
        L_inv[k, 1:k-1] = mod.(new_row, A.N)
    end
    
    # Calculate U⁻¹ (U is upper triangular)
    U_inv = CUDA.zeros(Int, n, n)
    diag_U = CUDA.diag(U_gpu)
    # a^(p-2) ≡ a^(-1) (mod p) for prime p
    diag_U_inv = mod.(powermod.(diag_U, A.N-2, A.N), A.N)
    for i in 1:n
        U_inv[i, i] = diag_U_inv[i]
    end
    
    # Using iterative approach for U⁻¹
    for k in (n-1):-1:1
        for j in (k+1):n
            sum_term = CUDA.sum(U_gpu[k, (k+1):j] .* U_inv[(k+1):j, j])
            U_inv[k, j] = mod(-sum_term * U_inv[k, k], A.N)
        end
    end
    
    # Calculate inverse: A⁻¹ = Qᵀ·U⁻¹·L⁻¹·Pᵀ
    UL_inv = mod.(U_inv * L_inv, A.N)
    A_inv = mod.(CUDA.transpose(Q) * UL_inv * CUDA.transpose(P), A.N)

    GpuMatrixModN(A_inv, A.N, new_rows=n, new_cols=n)
end

# Additional utility functions
"""
    identity(T, n, N)

Creates an n×n identity matrix in the finite field.
"""
function Base.identity(::Type{T}, n::Integer, N::Integer) where T
    padded_size = ceil(Int, n / TILE_WIDTH) * TILE_WIDTH
    GpuMatrixModN(Matrix{T}(I, n, n), N, new_rows=n, new_cols=n)
end

# function Base.identity(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
#     padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
#     padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
#     unsafe_GpuMatrixModN(CUDA.identity(T, padded_rows, padded_cols), N, new_rows=rows, new_cols=cols)
# end

"""
    zeros(T, rows, cols, N)

Creates a matrix of zeros in the finite field.
"""
function zeros(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
    GpuMatrixModN(CUDA.zeros(T, padded_rows, padded_cols), N, new_rows=rows, new_cols=cols)
end

"""
    rand(T, rows, cols, N)

Creates a random matrix with elements in the finite field.
"""
function rand(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
    padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
    # TODO: Here even the padding is nonzero.
    GpuMatrixModN(mod.(CUDA.rand(T, padded_rows, padded_cols), N), N, new_rows=rows, new_cols=cols)
end

# In-place operations with optional modulus
"""
    add!(C, A, B, [mod_N])

In-place addition: C = A + B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function add!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(MatrixModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(MatrixSizeMismatchException(
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
function sub!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(MatrixModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(MatrixSizeMismatchException(
            "All matrix dimensions must match"
        ))
    end
    
    #TODO: fuse the dots
    C.data .= mod.(A.data - B.data, N)
    #mod.(sub!(C.data, A.data, B.data), N)
    
    return C
end

"""
    elementwise_multiply!(C, A, B, [mod_N])

In-place element-wise multiplication: C = A .* B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function elementwise_multiply!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(MatrixModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(MatrixSizeMismatchException(
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
function negate!(B::GpuMatrixModN, A::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(MatrixModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end

    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match"
        ))
    end

    B.data .= mod.(-A.data .+ N, N)
    return B
end

"""
    scalar_add!(B, A, s, [mod_N])

In-place scalar addition: B = A + s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_add!(B::GpuMatrixModN, A::GpuMatrixModN, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(MatrixModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match"
        ))
    end

    @. B.data = mod(A.data + s, N)
    return B
end

"""
    scalar_subtract!(B, A, s, [mod_N])

In-place scalar subtraction: B = A - s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_subtract!(B::GpuMatrixModN, A::GpuMatrixModN, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(MatrixModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    @. B.data = mod(A.data - s, N)
    return B
end

"""
    scalar_multiply!(B, A, s, [mod_N])

In-place scalar multiplication: B = A * s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_multiply!(B::GpuMatrixModN, A::GpuMatrixModN, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        throw(MatrixModulusMismatchException(
            "Both matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    @. B.data = mod(A.data * s, N)
    return B
end

"""
    multiply!(C, A, B, [mod_N])

In-place matrix multiplication: C = A * B mod N.
If mod_N is provided, it will be used instead of C.N for the modulus.
This still involves some allocation internally due to the use of mat_mul_gpu.
"""
function multiply!(C::GpuMatrixModN, A::GpuMatrixModN, B::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        throw(MatrixModulusMismatchException(
            "All matrices must have the same modulus N or provide an override mod_N"
        ))
    end
    if A.cols != B.rows
        throw(MatrixSizeMismatchException(
            "Matrix dimensions do not match for multiplication"
        ))
    end
    if C.rows != A.rows || C.cols != B.cols
        throw(MatrixSizeMismatchException(
            "Output matrix C has incorrect dimensions"
        ))
    end
    
    mat_mul_gpu_type(A, B, N, C=C)
    return C
end

"""
    copy!(B, A)

In-place copy: B = A. No allocation is performed.
B is updated with the contents of A.
"""
function copy!(B::GpuMatrixModN, A::GpuMatrixModN)
    if size(A) != size(B)
        throw(MatrixSizeMismatchException(
            "Matrix dimensions must match"
        ))
    end
    
    CUDA.copy!(B.data, A.data)
    return B
end

"""
    mod_elements!(A, [mod_N])

Apply modulus to all elements of A in-place.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function mod_elements!(A::GpuMatrixModN, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    @. A.data = mod(A.data, N)
    return A
end

# Utility functions to change modulus
"""
    change_modulus(A, new_N)

Creates a new GpuMatrixModN with the same values as A but with a different modulus.
All elements are reduced modulo new_N.
"""
function change_modulus(A::GpuMatrixModN, new_N::Integer)
    (dataRows,dataCols) = size(A.data)
    result = GPUFiniteFieldMatrices.zeros(eltype(A.data), dataRows, dataCols, new_N)
    
    if new_N < A.N
        @. result.data = mod(A.data, new_N)
    else
        @. result.data = A.data
    end
     
    return GpuMatrixModN(result.data, new_N, new_rows=A.rows, new_cols=A.cols)
end

"""
    change_modulus!(A, new_N)

Changes the modulus of A in-place to new_N.
All elements are reduced modulo new_N.
"""
#TODO: we are not allowed to actually change an immutable struct.
#This returns a new struct, but modifies the underlying data
function change_modulus_no_alloc!(A::GpuMatrixModN, new_N::Integer)
    if new_N < A.N
        @. A.data = mod(A.data, new_N)
    end

    return GpuMatrixModN(A.data,new_N,new_rows=A.rows,new_cols=A.cols)
end

function backsubstitution_shared_kernel(U::CuArray{T, 2}, x, b, N) where T
    tid = threadIdx().x
    bid = blockIdx().x
    n = size(U, 1)
    
    shared_sum = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))
    
    for i = n:-1:1
        if tid == 1 && bid == 1
            sum = b[i]
            for j = i+1:n
                sum = (sum - (U[i, j] * x[j]) % N) % N
                if sum < 0
                    sum += N
                end
            end
            
            diag_inv = mod_inv(U[i, i], N)
            x[i] = (sum * diag_inv) % N
        end
        
        CUDA.sync_threads()
    end
    
    return nothing
end

function backsubstitution_optimized_gpu(U::GpuMatrixModN{T}, b) where T
    n = U.rows
    N = U.N
    x = CUDA.zeros(Int, n)
    d_b = CuArray(b)
    
    @cuda threads=1 blocks=1 backsubstitution_shared_kernel(U.data, x, d_b, N)
    
    return Array(x)
end

function backsubstitution_parallel_kernel(U::CuArray{T, 2}, x, b, N) where T
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(U, 1)
    
    if tid <= n
        for i = n:-1:1
            CUDA.sync_blocks()
            
            if tid == i
                sum = b[i]
                for j = i+1:n
                    sum = (sum - (U[i, j] * x[j]) % N) % N
                    if sum < 0
                        sum += N
                    end
                end
                
                diag_inv = mod_inv(U[i, i], N)
                x[i] = (sum * diag_inv) % N
            end
        end
    end
    
    return nothing
end

function backsubstitution_fully_parallel_gpu(U::GpuMatrixModN{T}, b) where T
    n = U.rows
    N = U.N
    x = CUDA.zeros(Int, n)
    d_b = CuArray(b)
    
    threads_per_block = min(256, n)
    num_blocks = ceil(Int, n / threads_per_block)
    
    @cuda threads=threads_per_block blocks=num_blocks backsubstitution_parallel_kernel(U.data, x, d_b, N)
    
    return Array(x)
end

function hensel_pseudoinverse(p, precision, A, T0)
    i = 1
    T = T0
    while i < precision
        T = 2*T - T * (A*T)
        i *= 2
    end
    R, pi = residue_ring(ZZ, p^precision)
    return matrix(R, [R(x) for x in Array(T)])
end
