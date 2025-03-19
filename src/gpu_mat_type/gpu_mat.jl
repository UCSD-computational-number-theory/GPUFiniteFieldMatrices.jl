using CUDA, LinearAlgebra

const TILE_WIDTH = 32
const DEFAULT_TYPE = Float32

"""
    GPUFiniteFieldMatrix{T}

A matrix over a finite field mod N, implemented using CuArray. The internal representation
pads to multiples of TILE_WIDTH (32) for efficient GPU computation.
"""
struct GPUFiniteFieldMatrix{T}
    data::CuArray{T, 2}  # Padded CuArray data
    rows::Int            # Actual row count
    cols::Int            # Actual column count
    N::Int               # The modulus N
    
    function GPUFiniteFieldMatrix(A::AbstractMatrix{T}, N::Int, elem_type::DataType=DEFAULT_TYPE) where T
        rows, cols = size(A)
        
        # Check if matrix multiplication would overflow
        max_ops = find_max_ops(elem_type, N)
        if rows > max_ops
            error("Matrix size exceeds maximum safe size for N $N with $elem_type. Max row count: $max_ops")
        end
        
        padded_rows = ceil(Int, rows / TILE_WIDTH) * TILE_WIDTH
        padded_cols = ceil(Int, cols / TILE_WIDTH) * TILE_WIDTH
        
        data = CUDA.CuArray{T}(undef, (padded_rows, padded_cols))
        
        # Initialize the padded areas to zero
        data .= T(0)
        
        A_inds = CartesianIndices(A)
        data_inds = CartesianIndices((1:rows, 1:cols))
        copyto!(data, data_inds, A, A_inds)
        
        # TODO: Do we need this?
        # data .= mod.(data, N)
        
        new{T}(data, rows, cols, N)
    end
    
    # New constructor that takes a CuArray directly without allocation
    function GPUFiniteFieldMatrix(data::CuArray{T, 2}, rows::Int, cols::Int, N::Int) where T
        # TODO: Check if data is already padded properly for GPU computation
        # Right now, only our functions use this, and they all pad to 32 already.

        # Check if matrix multiplication would overflow
        max_ops = find_max_ops(T, N)
        if rows > max_ops
            error("Matrix size exceeds maximum safe size for N $N with $T. Max row count: $max_ops")
        end
        
        # Assume data is already padded properly for GPU computation
        new{T}(data, rows, cols, N)
    end
end

"""
    find_max_ops(type, N)

Returns the maximum number of operations before a N is necessary given a datatype and N N.
"""
function find_max_ops(type, N)
    # I just copied this from the old code

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

    return floor(Int, (2^bits - 1) / N^2) - 1
end

# Get size
Base.size(A::GPUFiniteFieldMatrix) = (A.rows, A.cols)

# Technically, these can be implemented by enabling scalar indexing
# so I put them in, but they really should never be used.
Base.getindex(A::GPUFiniteFieldMatrix, i::Int, j::Int) = 
@CUDA.@allowscalar A.data[i, j]
Base.setindex!(A::GPUFiniteFieldMatrix, v, i::Int, j::Int) = 
@CUDA.@allowscalar A.data[i, j] = mod(v, A.N)

# Same as above; these really shouldn't be used outright.
Base.getindex(A::GPUFiniteFieldMatrix, I::AbstractArray{Int}, J::AbstractArray{Int}) = 
    GPUFiniteFieldMatrix(Array(A.data[I, J]), A.N)

Base.getindex(A::GPUFiniteFieldMatrix, I::AbstractArray{Int}, j::Int) = 
    GPUFiniteFieldMatrix(reshape(Array(A.data[I, j]), length(I), 1), A.N)

Base.getindex(A::GPUFiniteFieldMatrix, i::Int, J::AbstractArray{Int}) = 
    GPUFiniteFieldMatrix(reshape(Array(A.data[i, J]), 1, length(J)), A.N)

# Display function
function Base.show(io::IO, A::GPUFiniteFieldMatrix)
    println(io, "$(A.rows)×$(A.cols) GPUFiniteFieldMatrix modulo $(A.N):")

    # Only print the actual data, not the padding
    show(io, Array(A.data[1:A.rows, 1:A.cols]))
end

# Convert back to CPU array, without padding
Base.Array(A::GPUFiniteFieldMatrix) = Array(A.data[1:A.rows, 1:A.cols])

# Basic arithmetic operations with delayed reduction
import Base: +, -, *, /

function +(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data[1:A.rows, 1:A.cols] + B.data[1:B.rows, 1:B.cols], A.N)
    
    return GPUFiniteFieldMatrix(result, A.N)
end

function -(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix)
    if A.N != B.N
        error("Matrices must have the same modulus N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data[1:A.rows, 1:A.cols] - B.data[1:B.rows, 1:B.cols], A.N)
    
    return GPUFiniteFieldMatrix(result, A.N)
end

# Scalar operations
function +(a::Number, A::GPUFiniteFieldMatrix)
    result = mod.(a .+ A.data[1:A.rows, 1:A.cols], A.N)
    return GPUFiniteFieldMatrix(result, A.N)
end

function +(A::GPUFiniteFieldMatrix, a::Number)
    return a + A
end

function -(a::Number, A::GPUFiniteFieldMatrix)
    result = mod.(a .- A.data[1:A.rows, 1:A.cols], A.N)
    return GPUFiniteFieldMatrix(result, A.N)
end

function -(A::GPUFiniteFieldMatrix, a::Number)
    result = mod.(A.data[1:A.rows, 1:A.cols] .- a, A.N)
    return GPUFiniteFieldMatrix(result, A.N)
end

function *(a::Number, A::GPUFiniteFieldMatrix)
    result = mod.(a .* A.data[1:A.rows, 1:A.cols], A.N)
    return GPUFiniteFieldMatrix(result, A.N)
end

function *(A::GPUFiniteFieldMatrix, a::Number)
    return a * A
end

# Matrix multiplication using the existing mat_mul_gpu function
function *(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix)
    if A.N != B.N
        error("Matrices must have the same N")
    end
    if A.cols != B.rows
        error("Matrix dimensions do not match for multiplication")
    end
    
    return matmul_gpu_direct(A, B)
end

function .*(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix)
    if A.N != B.N
        error("Matrices must have the same N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    result = mod.(A.data[1:A.rows, 1:A.cols] .* B.data[1:B.rows, 1:B.cols], A.N)
    
    return GPUFiniteFieldMatrix(result, A.N)
end

function -(A::GPUFiniteFieldMatrix)
    result = mod.(-A.data[1:A.rows, 1:A.cols], A.N)
    return GPUFiniteFieldMatrix(result, A.N)
end

function Base.:^(A::GPUFiniteFieldMatrix, n::Integer)
    if A.rows != A.cols
        error("Matrix must be square for power operation")
    end
    
    if n == 0
        I_mat = Matrix{eltype(A.data)}(I, A.rows, A.cols)
        return GPUFiniteFieldMatrix(I_mat, A.N)
    elseif n < 0
        return inverse(A)^(-n)
    else
        # Binary exponentiation alg
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
    is_invertible(A::GPUFiniteFieldMatrix)

Checks if a matrix is invertible mod p (only works if N is prime).
Returns a boolean indicating invertibility.
"""
function is_invertible(A::GPUFiniteFieldMatrix)
    if A.rows != A.cols
        return false  # Only square matrices can be invertible
    end
    
    # Compute PLUP decomposition
    U, L, P_rows, P_cols = plup_gpu(Array(A), A.N)
    
    # Check if all diagonal elements of U are non-zero
    for i in 1:A.rows
        if U[i, i] == 0
            return false
        end
    end
    
    return true
end

"""
    inverse(A::GPUFiniteFieldMatrix)

Computes the inverse of a matrix mod p (only works if N is prime).
Throws an error if the matrix is not square (this is not a pseudo-inverse).
"""
function inverse(A::GPUFiniteFieldMatrix)
    if !is_invertible(A)
        error("Matrix is not invertible mod $(A.N)")
    end
    
    if A.rows != A.cols
        error("Only square matrices can be inverted")
    end
    
    n = A.rows
    
    augmented = zeros(Int, n, 2*n)
    A_array = Array(A)
    
    for i in 1:n
        for j in 1:n
            augmented[i, j] = A_array[i, j]
        end
        augmented[i, i+n] = 1
    end
    
    # Use the direct rref function that returns a GPUFiniteFieldMatrix
    result = rref_gpu_direct(augmented, A.N)
    
    # Extract the inverse matrix
    return result[1:n, (n+1):(2*n)]
end

# Additional utility functions
"""
    identity(T, n, N)

Creates an n×n identity matrix in the finite field.
"""
function identity(::Type{T}, n::Integer, N::Integer) where T
    I_mat = Matrix{T}(I, n, n)
    return GPUFiniteFieldMatrix(I_mat, N)
end

"""
    zeros(T, rows, cols, N)

Creates a matrix of zeros in the finite field.
"""
function zeros(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    Z_mat = zeros(T, rows, cols)
    return GPUFiniteFieldMatrix(Z_mat, N)
end

"""
    rand(T, rows, cols, N)

Creates a random matrix with elements in the finite field.
"""
function rand(::Type{T}, rows::Integer, cols::Integer, N::Integer) where T
    R_mat = rand(T(0):T(N-1), rows, cols)
    return GPUFiniteFieldMatrix(R_mat, N)
end

# In-place operations with optional modulus
"""
    add!(C, A, B, [mod_N])

In-place addition: C = A + B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function add!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        error("All matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B) || size(A) != size(C)
        error("All matrix dimensions must match")
    end
    
    # Perform addition and apply modulus in-place
    C.data[1:C.rows, 1:C.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .+ B.data[1:B.rows, 1:B.cols], N)
    
    return C
end

"""
    subtract!(C, A, B, [mod_N])

In-place subtraction: C = A - B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function subtract!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        error("All matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B) || size(A) != size(C)
        error("All matrix dimensions must match")
    end
    
    # Perform subtraction and apply modulus in-place
    C.data[1:C.rows, 1:C.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .- B.data[1:B.rows, 1:B.cols], N)
    
    return C
end

"""
    elementwise_multiply!(C, A, B, [mod_N])

In-place element-wise multiplication: C = A .* B mod N. No allocation is performed.
If mod_N is provided, it will be used instead of C.N for the modulus.
"""
function elementwise_multiply!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        error("All matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B) || size(A) != size(C)
        error("All matrix dimensions must match")
    end
    
    # Perform element-wise multiplication and apply modulus in-place
    C.data[1:C.rows, 1:C.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .* B.data[1:B.rows, 1:B.cols], N)
    
    return C
end

"""
    negate!(B, A, [mod_N])

In-place negation: B = -A mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function negate!(B::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        error("Both matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    # Perform negation and apply modulus in-place
    B.data[1:B.rows, 1:B.cols] .= mod.(-A.data[1:A.rows, 1:A.cols], N)
    
    return B
end

"""
    scalar_add!(B, A, s, [mod_N])

In-place scalar addition: B = A + s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_add!(B::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        error("Both matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    # Perform scalar addition and apply modulus in-place
    B.data[1:B.rows, 1:B.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .+ s, N)
    
    return B
end

"""
    scalar_subtract!(B, A, s, [mod_N])

In-place scalar subtraction: B = A - s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_subtract!(B::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        error("Both matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    # Perform scalar subtraction and apply modulus in-place
    B.data[1:B.rows, 1:B.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .- s, N)
    
    return B
end

"""
    scalar_multiply!(B, A, s, [mod_N])

In-place scalar multiplication: B = A * s mod N. No allocation is performed.
If mod_N is provided, it will be used instead of B.N for the modulus.
"""
function scalar_multiply!(B::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, s::Number, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : B.N
    
    if mod_N <= 0 && A.N != B.N
        error("Both matrices must have the same modulus N or provide an override mod_N")
    end
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    # Perform scalar multiplication and apply modulus in-place
    B.data[1:B.rows, 1:B.cols] .= mod.(A.data[1:A.rows, 1:A.cols] .* s, N)
    
    return B
end

"""
    multiply!(C, A, B, [mod_N])

In-place matrix multiplication: C = A * B mod N.
If mod_N is provided, it will be used instead of C.N for the modulus.
This still involves some allocation internally due to the use of mat_mul_gpu.
"""
function multiply!(C::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : C.N
    
    if mod_N <= 0 && (A.N != B.N || A.N != C.N)
        error("All matrices must have the same modulus N or provide an override mod_N")
    end
    if A.cols != B.rows
        error("Matrix dimensions do not match for multiplication")
    end
    if C.rows != A.rows || C.cols != B.cols
        error("Output matrix C has incorrect dimensions")
    end
    
    # Use the existing mat_mul_gpu function with the specified modulus
    # and copy the result to C
    result = mat_mul_gpu(Array(A), Array(B), N)
    
    # Copy the result to C
    result_inds = CartesianIndices(result)
    C_inds = CartesianIndices((1:C.rows, 1:C.cols))
    copyto!(C.data, C_inds, result, result_inds)
    
    return C
end

"""
    copy!(B, A)

In-place copy: B = A. No allocation is performed.
B is updated with the contents of A.
"""
function copy!(B::GPUFiniteFieldMatrix, A::GPUFiniteFieldMatrix)
    if size(A) != size(B)
        error("Matrix dimensions must match")
    end
    
    # Copy data from A to B
    B.data[1:B.rows, 1:B.cols] .= A.data[1:A.rows, 1:A.cols]
    
    return B
end

"""
    mod_elements!(A, [mod_N])

Apply modulus to all elements of A in-place.
If mod_N is provided, it will be used instead of A.N for the modulus.
"""
function mod_elements!(A::GPUFiniteFieldMatrix, mod_N::Integer=-1)
    N = mod_N > 0 ? mod_N : A.N
    
    A.data[1:A.rows, 1:A.cols] .= mod.(A.data[1:A.rows, 1:A.cols], N)
    return A
end

# Utility functions to change modulus
"""
    change_modulus(A, new_N)

Creates a new GPUFiniteFieldMatrix with the same values as A but with a different modulus.
All elements are reduced modulo new_N.
"""
function change_modulus(A::GPUFiniteFieldMatrix, new_N::Integer)
    result = zeros(eltype(A.data), A.rows, A.cols, new_N)
    
    # Copy data and apply new modulus
    result.data[1:result.rows, 1:result.cols] .= mod.(A.data[1:A.rows, 1:A.cols], new_N)
    
    return result
end

"""
    change_modulus!(A, new_N)

Changes the modulus of A in-place to new_N.
All elements are reduced modulo new_N.
"""
function change_modulus!(A::GPUFiniteFieldMatrix, new_N::Integer)
    # Check if multiplication would overflow with the new modulus
    max_ops = find_max_ops(eltype(A.data), new_N)
    if A.rows > max_ops
        error("Matrix size exceeds maximum safe size for modulus $new_N with $(eltype(A.data)). Max row count: $max_ops")
    end
    
    # Apply new modulus to all elements
    A.data[1:A.rows, 1:A.cols] .= mod.(A.data[1:A.rows, 1:A.cols], new_N)
    
    # Update the modulus
    setfield!(A, :N, new_N)
    
    return A
end

# Direct matrix multiplication that returns GPUFiniteFieldMatrix
"""
    matmul_gpu_direct(A, B)

Matrix multiplication that directly returns a GPUFiniteFieldMatrix without intermediate Array conversion.
"""
function matmul_gpu_direct(A::GPUFiniteFieldMatrix, B::GPUFiniteFieldMatrix)
    if A.N != B.N
        error("Matrices must have the same modulus")
    end
    if A.cols != B.rows
        error("Matrix dimensions do not match for multiplication")
    end
    
    # Define dimensions for the result
    result_rows = A.rows
    result_cols = B.cols
    
    # Calculate number of tiles for each dimension
    A_padded_rows = ceil(Int, A.rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A.cols / TILE_WIDTH) * TILE_WIDTH
    B_padded_cols = ceil(Int, B.cols / TILE_WIDTH) * TILE_WIDTH
    
    # Create an output GPUFiniteFieldMatrix with proper dimensions
    t = eltype(A.data)
    result = zeros(t, result_rows, result_cols, A.N)
    
    # Modify the internal padded dimensions to match what's needed for matmul
    result_data_padded = CUDA.CuArray{t}(undef, (A_padded_rows, B_padded_cols))
    
    # Call the appropriate kernel based on the regime
    # For simplicity, we'll just use mat_mul_plain for now and later add support for other regimes
    d_C = A.data * B.data
    d_C = mod.(d_C, A.N)
    
    # Copy results to our properly sized matrix
    result_inds = CartesianIndices((1:result_rows, 1:result_cols))
    d_C_inds = CartesianIndices((1:result_rows, 1:result_cols))
    copyto!(result.data, result_inds, d_C, d_C_inds)
    
    return result
end

# Direct RREF that returns GPUFiniteFieldMatrix
"""
    rref_gpu_direct(A, N)

Row reduction that directly returns a GPUFiniteFieldMatrix without intermediate Array conversion.
"""
function rref_gpu_direct(A::AbstractMatrix, N::Integer)
    # Call the existing rref_gpu function
    result = rref_gpu(A, N)
    
    # Convert the result to a GPUFiniteFieldMatrix
    return GPUFiniteFieldMatrix(result, N)
end

"""
    rref_gpu_direct(A::GPUFiniteFieldMatrix)

Row reduction that directly returns a GPUFiniteFieldMatrix without intermediate Array conversion.
"""
function rref_gpu_direct(A::GPUFiniteFieldMatrix)
    # Call the existing rref_gpu function on the array data
    result = rref_gpu(Array(A), A.N)
    
    # Convert the result to a GPUFiniteFieldMatrix
    return GPUFiniteFieldMatrix(result, A.N)
end

# Add direct plup decomposition function as well
"""
    plup_gpu_direct(A, N)

PLUP decomposition that returns the U and L matrices as GPUFiniteFieldMatrix types.
"""
function plup_gpu_direct(A::AbstractMatrix, N::Integer)
    U, L, P_rows, P_cols = plup_gpu(A, N)
    
    # Convert the U and L matrices to GPUFiniteFieldMatrix
    U_gpu = GPUFiniteFieldMatrix(U, N)
    L_gpu = GPUFiniteFieldMatrix(L, N)
    
    return U_gpu, L_gpu, P_rows, P_cols
end

"""
    plup_gpu_direct(A::GPUFiniteFieldMatrix)

PLUP decomposition that returns the U and L matrices as GPUFiniteFieldMatrix types.
"""
function plup_gpu_direct(A::GPUFiniteFieldMatrix)
    U, L, P_rows, P_cols = plup_gpu(Array(A), A.N)
    
    # Convert the U and L matrices to GPUFiniteFieldMatrix
    U_gpu = GPUFiniteFieldMatrix(U, A.N)
    L_gpu = GPUFiniteFieldMatrix(L, A.N)
    
    return U_gpu, L_gpu, P_rows, P_cols
end
