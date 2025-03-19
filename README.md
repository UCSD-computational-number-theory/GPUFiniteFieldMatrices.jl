# GPU Finite Field Matrices

A specialized computational linear algebra library designed for optimized and fast matrix multiplication, tailored for cohomology computations over $\mathbb{F}_p$ and $\mathbb{Z}/p^n\mathbb{Z}$. This library empowers researchers with efficient tools for solving systems over finite fields to enhance the computational capabilities in algebraic geometry studies.

## Features

- **GPUFiniteFieldMatrix**: A Julia wrapper type for matrices over finite fields that leverages CUDA for GPU acceleration
- Automatic padding to multiples of 32 for optimal CUDA performance
- Matrix operations with delayed reduction (mod N)
- Invertibility checking and matrix inversion (for prime moduli)
- RREF and PLUP decompositions for matrices over finite fields
- In-place operations that avoid memory allocations
- Modulus-changing operations to convert between different finite fields
- Direct GPU operations that avoid CPU-GPU transfers

## Usage

### Basic Operations

```julia
using GPUFiniteFieldMatrices

# Create a matrix mod 11
A = GPUFiniteFieldMatrix([1 2 3; 4 5 6; 7 8 9], 11)

# Create another matrix
B = GPUFiniteFieldMatrix([9 8 7; 6 5 4; 3 2 1], 11)

# Basic operations
C = A + B       # Addition
D = A - B       # Subtraction
E = A * B       # Matrix multiplication
F = A .* B      # Element-wise multiplication

# Scalar operations
G = 3 * A       # Scalar multiplication
H = A + 5       # Scalar addition

# Utility functions
I3 = identity(Int, 3, 11)    # 3×3 identity matrix mod 11
Z = zeros(Int, 2, 3, 11)     # 2×3 zero matrix mod 11
R = rand(Int, 3, 3, 11)      # Random 3×3 matrix with elements in [0,10]

# Check if a matrix is invertible (only works for prime moduli)
if is_invertible(A)
    A_inv = inverse(A)    # Compute the inverse
    I_check = A * A_inv   # Should be identity matrix
end
```

### In-place Operations (No Allocations)

```julia
# Pre-allocate result matrices
C = zeros(Int, 3, 3, 11)
D = zeros(Int, 3, 3, 11)

# In-place matrix operations
add!(C, A, B)                 # C = A + B
subtract!(D, A, B)            # D = A - B
elementwise_multiply!(C, A, B) # C = A .* B
multiply!(D, A, B)            # D = A * B (still has internal allocations)

# In-place scalar operations
scalar_add!(C, A, 5)          # C = A + 5
scalar_subtract!(D, A, 3)     # D = A - 3
scalar_multiply!(C, A, 7)     # C = A * 7

# Other in-place operations
negate!(D, A)                 # D = -A
copy!(C, A)                   # C = A
mod_elements!(A)              # Apply modulus to all elements
```

### Modulus Override for In-place Operations

All in-place operations can take an optional modulus parameter to override the default modulus:

```julia
# Matrices with different moduli
A = GPUFiniteFieldMatrix([1 2 3; 4 5 6; 7 8 9], 11)
B = GPUFiniteFieldMatrix([9 8 7; 6 5 4; 3 2 1], 7)
C = zeros(Int, 3, 3, 13)

# Use a different modulus for the operation
add!(C, A, B, 5)                # C = (A + B) mod 5
subtract!(C, A, B, 5)           # C = (A - B) mod 5
elementwise_multiply!(C, A, B, 5) # C = (A .* B) mod 5

# Scalar operations with modulus override
scalar_add!(C, A, 3, 5)         # C = (A + 3) mod 5
scalar_subtract!(C, A, 3, 5)    # C = (A - 3) mod 5
scalar_multiply!(C, A, 3, 5)    # C = (A * 3) mod 5

# Other operations with modulus override
negate!(C, A, 5)                # C = (-A) mod 5
mod_elements!(A, 5)             # Apply mod 5 to all elements
```

### Changing Modulus

```julia
# Create a matrix mod 11
A = GPUFiniteFieldMatrix([1 2 3; 4 5 6; 7 8 9], 11)

# Create a new matrix with a different modulus
B = change_modulus(A, 7)   # B contains A's values but mod 7

# Change modulus in-place
change_modulus!(A, 7)      # A is now mod 7
```

### Direct GPU Operations

These functions avoid unnecessary CPU-GPU transfers by working directly with GPUFiniteFieldMatrix:

```julia
# Matrix multiplication that returns a GPUFiniteFieldMatrix directly
C = matmul_gpu_direct(A, B)

# Row reduction that returns a GPUFiniteFieldMatrix directly
D = rref_gpu_direct(A)

# PLUP decomposition that returns GPUFiniteFieldMatrix matrices
U, L, P_rows, P_cols = plup_gpu_direct(A)
```

## Performance Benefits

- Using in-place operations can significantly reduce memory allocations
- Direct GPU operations avoid unnecessary CPU-GPU transfers
- Modulus-changing operations allow efficient conversion between different finite fields

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/GPUFiniteFieldMatrices.jl")
```

## Requirements

- Julia 1.6 or higher
- CUDA.jl
- A CUDA-capable GPU

## License

[MIT License](LICENSE)
