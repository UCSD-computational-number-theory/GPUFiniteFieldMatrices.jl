module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using SparseArrays
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles
using Unroll

# --- harden: DispatchDoctor return-type-stability gate (bead gfm-kvf.4.5) ---
# `@stable` wraps every function defined in the module body below (propagating
# through `include`) with a return-type-stability check. `default_mode="disable"`
# keeps it a no-op for downstream users; the test run flips it on via the
# `dispatch_doctor_mode` preference set in test/runtests.jl before this package
# loads. Macro imports (CUDA/Unroll) stay OUTSIDE the `@stable begin` block per
# DispatchDoctor's guidance. (If a wrapped function is ever legitimately unstable,
# import `@unstable` here too and annotate that function — none needed yet.)
using DispatchDoctor: @stable

const DEBUG = false

@stable default_mode = "disable" begin

include("CuModMatrix/CuModMatrix.jl")

include("CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
include("CuModMatrix/kernel_mul/mat_mul_ops.jl")
include("CuModMatrix/kernel_mul/stripe_mul.jl")

include("KaratsubaMatrix/KaratsubaMatrix.jl")
include("KaratsubaMatrix/KaratsubaKernels.jl")

include("CuModMatrix/rref_lu_pluq/permutations.jl")
include("CuModMatrix/rref_lu_pluq/pluq_kernels.jl")
include("CuModMatrix/kernel_ops/common.jl")
include("CuModMatrix/kernel_ops/add_ops.jl")
include("CuModMatrix/kernel_ops/sub_ops.jl")
include("CuModMatrix/kernel_ops/mul_ops.jl")
include("CuModMatrix/kernel_ops/div_ops.jl")
include("CuModMatrix/kernel_ops/mod_ops.jl")

include("CuModMatrix/triangular/triangular_inverse_no_copy.jl")
include("CuModMatrix/triangular/substitution_inplace.jl")

end # @stable default_mode = "disable"

# Export the main type and its operations
export CuModArray, CuModMatrix, CuModVector
export inverse

export KaratsubaArray, KaratsubaMatrix, KaratsubaVector

# Export utility functions
export eye 
export change_modulus, change_modulus_no_alloc!
export elementwise_multiply!, negate!
export scalar_add!, scalar_sub!, rmul!, lmul!
export mod_elements!, fill! 

# do not export: add!, sub!, zero!, is_invertible, is_invertible_with_inverse
#     (since they conflict with AbstractAlgebra

# Export GPU operations
export mat_mul_gpu_type, mat_mul_type_inplace!
export perm_array_to_matrix
export is_invertible, inverse, is_invertible_with_inverse
export perm_array_to_matrix
export apply_col_perm!, apply_row_perm!
export mod_inv
export pluq_gpu_kernel
export upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
export forward_sub_gpu_type_32, backward_sub_gpu_type_32

end
