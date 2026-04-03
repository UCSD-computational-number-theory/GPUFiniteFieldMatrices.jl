using KernelAbstractions
import KernelAbstractions as KA

const _HAS_GEMMKERNELS = Base.find_package("GemmKernels") !== nothing
if _HAS_GEMMKERNELS
    @eval using GemmKernels
end

include("common.jl")
include("types_ka.jl")
include("mod_arith_ka.jl")
include("perm_vectors_ka.jl")
include("matops_ka.jl")
include("basecase_pluq_ka.jl")
include("rectangular_pluq_ka.jl")
include("trsm_ka.jl")
include("schur_update_ka.jl")
include("blocked_recursive_pluq_ka.jl")
include("extract_ka.jl")
include("validation_ka.jl")
include("api_ka.jl")
include("batched_tiny_ka.jl")
