"""
File to compare the speed of different matrix multiplications,
using @time for trials of specified matrix types.
"""
module MatMulComp

using LinearAlgebra

"""
HOF to generate num of pairs of matrices and return them as an iterable.
"""
function genRandMats(num, func)
    return [func for _ in 1:num]
end

"""
Function that takes in vector of matrices and times total multiplication time
given some multiplication implementation.
"""
function timeMatMul(leftMatrices, rightMatrices)
    timeInSec = @time begin 
        for (lmat, rmat) in zip(leftMatrices, rightMatrices)
            lmat * rmat
        end 
    end
end

end