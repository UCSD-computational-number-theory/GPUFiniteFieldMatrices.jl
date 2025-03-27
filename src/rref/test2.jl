using RowEchelon
using Test
using CSV, DataFrames

using DelimitedFiles
A = readdlm("large.csv", ',', Int64)

@testset "Large Matrix 141MB" begin
    t = @elapsed begin
        C = rref(A)
    end
    #println("Time taken by rref function: $t seconds")
    println("Time:",t)
    writedlm( "resultC.csv",  C, ',')
end