using RowEchelon
using Test
using CSV, DataFrames

using DelimitedFiles
A = readdlm("medium.csv", ',', Int64)

@testset "Matrix 10MB" begin
    t = @elapsed begin
        C = rref(A)
    end
    #println("Time taken by rref function: $t seconds")
    println("Time:",t)
    writedlm( "resultD.csv",  C, ',')
end