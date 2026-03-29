using StaticLint

include("common.jl")

target = parse_target(ARGS)
files = target_julia_files(target)
println("StaticLint loaded for TARGET=$(target)")
println("Julia files discovered: $(length(files))")
for file in files
    println(file)
end
