using JuliaFormatter

include("common.jl")

mode = length(ARGS) >= 1 ? ARGS[1] : "check"
target = length(ARGS) >= 2 ? ARGS[2] : "."
files = target_julia_files(target)

if mode == "write"
    for file in files
        src = read(file, String)
        out = format_text(src)
        if out != src
            write(file, out)
            println("formatted: $(file)")
        end
    end
    println("Formatting complete for $(length(files)) file(s)")
elseif mode == "check"
    not_formatted = String[]
    for file in files
        src = read(file, String)
        out = format_text(src)
        if out != src
            push!(not_formatted, file)
        end
    end
    if isempty(not_formatted)
        println("All checked files are formatted")
    else
        println("Unformatted files:")
        for file in not_formatted
            println(file)
        end
        error("Formatting check failed for $(length(not_formatted)) file(s)")
    end
else
    error("Unknown mode $(mode). Use check or write.")
end
