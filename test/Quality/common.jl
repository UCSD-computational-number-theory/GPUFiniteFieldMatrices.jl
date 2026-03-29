function parse_target(args)
    return isempty(args) ? "." : args[1]
end

function target_julia_files(target)
    if isfile(target)
        return endswith(target, ".jl") ? [abspath(target)] : String[]
    end
    if isdir(target)
        files = String[]
        for (root, _, names) in walkdir(target)
            for name in names
                if endswith(name, ".jl")
                    push!(files, joinpath(root, name))
                end
            end
        end
        return sort(files)
    end
    error("Target does not exist: $(target)")
end
