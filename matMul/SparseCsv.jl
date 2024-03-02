"""
File containing methods to convert csv to sparse matrix and vice versa.

!WARNING: The following methods are CPU parallelized.
"""

# using Pkg
# Pkg.add("CSV")

using CSV
using Base.Threads

function read_dense_csv_P(type, filename, p)
    """ Reads a dense csv matrix, returns SparseMatModP representation """
    data = CSV.read(filename)
    nrows, ncols = size(data)

    rows = Arrays{type,1}
    cols = Arrays{type,1}
    vals = Arrays{type,1}

    Threads.@threads for i in 1:nrows
        for j in 1:ncols
            if data[i, j] != "0"
                push!(rows, i)
                push!(cols, j)
                push!(vals, data[i, j])
            end
        end
    end

    return SparseMatModP{type}(rows, cols, vals, nrows, ncols, p, find_P_Mults(p))
end

function read_dense_csv_Pn(type, filename, p, n)
    """ Reads a dense csv matrix, returns SparseMatModP representation """
    data = CSV.read(filename)
    nrows, ncols = size(data)

    rows = Arrays{type,1}
    cols = Arrays{type,1}
    vals = Arrays{type,1}

    Threads.@threads for i in 1:nrows
        for j in 1:ncols
            if data[i, j] != "0"
                push!(rows, i)
                push!(cols, j)
                push!(vals, data[i, j])
            end
        end
    end

    return SparseMatModP{type}(rows, cols, vals, nrows, ncols, p, n, find_P_Mults(p))
end

function read_sparse_csv_P(type, filename, p)
    """ Reads a sparse csv matrix, returns SparseMatModP representation """
    data = CSV.read(filename)

    rows = Arrays{type,1}
    cols = Arrays{type,1}
    vals = Arrays{type,1}

    Threads.@threads for i in 1:nrow(data)
        push!(rows, data[i, 1])
        push!(cols, data[i, 2])
        push!(vals, data[i, 3])
    end

    return SparseMatModP{type}(rows, cols, vals, nrows, ncols, p, find_P_Mults(p))
end

function read_sparse_csv_Pn(type, filename, p, n)
    """ Reads a sparse csv matrix, returns SparseMatModPn representation """
    data = CSV.read(filename)

    rows = Arrays{type,1}
    cols = Arrays{type,1}
    vals = Arrays{type,1}

    Threads.@threads for i in 1:nrow(data)
        push!(rows, data[i, 1])
        push!(cols, data[i, 2])
        push!(vals, data[i, 3])
    end

    return SparseMatModP{type}(rows, cols, vals, nrows, ncols, p, n, find_P_Mults(p))
end

function make_sparse_csv_P(filename, mat::SparseMatModP)
    data = [mat.rows mat.cols mat.vals]
    CSV.write(filename + "P$mat.p", data)
end

function make_sparse_csv_Pn(filename, mat::SparseMatModPn)
    data = [mat.rows mat.cols mat.vals]
    CSV.write(filename + "P$mat.pN+$mat.n", data)
end