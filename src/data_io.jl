"""
Load data from disk into a vector of tuples of the form (X2, T1, X1, T1, Y)
and a vector of frequency counts. This compressed representation of the data
allows approximately a 10x reduction in processing time because redundant
computations are avoided.
"""
function load()
    # Load raw data from CSV file.
    df = DataFrames.readtable("data/choices.csv")

    # Drop nulls.
    df = df[!DataFrames.isna(df[:LaterOptionChosen]), :]

    # Convert to non-nullable arrays.
    X = hcat(
        convert(Array{Float64}, df[:X2]),
        convert(Array{Float64}, df[:T2]),
        convert(Array{Float64}, df[:X1]),
        convert(Array{Float64}, df[:T1]),
    )
    y = convert(Array{Float64}, df[:LaterOptionChosen])

    # Normalize money measurements.
    X[:, 1] = X[:, 1] ./ maximum(X[:, 1])
    X[:, 3] = X[:, 3] ./ maximum(X[:, 3])

    # Translate data into a tuple format to maximize cache locality.
    n = size(X, 1)
    all_tuples = Array(NTuple{5, Float64}, n)
    for i in 1:n
        all_tuples[i] = (X[i, 1], X[i, 2], X[i, 3], X[i, 4], y[i])
    end

    # Work with the unique values in the input and their frequencies to
    # compress data for faster computations.
    unique_tuples_dict = StatsBase.countmap(all_tuples)
    unique_tuples = collect(keys(unique_tuples_dict))
    w = collect(values(unique_tuples_dict))

    return unique_tuples, w
end
