import IntertemporalChoiceHeuristics:
    load,
    update_weights!,
    fit,
    models,
    canonical_parameters

import Optim: minimizer

using ProgressMeter
using ArgParse

function main(demo::Bool, output_path::String)
    io = open(output_path, "w")

    inputs, weights = load()

    new_weights = copy(weights)

    if demo
        n_replicates = 100
    else
        n_replicates = 10_000
    end

    @showprogress 1 "Bootstrapping:" for replicate in 1:n_replicates
        update_weights!(new_weights, weights, 1.0)

        for model in models
            try
                res = fit(model, inputs, new_weights)
                parameters = minimizer(res)
                m = model(parameters)
                params = canonical_parameters(m)
                K = length(params)
                @printf(
                    io,
                    "%d\t%s\t%s\n",
                    replicate,
                    model,
                    join([i > K ? "NA" : string(params[i]) for i in 1:5], '\t')
                )
            end
        end
    end

    close(io)
end

s = ArgParseSettings(description = "generate_replicates.jl")

@add_arg_table s begin
    "--demo"
        action = :store_true
    "output_path"
        required = true
end

args = parse_args(s)

main(args["demo"], args["output_path"])
