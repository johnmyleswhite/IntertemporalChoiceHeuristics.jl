import IntertemporalChoiceHeuristics:
    load,
    update_weights!,
    fit,
    models,
    initial_parameters,
    evaluation_losses,
    training_losses,
    average_loss

import Optim: minimizer

using ProgressMeter
using ArgParse

function main(demo::Bool, output_path::String)
    io = open(output_path, "w")

    inputs, weights = load()

    training_weights = copy(weights)
    test_weights = copy(weights)

    if demo
        n_iterations = 5
    else
        n_iterations = 500
    end

    @showprogress 1 "Cross-validating:" for iteration in 1:n_iterations
        for training_proportion in (0.0007, 0.75)
            test_proportion = 1 - training_proportion

            # TODO: Condition on weights never being greater than
            # weights in full dataset to make this closer to the
            # original sampling without replacement scheme we used.

            # Algorithm: Generate a sample and find all elements where
            # where training weights are larger than weights. Cap these and
            # count them. Then repeatedly draw from multinomial until
            # the remaining weights have been filled in.
            update_weights!(training_weights, weights, training_proportion)
            # force_elementwise_upper_bound!(training_weights, weights)

            # TODO: Make these weights the "mirror" of the training
            # weights so there's no possibility that an "observation"
            # is in both sets.
            #
            # In practice this problem is irrelevant because we have
            # so many observations with identical covariates that the
            # two sets always share "observations" -- the two data sets
            # differ only in the weights assigned to each tuple of
            # covariates and outcomes.
            update_weights!(test_weights, weights, test_proportion)

            for training_loss in training_losses
                for model in models
                    for λ in (0.0, 0.05)
                        try
                            res = fit(
                                model,
                                inputs,
                                training_weights,
                                training_loss,
                                λ,
                            )
                            parameters = minimizer(res)
                            m = model(parameters)
                            for loss in evaluation_losses
                                l = average_loss(m, loss, inputs, test_weights)
                                @printf(
                                    io,
                                    "%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
                                    iteration,
                                    model,
                                    training_loss,
                                    λ,
                                    training_proportion,
                                    loss,
                                    l,
                                )
                            end
                        catch
                            for loss in evaluation_losses
                                @printf(
                                    io,
                                    "%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
                                    iteration,
                                    model,
                                    training_loss,
                                    λ,
                                    training_proportion,
                                    loss,
                                    "NA",
                                )
                            end
                        end
                    end
                end
            end
        end
    end

    close(io)

    return
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
