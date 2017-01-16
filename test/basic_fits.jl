module TestBasicFits
    verbose = false

    import IntertemporalChoiceHeuristics:
        load,
        fit,
        models,
        training_losses

    inputs, weights = load()

    for model in models
        for loss in training_losses
            @time res = fit(model, inputs, weights, loss)
            @time res = fit(model, inputs, weights, loss)
            if verbose
                println(model)
                println(loss)
                println(res)
                println()
            end
        end
    end
end
