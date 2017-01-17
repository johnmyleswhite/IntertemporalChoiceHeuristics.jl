module TestConsistency
    verbose = false

    using Base.Test

    import IntertemporalChoiceHeuristics:
        load,
        fit,
        models,
        simulate,
        make_closures,
        stderrs

    import Optim: minimizer

    import ForwardDiff: hessian

    import Distributions: ccdf, Chisq

    inputs, weights = load()
    n_trials = 1_000

    for model in models
        res = fit(model, inputs, weights)
        parameters = minimizer(res)
        m = model(parameters)

        for sim in 1:10
            sim_inputs, sim_weights = simulate(m, inputs, n_trials)

            sim_res = fit(model, sim_inputs, sim_weights)
            sim_parameters = minimizer(sim_res)
            sim_m = model(sim_parameters)

            p_value = NaN
            try
                se = stderrs(model, sim_inputs, sim_weights)
                z = (sim_parameters .- parameters) ./ se
                χ = sum(z.^2)
                p_value = ccdf(Chisq(length(parameters)), χ)
            end

            if verbose
                @printf(
                    "%s\t%.2f\t%.4f\n",
                    model,
                    norm(
                        100 * (parameters .- sim_parameters) ./ parameters,
                        Inf,
                    ),
                    p_value,
                )
            end
        end
    end
end
