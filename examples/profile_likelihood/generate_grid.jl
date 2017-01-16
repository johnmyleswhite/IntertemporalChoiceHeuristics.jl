import IntertemporalChoiceHeuristics:
    load,
    fit,
    models,
    make_closures,
    LogLoss,
    invlogit,
    stderrs

inputs, weights = load()

loss = LogLoss()
λ = 0.0

for model in models
    res = fit(model, inputs, weights)

    nll, nll_gr! = make_closures(model, loss, λ, inputs, weights)

    mle = Optim.minimizer(res)

    se = stderrs(model, inputs, weights)

    K = length(mle)

    n_points = 256
    for j in 1:K
        parameters = copy(mle)
        grid = linspace(
            mle[j] - 6 * se[j],
            mle[j] + 6 * se[j],
            n_points,
        )
        for i in 1:n_points
            parameters[j] = grid[i]
            @printf(
                "%s\t%d\t%s\t%s\t%s\t%s\n",
                model,
                j,
                grid[i],
                nll(parameters),
                mle[j],
                se[j],
            )
        end
    end
end
