module TestOptimizationAlgorithms
    verbose = false

    using Base.Test

    import IntertemporalChoiceHeuristics:
        load,
        fit,
        models,
        initial_parameters,
        LogLoss

    import Optim

    inputs, weights = load()

    loss = LogLoss()
    λ = 0.0
    algorithms = (Optim.Newton(), Optim.LBFGS(), Optim.BFGS())

    @testset "Identical optima found by different algorithms" begin
        for model in models
            Θ = initial_parameters(model)

            costs = Array(Float64, 3)
            for (i, algorithm) in enumerate(algorithms)
                res = fit(model, inputs, weights, loss, λ, Θ, algorithm)
                costs[i] = Optim.minimum(res)
                if verbose
                    @printf(
                        "%s\t%s\t%s\n",
                        model,
                        algorithm,
                        Optim.minimum(res),
                    )
                end
            end

            @test all(x -> x < 1e-10, (costs ./ minimum(costs)) - 1.0)
        end
    end
end
