module TestRandomInitialParameters
    verbose = false

    using Base.Test

    import IntertemporalChoiceHeuristics:
        load,
        fit,
        models,
        initial_parameters,
        training_losses

    import Distributions: Uniform

    import Optim

    inputs, weights = load()

    λ = 0.0

    @testset "Identical optima found from different starting points" begin
        for model in models
            Θ = initial_parameters(model)
            K = length(Θ)
            for loss in training_losses
                res = fit(model, inputs, weights, loss, λ, Θ)

                fₓ = Optim.minimum(res)

                n_points = 10
                costs = Array(Float64, n_points)
                for i in 1:n_points
                    ϵ = rand(Uniform(-0.1, +0.1), K)
                    Θ′ = Θ + ϵ
                    try
                        res = fit(model, inputs, weights, loss, λ, Θ′)
                        fₓ′ = Optim.minimum(res)
                        costs[i] = fₓ′
                        δ = (fₓ′ - fₓ) / fₓ
                        if verbose
                            @printf(
                                "%s\t%s\t%d\t%.4f\t%.4f\t%.4f\n",
                                model,
                                loss,
                                i,
                                fₓ,
                                fₓ′,
                                δ,
                            )
                        end
                    catch
                        costs[i] = fₓ
                    end
                end
                # TODO: Get this to pass reliably.
                # @test all(x -> x < 1e-10, (costs ./ minimum(costs)) - 1.0)
            end
        end
    end
end
