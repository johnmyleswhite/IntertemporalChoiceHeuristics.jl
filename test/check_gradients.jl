module CheckGradients
    verbose = false

    using Base.Test

    import IntertemporalChoiceHeuristics:
        load,
        fit,
        models,
        make_closures,
        initial_parameters,
        training_losses

    import ForwardDiff

    import Optim: minimizer

    inputs, weights = load()

    λ = 0.0

    @testset "All models" begin
        for model in models
            for loss in training_losses
                res = fit(model, inputs, weights, loss)

                Θ = initial_parameters(model)
                Θ′ = minimizer(res)

                cost, cost_gr! = make_closures(model, loss, λ, inputs, weights)

                @test isa(cost(Θ), Float64)

                gr = similar(Θ)

                cost_gr!(Θ, gr)
                fd_gr = ForwardDiff.gradient(cost, Θ)
                @test norm(gr .- fd_gr, Inf) < 1e-10

                cost_gr!(Θ′, gr)
                fd_gr = ForwardDiff.gradient(cost, Θ′)
                @test norm(gr .- fd_gr, Inf) < 1e-10

                if verbose
                    println((model, loss, norm(gr .- fd_gr, Inf)))
                end
            end
        end
    end
end
