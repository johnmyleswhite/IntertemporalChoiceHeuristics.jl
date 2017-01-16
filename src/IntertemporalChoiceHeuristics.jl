module IntertemporalChoiceHeuristics
    import DataFrames, Distributions, ForwardDiff, Optim, StatsBase

    include("models/base_model.jl")
    include("losses/base_loss.jl")

    include("links.jl")
    include("data_io.jl")
    include("weighting.jl")
    include("fitting.jl")
    include("average_loss.jl")
    include("simulate.jl")
    include("cov.jl")
    include("stderrs.jl")

    include("models/heuristics/baseline.jl")
    include("models/heuristics/ITCH.jl")
    include("models/heuristics/DRIFT.jl")
    include("models/heuristics/tradeoff.jl")

    include("models/exponential/standard.jl")
    include("models/exponential/homothetic.jl")
    include("models/exponential/intercept.jl")

    include("models/hyperbolic/standard.jl")
    include("models/hyperbolic/homothetic.jl")
    include("models/hyperbolic/intercept.jl")

    include("models/hyperboloid/standard.jl")
    include("models/hyperboloid/homothetic.jl")
    include("models/hyperboloid/intercept.jl")

    include("models/quasihyperbolic/standard.jl")
    include("models/quasihyperbolic/homothetic.jl")
    include("models/quasihyperbolic/intercept.jl")

    include("models/system2/standard.jl")
    include("models/system2/homothetic.jl")
    include("models/system2/intercept.jl")

    include("losses/absolute_error.jl")
    include("losses/extremity.jl")
    include("losses/log_loss.jl")
    include("losses/squared_error.jl")
    include("losses/zero_one_loss.jl")

    models = (
        Baseline,
        ITCH,
        DRIFT,
        Tradeoff,
        Exponential,
        HomotheticExponential,
        ExponentialIntercept,
        Hyperbolic,
        HomotheticHyperbolic,
        HyperbolicIntercept,
        Hyperboloid,
        HomotheticHyperboloid,
        HyperboloidIntercept,
        QuasiHyperbolic,
        HomotheticQuasiHyperbolic,
        QuasiHyperbolicIntercept,
        System2,
        HomotheticSystem2,
        System2Intercept,
    )

    training_losses = (
        SquaredError(),
        LogLoss(),
    )

    evaluation_losses = (
        AbsoluteError(),
        SquaredError(),
        LogLoss(),
        ZeroOneLoss(),
        Extremity(),
    )
end
