abstract Model

function (::Type{T}){T <: Model}()
    return T(initial_parameters(T))
end

function (::Type{T}){T <: Model}(res::Optim.MultivariateOptimizationResults)
    return T(Optim.minimizer(res))
end
