function Base.cov{T <: Model}(
    ::Type{T},
    inputs::Vector,
    weights::Vector,
)
    # Construct the negative log likelihood function and its gradient.
    loss = LogLoss()
    λ = 0.0
    nll, nll_gr! = make_closures(T, loss, λ, inputs, weights)

    # Find the MLE for the model.
    res = fit(T, inputs, weights)
    Θ = Optim.minimizer(res)

    # Compute the Hessian at the MLE.
    H = ForwardDiff.hessian(nll, Θ)

    # Construct the covariance matrix.
    return inv(H)
end
