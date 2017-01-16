"""
Compute the cost function for a model and loss function given a vector of
parameters, a dataset and a vector of frequency counts as weights.
"""
function cost_function{T <: Model}(
    ::Type{T},
    loss::Loss,
    λ::Real,
    parameters::Vector,
    inputs::Vector,
    weights::Vector,
)
    # Construct a model object that allows stack allocation of model parameters.
    m = T(parameters)

    # We iterate over all observations in the dataset.
    n = length(inputs)

    # Iterate over the full dataset accumating the total cost.
    total_cost = 0.0
    for i in 1:n
        @inbounds x2, t2, x1, t1, y = inputs[i]
        @inbounds w = weights[i]
        p = predict(m, x2, t2, x1, t1)
        total_cost += w * loss_value(loss, p, y)
    end

    # Add L2 regularization.
    total_cost += λ * norm(parameters)^2

    # Return the total cost.
    return total_cost
end

"""
Generate a closure that computes the cost function for a model given only a
vector of parameters as an argument.
"""
function cost_function_closure{T <: Model}(
    ::Type{T},
    loss::Loss,
    λ::Real,
    inputs::Vector,
    weights::Vector,
)
    return parameters -> cost_function(T, loss, λ, parameters, inputs, weights)
end

"""
Compute the gradient of a cost function for a model and loss function given a
vector of parameters, a dataset and a vector of frequency counts as weights.
"""
function cost_function_gradient!{T <: Model}(
    gr::Vector,
    ::Type{T},
    loss::Loss,
    λ::Real,
    parameters::Vector,
    inputs::Vector,
    weights::Vector,
)
    # Set the gradient to zero because we will accumulate the gradient by
    # mutating summation.
    fill!(gr, 0.0)

    # Construct a model object that allows stack allocation of model parameters.
    m = T(parameters)

    # We iterate over all observations in the dataset.
    n = length(inputs)
    K = length(parameters)

    # Iterate over the full dataset accumating the gradient.
    for i in 1:n
        @inbounds x2, t2, x1, t1, y = inputs[i]
        @inbounds w = weights[i]
        p = predict(m, x2, t2, x1, t1)
        gradient_tuple = gradient_component(m, x2, t2, x1, t1)
        σ = w * loss_derivative(loss, p, y)
        for j in 1:K
            @inbounds gr[j] += σ * gradient_tuple[j]
        end
    end

    # Add L2 regularization.
    for j in 1:K
        @inbounds gr[j] += λ * 2.0 * parameters[j]
    end

    # Return nothing. The gradient is already stored in the mutated argument.
    return
end

"""
Generate a closure that computes the gradient of the cost function for a model
given only a vector of parameters and a mutable vector that will be updated to
contain the gradient as arguments.
"""
function cost_function_gradient_closure{T <: Model}(
    ::Type{T},
    loss::Loss,
    λ::Real,
    inputs::Vector,
    weights::Vector,
)
    return (parameters, gr) -> cost_function_gradient!(
        gr,
        T,
        loss,
        λ,
        parameters,
        inputs,
        weights,
    )
end

"""
Generate the closures we use for model fitting.
"""
function make_closures{T <: Model}(
    ::Type{T},
    loss::Loss,
    λ::Real,
    inputs::Vector,
    weights::Vector,
)
    return (
        cost_function_closure(T, loss, λ, inputs, weights),
        cost_function_gradient_closure(T, loss, λ, inputs, weights),
    )
end

"""
Fit a model given inputs and weights.

Set loss to the type of loss function that will be minimized when fitting the
model's parameters.

Set λ to the regularization parameter to increase the L2 penalty on the model
parameters in the unconstrained space.

Set Θ₀ to any vector of initial parameters for the model being fit. Defaults to
initial_parameters(T) for a model of type T.

Set method to any method provided by the Optim package. Defaults to
Optim.Newton(), which is a straight-forward implementation of Newton's method.
"""
function fit{T <: Model}(
    ::Type{T},
    inputs::Vector,
    weights::Vector,
    loss::Loss = LogLoss(),
    λ::Real = 0.0,
    Θ₀::Vector = initial_parameters(T),
    method::Optim.Optimizer = Optim.Newton(),
)
    cost, cost_gradient! = make_closures(T, loss, λ, inputs, weights)

    opts = Optim.OptimizationOptions(
        f_tol = 0.0,
        show_trace = false,
        autodiff = true,
    )

    # TODO: Provide analytic Hessians to Optim.optimize.
    res = Optim.optimize(cost, cost_gradient!, Θ₀, method, opts)

    # Require convergence.
    if !Optim.converged(res)
        error("Failed to converge")
    end

    return res
end
