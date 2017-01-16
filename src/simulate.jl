"""
Given a dataset, generate a closure that computes the negative log likelihood
for a model and takes only a vector of parameters as an argument.
"""
function simulate(
    m::Model,
    inputs::Vector,
    n_trials::Integer,
)
    sim_inputs = similar(inputs, 0)
    sim_weights = Int[]
    n = length(inputs)

    for i in 1:n
        @inbounds x2, t2, x1, t1, y = inputs[i]
        p = predict(m, x2, t2, x1, t1)
        w = rand(Distributions.Binomial(n_trials, p))
        push!(sim_inputs, (x2, t2, x1, t1, 1.0))
        push!(sim_weights, w)
        push!(sim_inputs, (x2, t2, x1, t1, 0.0))
        push!(sim_weights, n_trials - w)
    end

    return sim_inputs, sim_weights
end
