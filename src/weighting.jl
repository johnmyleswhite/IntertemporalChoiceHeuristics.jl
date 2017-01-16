function update_weights!(
    new_weights::Vector,
    weights::Vector,
    training_proportion::Real,
)
    n = round(Int, training_proportion * sum(weights))
    π = weights ./ sum(weights)
    d = Distributions.Multinomial(n, π)
    rand!(d, new_weights)
    return
end
