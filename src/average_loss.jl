function average_loss(m::Model, loss::Loss, inputs::Vector, weights::Vector)
    l, n = 0.0, 0
    n_inputs = length(inputs)
    for i in 1:n_inputs
        @inbounds x2, t2, x1, t1, y = inputs[i]
        @inbounds w = weights[i]
        p = predict(m, x2, t2, x1, t1)
        l += w * loss_value(loss, p, y)
        n += w
    end
    return l / n
end
