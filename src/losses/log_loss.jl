immutable LogLoss <: Loss
end

@inline function loss_value(loss::LogLoss, p::Real, y::Real)
    return -log(ifelse(y == 1.0, p, 1 - p))
end

Base.show(io::IO, loss::LogLoss) = print(io, "Log-Loss")

@inline function loss_derivative(loss::LogLoss, p::Real, y::Real)
    return p - y
end
