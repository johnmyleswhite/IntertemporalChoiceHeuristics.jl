immutable SquaredError <: Loss
end

@inline function loss_value(loss::SquaredError, p::Real, y::Real)
    return (p - y) * (p - y) # == (p - y)^2
end

Base.show(io::IO, loss::SquaredError) = print(io, "Squared Error")

@inline function loss_derivative(loss::SquaredError, p::Real, y::Real)
    return 2 * (p - y) * (p * (1 - p))
end
