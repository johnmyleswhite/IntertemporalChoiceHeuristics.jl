immutable AbsoluteError <: Loss
end

@inline function loss_value(loss::AbsoluteError, p::Real, y::Real)
    return abs(p - y)
end

Base.show(io::IO, loss::AbsoluteError) = print(io, "Absolute Error")
