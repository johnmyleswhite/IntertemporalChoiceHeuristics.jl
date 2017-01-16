immutable ZeroOneLoss <: Loss
end

@inline function loss_value(loss::ZeroOneLoss, p::Real, y::Real)
    return Float64(round(Int, p) != y)
end

Base.show(io::IO, loss::ZeroOneLoss) = print(io, "0-1 Loss")
