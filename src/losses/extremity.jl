immutable Extremity <: Loss
end

@inline function loss_value(loss::Extremity, p::Real, y::Real)
    return abs(p - 0.5)
end

Base.show(io::IO, loss::Extremity) = print(io, "Extremity")
