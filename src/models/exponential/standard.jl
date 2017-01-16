immutable Exponential{T} <: Model
    a::T
    δ::T
end

Base.show(io::IO, ::Type{Exponential}) = print(io, "Classical Exponential")

function Base.show(io::IO, m::Exponential)
    @printf(
        io,
        "Classical Exponential (a = %.3f, δ = %.3f)",
        m.a,
        m.δ,
    )
end

function initial_parameters(::Type{Exponential})
    return Float64[log(1.0), logit(0.5)]
end

canonical_parameters(m::Exponential) = (m.a, m.δ)

function Exponential(parameters::Vector)
    a = exp(parameters[1])
    δ = invlogit(parameters[2])
    return Exponential(a, δ)
end

@inline function predict(
    m::Exponential,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, δ = m.a, m.δ
    u1 = x1 * δ^t1
    u2 = x2 * δ^t2
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::Exponential,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, δ = m.a, m.δ
    u2 = x2 * δ^t2
    u1 = x1 * δ^t1
    gr = (
        (u2 - u1) * a,
        a * ((x2 * t2 * δ^t2 - x1 * t1 * δ^t1) / δ) * δ * (1 - δ),
    )
    return gr
end
