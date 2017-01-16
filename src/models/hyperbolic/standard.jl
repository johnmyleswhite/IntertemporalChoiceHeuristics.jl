immutable Hyperbolic{T} <: Model
    a::T
    α::T
end

Base.show(io::IO, ::Type{Hyperbolic}) = print(io, "Classical Hyperbolic")

function Base.show(io::IO, m::Hyperbolic)
    @printf(
        io,
        "Classical Hyperbolic (a = %.3f, α = %.3f)",
        m.a,
        m.α,
    )
end

initial_parameters(::Type{Hyperbolic}) = Float64[log(1.0), log(1.0)]

canonical_parameters(m::Hyperbolic) = (m.a, m.α)

function Hyperbolic(parameters::Vector)
    a = exp(parameters[1])
    α = exp(parameters[2])
    return Hyperbolic(a, α)
end

@inline function predict(
    m::Hyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α = m.a, m.α
    u1 = x1 * (1 / (1 + α * t1))
    u2 = x2 * (1 / (1 + α * t2))
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::Hyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α = m.a, m.α
    d_t2 = 1 / (1 + α * t2)
    d_t1 = 1 / (1 + α * t1)
    gr = (
        (x2 * d_t2 - x1 * d_t1) * a,
        a * (x1 * t1 * d_t1^2 - x2 * t2 * d_t2^2) * α,
    )
    return gr
end
