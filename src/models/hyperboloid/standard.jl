immutable Hyperboloid{T} <: Model
    a::T
    α::T
    μ::T
end

Base.show(io::IO, ::Type{Hyperboloid}) = print(io, "Classical Hyperboloid")

function Base.show(io::IO, m::Hyperboloid)
    @printf(
        io,
        "Classical Hyperboloid (a = %.3f, α = %.3f, μ = %.3f)",
        m.a,
        m.α,
        m.μ,
    )
end

function initial_parameters(::Type{Hyperboloid})
    return Float64[log(1.0), log(1.0), log(1.0)]
end

canonical_parameters(m::Hyperboloid) = (m.a, m.α, m.μ)

function Hyperboloid(parameters::Vector)
    a = exp(parameters[1])
    α = exp(parameters[2])
    μ = exp(parameters[3])
    return Hyperboloid(a, α, μ)
end

@inline function predict(
    m::Hyperboloid,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α, μ = m.a, m.α, m.μ
    u1 = x1 * (1 + α * t1)^(-μ)
    u2 = x2 * (1 + α * t2)^(-μ)
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::Hyperboloid,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α, μ = m.a, m.α, m.μ
    u1 = x1 * (1 + α * t1)^(-μ)
    u2 = x2 * (1 + α * t2)^(-μ)
    gr = (
        (u2 - u1) * a,
        a * (x2 * -μ * (1 + α * t2)^(-μ - 1) * t2 - x1 * -μ * (1 + α * t1)^(-μ - 1) * t1) * α,
        a * (-x2 * (1 + α * t2)^(-μ) * log(1 + α * t2) + x1 * (1 + α * t1)^(-μ) * log(1 + α * t1)) * μ,
    )
    return gr
end
