immutable HomotheticHyperboloid{T} <: Model
    a::T
    α::T
    μ::T
end

function Base.show(io::IO, ::Type{HomotheticHyperboloid})
    return print(io, "Homothetic Hyperboloid")
end

function Base.show(io::IO, m::HomotheticHyperboloid)
    @printf(
        io,
        "Homothetic Hyperboloid (a = %.3f, α = %.3f, μ = %.3f)",
        m.a,
        m.α,
        m.μ,
    )
end

function initial_parameters(::Type{HomotheticHyperboloid})
    return Float64[log(1.0), log(1.0), log(1.0)]
end

canonical_parameters(m::HomotheticHyperboloid) = (m.a, m.α, m.μ)

function HomotheticHyperboloid(parameters::Vector)
    a = exp(parameters[1])
    α = exp(parameters[2])
    μ = exp(parameters[3])
    return HomotheticHyperboloid(a, α, μ)
end

@inline function predict(
    m::HomotheticHyperboloid,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α, μ = m.a, m.α, m.μ
    u1 = log(x1 * (1 + α * t1)^(-μ))
    u2 = log(x2 * (1 + α * t2)^(-μ))
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HomotheticHyperboloid,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α, μ = m.a, m.α, m.μ
    gr = (
        (log(x2 * (1 + α * t2)^(-μ)) - log(x1 * (1 + α * t1)^(-μ))) * a,
        a * μ * ((1 + α * t1)^(-1) * t1 - (1 + α * t2)^(-1) * t2) * α,
        a * (log(1 + α * t1) - log(1 + α * t2)) * μ,
    )
    return gr
end
