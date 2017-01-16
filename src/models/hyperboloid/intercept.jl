immutable HyperboloidIntercept{T} <: Model
    β₀::T
    a::T
    α::T
    μ::T
end

function Base.show(io::IO, ::Type{HyperboloidIntercept})
    return print(io, "Intercept Hyperboloid")
end

function Base.show(io::IO, m::HyperboloidIntercept)
    @printf(
        io,
        "Intercept Hyperboloid (β₀ = %.3f, a = %.3f, α = %.3f, μ = %.3f)",
        m.β₀,
        m.a,
        m.α,
        m.μ,
    )
end

function initial_parameters(::Type{HyperboloidIntercept})
    return Float64[0.0, log(1.0), log(1.0), log(1.0)]
end

canonical_parameters(m::HyperboloidIntercept) = (m.a, m.α, m.μ, m.β₀)

function HyperboloidIntercept(parameters::Vector)
    β₀ = parameters[1]
    a = exp(parameters[2])
    α = exp(parameters[3])
    μ = exp(parameters[4])
    return HyperboloidIntercept(β₀, a, α, μ)
end

@inline function predict(
    m::HyperboloidIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, α, μ = m.β₀, m.a, m.α, m.μ
    u1 = x1 * (1 + α * t1)^(-μ)
    u2 = x2 * (1 + α * t2)^(-μ)
    z = β₀ + a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HyperboloidIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, α, μ = m.β₀, m.a, m.α, m.μ
    u1 = x1 * (1 + α * t1)^(-μ)
    u2 = x2 * (1 + α * t2)^(-μ)
    gr = (
        1.0,
        (u2 - u1) * a,
        a * (x2 * -μ * (1 + α * t2)^(-μ - 1) * t2 - x1 * -μ * (1 + α * t1)^(-μ - 1) * t1) * α,
        a * (-x2 * (1 + α * t2)^(-μ) * log(1 + α * t2) + x1 * (1 + α * t1)^(-μ) * log(1 + α * t1)) * μ,
    )
    return gr
end
