immutable HyperbolicIntercept{T} <: Model
    β₀::T
    a::T
    α::T
end

function Base.show(io::IO, ::Type{HyperbolicIntercept})
    return print(io, "Intercept Hyperbolic")
end

function Base.show(io::IO, m::HyperbolicIntercept)
    @printf(
        io,
        "Intercept Hyperbolic (β₀ = %.3f, a = %.3f, α = %.3f)",
        m.β₀,
        m.a,
        m.α,
    )
end

function initial_parameters(::Type{HyperbolicIntercept})
    return Float64[0.0, log(1.0), log(1.0)]
end

canonical_parameters(m::HyperbolicIntercept) = (m.β₀, m.a, m.α)

function HyperbolicIntercept(parameters::Vector)
    β₀ = parameters[1]
    a = exp(parameters[2])
    α = exp(parameters[3])
    return HyperbolicIntercept(β₀, a, α)
end

@inline function predict(
    m::HyperbolicIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, α = m.β₀, m.a, m.α
    u1 = x1 * (1 / (1 + α * t1))
    u2 = x2 * (1 / (1 + α * t2))
    z = β₀ + a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HyperbolicIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, α = m.β₀, m.a, m.α
    d_t2 = 1 / (1 + α * t2)
    d_t1 = 1 / (1 + α * t1)
    gr = (
        1.0,
        (x2 * d_t2 - x1 * d_t1) * a,
        a * (x1 * t1 * d_t1^2 - x2 * t2 * d_t2^2) * α,
    )
    return gr
end
