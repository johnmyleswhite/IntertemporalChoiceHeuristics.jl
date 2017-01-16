immutable ExponentialIntercept{T} <: Model
    β₀::T
    a::T
    δ::T
end

function Base.show(io::IO, ::Type{ExponentialIntercept})
    return print(io, "Intercept Exponential")
end

function Base.show(io::IO, m::ExponentialIntercept)
    @printf(
        io,
        "Intercept Exponential (β₀ = %.3f, a = %.3f, δ = %.3f)",
        m.β₀,
        m.a,
        m.δ,
    )
end

function initial_parameters(::Type{ExponentialIntercept})
    return Float64[0.0, log(1.0), logit(0.5)]
end

canonical_parameters(m::ExponentialIntercept) = (m.β₀, m.a, m.δ)

function ExponentialIntercept(parameters::Vector)
    β₀ = parameters[1]
    a = exp(parameters[2])
    δ = invlogit(parameters[3])
    return ExponentialIntercept(β₀, a, δ)
end

@inline function predict(
    m::ExponentialIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, δ = m.β₀, m.a, m.δ
    u1 = x1 * δ^t1
    u2 = x2 * δ^t2
    z = β₀ + a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::ExponentialIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, δ = m.β₀, m.a, m.δ
    u2 = x2 * δ^t2
    u1 = x1 * δ^t1
    gr = (
        1.0,
        (u2 - u1) * a,
        a * ((x2 * t2 * δ^t2 - x1 * t1 * δ^t1) / δ) * δ * (1 - δ),
    )
    return gr
end
