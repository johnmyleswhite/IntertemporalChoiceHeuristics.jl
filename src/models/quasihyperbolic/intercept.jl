immutable QuasiHyperbolicIntercept{T} <: Model
    β₀::T
    a::T
    β::T
    δ::T
end

function Base.show(io::IO, ::Type{QuasiHyperbolicIntercept})
    return print(io, "Intercept Quasi-Hyperbolic")
end

function Base.show(io::IO, m::QuasiHyperbolicIntercept)
    @printf(
        io,
        "Intercept Quasi-Hyperbolic (β₀ = %.3f, a = %.3f, β = %.3f, δ = %.3f)",
        m.β₀,
        m.a,
        m.β,
        m.δ,
    )
end

function initial_parameters(::Type{QuasiHyperbolicIntercept})
    return Float64[0.0, log(1.0), logit(0.5), logit(0.5)]
end

canonical_parameters(m::QuasiHyperbolicIntercept) = (m.β₀, m.a, m.β, m.δ)

function QuasiHyperbolicIntercept(parameters::Vector)
    β₀ = parameters[1]
    a = exp(parameters[2])
    β = invlogit(parameters[3])
    δ = invlogit(parameters[4])
    return QuasiHyperbolicIntercept(β₀, a, β, δ)
end

@inline function predict(
    m::QuasiHyperbolicIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, β, δ = m.β₀, m.a, m.β, m.δ
    u1 = x1 * ifelse(t1 == 0.0, 1.0, β) * δ^t1
    u2 = x2 * ifelse(t2 == 0.0, 1.0, β) * δ^t2
    z = β₀ + a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::QuasiHyperbolicIntercept,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    β₀, a, β, δ = m.β₀, m.a, m.β, m.δ
    βt2 = ifelse(t2 == 0.0, 1.0, β)
    βt1 = ifelse(t1 == 0.0, 1.0, β)
    Δβt2 = ifelse(t2 == 0.0, 0.0, 1.0)
    Δβt1 = ifelse(t1 == 0.0, 0.0, 1.0)
    δt2 = δ^t2
    δt1 = δ^t1
    gr = (
        1.0,
        (x2 * βt2 * δt2 - x1 * βt1 * δt1) * a,
        a * (x2 * Δβt2 * δt2 - x1 * Δβt1 * δt1) * β * (1 - β),
        a * ((x2 * βt2 * t2 * δt2 - x1 * βt1 * t1 * δt1) / δ) * δ * (1 - δ),
    )
    return gr
end
