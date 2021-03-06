immutable QuasiHyperbolic{T} <: Model
    a::T
    β::T
    δ::T
end

Base.show(io::IO, ::Type{QuasiHyperbolic}) = print(io, "Classical Quasi-Hyperbolic")

function Base.show(io::IO, m::QuasiHyperbolic)
    @printf(
        io,
        "Classical Quasi-Hyperbolic (a = %.3f, β = %.3f, δ = %.3f)",
        m.a,
        m.β,
        m.δ,
    )
end

function initial_parameters(::Type{QuasiHyperbolic})
    return Float64[log(1.0), logit(0.5), logit(0.5)]
end

canonical_parameters(m::QuasiHyperbolic) = (m.a, m.β, m.δ)

function QuasiHyperbolic(parameters::Vector)
    a = exp(parameters[1])
    β = invlogit(parameters[2])
    δ = invlogit(parameters[3])
    return QuasiHyperbolic(a, β, δ)
end

@inline function predict(
    m::QuasiHyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, β, δ = m.a, m.β, m.δ
    u1 = x1 * ifelse(t1 == 0.0, 1.0, β) * δ^t1
    u2 = x2 * ifelse(t2 == 0.0, 1.0, β) * δ^t2
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::QuasiHyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, β, δ = m.a, m.β, m.δ
    βt2 = ifelse(t2 == 0.0, 1.0, β)
    βt1 = ifelse(t1 == 0.0, 1.0, β)
    Δβt2 = ifelse(t2 == 0.0, 0.0, 1.0)
    Δβt1 = ifelse(t1 == 0.0, 0.0, 1.0)
    δt2 = δ^t2
    δt1 = δ^t1
    gr = (
        (x2 * βt2 * δt2 - x1 * βt1 * δt1) * a,
        a * (x2 * Δβt2 * δt2 - x1 * Δβt1 * δt1) * β * (1 - β),
        a * ((x2 * βt2 * t2 * δt2 - x1 * βt1 * t1 * δt1) / δ) * δ * (1 - δ),
    )
    return gr
end
