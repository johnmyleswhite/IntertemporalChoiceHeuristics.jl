immutable HomotheticQuasiHyperbolic{T} <: Model
    a::T
    β::T
    δ::T
end

function Base.show(io::IO, ::Type{HomotheticQuasiHyperbolic})
    return print(io, "Homothetic Quasi-Hyperbolic")
end

function Base.show(io::IO, m::HomotheticQuasiHyperbolic)
    @printf(
        io,
        "Homothetic Hyperboloid (a = %.3f, β = %.3f, δ = %.3f)",
        m.a,
        m.β,
        m.δ,
    )
end

function initial_parameters(::Type{HomotheticQuasiHyperbolic})
    return Float64[log(1.0), logit(0.5), logit(0.5)]
end

canonical_parameters(m::HomotheticQuasiHyperbolic) = (m.a, m.β, m.δ)

function HomotheticQuasiHyperbolic(parameters::Vector)
    a = exp(parameters[1])
    β = invlogit(parameters[2])
    δ = invlogit(parameters[3])
    return HomotheticQuasiHyperbolic(a, β, δ)
end

@inline function predict(
    m::HomotheticQuasiHyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, β, δ = m.a, m.β, m.δ
    u1 = log(x1 * ifelse(t1 == 0.0, 1.0, β) * δ^t1)
    u2 = log(x2 * ifelse(t2 == 0.0, 1.0, β) * δ^t2)
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HomotheticQuasiHyperbolic,
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
    gr = (
        (log(x2 * βt2 * δ^t2) - log(x1 * βt1 * δ^t1)) * a,
        a * (Δβt2 / βt2 - Δβt1 / βt1) * β * (1 - β),
        a * ((t2 - t1) / δ) * δ  * (1 - δ),
    )
    return gr
end
