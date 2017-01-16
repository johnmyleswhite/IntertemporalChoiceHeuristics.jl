immutable HomotheticSystem2{T} <: Model
    a::T
    ω::T
    δ₁::T
    δ₂::T
end

Base.show(io::IO, ::Type{HomotheticSystem2}) = print(io, "Homothetic System-2")

function Base.show(io::IO, m::HomotheticSystem2)
    @printf(
        io,
        "Homothetic System-2 (a = %.3f, ω = %.3f, δ₁ = %.3f, δ₂ = %.3f)",
        m.a,
        m.ω,
        m.δ₁,
        m.δ₂,
    )
end

function initial_parameters(::Type{HomotheticSystem2})
    return Float64[log(1.0), logit(0.5), logit(0.25), logit(0.75)]
end

canonical_parameters(m::HomotheticSystem2) = (m.a, m.ω, m.δ₁, m.δ₂)

function HomotheticSystem2(parameters::Vector)
    a = exp(parameters[1])
    ω = invlogit(parameters[2])
    δ₁ = invlogit(parameters[3])
    δ₂ = invlogit(parameters[4])
    return HomotheticSystem2(a, ω, δ₁, δ₂)
end

@inline function predict(
    m::HomotheticSystem2,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, ω, δ₁, δ₂ = m.a, m.ω, m.δ₁, m.δ₂
    u1 = log(x1 * (ω * δ₁^t1 + (1 - ω) * δ₂^t1))
    u2 = log(x2 * (ω * δ₁^t2 + (1 - ω) * δ₂^t2))
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HomotheticSystem2,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, ω, δ₁, δ₂ = m.a, m.ω, m.δ₁, m.δ₂
    d_t2 = ω * δ₁^t2 + (1 - ω) * δ₂^t2
    d_t1 = ω * δ₁^t1 + (1 - ω) * δ₂^t1
    u2 = log(x2 * d_t2)
    u1 = log(x1 * d_t1)
    gr = (
        (u2 - u1) * a,
        a * ((δ₁^t2 - δ₂^t2) / d_t2 - (δ₁^t1 - δ₂^t1) / d_t1) * ω * (1 - ω),
        a * ((ω * t2 * δ₁^(t2 - 1)) / d_t2 - (ω * t1 * δ₁^(t1 - 1)) / d_t1) * δ₁ * (1 - δ₁),
        a * (((1 - ω) * t2 * δ₂^(t2 - 1)) / d_t2 - ((1 - ω) * t1 * δ₂^(t1 - 1)) / d_t1) * δ₂ * (1 - δ₂),
    )
    return gr
end
