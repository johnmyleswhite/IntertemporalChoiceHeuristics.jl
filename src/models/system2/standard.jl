immutable System2{T} <: Model
    a::T
    ω::T
    δ₁::T
    δ₂::T
end

Base.show(io::IO, ::Type{System2}) = print(io, "Classical System-2")

function Base.show(io::IO, m::System2)
    @printf(
        io,
        "Classical System-2 (a = %.3f, ω = %.3f, δ₁ = %.3f, δ₂ = %.3f)",
        m.a,
        m.ω,
        m.δ₁,
        m.δ₂,
    )
end

function initial_parameters(::Type{System2})
    return Float64[log(1.0), logit(0.5), logit(0.25), logit(0.75)]
end

canonical_parameters(m::System2) = (m.a, m.ω, m.δ₁, m.δ₂)

function System2(parameters::Vector)
    a = exp(parameters[1])
    ω = invlogit(parameters[2])
    δ₁ = invlogit(parameters[3])
    δ₂ = invlogit(parameters[4])
    return System2(a, ω, δ₁, δ₂)
end

@inline function predict(
    m::System2,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, ω, δ₁, δ₂ = m.a, m.ω, m.δ₁, m.δ₂
    u1 = x1 * (ω * δ₁^t1 + (1 - ω) * δ₂^t1)
    u2 = x2 * (ω * δ₁^t2 + (1 - ω) * δ₂^t2)
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::System2,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, ω, δ₁, δ₂ = m.a, m.ω, m.δ₁, m.δ₂
    gr = (
        (x2 * (ω * δ₁^t2 + (1 - ω) * δ₂^t2) - x1 * (ω * δ₁^t1 + (1 - ω) * δ₂^t1)) * a,
        a * (x2 * (δ₁^t2 - δ₂^t2) - x1 * (δ₁^t1 - δ₂^t1)) * ω * (1 - ω),
        a * (x2 * ω * t2 * δ₁^(t2 - 1) - x1 * ω * t1 * δ₁^(t1 - 1)) * δ₁ * (1 - δ₁),
        a * (x2 * (1 - ω) * t2 * δ₂^(t2 - 1) - x1 * (1 - ω) * t1 * δ₂^(t1 - 1)) * δ₂ * (1 - δ₂),
    )
    return gr
end
