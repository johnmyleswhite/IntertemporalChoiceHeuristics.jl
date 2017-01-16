immutable HomotheticHyperbolic{T} <: Model
    a::T
    α::T
end

function Base.show(io::IO, ::Type{HomotheticHyperbolic})
    return print(io, "Homothetic Hyperbolic")
end

function Base.show(io::IO, m::HomotheticHyperbolic)
    @printf(
        io,
        "Homothetic Hyperbolic (a = %.3f, α = %.3f)",
        m.a,
        m.α,
    )
end

function initial_parameters(::Type{HomotheticHyperbolic})
    return Float64[log(1.0), log(1.0)]
end

canonical_parameters(m::HomotheticHyperbolic) = (m.a, m.α)

function HomotheticHyperbolic(parameters::Vector)
    a = exp(parameters[1])
    α = exp(parameters[2])
    return HomotheticHyperbolic(a, α)
end

@inline function predict(
    m::HomotheticHyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α = m.a, m.α
    u1 = log(x1 * (1 / (1 + α * t1)))
    u2 = log(x2 * (1 / (1 + α * t2)))
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HomotheticHyperbolic,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, α = m.a, m.α
    d_t2 = 1 / (1 + α * t2)
    d_t1 = 1 / (1 + α * t1)
    gr = (
        (log(x2 * d_t2) - log(x1 * d_t1)) * a,
        a * (t1 * d_t1 - t2 * d_t2) * α,
    )
    return gr
end
