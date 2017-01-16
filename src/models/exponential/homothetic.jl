immutable HomotheticExponential{T} <: Model
    a::T
    δ::T
end

function Base.show(io::IO, ::Type{HomotheticExponential})
    return print(io, "Homothetic Exponential")
end

function Base.show(io::IO, m::HomotheticExponential)
    @printf(
        io,
        "Homothetic Exponential (a = %.3f, δ = %.3f)",
        m.a,
        m.δ,
    )
end

function initial_parameters(::Type{HomotheticExponential})
    return Float64[log(1.0), logit(0.5)]
end

canonical_parameters(m::HomotheticExponential) = (m.a, m.δ)

function HomotheticExponential(parameters::Vector)
    a = exp(parameters[1])
    δ = invlogit(parameters[2])
    return HomotheticExponential(a, δ)
end

@inline function predict(
    m::HomotheticExponential,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, δ = m.a, m.δ
    u1 = log(x1 * δ^t1)
    u2 = log(x2 * δ^t2)
    z = a * (u2 - u1)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::HomotheticExponential,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, δ = m.a, m.δ
    u2 = log(x2 * δ^t2)
    u1 = log(x1 * δ^t1)
    gr = (
        (u2 - u1) * a,
        a * ((t2 - t1) / δ) * δ * (1 - δ),
    )
    return gr
end
