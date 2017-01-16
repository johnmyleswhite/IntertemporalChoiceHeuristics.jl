immutable Tradeoff{T} <: Model
    a::T
    k::T
    γx::T
    γt::T
end

Base.show(io::IO, ::Type{Tradeoff}) = print(io, "Heuristic Tradeoff")

function Base.show(io::IO, m::Tradeoff)
    @printf(
        io,
        "Heuristic Tradeoff (a = %.3f, k = %.3f, γx = %.3f, γt = %.3f)",
        m.a,
        m.k,
        m.γx,
        m.γt,
    )
end

function initial_parameters(::Type{Tradeoff})
    return Float64[log(1.0), log(1.0), log(1.0), log(1.0)]
end

canonical_parameters(m::Tradeoff) = (m.a, m.k, m.γx, m.γt)

function Tradeoff(parameters::Vector)
    a = exp(parameters[1])
    k = exp(parameters[2])
    γx = exp(parameters[3])
    γt = exp(parameters[4])
    return Tradeoff(a, k, γx, γt)
end

@inline ϕ(χ, γ) = log(1 + γ * χ) / γ

@inline ϕ′(χ, γ) = ((γ * χ) / (1 + γ * χ) - log(1 + γ * χ)) / γ^2

@inline function predict(m::Tradeoff, x2::Real, t2::Real, x1::Real, t1::Real)
    a, k, γx, γt = m.a, m.k, m.γx, m.γt
    δϕx = ϕ(x2, γx) - ϕ(x1, γx)
    δϕt = ϕ(t2, γt) - ϕ(t1, γt)
    z = a * (δϕx - k * δϕt)
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::Tradeoff,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    a, k, γx, γt = m.a, m.k, m.γx, m.γt
    δϕx = ϕ(x2, γx) - ϕ(x1, γx)
    δϕt = ϕ(t2, γt) - ϕ(t1, γt)
    δϕ′x = ϕ′(x2, γx) - ϕ′(x1, γx)
    δϕ′t = ϕ′(t2, γt) - ϕ′(t1, γt)
    gr = (
        (δϕx - k * δϕt) * a,
        -a * δϕt * k,
        a * δϕ′x * γx,
        a * -k * δϕ′t * γt,
    )
    return gr
end
