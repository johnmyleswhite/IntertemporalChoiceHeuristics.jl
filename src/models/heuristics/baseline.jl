immutable Baseline{T} <: Model
    β₀::T
end

Base.show(io::IO, ::Type{Baseline}) = print(io, "Heuristic Baseline")

function Base.show(io::IO, m::Baseline)
    @printf(
        io,
        "Heuristic Baseline (β₀ = %.3f)",
        m.β₀,
    )
end

initial_parameters(::Type{Baseline}) = Float64[0.0]

canonical_parameters(m::Baseline) = (m.β₀, )

function Baseline(parameters::Vector)
    β₀ = parameters[1]
    return Baseline(β₀)
end

@inline function predict(m::Baseline, x2::Real, t2::Real, x1::Real, t1::Real)
    β₀ = m.β₀
    z = β₀
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::Baseline,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    gr = (1.0, )
    return gr
end
