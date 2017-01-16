immutable DRIFT{T} <: Model
    β₀::T
    β₁::T
    β₂::T
    β₃::T
    β₄::T
end

Base.show(io::IO, ::Type{DRIFT}) = print(io, "Heuristic DRIFT")

function Base.show(io::IO, m::DRIFT)
    @printf(
        io,
        "Heuristic DRIFT (β₀ = %.3f, β₁ = %.3f, β₂ = %.3f, β₃ = %.3f, β₄ = %.3f)",
        m.β₀,
        m.β₁,
        m.β₂,
        m.β₃,
        m.β₄,
    )
end

initial_parameters(::Type{DRIFT}) = Float64[0.0, 0.0, 0.0, 0.0, 0.0]

canonical_parameters(m::DRIFT) = (m.β₀, m.β₁, m.β₂, m.β₃, m.β₄)

function DRIFT(parameters::Vector)
    β₀, β₁, β₂, β₃, β₄ = (
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
    )
    return DRIFT(β₀, β₁, β₂, β₃, β₄)
end

@inline function predict(m::DRIFT, x2::Real, t2::Real, x1::Real, t1::Real)
    β₀, β₁, β₂, β₃, β₄ = m.β₀, m.β₁, m.β₂, m.β₃, m.β₄
    δx = x2 - x1
    δt = t2 - t1
    z = (
        β₀ +
        β₁ * δx +
        β₂ * (δx / x1) +
        β₃ * ((x2 / x1)^(1 / δt) - 1) +
        β₄ * δt
    )
    p = invlogit(z)
    return p
end

@inline function gradient_component(
    m::DRIFT,
    x2::Real,
    t2::Real,
    x1::Real,
    t1::Real,
)
    δx = x2 - x1
    δt = t2 - t1
    gr = (
        1.0,
        δx,
        (δx / x1),
        ((x2 / x1)^(1 / δt) - 1),
        δt,
    )
    return gr
end
