function stderrs{T <: Model}(::Type{T}, inputs::Vector, weights::Vector)
    se = sqrt.(diag(cov(T, inputs, weights)))
    return se
end
