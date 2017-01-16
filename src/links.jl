"""
The inverse logit function, which is also the CDF of the Logistic(0, 1)
distribution. Note that this function will generate an exact value of 0 when
z <= -710 and will generate an exact value of 1 when z >= 37. This introduces
numeric problems if you provide inputs outside of that range.
"""
@inline invlogit(z::Real) = 1 / (1 + exp(-z))

"""
The logit function, which is also the quantile function of the Logistic(0, 1)
distribution.
"""
@inline logit(p::Real) = log(p / (1 - p))
