#
# Correctness Tests
#

using Base.Test
using IntertemporalChoiceHeuristics

my_tests = (
    "show.jl",
    "fits.jl",
    "gradients.jl",
    "optimization_algorithms.jl",
    "random_initial_parameters.jl",
    "consistency.jl",
)

println("Running tests:")

@testset "All tests" begin
    for my_test in my_tests
        @printf(" * %s\n", my_test)
        include(my_test)
    end
end
