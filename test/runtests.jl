#
# Correctness Tests
#

using Base.Test
using IntertemporalChoiceHeuristics

my_tests = (
    "show_methods.jl",
    "basic_fits.jl",
    "check_gradients.jl",
    # "random_initial_parameters.jl",
    # "consistent_estimators.jl",
)

println("Running tests:")

for my_test in my_tests
    @printf(" * %s\n", my_test)
    include(my_test)
end
