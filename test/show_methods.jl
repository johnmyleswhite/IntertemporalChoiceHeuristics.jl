module TestShowMethods
    verbose = false

    import IntertemporalChoiceHeuristics: models, initial_parameters

    io = IOBuffer()
    for model in models
        println(io, model)
        println(io, model(initial_parameters(model)))
        println(io)
    end

    if verbose
        print(takebuf_string(io))
    end
end
