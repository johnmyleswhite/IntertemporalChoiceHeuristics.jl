import IntertemporalChoiceHeuristics:
    load,
    fit,
    models,
    evaluation_losses,
    average_loss

import Optim: minimizer

inputs, weights = load()

for model in models
    res = fit(model, inputs, weights)
    parameters = minimizer(res)
    m = model(parameters)
    for loss in evaluation_losses
        l = average_loss(m, loss, inputs, weights)
        @printf(
            "%s\t%s\t%s\t%s\t%s\n",
            m,
            loss,
            l,
            Optim.minimum(res),
            norm(parameters),
        )
    end
end
