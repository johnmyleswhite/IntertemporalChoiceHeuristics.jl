# Changes from R Code

* All model fitting procedures use frequency weights, which alters the behavior
    of the cross-validation procedure as it no longer performs sampling without
    replacement. But the performance gains from using frequency weights are
    considerable because the data can be compressed by a factor of 10x. A
    future iteration may ensure that the cross-validation procedure exactly
    matches the original R code's approach.

* Several new models were implemented:
    * The System-2 model of [van den Bos and McClure](http://onlinelibrary.wiley.com/doi/10.1002/jeab.6/abstract).
    * The Hyperboloid model of [Green and Myerseon](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1382186), which
        is equivalent to the Generalized Hyperbolic model of [Loewenstein and
        Prelec](http://qje.oxfordjournals.org/content/107/2/573.short).
    * Variants of the classical discounting models were added that included
        intercept terms in their logistic noise formulation.

* Analytic gradients were implemented and are used to speed up optimization and
    improve the convergence of the model fitting procedure.

* Convergence is required or an error is thrown. Model fits from an fitting
    process that fail to converge are never silently passed through the code.

* The ad hoc box constraints in the original code were replaced with open
    interval constraints that have been formulated as an unconstrained
    optimization problem to allow the use of a variety of optimization
    algorithms.

* The additional epsilon noise that was used in the original paper to prevent
    the log likelihood function from evaluating to non-finite values was
    removed.

* The approach to rescaling inputs was simplified and is now applied to the
    ITCH and DRIFT in the same way as it is applied to the classical
    discounting models.

* The ITCH and DRIFT models are now fit using the same optimization procedure
    that is applied to the other models.

* The code is much less repetitive because of the use of generic functions.

* The models can be fit to minimize the negative log likelihood of the data or
    to minimize the mean squared error of the model's predictions.

* The initial parameter values for all parameters are chosen systematically as
    follows:
    * Parameters that lie in `(0, 1)` are set to `0.5`.
    * Parameters that lie in `(0, Inf)` are set to `1.0`.
    * Parameters that lie in `(-Inf, +Inf)` are set to `0.0`.
    * One exception is made for the System-2 model, which exhibits perfect
        symmetry and would have symmetric gradients according to the usual
        defaults. The non-identified parameters are both in the interval
        `(0, 1)` and are therefore set to `0.25` and `0.75`.

* The code uses forward-mode automatic differentiation rather than finite
    differences when analytic functions are not available. This substantially
    improves the accuracy of these automatically computed quantities and is
    particularly relevant in this re-implementation as the code now defaults to
    using Newton's method in the unconstrained parameter space.

* A pseudo-loss function that we call extremity was added to our analysis,
    which measures the distance of a predicted probability from `0.5`. This is
    used to diagnose pathologically over-confident model fits.

* New functions were added to use the fitted models as generative models and
    additional functionality was added to evaluate the CLT-derived standard
    errors of model fit using Fisher information.
