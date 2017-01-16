IntertemporalChoiceHeuristics.jl
================================

Julia re-implementation of the models analyzed in the Psychological Science
paper called "Money Earlier or Later? Simple Heuristics Explain Intertemporal
Choices Better than Delay Discounting". The original repository is at
[https://github.com/johnmyleswhite/IntertemporalChoiceHeuristics](https://github.com/johnmyleswhite/IntertemporalChoiceHeuristics).

This re-implementation differs from the original in many ways. See the
CHANGELOG.md file for details.

# Usage

```jl
import IntertemporalChoiceHeuristics: load, fit, ITCH

inputs, weights = load()

res = fit(ITCH, inputs, weights)

ITCH(res)
```
