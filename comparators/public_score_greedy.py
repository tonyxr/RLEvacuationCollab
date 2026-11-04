"""Reference plug-in: the front-aware public-score greedy rule.

It reproduces ``objective_greedy`` through the plug-in path and exists for
two reasons: the hook's bitwise regression test, and as the template for
new comparators (copy this file, keep the constructor and ``select``
signatures, replace the scoring).
"""

from __future__ import annotations

from DecisionInterface import (
    ObjectiveMatchedGreedyHeuristic,
    public_objective_kwargs,
    public_regional_objective_values,
)


class PublicScoreGreedy:
    INFORMATION_CONTRACT = {
        "role": "plug_in_reference",
        "label": "front-aware public-score greedy via the plug-in path",
        "information": "the public enriched regional observation only",
        "timing": "installs at every epoch",
    }

    def __init__(self, *, core, params):
        if params:
            raise ValueError(f"PublicScoreGreedy takes no parameters, got {sorted(params)}")
        self._core = core
        self._policy = ObjectiveMatchedGreedyHeuristic()

    def select(self, observation, deterministic=True):
        return self._policy.select(
            observation,
            candidate_values=public_regional_objective_values(
                observation, **public_objective_kwargs(self._core)
            ),
            deterministic=True,
        )
