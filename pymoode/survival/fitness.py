# Native
from typing import Optional

# External
import numpy as np

# pymoo imports
from pymoo.core.survival import Survival
from pymoo.core.population import Population
from pymoo.core.problem import Problem


# =========================================================================================================
# Implementation
# =========================================================================================================

class BaseFitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(
        self,
        problem: Problem,
        pop: Population,
        n_survive: Optional[int]=None,
        **kwargs
    ):
        return pop[:n_survive]


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(
        self,
        problem: Problem,
        pop: Population,
        n_survive: Optional[int]=None,
        **kwargs
    ):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]
