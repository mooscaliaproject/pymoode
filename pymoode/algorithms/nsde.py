from typing import Callable, Optional, Tuple, Union

from pymoo.core.population import Population
from pymoo.core.survival import Survival

# pymoode imports
from pymoode.algorithms.base.differential import MODE

# =============================================================================
# Implementation
# =============================================================================


class NSDE(MODE):
    """
    NSDE is an algorithm that combines that combines NSGA-II sorting
    and survival strategies
    to DE mutation and crossover.

    For many-objective problems, try using NSDE-R, GDE3-MNN, or GDE3-2NN.

    For Bi-objective problems, survival = RankAndCrowding(crowding_func='pcd')
    is very effective.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.7,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        de_repair: Union[Callable, str] = 'bounce-back',
        survival: Optional[Survival] = None,
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            survival=survival,
            **kwargs,
        )

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            'This algorithms uses the AskAndTell interface thus'
            " 'infills' must to be provided."
        )

        # Merge in mu + lambda style
        pop = Population.merge(self.pop, infills)

        # Perform a survival to reduce to pop size
        self.pop = self.survival.do(
            self.problem, pop, n_survive=self.n_offsprings
        )
