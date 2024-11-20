# pymoo imports
from typing import Callable, Optional, Tuple, Union

from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.util.display.display import Display
from pymoo.util.dominator import get_relation

# pymoode imports
from pymoode.algorithms.base.differential import MODE
from pymoode.survival import RankAndCrowding

# =============================================================================
# Implementation
# =============================================================================


class GDE3(MODE):
    """GDE3 is an extension of DE to multi-objective problems
    using a mixed type survival strategy.
    It is implemented in this version with the same constraint
    handling strategy of NSGA-II by default.

    Derived algorithms GDE3-MNN and GDE3-2NN use by default
    survival `RankAndCrowding` with metrics 'mnn' and '2nn'.

    For many-objective problems, try using NSDE-R, GDE3-MNN, or GDE3-2NN.

    For Bi-objective problems,
    survival = RankAndCrowding(crowding_func='pcd') is very effective.

    Kukkonen, S. & Lampinen, J., 2005.
    GDE3: The third evolution step of generalized differential evolution.
    2005 IEEE congress on evolutionary computation, Volume 1, pp. 443-450.
    """
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        sampling: Optional[Sampling] = None,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.5,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        de_repair: Union[Callable, str] = 'bounce-back',
        survival: Optional[Survival] = None,
        output: Optional[Display] = None,
        **kwargs,
    ):
        super().__init__(
            pop_size,
            sampling,
            variant,
            CR,
            F,
            gamma,
            de_repair,
            survival,
            output,
            **kwargs,
        )

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            'This algorithms uses the AskAndTell interface'
            " thus 'infills' must to be provided."
        )

        # The individuals that are considered for the survival later
        # and final survive
        survivors = []

        # now for each of the infill solutions
        for k in range(len(self.pop)):
            # Get the offspring an the parent it is coming from
            off, parent = infills[k], self.pop[k]

            # Check whether the new solution dominates the parent or not
            rel = get_relation(parent, off)

            # If indifferent we add both
            if rel == 0:
                survivors.extend([parent, off])

            # If offspring dominates parent
            elif rel == -1:
                survivors.append(off)

            # If parent dominates offspring
            else:
                survivors.append(parent)

        # Create the population
        survivors = Population.create(*survivors)

        # Perform a survival to reduce to pop size
        self.pop = self.survival.do(
            self.problem, survivors, n_survive=self.n_offsprings
        )


class GDE3MNN(GDE3):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.5,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        survival: Optional[Survival] = None,
        **kwargs,
    ):
        survival = RankAndCrowding(crowding_func='mnn')
        super().__init__(
            pop_size, variant, CR, F, gamma, survival=survival, **kwargs
        )


class GDE32NN(GDE3):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.5,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        survival: Optional[Survival] = None,
        **kwargs,
    ):
        survival = RankAndCrowding(crowding_func='2nn')
        super().__init__(
            pop_size, variant, CR, F, gamma, survival=survival, **kwargs
        )


class GDE3P(GDE3):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.5,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        survival: Optional[Survival] = None,
        **kwargs,
    ):
        survival = RankAndCrowding(crowding_func='pcd')
        super().__init__(
            pop_size, variant, CR, F, gamma, survival=survival, **kwargs
        )
