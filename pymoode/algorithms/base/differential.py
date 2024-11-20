# Native
from typing import Callable, Optional, Tuple, Union

# External
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import (
    DefaultMultiObjectiveTermination,
    DefaultSingleObjectiveTermination,
)
from pymoo.util.display.display import Display
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.misc import has_feasible

from pymoode.algorithms.base.evolutionary import EvolutionaryAlgorithm
from pymoode.operators.variant import DifferentialVariant
from pymoode.survival import RankAndCrowding
from pymoode.survival.replacement import ImprovementReplacement

# =============================================================================
# Implementation
# =============================================================================


class DifferentialEvolution(EvolutionaryAlgorithm):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        sampling: Optional[Sampling] = None,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.7,
        F: Union[Tuple[float, float], float, None] = (0.5, 1.0),
        gamma: float = 1e-4,
        de_repair: Union[Callable, str] = 'bounce-back',
        survival: Optional[Survival] = None,
        advance_after_initial_infill: bool = True,
        output: Optional[Display] = None,
        **kwargs,
    ):
        """
        Base class for Differential Evolution algorithms

        Parameters
        ----------
        pop_size : int, optional
            Population size. Defaults to 100.

        sampling : Sampling, optional
            Sampling strategy of pymoo. Defaults to LHS().

        variant : str, optional
            Differential evolution strategy. Must be a string in the format:
            "DE/selection/n/crossover", in which, n in an integer
            of number of difference vectors, and crossover
            is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'

            The selection strategy 'ranked' might be helpful to improve
            convergence speed without much harm to diversity.
            Defaults to 'DE/rand/1/bin'.

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control
            convergence speed, use lower values.

        F : iterable of float or float, optional
            Scale factor or mutation parameter. Defined in the range (0, 2]
            To reinforce exploration, use higher values; for exploitation,
            use lower values.

        gamma : float, optional
            Jitter deviation parameter. Should be in the range (0, 2).
            Defaults to 1e-4.

        de_repair : str, optional
            Repair of DE mutant vectors. Is either callable or one of:

                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'

            If callable, has the form fun(X, Xb, xl, xu) in which X contains
            mutated vectors including violations, Xb contains reference
            vectors for repair in feasible space, xl is a 1d vector
            of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.

        genetic_mutation, optional
            Pymoo's genetic mutation operator after crossover.
            Defaults to NoMutation().

        survival : Survival, optional
            Replacement survival operator.
            Defaults to ImprovementReplacement().

        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """
        # Set defaults
        if sampling is None:
            sampling = LHS()
        if survival is None:
            survival = ImprovementReplacement()
        if output is None:
            output = SingleObjectiveOutput()

        # Mating
        mating = DifferentialVariant(
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            **kwargs,
        )

        # Number of offsprings at each generation
        n_offsprings = pop_size

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            mating=mating,
            n_offsprings=n_offsprings,
            eliminate_duplicates=None,
            survival=survival,
            output=output,
            advance_after_initial_infill=advance_after_initial_infill,
            **kwargs,
        )

        self.termination = DefaultSingleObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(
            self.problem, infills, None, n_survive=self.pop_size
        )

    def _infill(self):
        infills = self.mating(
            self.problem, self.pop, self.n_offsprings, algorithm=self
        )
        return infills

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, (
            'This algorithms uses the AskAndTell interface'
            ' thus infills must be provided.'
        )

        # One-to-one replacement survival
        self.pop = self.survival.do(self.problem, self.pop, infills)

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get('CV'))]]
        else:
            self.opt = self.pop[self.pop.get('rank') == 0]


class MODE(DifferentialEvolution):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pop_size: int = 100,
        sampling: Optional[Sampling] = None,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.7,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        de_repair: Union[Callable, str] = 'bounce-back',
        survival: Optional[Survival] = None,
        output: Optional[Display] = None,
        **kwargs,
    ):
        # Set defaults
        if sampling is None:
            sampling = LHS()
        if survival is None:
            survival = RankAndCrowding()
        if output is None:
            output = MultiObjectiveOutput()

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            survival=survival,
            output=output,
            **kwargs,
        )

        self.termination = DefaultMultiObjectiveTermination()

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = self.survival.do(
            self.problem, infills, None, n_survive=self.pop_size
        )
