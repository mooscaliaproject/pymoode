# External
from typing import Tuple, Union

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.misc import has_feasible

# pymoode imports
from pymoode.algorithms.nsde import NSDE

# =============================================================================
# Implementation
# =============================================================================


class NSDER(NSDE):
    """
    NSDE-R is an extension of NSDE to many-objective problems
    (Reddy & Dulikravich, 2019) using NSGA-III survival.

    S. R. Reddy and G. S. Dulikravich,
    "Many-objective differential evolution optimization
    based on reference points: NSDE-R,"
    Struct. Multidisc. Optim., vol. 60, pp. 1455-1473, 2019.
    """
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        ref_dirs: np.ndarray,
        pop_size: int = 100,
        variant: str = 'DE/rand/1/bin',
        CR: float = 0.5,
        F: Union[Tuple[float, float], float, None] = None,
        gamma: float = 1e-4,
        **kwargs,
    ):

        self.ref_dirs = ref_dirs

        if self.ref_dirs is not None:
            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f'WARNING: pop_size={pop_size} is less than'
                    ' the number of reference directions'
                    ' ref_dirs={len(self.ref_dirs)}.\n'
                    'This might cause unwanted behavior of the algorithm. \n'
                    'Please make sure pop_size is equal or'
                    ' larger than the number of reference directions. '
                )

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = DERSurvival(ref_dirs)

        super().__init__(
            pop_size=pop_size,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            survival=survival,
            **kwargs,
        )

    def _setup(self, problem, **kwargs):
        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    'Dimensionality of reference points '
                    'must be equal to the number of objectives: %s != %s'
                    % (self.ref_dirs.shape[1], problem.n_obj)
                )

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get('CV'))]]
        else:
            self.opt = self.survival.opt


class DERSurvival(ReferenceDirectionSurvival):
    def _do(self, problem, pop, *args, n_survive=None, D=None, **kwargs):
        return super()._do(problem, pop, n_survive, D, **kwargs)
