# Native
from abc import abstractmethod
from typing import Optional, Union

# External
import numpy as np

# pymoo
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.core.problem import Problem


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialOperator(Crossover):

    def __init__(self, n_parents=None, **kwargs):
        """White label for differential evolution operators

        Parameters
        ----------
        n_parents : int | None, optional
            Number of parents necessary in its operations. Useful for compatibility with pymoo.
        """
        # __init__ operator
        super().__init__(n_parents=n_parents, n_offsprings=1, prob=1.0, **kwargs)

    @staticmethod
    def default_prepare(pop: Population, parents: Union[Population, np.ndarray]):
        """Utility function that converts population and parents from pymoo Selection to pop and X

        Parameters
        ----------
        pop : Population
            pymoo population

        parents : Population | np.ndarray (n_samples, n_parents) | None
            Parent population or indices

        Returns
        -------
        pop, X : Population (n_samples, n_parents), np.ndarray (n_parents, n_samples, n_var)
            Population and corresponding decision variables
        """
        # Convert pop if parents is not None
        if parents is not None:
            pop = pop[parents]

        # Get all X values for mutation parents
        X = np.swapaxes(pop, 0, 1).get("X")
        return pop, X

    @abstractmethod
    def do(
        self,
        problem: Problem,
        pop: Population,
        parents: Optional[Union[Population, np.ndarray]] = None,
        **kwargs
    ):
        pass

    @abstractmethod
    def _do(self, problem: Problem, X: np.ndarray, **kwargs):
        pass
