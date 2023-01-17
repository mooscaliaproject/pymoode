import numpy as np
from abc import abstractmethod
from pymoo.core.crossover import Crossover


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
    def default_prepare(pop, parents):
        """Utility function that converts population and parents from pymoo Selection to pop and X

        Parameters
        ----------
        pop : Population
            pymoo population
        
        parents : Population | np.array (n_samples, n_parents) | None
            Parent population or indices

        Returns
        -------
        pop, X : Population (n_samples, n_parents), np.array (n_parents, n_samples, n_var)
            Population and corresponding decision variables
        """
        # Convert pop if parents is not None
        if parents is not None:
            pop = pop[parents]

        # Get all X values for mutation parents
        X = np.swapaxes(pop, 0, 1).get("X")
        return pop, X

    @abstractmethod
    def do(self, problem, pop, parents=None, **kwargs):
        pass
    
    @abstractmethod
    def _do(self, problem, X, **kwargs):
        pass