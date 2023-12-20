# Native
from abc import abstractmethod
from typing import Optional, Union

# External
import numpy as np

# pymoo imports
from pymoo.core.population import Population
from pymoo.core.problem import Problem

# pymoode imports
from pymoode.operators.deop import DifferentialOperator


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialCrossover(DifferentialOperator):

    def __init__(self,
                 variant="bin",
                 CR=0.7,
                 at_least_once=True,
                 **kwargs):
        """Differential evolution crossover
        (DE mutation is considered a part of this operator)

        Parameters
        ----------
        variant : str | callable, optional
            Crossover variant. Must be either "bin", "exp", or callable. By default "bin".
            If callable, it has the form:
            ``cross_function(n_matings, n_var, CR, at_least_once=True)``

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        at_least_once : bool, optional
            Either or not offsprings must inherit at least one attribute from mutant vectors, by default True
        """

        # __init__ operator
        super().__init__(n_parents=2, **kwargs)

        self.CR = CR
        self.variant = variant
        self.at_least_once = at_least_once

    def do(
        self,
        problem: Problem,
        pop: Population,
        parents: Optional[Union[Population, np.ndarray]] = None,
        **kwargs
    ):

        # Convert pop if parents is not None
        pop, X = self.default_prepare(pop, parents)

        # Create child vectors
        U = self._do(problem, X, **kwargs)

        return Population.new("X", U)

    @abstractmethod
    def _do(self, problem: Problem, X: np.ndarray, **kwargs):
        pass


class DEX(DifferentialCrossover):

    def __init__(self,
                 variant="bin",
                 CR=0.7,
                 at_least_once=True,
                 **kwargs):

        super().__init__(
            variant=variant, CR=CR,
            at_least_once=at_least_once,
            **kwargs,
        )

        if self.variant == "bin":
            self.cross_function = cross_binomial
        elif self.variant == "exp":
            self.cross_function = cross_exp
        elif hasattr(self.variant, "__call__"):
            self.cross_function = self.variant
        else:
            raise ValueError("Crossover variant must be either 'bin', 'exp', or callable")

    def _do(self, problem: Problem, X: np.ndarray, **kwargs):

        # Decompose input vector
        V = X[1]
        X_ = X[0]
        U = np.array(X_, copy=True)

        # About X
        n_matings, n_var = X_.shape

        # Mask
        M = self.cross_function(n_matings, n_var, self.CR, self.at_least_once)
        U[M] = V[M]

        return U


# =========================================================================================================
# Crossovers
# =========================================================================================================


# From version 0.5.0 of pymoo
def row_at_least_once_true(M: np.ndarray):

    _, d = M.shape

    for k in np.where(~np.any(M, axis=1))[0]:
        M[k, np.random.randint(d)] = True

    return M


def cross_binomial(n_matings, n_var, prob, at_least_once=True):

    M = np.random.random((n_matings, n_var)) < prob

    if at_least_once:
        M = row_at_least_once_true(M)

    return M


def cross_exp(n_matings, n_var, prob, at_least_once=True):

    # the mask do to the crossover
    M = np.full((n_matings, n_var), False)

    # start point of crossover
    s = np.random.randint(0, n_var, size=n_matings)

    # create for each individual the crossover range
    for i in range(n_matings):

        # the actual index where we start
        start = s[i]
        for j in range(n_var):

            # the current position where we are pointing to
            current = (start + j) % n_var

            # replace only if random value keeps being smaller than CR
            if np.random.random() <= prob:
                M[i, current] = True
            else:
                break

    if at_least_once:
        M = row_at_least_once_true(M)

    return M
