# Native
import warnings

# External
import numpy as np

# pymoo imports
from pymoo.operators.mutation.nom import NoMutation
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population
from pymoo.core.problem import Problem

# pymoode imports
from pymoode.operators.des import DES
from pymoode.operators.dex import DEX
from pymoode.operators.dem import DEM


# =========================================================================================================
# Implementation
# =========================================================================================================

class DifferentialVariant(InfillCriterion):

    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 genetic_mutation=None,
                 **kwargs):
        """InfillCriterion class for Differential Evolution

        Parameters
        ----------
        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'

            The selection strategy 'ranked' might be helpful to improve convergence speed without much harm to diversity. Defaults to 'DE/rand/1/bin'.

        CR : float, optional
            Crossover parameter. Defined in the range [0, 1]
            To reinforce mutation, use higher values. To control convergence speed, use lower values.

        F : iterable of float or float, optional
            Scale factor or mutation parameter. Defined in the range (0, 2]
            To reinforce exploration, use higher values; for exploitation, use lower values.

        gamma : float, optional
            Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.

        de_repair : str, optional
            Repair of DE mutant vectors. Is either callable or one of:

                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'

            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.

        genetic_mutation : Mutation, optional
            Pymoo's genetic algorithm's mutation operator after crossover. Defaults to NoMutation().

        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """

        # Fix deprecated pm kwargs
        kwargs, genetic_mutation = _fix_deprecated_pm_kwargs(kwargs, genetic_mutation)

        # Default initialization of InfillCriterion
        super().__init__(eliminate_duplicates=None, **kwargs)

        # Parse the information from the string
        _, selection_variant, n_diff, crossover_variant, = variant.split("/")
        n_diffs = int(n_diff)

        # When "to" in variant there are more than 1 difference vectors
        if "-to-" in variant:
            n_diffs =  n_diffs + 1

        # Define parent selection operator
        self.selection = DES(selection_variant)

        # Define differential evolution mutation
        self.de_mutation = DEM(F=F, gamma=gamma, de_repair=de_repair, n_diffs=n_diffs)

        # Define crossover strategy (DE mutation is included)
        self.crossover = DEX(variant=crossover_variant, CR=CR, at_least_once=True)

        # Define posterior mutation strategy and repair
        self.genetic_mutation = genetic_mutation if genetic_mutation is not None else NoMutation()

    def _do(
        self,
        problem: Problem,
        pop: Population,
        n_offsprings: int,
        **kwargs
    ):

        # Select parents including donor vector
        parents = self.selection(
            problem, pop,
            n_offsprings, self.n_parents,
            to_pop=False, **kwargs,
        )

        # Mutant vectors from DE
        mutants = self.de_mutation(problem, pop, parents, **kwargs)

        # Crossover between mutant vector and target
        matings = self.merge_columnwise(pop, mutants)
        trials = self.crossover(problem, matings, **kwargs)

        # Perform posterior mutation and repair if passed
        off = self.genetic_mutation(problem, trials, **kwargs)

        return off

    @property
    def n_parents(self):
        return self.de_mutation.n_parents

    @staticmethod
    def merge_columnwise(*populations):
        matings = np.column_stack(populations).view(Population)
        return matings



def _fix_deprecated_pm_kwargs(kwargs, genetic_mutation):
    if "pm" in kwargs:
        warnings.warn(
            "pm is deprecated; use 'genetic_mutation'",
            DeprecationWarning, 2
        )
        if genetic_mutation is None:
            genetic_mutation = kwargs["pm"]
    elif "mutation" in kwargs:
        warnings.warn(
            "mutation is deprecated; use 'genetic_mutation'",
            DeprecationWarning, 2
        )
        if genetic_mutation is None:
            genetic_mutation = kwargs["mutation"]

    return kwargs, genetic_mutation
