# External
import warnings

# pymoo imports
from pymoo.operators.mutation.nom import NoMutation
from pymoo.core.infill import InfillCriterion

# pymoode imports
from pymoode.operators.des import DES
from pymoode.operators.dex import DEX


# =========================================================================================================
# Implementation
# =========================================================================================================

class Variant(InfillCriterion):

    def __init__(self,
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 mutation=None,
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

        mutation : Mutation, optional
            Pymoo's mutation operator after crossover. Defaults to NoMutation().
        
        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """

        # Fix deprecated pm kwargs
        kwargs, mutation = _fix_deprecated_pm_kwargs(kwargs, mutation)

        # Default initialization of InfillCriterion
        super().__init__(eliminate_duplicates=None, **kwargs)

        # Parse the information from the string
        _, selection_variant, n_diff, crossover_variant, = variant.split("/")
        n_diffs = int(n_diff)

        # When "to" in variant there are more than 1 difference vectors
        if "-to-" in variant:
            n_diffs += 1

        # Define parent selection operator
        self.selection = DES(selection_variant)

        # Default value for F
        if F is None:
            F = (0.0, 1.0)

        # Define crossover strategy (DE mutation is included)
        self.crossover = DEX(variant=crossover_variant,
                             CR=CR,
                             F=F,
                             gamma=gamma,
                             n_diffs=n_diffs,
                             at_least_once=True,
                             de_repair=de_repair)

        # Define posterior mutation strategy and repair
        self.mutation = mutation if mutation is not None else NoMutation()

    def _do(self, problem, pop, n_offsprings, **kwargs):

        # Select parents including donor vector
        parents = self.selection(problem, pop, n_offsprings, self.crossover.n_parents, to_pop=True, **kwargs)

        # Perform mutation included in DEX and crossover
        off = self.crossover(problem, parents, **kwargs)

        # Perform posterior mutation and repair if passed
        off = self.mutation(problem, off)

        return off


def _fix_deprecated_pm_kwargs(kwargs, mutation):
    if "pm" in kwargs:
        warnings.warn(
            "pm is deprecated; use 'mutation' for compatibility purposes with pymoo",
            DeprecationWarning, 2
        )
        if mutation is None:
            mutation = kwargs["pm"]

    return kwargs, mutation
