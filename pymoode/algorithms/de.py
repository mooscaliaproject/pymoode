# pymoo imports
from pymoo.operators.sampling.lhs import LHS

# pymoode imports
from pymoode.survival.replacement import ImprovementReplacement
from pymoode.algorithms.base.differential import DifferentialEvolution


# =========================================================================================================
# Implementation
# =========================================================================================================

class DE(DifferentialEvolution):

    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=(0.5, 1.0),
                 gamma=1e-4,
                 de_repair="bounce-back",
                 survival=ImprovementReplacement(),
                 **kwargs):
        """
        Single-objective Differential Evolution proposed by Storn and Price (1997).

        Storn, R. & Price, K., 1997. Differential evolution-a simple and efficient heuristic for global optimization over continuous spaces. J. Glob. Optim., 11(4), pp. 341-359.

        Parameters
        ----------
        pop_size : int, optional
            Population size. Defaults to 100.

        sampling : Sampling, optional
            Sampling strategy of pymoo. Defaults to LHS().

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

        genetic_mutation, optional
            Pymoo's genetic mutation operator after crossover. Defaults to NoMutation().

        survival : Survival, optional
            Replacement survival operator. Defaults to ImprovementReplacement().

        repair : Repair, optional
            Pymoo's repair operator after mutation. Defaults to NoRepair().
        """
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            variant=variant,
            CR=CR,
            F=F,
            gamma=gamma,
            de_repair=de_repair,
            survival=survival,
            **kwargs,
        )

