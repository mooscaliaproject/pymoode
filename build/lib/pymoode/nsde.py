"""Created by

Bruno Scalia C. F. Leite, 2021

"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoode.de import InfillDE
from pymoode.survivors import RankSurvival


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSDE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.9,
                 F=None,
                 gamma=1e-4,
                 SA=None,
                 pm=None,
                 repair="bounce-back",
                 survival=RankSurvival(),
                 **kwargs):
        
        """NSDE is an algorithm that combines that combines NSGA-II sorting and survival strategies
        to DE mutation and crossover following the implementation by Leite et al. (2022)
        with a self-adaptative mutation (scale factor) F parameter as in SA-NSDE.
        When using low CR values (~0.2), try using GDE3.
        For many-objective problems, try using NSDER or RankSurvival with 'mnn' crowding metric.

        Args:
            pop_size (int, optional): Population size. Defaults to 100.
            sampling (Sampling, optional): Sampling strategy of pymoo. Defaults to LHS().
            variant (str, optional): Differential evolution strategy. Must be a string in the format:
                "DE/selection/n/crossover", in which, n in an integer of number of difference vectors,
                and crossover is either "bin" or "exp".
                Selection variants are:
                    - "ranked"
                    - "rand"
                    - "best"
                    - "current-to-best"
                    - "current-to-rand"
                    - "rand-to-best"
                Defaults to "DE/rand/1/bin"
            CR (float, optional): Crossover parameter. Defined in the range [0, 1]
                To reinforce mutation, use higher values. To control convergence speed, use lower values.
                Defaults to 0.9.
            F (iterable of float or float, optional): Scale factor or mutation parameter. Defined in the range (0, 2]
                To reinforce exploration, use higher lower bounds; for exploitation, use lower values.
                Defaults to (0.0, 1.0).
            gamma (float, optional): Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.
            SA (float, optional): Probability of using self-adaptive scale factor. Defaults to None.
            pm (Mutation, optional): Pymoo's mutation operators after crossover. Defaults to NoMutation().
            reapair (Repair, optional): Repair of mutant vectors. Is either callable or one of:
                "bounce-back"
                "midway"
                "rand-init"
                "to-bounds"
                If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors
                including violations and Xb contains reference vectors for repair in feasible space.
                Defaults to "bounce-back".
            survival (Survival, optional): Pymoo's survival strategy. Defaults to RankSurvival() with bulk removal ("full")
                and crowding distances ("cd").
        """
        
        #Number of offsprings at each generation
        n_offsprings = pop_size
        
        #Mating
        mating = InfillDE(variant=variant,
                          CR=CR,
                          F=F,
                          gamma=gamma,
                          SA=SA,
                          pm=pm,
                          repair=repair)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         mating=mating,
                         survival=survival,
                         eliminate_duplicates=False,
                         n_offsprings=n_offsprings,
                         **kwargs)
