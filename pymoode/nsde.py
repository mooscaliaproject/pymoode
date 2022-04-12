"""Created by

Bruno Scalia C. F. Leite, 2021

"""

from pymoo.operators.mutation.nom import NoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.repair import NoRepair
from pymoode.de import InfillDE


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSDE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/ranked/1/bin",
                 CR=0.9,
                 F=None,
                 gamma=1e-4,
                 SA=0.5,
                 refpoint=None,
                 posterior=NoMutation(),
                 repair=NoRepair(),
                 survival=RankAndCrowdingSurvival(),
                 rnd_iter=1,
                 **kwargs):
        """NSDE is an algorithm that combines that combines NSGA-II sorting and survival strategies
        to DE mutation and crossover following the implementation by Leite et al. (2022)
        with a self-adaptative mutation (scale factor) F parameter.
        When using low CR values (~0.2), try using GDE3.
        For many-objective problems, try using NSDER.

        Args:
            pop_size (int, optional):Population size. Defaults to 100.
            sampling (Sampling, optional): Sampling strategy. Defaults to LHS().
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
                Defaults to "DE/ranked/1/bin"
            CR (float, optional): Crossover parameter. Defined in the range [0, 1]
                To reinforce mutation, use higher values. To control convergence speed, use lower values.
                Defaults to 0.9.
            F (iterable of float or float, optional): Scale factor or mutation parameter. Defined in the range (0, 2]
                To reinforce exploration, use higher lower bounds; for exploitation, use lower values.
                Defaults to (0.0, 1.0).
            gamma (float, optional): Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.
            SA (float, optional): Probability of using self-adaptive scale factor. Defaults to 0.5.
            refpoint (float or array, optional): Reference point for distances in self-adapting strategy. Defaults to None.
            posterior (Mutation, optional): Pymoo's mutation operators after crossover. Defaults to NoMutation().
            reapair (Repair, optional): Pymoo's repair operators after mating. Defaults to NoRepair().
            rnd_iter (int, optional): Number of random repairs to difference vectors violating boundaries. Defaults to 1.
            survival (Survival, optional): Pymoo's survival strategy. Defaults to RankAndCrowdingSurvival().
        """
        
        #Number of offsprings at each generation
        n_offsprings = pop_size
        
        #Mating
        mating = InfillDE(variant=variant,
                          CR=CR,
                          F=F,
                          gamma=gamma,
                          SA=SA,
                          refpoint=refpoint,
                          posterior=posterior,
                          repair=repair,
                          rnd_iter=rnd_iter)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         mating=mating,
                         survival=survival,
                         eliminate_duplicates=False,
                         n_offsprings=n_offsprings,
                         **kwargs)
