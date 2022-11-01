from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoode.algorithms._de import InfillDE
from pymoode.survival._classes import RankAndCrowding
from pymoode.operators.dex import _validate_deprecated_repair


# =========================================================================================================
# Implementation
# =========================================================================================================


class NSDE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=None,
                 gamma=1e-4,
                 pm=None,
                 de_repair="bounce-back",
                 survival=RankAndCrowding(),
                 **kwargs):
        """
        NSDE is an algorithm that combines that combines NSGA-II sorting and survival strategies 
        to DE mutation and crossover.
        
        For many-objective problems, try using NSDE-R, GDE3-MNN, or GDE3-2NN.
        
        For Bi-objective problems, survival = RankAndCrowding(crowding_func='pcd') is very effective.

        Parameters
        ----------
        pop_size : int, optional
            Population size. Defaults to 100.
            
        sampling : Sampling, optional
            Sampling strategy of pymoo. Defaults to LHS().
            
        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:
            
                - "ranked'
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
            
        pm : Mutation, optional
            Pymoo's mutation operators after crossover. Defaults to NoMutation().
            
        de_reapair : str or callable, optional
            Repair of mutant vectors. Is either callable or one of:
        
                - 'bounce-back'
                - 'midway'
                - 'rand-init'
                - 'to-bounds'
            
            If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for de_repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
            Defaults to 'bounce-back'.
            
        survival : Survival, optional
            Pymoo's survival strategy.
            Defaults to RankAndCrowding() with crowding distances ('cd').
            In GDE3, the survival strategy is applied after a one-to-one comparison between child vector and corresponding parent when both are non-dominated by the other.
        """
        
        de_repair, kwargs["repair"] = _validate_deprecated_repair(de_repair, **kwargs)
        
        # Number of offsprings at each generation
        n_offsprings = pop_size
        
        # Mating
        mating = InfillDE(variant=variant,
                          CR=CR,
                          F=F,
                          gamma=gamma,
                          pm=pm,
                          de_repair=de_repair)
        
        # Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         mating=mating,
                         survival=survival,
                         eliminate_duplicates=False,
                         n_offsprings=n_offsprings,
                         **kwargs)
