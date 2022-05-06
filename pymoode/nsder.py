"""Created by

Bruno Scalia C. F. Leite, 2022

"""
import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.misc import has_feasible
from pymoode.nsde import NSDE

# =========================================================================================================
# Implementation
# =========================================================================================================

class NSDER(NSDE):
    
    def __init__(self,
                 ref_dirs,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/ranked/1/bin",
                 CR=0.9,
                 F=None,
                 gamma=1e-4,
                 SA=0.5,
                 **kwargs):
        
        """NSDE-R is an extension of NSDE to many-objective problems (Reddy & Dulikravich, 2019) using NSGA-III survival.
        In this implementation, features of SA-NSDE (Leite et al., 2022) are incorporated.
        
        S. R. Reddy and G. S. Dulikravich, "Many-objective differential evolution optimization based on reference points: NSDE-R," Struct. Multidisc. Optim., vol. 60, pp. 1455-1473, 2019.

        Args:
            ref_dirs (array like): The reference direction that should be used during the optimization.
                Each row represents a reference line and each column a variable.
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
            rnd_iter (int, optional): Number of random repairs to difference vectors violating boundaries
                before bounce back. Defaults to 1.
            survival (Survival, optional): Pymoo's survival strategy. Defaults to RankAndCrowdingSurvival().
        """
        
        self.ref_dirs = ref_dirs

        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)
            
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         variant=variant,
                         CR=CR,
                         F=F,
                         gamma=gamma,
                         SA=SA,
                         survival=survival,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))
    
    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt