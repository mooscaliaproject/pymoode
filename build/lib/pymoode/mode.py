from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoode.de import InfillDE
from pymoode.survivors import RankSurvival
from pymoo.core.population import Population
from pymoo.util.dominator import get_relation



# =========================================================================================================
# Implementation
# =========================================================================================================


class MODE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.7,
                 F=None,
                 gamma=1e-4,
                 SA=None,
                 pm=None,
                 repair="bounce-back",
                 survival=RankSurvival(),
                 survival_type="combined",
                 **kwargs):
        
        """
        This is a white label class for multi-objective differential evolution. It is used to create GDE3
        and NSDE algorithms.

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
                Defaults to 0.2.
            F (iterable of float or float, optional): Scale factor or mutation parameter. Defined in the range (0, 2]
                To reinforce exploration, use higher lower bounds; for exploitation, use lower values.
                Defaults to (0.0, 1.0).
            gamma (float, optional): Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.
            SA (float, optional): Probability of using self-adaptive scale factor. Defaults to None.
            refpoint (float or array, optional): Reference point for distances in self-adapting strategy. Defaults to None.
            posterior (Mutation, optional): Pymoo's mutation operators after crossover. Defaults to NoMutation().
            pm (Mutation, optional): Pymoo's mutation operators after crossover. Defaults to NoMutation().
            reapair (Repair, optional): Repair of mutant vectors. Is either callable or one of:
                "bounce-back"
                "midway"
                "rand-init"
                "to-bounds"
                If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors
                including violations, Xb contains reference vectors for repair in feasible space, 
                xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
                Defaults to "bounce-back".
            survival (Survival, optional): Pymoo's survival strategy. Defaults to RankSurvival() with bulk removal ("full")
                and crowding distances ("cd").
                In GDE3, the survival strategy is applied after a one-to-one comparison between child vector
                and corresponding parent when both are non-dominated by the other.
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
        
        self._advance = self._advance_combined if survival_type == "combined" \
            else self._advance_mixed
    
    def _advance_combined(self, infills=None, **kwargs):
        
        return super()._advance(infills=infills, **kwargs)

    def _advance_mixed(self, infills=None, **kwargs):
        
        assert infills is not None, "This algorithms uses the AskAndTell interface thus 'infills' must to be provided."

        #The individuals that are considered for the survival later and final survive
        survivors = []

        # now for each of the infill solutions
        for k in range(len(self.pop)):

            #Get the offspring an the parent it is coming from
            off, parent = infills[k], self.pop[k]

            #Check whether the new solution dominates the parent or not
            rel = get_relation(parent, off)

            #If indifferent we add both
            if rel == 0:
                survivors.extend([parent, off])

            #If offspring dominates parent
            elif rel == -1:
                survivors.append(off)

            #If parent dominates offspring
            else:
                survivors.append(parent)

        #Create the population
        survivors = Population.create(*survivors)

        #Perform a survival to reduce to pop size
        self.pop = self.survival.do(self.problem, survivors, n_survive=self.n_offsprings)