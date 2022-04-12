"""Created by

Bruno Scalia C. F. Leite, 2021

"""

import numpy as np
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.selection import Selection
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from mooscalia.moodex import MOODEX
from mooscalia.nsde import MOODES


class NSGADE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 variant="DE/front-to-front/1/bin",
                 F=None,
                 SA=None,
                 min_nds=0.2,
                 dither="vector",
                 jitter=False,
                 refpoint=None,
                 sbx=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=NoMutation(),
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 **kwargs):
        """
        NSGA-II-DE is an algorithm proposed by Leite et al. (2021) that combines NSGA-II sorting and survival strategies \
            to DE mutation and crossover with self-adaptative mutation (scale factor) F parameter. It is implemented using \
                pymoo's basic structure.

        Args:
            pop_size (int, optional): Population size. Defaults to 100.
            variant (str, optional): Differential evolution strategy. Must be a string in the format: \
                "DE/selection/n/crossover", in which, n in an integer, crossover is either "bin" or "exp", \
                    and selection is one of the following:\
                            "rand"
                            "front"
                            "current-to-front"
                            "rand-to-front"
                            "front-to-front"
                            "current-to-rand"
                            "current-to-pbest"
                Defaults to "DE/front-to-front/1/bin".
            CR (float, optional): Crossover parameter. Defaults to 0.8.
            F (float, tuple, or None, optional): Mutation parameter. If using self-adaptative strategy, \
                must be either a tuple of None. If None, it is set to (0, 1). Defaults to None.
            saF (float or None, optional): Self-adaptative F probability. If None, the strategy is avoided. Defaults to None.
            dither (str, optional): Type of dither operation. "vector" is strogly recommended. Defaults to "vector".
            jitter (bool, optional): Either or not to use jitter. Defaults to False.
            refpoint (float, array-like, or None, optional): Reference point in self-adaptative strategy. \
                If None, it is set to 1.0. For convex Pareto fronts, 1.0 is recommended, whereas for concave, 0.0 is recommended. \
                    Defaults to None.
            pselection (float or None, optional): Probability of elitism in selection. If None, full elitism is used. Defaults to None.
            mutation (Mutation, optional): Mutation operator from pymoo. Defaults to NoMutation().
            survival (Survival, optional): Survival operator from pymoo. Defaults to RankAndCrowdingSurvival().
            eliminate_duplicates (bool, optional): NSGA-II hyperparameter. Defaults to True.
            n_offsprings (int or None, optional): Number of offspring population size. If None, it is equal to population size. Defaults to None.
        """
        
        #Parse the information from the string
        _, selection_variant, n_diff, cross_over_variant, = variant.split("/")
        n_diffs = int(n_diff)
        
        #Number of offsprings at each generation
        if n_offsprings is None:
            n_offsprings = pop_size
        
        #Reference point for self-adaptative mutation hyperparameter
        if refpoint is None:
            refpoint = 1.0
        
        #Avoid unnecessary calculations for duplicates
        only_mutants = not eliminate_duplicates
        
        #Define parent selection operator
        selection = MOODES(selection_variant, min_nds=min_nds)
        
        #Make F a tuple
        if isinstance(F, (float, int)):
            F = (F, F)
            
        #Define crossover strategy
        crossover = SBXDE(F=F,
                          SA=SA,
                          variant=cross_over_variant,
                          dither=dither,
                          jitter=jitter,
                          refpoint=refpoint,
                          sbx=sbx,
                          n_diffs=n_diffs,
                          n_iter=1)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         **kwargs)


class SBXDE(Crossover):
    
    def __init__(self,
                 F=None,
                 SA=None,
                 variant="bin",
                 dither="vector",
                 jitter=False,
                 refpoint=1.0,
                 sbx=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 n_diffs=1,
                 n_iter=1,
                 at_least_once=True,
                 only_mutants=False,
                 **kwargs):
        
        #Define crossover strategy
        moodex = MOODEX(CR=1.0,
                        F=F,
                        SA=SA,
                        variant=variant,
                        n_diffs=n_diffs,
                        dither=dither,
                        jitter=jitter,
                        refpoint=refpoint,
                        n_iter=n_iter,
                        at_least_once=at_least_once,
                        only_mutants=only_mutants)
        
        self.moodex = moodex
        self.sbx = sbx

        super().__init__(1 + 2 * n_diffs, 1, **kwargs)
    
    def do(self, problem, pop, parents, **kwargs):
        
        #Obtain offsprings from DE
        _off_de = self.moodex.do(problem, pop, parents, **kwargs)
        
        #Make intermediate population
        _intermediates = Population.merge(pop, _off_de)
        
        #Number of individuals at each
        n_pop = len(pop)
        n_off = len(_off_de)
        
        #Define new parents indexes for SBX
        _parents_sbx = np.zeros((n_pop, 2), dtype=int)
        _parents_sbx[:, 0] = np.random.permutation(n_pop)
        _parents_sbx[:, 1] = np.random.permutation(n_off) + n_pop
        
        #Obtain offsprings from SBX
        _off = self.sbx.do(problem, _intermediates, _parents_sbx, **kwargs)
        
        return _off
        
        
