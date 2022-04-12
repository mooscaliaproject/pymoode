"""Created by

Bruno Scalia C. F. Leite, 2021

"""

import numpy as np
from pymoo.operators.mutation.nom import NoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.selection import Selection
from mooscalia.moodex import MOODEX


class NSDE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 variant="DE/ranked/1/bin",
                 CR=0.9,
                 F=None,
                 SA=None,
                 dither="vector",
                 jitter=True,
                 gamma=1e-4,
                 refpoint=None,
                 mutation=NoMutation(),
                 survival=RankAndCrowdingSurvival(),
                 n_offsprings=None,
                 **kwargs):
        
        """NSDE is an algorithm proposed by Leite et al. (2021) that combines NSGA-II sorting and survival strategies \
            to DE mutation and crossover with self-adaptative mutation (scale factor) F parameter. It is implemented using \
                pymoo's basic structure.

        Args:
            pop_size (int, optional): Population size. Defaults to 100.
            variant (str, optional): Differential evolution strategy. Must be a string in the format: \
                "DE/selection/n/crossover", in which, n in an integer, crossover is either "bin" or "exp", \
                    and selection is one of the following:\
                            "ranked"
                            "rand"
                Defaults to "DE/ranked/1/bin".
            CR (float, optional): Crossover parameter. Defaults to 0.9.
            F (float, tuple, or None, optional): Mutation parameter. If using self-adaptative strategy, \
                must be either a tuple of None. If None, it is set to (0, 1). Defaults to None.
            SA (float or None, optional): Self-adaptative F probability. If None, the strategy is avoided. Defaults to None.
            min_nds (float, optional): Minimum relative size of elite members in front strategies. Defaults to 0.2.
            dither (str, optional): Type of dither operation. "vector" is strongly recommended. Defaults to "vector".
            jitter (bool, optional): Either or not to use jitter. Defaults to False.
            refpoint (float, array-like, or None, optional): Reference point in self-adaptative strategy. \
                If None, it is set to 1.0. Defaults to None.
            mutation ([type], optional): [description]. Defaults to NoMutation().
            survival ([type], optional): [description]. Defaults to RankAndCrowdingSurvival().
            eliminate_duplicates (bool, optional): [description]. Defaults to True.
            n_offsprings ([type], optional): [description]. Defaults to None.
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
        
        #Define parent selection operator
        selection = MOODES(selection_variant)
        
        #Make F a tuple
        if isinstance(F, (float, int)):
            F = (F, F)
        
        #Define crossover strategy
        crossover = MOODEX(CR=CR,
                            F=F,
                            SA=SA,
                            variant=cross_over_variant,
                            dither=dither,
                            jitter=jitter,
                            gamma=gamma,
                            refpoint=refpoint,
                            n_diffs=n_diffs,
                            n_iter=1)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=False,
                         n_offsprings=n_offsprings,
                         **kwargs)

#This is the core selection class
class MOODES(Selection):

    def __init__(self,
                 variant,
                 **kwargs):
        
        super().__init__()
        self.variant = variant

    def _do(self, pop, n_select, n_parents, **kwargs):
        
        #Obtain number of elements in population
        n_pop = len(pop)
        
        #For most variants n_select must be equal to len(pop)
        variant = self.variant
            
        if variant == "ranked":
            """Proposed by Zhang et al. (2021). doi.org/10.1016/j.asoc.2021.107317"""
            P = self._ranked(pop, n_select, n_parents)
            
        else:
            P = self._rand(pop, n_select, n_parents)

        return P
    
    def _rand(self, pop, n_select, n_parents, **kwargs):
        
        #len of pop
        n_pop = len(pop)

        #Base form
        P = np.empty([n_select, n_parents], dtype=int)
        P.fill(np.nan)
        
        #Fill first column with corresponding parent
        P[:, 0] = np.arange(n_pop)

        #Fill next columns in loop
        for j in range(1, n_parents):
            
            P[:, j] = np.random.choice(n_pop, n_select)            
            reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
            
            while np.any(reselect):
                P[reselect, j] = np.random.choice(n_pop, reselect.sum())
                reselect = (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)
        
        return P
    
    def _ranked(self, pop, n_select, n_parents, **kwargs):
        
        P = self._rand(pop, n_select, n_parents, **kwargs)
        P[:, 1:] = rank_sort(P[:, 1:], pop)
        
        return P
    

def ranks_from_cv(pop):
    
    ranks = pop.get("rank")
    cv_elements = ranks == None
    
    if np.any(cv_elements):
        ranks[cv_elements] = np.arange(len(pop))[cv_elements]
    
    return ranks

def rank_sort(P, pop):
    
    ranks = ranks_from_cv(pop)
    
    sorted = np.argsort(ranks[P], axis=1, kind="stable")    
    S = np.take_along_axis(P, sorted, axis=1)

    P[:, 0] = S[:, 0]
    
    n_diffs = int((P.shape[1] - 1) / 2)

    for j in range(1, n_diffs + 1):
        P[:, 2*j - 1] = S[:, j]
        P[:, 2*j] = S[:, -j]
    
    return P

def reiforce_directions(P, pop):
    
    ranks = ranks_from_cv(pop)
    
    ranks = ranks[P] 
    S = P.copy()
    
    n_diffs = int(P.shape[1] / 2)

    for j in range(0, n_diffs):
        bad_directions = ranks[:, 2*j] > ranks[:, 2*j + 1]
        P[bad_directions, 2*j] = S[bad_directions, 2*j + 1]
        P[bad_directions, 2*j + 1] = S[bad_directions, 2*j]
    
    return P
