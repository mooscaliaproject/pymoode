"""Created by

Bruno Scalia C. F. Leite, 2021

"""

import numpy as np
from pymoo.operators.mutation.nom import NoMutation
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.selection import Selection
from mooscalia.moodex import MOODEX


class NSGA2DE(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 variant="DE/front-to-front/1/bin",
                 CR=0.8,
                 F=None,
                 SA=None,
                 min_nds=0.2,
                 dither="vector",
                 jitter=False,
                 refpoint=None,
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
        crossover = MOODEX(CR=CR,
                            F=F,
                            SA=SA,
                            variant=cross_over_variant,
                            n_diffs=n_diffs,
                            dither=dither,
                            jitter=jitter,
                            refpoint=refpoint,
                            only_mutants=only_mutants)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         **kwargs)

#This is the core selection class
class MOODES(Selection):

    def __init__(self,
                 variant,
                 min_nds=None,
                 **kwargs):
        
        super().__init__()
        
        if min_nds is None:
            min_nds = 0.0
        
        self.variant = variant
        self.min_nds = min_nds
        self.random_selection = RandomSelection()

    def _do(self, pop, n_select, n_parents, **kwargs):
        
        #Obtain number of elements in population
        n_pop = len(pop)
        
        #Define n_diff
        n_diffs = int((n_parents-1)/2)
        
        #For most variants n_select must be equal to len(pop)
        variant = self.variant
        
        if "front" in variant:
            #Obtains elements in front and set at xp0
            n_front = int(sum(check_front_pop(pop)))
                
            if (n_front < self.min_nds*n_pop) | (n_front == 0):
                n_front = int(self.min_nds*n_pop)

            #the corresponding indexes
            pfront = np.arange(n_front)
            
        #Create offsprings and add it to the data of the algorithm
        P = self.random_selection.do(pop, n_select, n_parents)
        
        n = len(P)
        
        if variant == "front":
            P[:, 0] = np.random.choice(pfront, n)
        
        elif variant == "current-to-front":
            P[:, 0] = np.arange(n)
            P[:, 1] = np.random.choice(pfront, n)
            P[:, 2] = np.arange(n)
            P = reiforce_directions(P, pop, n_diffs)
        
        elif variant == "current-to-rand":
            P[:, 0] = np.arange(n)
            P[:, 2] = np.arange(n)
        
        elif variant == "rand-to-front":
            P[:, 1] = np.random.choice(pfront, n)
            P = reiforce_directions(P, pop, n_diffs)
        
        elif variant == "front-to-front":
            P[:, 0] = np.random.choice(pfront, n)
            P = to_front(P, pfront, n, n_diffs)
            P = reiforce_directions(P, pop, n_diffs)
        
        elif variant == "only-front":
            P = np.random.choice(pfront, P.shape)
            P = reiforce_directions(P, pop, n_diffs)
        
        elif variant == "current-to-pbest":
            #Best pselection fraction of the population
            n_pbest = int(np.ceil(self.min_nds * n_pop))

            #The corresponding indices to select from
            pbest = np.arange(n_pbest)

            P[:, 0] = np.arange(n)
            P[:, 1] = np.random.choice(pbest, n)
            P[:, 2] = np.arange(n)
            
        elif variant == "ranked":
            """Proposed by Zhang et al. (2021). doi.org/10.1016/j.asoc.2021.107317
            """
            P = rank_sort(P, pop, n_diffs)

        return P


def check_front_pop(pop):
    ranks = pop.get("rank")
    result = ranks == 0
    return result

def to_front(P, pfront, n, n_diffs):
    
    n_diffs = int((P.shape[-1] - 1)/2)
    
    for j in range(1, n_diffs + 1):
        P[:, 2*j-1] = np.random.choice(pfront, n)
    
    return P

def rank_sort(P, pop, n_diffs):
    
    ranks = pop.get("rank")
    cv_elements = ranks == None
    ranks = np.where(cv_elements, -1, ranks)
    max_rank = ranks.max()
    ranks = np.where(cv_elements, max_rank + 1, ranks)
    ranks[cv_elements] = ranks[cv_elements] + np.arange(sum(cv_elements))
    
    sorted = np.argsort(ranks[P], axis=1, kind="stable")    
    S = np.take_along_axis(P, sorted, axis=1)
    
    P[:, 0] = S[:, 0]

    for j in range(1, n_diffs + 1):
        P[:, 2*j-1] = S[:, j]
        P[:, 2*j] = S[:, -j]
    
    return P

def reiforce_directions(P, pop, n_diffs):
    
    ranks = pop.get("rank")
    cv_elements = ranks == None
    ranks = np.where(cv_elements, -1, ranks)
    max_rank = ranks.max()
    ranks = np.where(cv_elements, max_rank + 1, ranks)
    ranks[cv_elements] = ranks[cv_elements] + np.arange(sum(cv_elements))
    
    ranks = ranks[P]  
    S = P.copy()

    for j in range(1, n_diffs + 1):
        bad_directions = ranks[:, 2*j-1] > ranks[:, 2*j]
        P[bad_directions, 2*j-1] = S[bad_directions, 2*j]
        P[bad_directions, 2*j] = S[bad_directions, 2*j-1]
    
    return P
