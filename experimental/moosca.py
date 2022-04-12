from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.mating import Mating
from mooscalia.doublemating import DoubleMating
from mooscalia.moodex import MOODEX
from mooscalia.nsga2de import MOODES
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.repair import NoRepair


class MOOSCA(NSGA2):
    
    def __init__(self,
                 pop_size=100,
                 variant="DE/front-to-front/1/bin",
                 CR=0.8,
                 F=None,
                 saF=None,
                 sbx=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 dither="vector",
                 jitter=False,
                 refpoint=None,
                 min_nds=0.1,
                 min_ga=0.5,
                 mutation=PolynomialMutation(prob=None, eta=20),
                 survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 **kwargs):
        
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
        moodes = MOODES(selection_variant,
                        min_nds=min_nds)
        
        #Make F a tuple
        if isinstance(F, (float, int)):
            F = (F, F)
        
        #Define crossover strategy
        moodex = MOODEX(CR=CR,
                        F=F,
                        saF=saF,
                        variant=cross_over_variant,
                        n_diffs=n_diffs,
                        dither=dither,
                        jitter=jitter,
                        refpoint=refpoint,
                        only_mutants=only_mutants)
        
        primal_mating = Mating(TournamentSelection(func_comp=binary_tournament), sbx, mutation)
        dual_mating = Mating(moodes, moodex, NoMutation())
        
        mating = DoubleMating(primal_mating, dual_mating, min_ga=min_ga,
                              repair=NoRepair(),
                              eliminate_duplicates=DefaultDuplicateElimination(),
                              n_max_iterations=100)
        
        #Init from pymoo's NSGA2
        super().__init__(pop_size=pop_size,
                         mating=mating,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         **kwargs)