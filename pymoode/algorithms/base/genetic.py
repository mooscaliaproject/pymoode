from pymoode.algorithms.base.evolutionary import EvolutionaryAlgorithm
from pymoo.core.mating import Mating

class GeneticAlgorithm(EvolutionaryAlgorithm):
    
    def __init__(self,
                 pop_size=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=True,
                 repair=None,
                 **kwargs):
        
        mating = Mating(selection, crossover, mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates)
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            survival=survival,
            mating=mating,
            n_offsprings=n_offsprings,
            eliminate_duplicates=eliminate_duplicates,
            repair=repair, 
            **kwargs,
        )