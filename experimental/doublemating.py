import math

from pymoo.core.infill import InfillCriterion
from pymoo.core.mating import Mating
from pymoo.core.population import Population


class DoubleMating(InfillCriterion):

    def __init__(self,
                 primal,
                 dual,
                 min_ga=0.5,
                 **kwargs):

        super().__init__(**kwargs)
        self.primal = primal
        self.dual = dual
        self.min_ga = min_ga

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        
        n_feasible = int(sum(pop.get("feasible").flatten()))
        min_ga = int(self.min_ga*n_offsprings)
        n_off_primal = min_ga#max(n_offsprings - n_feasible, min_ga)
        n_off_dual = n_offsprings - n_off_primal

        # do the crossover using the parents index and the population - additional data provided if necessary
        _off_primal = self.primal.do(problem, pop, n_off_primal, parents=parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        _off_dual = self.dual.do(problem, pop, n_off_dual, parents=parents, **kwargs)
        
        _off = Population.merge(_off_primal, _off_dual)

        return _off