import numpy as np
from pymoo.core.population import Population
from pymoode.nsde import NSDE


class ENSDE(NSDE):
    
    def __init__(self, n_ensembles=2, **kwargs):
        self.n_ensembles = n_ensembles
        super().__init__(**kwargs)
    
    def _infill(self):

        return None

    def _advance(self, infills=None, **kwargs):
        
        #Next generation
        _pop = Population()
        
        #Define segregated groups
        groups = np.random.choice(np.arange(self.n_ensembles), self.pop_size, replace=True)
        
        #For each group do individual runs
        for g in np.arange(self.n_ensembles):
            
            #Filter by group
            g_ = groups == g
            
            if g_.sum() == 0:
                continue
            
            else:
                
                #Select population from group
                popg = self.pop[g_]
                
                #Offspring from group
                offg = self.mating.do(self.problem, popg, len(popg), algorithm=self)
                self.evaluator.eval(self.problem, offg, algorithm=self)
                
                #Survivors from group
                nextg = self.survival.do(self.problem, Population.merge(popg, offg),
                                         n_survive=len(popg), algorithm=self)
                
                _pop = Population.merge(_pop, nextg)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, _pop, n_survive=self.pop_size, algorithm=self)