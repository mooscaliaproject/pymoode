from pymoo.core.mating import Mating
from pymoo.core.mutation import Mutation

class NoMutation(Mutation):
    
    def _do(self, problem, pop, **kwargs):
        return pop
    
    def do(self, problem, pop, **kwargs):
        return pop

#Cria nondominated sorting alternativo para realizar em no método _advance do algoritimo
#O método salva pontos relevantes para reprodução e passa ao mating
#O mating deve ser um alternativo que já tem a população de breeding salva 

class SideMating(Mating):
    
    def __init__(self,
                 selection,
                 crossover,
                 mutation=NoMutation(),
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.breedings = None
    
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
        
        if self.breedings is None:
            breedings = pop
        else:
            breedings = self.breedings
        
        _off = super()._do(problem, breedings, n_offsprings, parents=parents, **kwargs)
        
        return _off
    
    def set_breedings(self, breedings):
        self.breedings = breedings