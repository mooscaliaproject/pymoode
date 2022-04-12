from pymoo.core.mating import Mating

class DEM(Mating):
    
    def do(self, problem, pop, n_offsprings, **kwargs):

        # do the mating
        _off = self._do(problem, pop, n_offsprings, **kwargs)

        # repair the individuals if necessary - disabled if repair is NoRepair
        _off = self.repair.do(problem, _off, **kwargs)

        # eliminate the duplicates - disabled if it is NoRepair
        _off = self.eliminate_duplicates.do(_off, pop, _off)

        return _off