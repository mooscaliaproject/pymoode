import numpy as np
from pymoo.core.indicator import Indicator
from scipy.spatial.distance import pdist, squareform

class SpacingIndicator(Indicator):
    
    def __init__(self,
                 zero_to_one=False,
                 ideal=None,
                 nadir=None):
        
        super().__init__(zero_to_one=zero_to_one,
                         ideal=ideal,
                         nadir=nadir)
    
    def _do(self, F, *args, **kwargs):
        
        #Get F dimensions
        n_points, n_obj = F.shape
        
        #knn
        D = squareform(pdist(F, metric="cityblock"))
        d = np.partition(D, 1, axis=1)[:, 1]
        dm = np.mean(d)
        
        #Get spacing
        S = np.sqrt(np.sum(np.square(d - dm)) / n_points)
        
        return S