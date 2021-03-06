import numpy as np
from sklearn.neighbors import NearestNeighbors
from pymoo.core.indicator import Indicator

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
        knn = NearestNeighbors(n_neighbors=2, metric="manhattan")
        knn.fit(F)
        
        #Get distances (nearest neighbor is the point itself)
        _d, _ = knn.kneighbors(F.copy(), return_distance=True)
        d = _d[:, -1]
        dm = np.mean(d)
        
        #Get spacing
        S = np.sqrt(np.sum(np.square(d - dm)) / n_points)
        
        return S