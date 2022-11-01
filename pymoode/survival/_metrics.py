import numpy as np
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance as _calc_crowding_distance
from pymoo.util.misc import find_duplicates
from scipy.spatial.distance import pdist, squareform

try:
    from pymoode.cython.info import info
    if info() == "This file was based on pymoo to check compilation success":
        IS_COMPILED = True
    else:
        IS_COMPILED = False
except:
    IS_COMPILED = False

#Redefine function to be able to support more kwargs
def calc_crowding_distance(F, **kwargs):
    return _calc_crowding_distance(F, filter_out_duplicates=True)

if IS_COMPILED:
    
    from pymoode.cython.mnn import calc_mnn, calc_2nn
    from pymoode.cython.pruning_cd import calc_pcd

    def calc_mnn_nds(F, n_remove: int=0):
        return calc_mnn(F, n_remove=n_remove)

    def calc_2nn_nds(F, n_remove: int=0):
        return calc_2nn(F, n_remove=n_remove)

    def calc_pcd_nds(F, n_remove: int=0):
        return calc_pcd(F, n_remove=n_remove)

else:
    
    print("WARNING: Failed to import cython modules. Slow implementation on.")
    from pymoode.misc.mnn import calc_mnn
    from pymoode.misc.pruning_cd import calc_pcd
    
    def calc_mnn_nds(F, n_remove: int=0):
        return calc_mnn(F, n_remove=n_remove, twonn=False)

    def calc_2nn_nds(F, n_remove: int=0):
        return calc_mnn(F, n_remove=n_remove, twonn=True)

    def calc_pcd_nds(F, n_remove: int=0):
        return calc_pcd(F, n_remove=n_remove)

def get_crowding_function(label):

    if label == "cd":
        fun = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=False)
    elif (label == "pcd") or (label == "pruning-cd"):
        fun = FunctionalDiversity(calc_pcd_nds, filter_out_duplicates=True)
    elif label == "ce":
        fun = FunctionalDiversity(calc_crowding_entropy, filter_out_duplicates=True)
    elif label == "mnn":
        fun = FuncionalDiversityMNN(calc_mnn_nds, filter_out_duplicates=True)
    elif label == "2nn":
        fun = FuncionalDiversityMNN(calc_2nn_nds, filter_out_duplicates=True)
    elif hasattr(label, "__call__"):
        fun = FunctionalDiversity(label, filter_out_duplicates=True)
    elif isinstance(label, CrowdingDiversity):
        fun = label
    else:
        raise KeyError("Crwoding function not defined")
    return fun
    

class CrowdingDiversity:
    
    def do(self, F, n_remove=0):
        #Converting types Python int to Cython int would fail in some cases converting to long instead
        n_remove = np.intc(n_remove)
        F = np.array(F, dtype=np.double)
        return self._do(F, n_remove=n_remove)
    
    def _do(self, F, n_remove=None):
        pass


class FunctionalDiversity(CrowdingDiversity):
    
    def __init__(self, function=None, filter_out_duplicates=True):
        self.function = function
        self.filter_out_duplicates = filter_out_duplicates
        super().__init__()
    
    def _do(self, F, **kwargs):
        
        n_points, n_obj = F.shape

        if n_points <= 2:
            return np.full(n_points, np.inf)

        else:

            if self.filter_out_duplicates:
                # filter out solutions which are duplicates - duplicates get a zero finally
                is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
            else:
                # set every point to be unique without checking it
                is_unique = np.arange(n_points)

            # index the unique points of the array
            _F = F[is_unique]
            
            _d = self.function(_F, **kwargs)
            
            d = np.zeros(n_points)
            d[is_unique] = _d
        
        return d


class FuncionalDiversityMNN(FunctionalDiversity):
    
    def _do(self, F, **kwargs):
        
        n_points, n_obj = F.shape

        if n_points <= n_obj:
            return np.full(n_points, np.inf)
        
        else:
            return super()._do(F, **kwargs)
            

def calc_crowding_entropy(F, filter_out_duplicates=True, **kwargs):
    """Wang, Y.-N., Wu, L.-H. & Yuan, X.-F., 2010. Multi-objective self-adaptive differential 
    evolution with elitist archive and crowding entropy-based diversity measure. 
    Soft Comput., 14(3), pp. 193-209.

    Parameters
    ----------
    F : 2d array like
        Objective functions.
        
    filter_out_duplicates : bool, optional
        Defaults to True.

    Returns
    -------
    crowding_enropies : 1d array
    """
    
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dl = dist.copy()[:-1]
    du = dist.copy()[1:]
    
    #Fix nan
    dl[np.isnan(dl)] = 0.0
    du[np.isnan(du)] = 0.0
    
    #Total distance
    cd = dl + du

    #Get relative positions
    pl = (dl[1:-1] / cd[1:-1])
    pu = (du[1:-1] / cd[1:-1])

    #Entropy
    entropy = np.row_stack([np.full(n_obj, np.inf),
                            -(pl * np.log2(pl) + pu * np.log2(pu)),
                            np.full(n_obj, np.inf)])
    
    #Crowding entropy
    J = np.argsort(I, axis=0)
    _cej = cd[J, np.arange(n_obj)] * entropy[J, np.arange(n_obj)] / norm
    _cej[np.isnan(_cej)] = 0.0
    ce = _cej.sum(axis=1)

    return ce


def calc_mnn_fast(F, **kwargs):
    return _calc_mnn_fast(F, F.shape[1], **kwargs)
    
    
def calc_2nn_fast(F, **kwargs):
    return _calc_mnn_fast(F, 2, **kwargs)


def _calc_mnn_fast(F, n_neighbors, **kwargs):

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = 1.0
    
    # F normalized
    F = (F - F.min(axis=0)) / norm
    
    # Distances pairwise (Inefficient)
    D = squareform(pdist(F, metric="sqeuclidean"))
    
    # M neighbors
    M = F.shape[1]
    _D = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]
    
    # Metric d
    d = np.prod(_D, axis=1)
    
    # Set top performers as np.inf
    _extremes = np.concatenate((np.argmin(F, axis=0), np.argmax(F, axis=0)))
    d[_extremes] = np.inf

    return d
