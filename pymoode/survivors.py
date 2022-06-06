import numpy as np
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance as _calc_crowding_distance
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.util.misc import find_duplicates
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

#Redefine function to be able to support more kwargs
def calc_crowding_distance(F, filter_out_duplicates=True, **kwargs):
    return _calc_crowding_distance(F, filter_out_duplicates=True)


def get_crowding_function(label):

    if label == "cd":
        fun = FunctionalDiversity(calc_crowding_distance)
    elif label == "ce":
        fun = FunctionalDiversity(calc_crowding_entropy)
    #Inefficient form based on full pairwise distances matrix
    elif label == "mnn_bak":
        fun = FunctionalDiversity(_calc_mnn)
    elif label == "mnn":
        fun = MNNDiversity(fast_mode=False)
    elif label == "mnn-fast":
        fun = MNNDiversity(fast_mode=True)
    elif label == "2nn":
        fun = MNNDiversity(fast_mode=False, twonn=True)
    elif label == "2nn-fast":
        fun = MNNDiversity(fast_mode=True, twonn=True)
    else:
        raise KeyError("Crwoding function not defined")
    return fun

class RankSurvival(Survival):

    def __init__(self,
                 nds=None,
                 rule="full",
                 crowding_func="cd"):
        
        """
        A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some crowding metric.

        Parameters
        ----------
        
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        rule : str, optional
            Rule to remove individuals exceeding popsize. Options are:
            
                - 'full'
                - 'sqrt'
                - 'single'
            
            The rule 'full' corresponds to the original version, in which all individuals exceeding the front are removed at onde. Other options remove individuals recursively and recalculate metrics at each iteration, which helps improving diversity, although is more computationally expensive. Defaults to 'full'.
            
        crowding_func : str or callable, optional
            Crowding metric. Options are:
            
                - 'cd': crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Neaest Neighbors
                - '2nn': 2-Neaest Neighbors
                - 'mnn-fast': M-Neaest Neighbors Bulk Removal
                - '2nn-fast': 2-Neaest Neighbors Bulk Removal
                
            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array. The options 'cd' and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using either 'mnn' or '2nn', individuals are already eliminated in a 'single' manner, 
            therefore 'rule' is ignored. To bulk removal, chose '-fast' variants.
            Defaults to 'cd'.
        """
        if crowding_func in ("mnn", "2nn"):
            rule = "full"
        
        if not hasattr(crowding_func, "__call__"):
            crowding_func = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.rule = rule
        self.crowding_func = crowding_func
        
    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # current front sorted by crowding distance if splitting
            while len(survivors) + len(front) > n_survive:
                
                #Define how many will be removed
                if self.rule == "full":
                    n_remove = len(survivors) + len(front) - n_survive
                elif self.rule == "single":
                    n_remove = 1
                elif self.rule == "sqrt":
                    n_remove = int(np.sqrt(len(survivors) + len(front) - n_survive))
                
                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(
                    F[front, :], n_remove=n_remove
                    )
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                
                I = I[:-n_remove]
                front = front[I]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(
                    F[front, :], n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return pop[survivors]


class ConstrainedRankSurvival(Survival):
    
    def __init__(self, nds=None, ranking=None):
        """
        The Rank and Crowding survival approach for handling constraints proposed on
        GDE3 by Kukkonen, S. & Lampinen, J. (2005).

        Parameters
        ----------
        
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.
            
        ranking : Survival, optional
            Basic survival operator that splts by feasibility. 
            Feasible and infeasible solutions are ranked by nds separately. Defaults to None,
            which creates a RankSurvival instance.
        """
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = ranking if ranking is not None else RankSurvival()
    
    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        #If the split should be done beforehand
        if problem.n_constr > 0:

            #Split by feasibility
            feas, infeas = split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True)

            #Obtain len of feasible
            n_feas = len(feas)

            #Assure there is at least_one survivor
            if n_feas == 0:
                survivors = Population()
            else:
                survivors = self.ranking.do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            #Calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            #If infeasible solutions need to be added
            if n_remaining > 0:
                
                #Constraints to new ranking
                G = pop[infeas].get("G")
                G = np.maximum(G, 0)
                
                #Fronts in infeasible population
                infeas_fronts = self.nds.do(G, n_stop_if_ranked=n_remaining)
                
                #Iterate over fronts
                for k, front in enumerate(infeas_fronts):

                    #Save ranks
                    pop[infeas][front].set("cv_rank", k)

                    #Current front sorted by CV
                    if len(survivors) + len(front) > n_survive:
                        
                        #Obtain CV of front
                        CV = pop[infeas][front].get("CV").flatten()
                        I = randomized_argsort(CV, order='ascending', method='numpy')
                        I = I[:(n_survive - len(survivors))]

                    #Otherwise take the whole front unsorted
                    else:
                        I = np.arange(len(front))

                    # extend the survivors by all or selected individuals
                    survivors = Population.merge(survivors, pop[infeas][front[I]])

        else:
            survivors = self.ranking.do(problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors
    

class CrowdingDiversity:
    
    def do(self, F, filter_out_duplicates=True, n_remove=None):
        return self._do(F, filter_out_duplicates=filter_out_duplicates, n_remove=n_remove)
    
    def _do(self, F, filter_out_duplicates=True, n_remove=None):
        pass


class FunctionalDiversity(CrowdingDiversity):
    
    def __init__(self, function=None) -> None:
        self.function = function
        super().__init__()
    
    def _do(self, F, filter_out_duplicates=True, **kwargs):
        return self.function(F, filter_out_duplicates=filter_out_duplicates, **kwargs)
    

class MNNDiversity(CrowdingDiversity):
    
    def __init__(self, fast_mode=False, twonn=False, **kwargs) -> None:
        """
        Kukkonen, S. & Deb, K., 2006. A fast and effective method for pruning of non-dominated solutions in many-objective problems. In: Parallel problem solving from nature-PPSN IX. Berlin: Springer, pp. 553-562

        Parameters
        ----------
        
        fast_mode : bool, optional
            Eliminate all individuals at once. Defaults to False.
            
        twonn : bool, optional
            Use 2-NN instead of than M-NN. Defaults to False.
        """
        super().__init__()
        self.fast_mode = fast_mode
        self.twonn = twonn
    
    def _do(self, F, filter_out_duplicates=True, n_remove=None, **kwargs):
        
        n_points, n_obj = F.shape

        if n_points <= n_obj:
            return np.full(n_points, np.inf)

        else:

            if filter_out_duplicates:
                # filter out solutions which are duplicates - duplicates get a zero finally
                is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
            else:
                # set every point to be unique without checking it
                is_unique = np.arange(n_points)

            # index the unique points of the array
            _F = F[is_unique].copy()
            
            #Break if many duplicates
            if _F.shape[0] <= n_obj:
                return np.full(n_points, np.inf)

            # calculate the norm for each objective - set to NaN if all values are equal
            norm = np.max(_F, axis=0) - np.min(_F, axis=0)
            norm[norm == 0] = 1.0
            
            #F normalized
            _F = (_F - _F.min(axis=0)) / norm
            
            #Create the heap H
            H = np.arange(_F.shape[0])
            
            #Find best at each function and assign np.inf
            ideal_idx = np.argmin(_F, axis=0)
            
            #Create knn instance
            if self.twonn:
                knn = NearestNeighbors(n_neighbors=3)
            else:
                knn = NearestNeighbors(n_neighbors=n_obj + 1)
            
            #Fit knn to front
            knn.fit(_F)
        
            #Get distances and indexes
            _D, _mnn = knn.kneighbors(X=_F, return_distance=True)
            D, Mnn = _D[:, 1:], _mnn[:, 1:]
            
            #Get d metrics
            d = np.prod(D, axis=1)
            d[ideal_idx] = np.inf
            
            #Avoid loop if in fast mode
            if self.fast_mode:
                
                #Initialize as zeros
                d2 = np.zeros(n_points)
                d2[is_unique] = d

                return d2
            
            #Elements to keep
            n_keep = _F.shape[0] - n_remove
            
            _iter = 0
            
            #Remove elements from heap recursively
            while ((H.shape[0] > n_keep) and (H.shape[0] > n_obj + 1)):
                
                #Most crowded element in H
                w = H[np.argmin(d[H])]
                
                #Remove w from H
                H = H[H != w]
                
                #Find neighbors of w
                _wnn = np.any(Mnn[H] == w, axis=1)
                wnn = H[_wnn]
                
                #If any items are to be re-calculated
                if np.any(_wnn):
                    
                    #Fit knn to remaining individuals
                    knn.fit(_F[H])
                
                    #Get distances and indexes
                    _D, _mnn = knn.kneighbors(X=_F[wnn], return_distance=True)
                    D[wnn], Mnn[wnn] = _D[:, 1:], _mnn[:, 1:]
                    
                    #Get d metrics
                    d[wnn] = np.prod(D[wnn], axis=1)
                    d[ideal_idx] = np.inf
                
                #New item removed
                _iter = _iter + 1
                
                #Break by excessive calls
                if _iter > 1e4:
                    print("Breaking max iter in MNN")
                    break
            
            #Initialize as zeros
            d2 = np.zeros(n_points)
            d2[is_unique] = d

            return d2
            

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

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
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
        _ce = _cej.sum(axis=1)

        #Save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        ce = np.zeros(n_points)
        ce[is_unique] = _ce

    return ce


def _calc_mnn(F, filter_out_duplicates=True, **kwargs):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique].copy()

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = 1.0
        
        # F normalized
        _F = (_F - _F.min(axis=0)) / norm
        
        # Distances pairwise (Inefficient)
        D = pdist(_F, metric="euclidean")
        D = squareform(D)
        
        # M neighbors
        M = _F.shape[1]
        _D = np.sort(D, axis=1)[:, 1:M+1]
        
        # Metric d
        _d = np.prod(_D, axis=1)
        
        # Set top performers as np.inf
        _extremes = np.argmin(_F, axis=0)
        _d[_extremes] = np.inf

        # Save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        d = np.zeros(n_points)
        d[is_unique] = _d

    return d
