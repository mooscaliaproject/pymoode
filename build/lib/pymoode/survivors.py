import numpy as np
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.util.misc import find_duplicates


def get_crowding_function(label):

    if label == "cd":
        fun = calc_crowding_distance
    elif label == "ce":
        fun = calc_crowding_entropy
    elif label == "dhv":
        fun = diag_hv_crowding_distance
    elif label == "cdsd":
        fun = calc_crowding_squared_diag
    elif label == "cde":
        fun = calc_crowding_diagonal_entropy
    else:
        raise KeyError("Crwoding function not defined")
    return fun

class RankSurvival(Survival):

    def __init__(self,
                 nds=None,
                 rule="full",
                 crowding_func="cd"):
        """A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some crowding metric.

        Args:
            nds (str or None, optional): Pymoo type of non-dominated sorting. Defaults to None.
            rule (str, optional): Rule to remove individuals exceeding popsize. Options are:
                "full", "sqrt", and "single". The rule "full" corresponds to the original version,
                in which all individuals exceeding the front are removed at onde. Other options remove
                individuals recursively and recalculate metrics at each iteration, which helps improving
                diversity, although is more computationally expensive. Defaults to "full".
            crowding_func (str or callable, optional): Crowding metric. Options are:
                "cd": crowding distances
                "ce": crowding entropy
                "cdsd": squared diagonal of crowding distances hypercube
                "cde": crowding diagonal entropy
                "dhv": diagonal of hypervolume added from neighbors.
                If callable, it takes objective functions as the unique argument and must return metrics.
                Defaults to "cd".
        """
        
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
                
                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func(F[front, :])
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                
                if self.rule == "full":
                    n_remove = len(survivors) + len(front) - n_survive
                elif self.rule == "single":
                    n_remove = 1
                elif self.rule == "sqrt":
                    n_remove = int(np.sqrt(len(survivors) + len(front) - n_survive))
                
                I = I[:-n_remove]
                front = front[I]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return pop[survivors]


class ConstrainedRankSurvival(Survival):
    
    def __init__(self, nds=None, ranking=None):
        """The Rank and Crowding survival approach for handling constraints proposed on
        GDE3 by Kukkonen, S. and Lampinen, J. (2005).

        Args:
            nds (str or None, optional): Pymoo type of non-dominated sorting. Defaults to None.
            ranking (Survival, optional): Basic survival operator. Defaults to None,
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


def calc_crowding_entropy(F, filter_out_duplicates=True):
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


def calc_crowding_squared_diag(F, filter_out_duplicates=True):
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
        dl = dist.copy()[:-1] / norm
        du = dist.copy()[1:] / norm
        
        #Fix nan
        dl[np.isnan(dl)] = 0.0
        du[np.isnan(du)] = 0.0
        
        #Total distance
        cd = dl + du

        #Get diagonals
        J = np.argsort(I, axis=0)
        _diag = np.square(cd[J, np.arange(n_obj)]).sum(axis=1)

        #Save the final vector which sets the crowding diagonals for duplicates to zero to be eliminated
        diag = np.zeros(n_points)
        diag[is_unique] = _diag

    return diag


def calc_crowding_diagonal_entropy(F, filter_out_duplicates=True):
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
        dl = dist.copy()[:-1] / norm
        du = dist.copy()[1:] / norm
        
        #Fix nan
        dl[np.isnan(dl)] = 0.0
        du[np.isnan(du)] = 0.0
        
        #Get diagonals
        J = np.argsort(I, axis=0)
        diag_lower = np.sqrt(np.square(dl[J, np.arange(n_obj)]).sum(axis=1))
        diag_upper = np.sqrt(np.square(du[J, np.arange(n_obj)]).sum(axis=1))
        
        #Fix nan
        _inf = np.isinf(diag_lower) | np.isinf(diag_upper)
        diag_lower[_inf] = 1.0
        diag_upper[_inf] = 1.0
        
        #Total distance
        diag = diag_lower + diag_upper

        #Get relative positions
        pl = (diag_lower / diag)
        pu = (diag_upper / diag)

        #Entropy
        entropy = -(pl * np.log2(pl) + pu * np.log2(pu))
        
        #Diagonal crowding entropy
        _ced = entropy * diag
        _ced[_inf] = np.inf

        #Save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        ced = np.zeros(n_points)
        ced[is_unique] = _ced

    return ced


def diag_hv_crowding_distance(F, filter_out_duplicates=True):
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
        dist_scaled = dist.copy()[1:] / norm
        dist_scaled[np.isnan(dist_scaled)] = 0.0
        
        #Diagonal of neighbors HV
        J = np.argsort(I, axis=0)
        _dhv = np.square(dist_scaled[J, np.arange(n_obj)]).sum(axis=1) / n_obj

        #Save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        dhv = np.zeros(n_points)
        dhv[is_unique] = _dhv

    return dhv

