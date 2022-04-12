import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, leaders
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival

class RankAndClusterSurvival(Survival):

    def __init__(self, nds=None, cluster="ward", dist="cityblock") -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.cluster = cluster
        self.dist = dist

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                
                t = n_survive - len(survivors)
                
                Fi = F[front]
                Fi = (Fi - Fi.min(axis=0)) / (Fi.max(axis=0) - Fi.min(axis=0))
                Ci, _ = kmeans2(Fi, t)
                Y = cdist(Ci, Fi, self.dist)
                I = np.argmin(Y, axis=1)
                #Zi = linkage(Fi, method=self.cluster, metric=self.dist, optimal_ordering=True)
                #Ti = fcluster(Zi, t, criterion="maxclust")
                #Li, Mi = leaders(Zi, Ti)
                #_, I = np.unique(Ti, return_index=True)
                I = I[:t]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]