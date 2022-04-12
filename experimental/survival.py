import numpy as np
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population


class ConsRankAndCrwoding(RankAndCrowdingSurvival):
    
    def do(self,
           problem,
           pop,
           *args,
           n_survive=None,
           return_indices=False,
           **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        #If the split should be done beforehand
        if problem.n_constr > 0:

            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True)

            #Obtain len of feasible
            n_feas = len(feas)

            #Assure there is at least_one survivor
            if n_feas == 0:
                survivors = Population()
            else:
                survivors = super()._do(problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            #Calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            #If infeasible solutions need to be added
            if n_remaining > 0:
                
                #Maximum rank among feasible solutions
                if n_feas == 0:
                    max_rank = 1
                else:
                    max_rank = pop[feas].get("rank").max() + 1
                
                #Constraints to new ranking
                G = pop[infeas].get("G")
                G = np.where(G < 0.0, 0.0, G)
                #G[G < 0.0] = 0.0
                
                #Fronts in infeasible population
                infeas_fronts = self.nds.do(G, n_stop_if_ranked=n_remaining)
                
                #Iterate over fronts
                for k, front in enumerate(infeas_fronts):

                    #Save ranks
                    pop[infeas][front].set("rank", k + max_rank)
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
            survivors = self._do(problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors
    

class ExtendedRCSurvival(RankAndCrowdingSurvival):

    def __init__(self, nds=None, limit=None) -> None:
        super().__init__(nds=nds)
        self.nds_archive = Population()
        self.limit = limit

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        
        if self.limit is None:
            limit = int(len(pop)/2)
        else:
            limit = self.limit
        
        pop = Population.merge(pop, self.nds_archive)

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []
        
        # nds archive
        J = []
        nds_to_archive = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                J = I[(n_survive - len(survivors)):]
                I = I[:(n_survive - len(survivors))]
                
                # move remaining individuals to external archive
                nds_to_archive.extend(front[J])

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        
        if len(nds_to_archive) <= limit:
            self.nds_archive = pop[nds_to_archive]
        else:
            print(len(nds_to_archive))
            nds_to_archive = nds_to_archive[:limit]
            self.nds_archive = pop[nds_to_archive]

        return pop[survivors]