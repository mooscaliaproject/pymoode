import numpy as np
from pymoo.core.selection import Selection


# =========================================================================================================
# Implementation
# =========================================================================================================


# This is the core differential evolution selection class
class DES(Selection):

    def __init__(self, variant, **kwargs):
        """Differential Evolution parent selection class

        Parameters
        ----------
        variant : str, optional
            Differential evolution strategy. Must be a string in the format: "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either 'bin' or 'exp'. Selection variants are:

                - 'ranked'
                - 'rand'
                - 'best'
                - 'current-to-best'
                - 'current-to-rand'
                - 'rand-to-best'
        """

        super().__init__()
        self.variant = variant

    def _do(self, problem, pop, n_select, n_parents, **kwargs):

        # Obtain number of elements in population
        n_pop = len(pop)

        # For most variants n_select must be equal to len(pop)
        variant = self.variant

        if variant == "ranked":
            """Proposed by Zhang et al. (2021). doi.org/10.1016/j.asoc.2021.107317"""
            P = self._ranked(pop, n_select, n_parents)

        elif variant == "best":
            P = self._best(pop, n_select, n_parents)

        elif variant == "current-to-best":
            P = self._current_to_best(pop, n_select, n_parents)

        elif variant == "current-to-rand":
            P = self._current_to_rand(pop, n_select, n_parents)
        
        elif variant == "rand-to-best":
            P = self._rand_to_best(pop, n_select, n_parents)

        else:
            P = self._rand(pop, n_select, n_parents)

        return P

    def _rand(self, pop, n_select, n_parents, **kwargs):

        # len of pop
        n_pop = len(pop)

        # Base form
        P = np.empty([n_select, n_parents], dtype=int)

        # Fill base vector with corresponding parent
        base = np.arange(n_pop)[:n_select]

        # Fill next columns in loop
        for j in range(n_parents):

            P[:, j] = np.random.choice(n_pop, n_select)
            reselect = get_reselect(P, base, j)

            while np.any(reselect):
                P[reselect, j] = np.random.choice(n_pop, reselect.sum())
                reselect = get_reselect(P, base, j)

        return P

    def _best(self, pop, n_select, n_parents, **kwargs):

        # len of pop
        n_pop = len(pop)

        # Base form
        P = np.empty([n_select, n_parents], dtype=int)

        # Fill base vector with corresponding parent
        base = np.arange(n_pop)[:n_select]

        # Fill first column with best candidate
        P[:, 0] = 0

        # Fill next columns in loop
        for j in range(1, n_parents):

            P[:, j] = np.random.choice(n_pop, n_select)
            reselect = get_reselect(P, base, j)

            while np.any(reselect):
                P[reselect, j] = np.random.choice(n_pop, reselect.sum())
                reselect = get_reselect(P, base, j)

        return P

    def _current_to_best(self, pop, n_select, n_parents, **kwargs):

        # len of pop
        n_pop = len(pop)

        # Base form
        P = np.empty([n_select, n_parents], dtype=int)

        # Fill base vector with corresponding parent
        base = np.arange(n_pop)[:n_select]

        # Fill first column with current candidate
        P[:, 0] = np.arange(n_pop)

        # Fill first direction from current
        P[:, 2] = np.arange(n_pop)

        # Towards best
        P[:, 1] = 0

        # Fill next columns in loop
        for j in range(3, n_parents):

            P[:, j] = np.random.choice(n_pop, n_select)
            reselect = get_reselect(P, base, j)

            while np.any(reselect):
                P[reselect, j] = np.random.choice(n_pop, reselect.sum())
                reselect = get_reselect(P, base, j)

        return P

    def _current_to_rand(self, pop, n_select, n_parents, **kwargs):

        # len of pop
        n_pop = len(pop)

        # Base form
        P = np.empty([n_select, n_parents], dtype=int)

        # Fill base vector with corresponding parent
        base = np.arange(n_pop)[:n_select]

        # Fill first column with current candidate
        P[:, 0] = np.arange(n_pop)

        # Fill first direction from current
        P[:, 2] = np.arange(n_pop)

        # Towards random
        P[:, 1] = np.random.choice(n_pop, n_select)
        reselect = get_reselect(P, base, 1)

        while np.any(reselect):
            P[reselect, 1] = np.random.choice(n_pop, reselect.sum())
            reselect = get_reselect(P, base, 1)

        # Fill next columns in loop
        for j in range(3, n_parents):

            P[:, j] = np.random.choice(n_pop, n_select)
            reselect = get_reselect(P, base, j)

            while np.any(reselect):
                P[reselect, j] = np.random.choice(n_pop, reselect.sum())
                reselect = get_reselect(P, base, j)

        return P
    
    def _rand_to_best(self, pop, n_select, n_parents, **kwargs):
        
        PB = self._best(pop, n_select, n_parents, **kwargs)
        P = PB.copy()
        
        P[:, 0] = PB[:, 1]
        P[:, 1] = PB[:, 0]
        
        return P

    def _ranked(self, pop, n_select, n_parents, **kwargs):

        P = self._rand(pop, n_select, n_parents, **kwargs)
        P[:, 1:] = rank_sort(P[:, 1:], pop)

        return P


def get_reselect(P, base, j):
    return (P[:, j] == base) | (P[:, j].reshape([-1, 1]) == P[:, :j]).any(axis=1)


def ranks_from_cv(pop):

    ranks = pop.get("rank")
    cv_elements = ranks == None

    if np.any(cv_elements):
        ranks[cv_elements] = np.arange(len(pop))[cv_elements]

    return ranks


def rank_sort(P, pop):

    ranks = ranks_from_cv(pop)

    sorted = np.argsort(ranks[P], axis=1, kind="stable")
    S = np.take_along_axis(P, sorted, axis=1)

    P[:, 0] = S[:, 0]

    n_diffs = int((P.shape[1] - 1) / 2)

    for j in range(1, n_diffs + 1):
        P[:, 2*j - 1] = S[:, j]
        P[:, 2*j] = S[:, -j]

    return P


def reiforce_directions(P, pop):

    ranks = ranks_from_cv(pop)

    ranks = ranks[P]
    S = P.copy()

    n_diffs = int(P.shape[1] / 2)

    for j in range(0, n_diffs):
        bad_directions = ranks[:, 2*j] > ranks[:, 2*j + 1]
        P[bad_directions, 2*j] = S[bad_directions, 2*j + 1]
        P[bad_directions, 2*j + 1] = S[bad_directions, 2*j]

    return P