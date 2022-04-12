
import numpy as np
from math import floor


class EpsNDS(object):

    def __init__(self, epsilon=None, method="fast_non_dominated_sort") -> None:
        
        FUNCS = {"fast_non_dominated_sort":fast_non_dominated_sort_epsilon,
                 "naive_non_dominated_sort":naive_non_dominated_sort_regular,
                 "efficient_non_dominated_sort":efficient_non_dominated_sort_epsilon}
        
        if not method in FUNCS.keys():
            raise KeyError(f"Method of sorting not available.\nPlease use any of {(FUNCS.keys())}")
        else:
            method = FUNCS[method]

        self.epsilon = epsilon
        self.method = method

    def do(self, F, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        
        F = F.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        
        func = self.method

        # set the epsilon if it should be set
        if self.epsilon is not None:
            kwargs["epsilon"] = float(self.epsilon)

        fronts = func(F, **kwargs)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = rank_from_fronts(fronts, F.shape[0])
            return fronts, rank

        return fronts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


class EpsDominator(object):

    @staticmethod
    def get_relation(a, b, cva=None, cvb=None, epsilon=0):

        if cva is not None and cvb is not None:
            if cva < cvb:
                return 1
            elif cvb < cva:
                return -1

        val = 0
        for i in range(len(a)):
            if a[i] + epsilon < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] + epsilon < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix_loop(F, G, epsilon=0):
        n = F.shape[0]
        CV = np.sum(G * (G > 0).astype(np.float), axis=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = EpsDominator.get_relation(F[i, :], F[j, :], CV[i], CV[j], epsilon=epsilon)
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(F, _F=None, epsilon=0):

        """
        if G is None or len(G) == 0:
            constr = np.zeros((F.shape[0], F.shape[0]))
        else:
            # consider the constraint violation
            # CV = Problem.calc_constraint_violation(G)
            # constr = (CV < CV) * 1 + (CV > CV) * -1
            CV = Problem.calc_constraint_violation(G)[:, 0]
            constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
        """

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M


def fast_non_dominated_sort_epsilon(F, epsilon=1e-6, **kwargs):
    
    M = EpsDominator.calc_domination_matrix(F, epsilon=epsilon)

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts

def efficient_non_dominated_sort_epsilon(F, strategy="sequential", epsilon=0):
    """
    Efficient Non-dominated Sorting (ENS)
    Parameters
    ----------
    F: numpy.ndarray
        objective values for each individual.
    strategy: str
        search strategy, can be "sequential" or "binary".
    Returns
    -------
        fronts: list
            Indices of the individuals in each front.
    References
    ----------
    X. Zhang, Y. Tian, R. Cheng, and Y. Jin,
    An efficient approach to nondominated sorting for evolutionary multiobjective optimization,
    IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
    """

    assert (strategy in ["sequential", 'binary']), "Invalid search strategy"

    # the shape of the input
    N, M = F.shape

    # do a lexicographic ordering
    I = np.lexsort(F.T[::-1])
    F = F[I]

    # front ranks for each individual
    fronts = []

    for i in range(N):

        if strategy == 'sequential':
            k = sequential_search_regular(F, i, fronts, epsilon=epsilon)
        else:
            k = binary_search_regular(F, i, fronts, epsilon=epsilon)

        # create empty fronts if necessary
        if k >= len(fronts):
            fronts.append([])

        # append the current individual to a front
        fronts[k].append(i)

    # now map the fronts back to the originally sorting
    ret = []
    for front in fronts:
        ret.append(I[front])

    return ret


def sequential_search_regular(F, i, fronts, epsilon=0) -> int:
    """
    Find the front rank for the i-th individual through sequential search.
    Parameters
    ----------
    F: np.ndarray
        the objective values
    i: int
        the index of the individual
    fronts: list
        individuals in each front
    """

    num_found_fronts = len(fronts)
    k = 0  # the front now checked
    current = F[i]
    while True:
        if num_found_fronts == 0:
            return 0
        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k]
        solutions = F[fk_indices[::-1]]
        non_dominated = True
        for f in solutions:
            relation = EpsDominator.get_relation(current, f, epsilon=epsilon)
            if relation == -1:
                non_dominated = False
                break
        if non_dominated:
            return k
        else:
            k += 1
            if k >= num_found_fronts:
                # move the individual to a new front
                return num_found_fronts


def binary_search_epsilon(F, i, fronts, epsilon=0):
    """
    Find the front rank for the i-th individual through binary search.
    Parameters
    ----------
    F: np.ndarray
        the objective values
    i: int
        the index of the individual
    fronts: list
        individuals in each front
    """

    num_found_fronts = len(fronts)
    if num_found_fronts == 0:
        return 0

    k_min = 0  # the lower bound for checking
    k_max = num_found_fronts  # the upper bound for checking
    k = floor((k_max + k_min) / 2 + 0.5)  # the front now checked
    current = F[i]
    while True:

        # solutions in the k-th front, examine in reverse order
        fk_indices = fronts[k - 1]
        solutions = F[fk_indices[::-1]]
        non_dominated = True

        for f in solutions:
            relation = EpsDominator.get_relation(current, f, epsilon=epsilon)
            if relation == -1:
                non_dominated = False
                break

        # binary search
        if non_dominated:
            if k == k_min + 1:
                return k - 1
            else:
                k_max = k
                k = floor((k_max + k_min) / 2 + 0.5)
        else:
            k_min = k
            if k_max == k_min + 1 and k_max < num_found_fronts:
                return k_max - 1
            elif k_min == num_found_fronts:
                return num_found_fronts
            else:
                k = floor((k_max + k_min) / 2 + 0.5)


def naive_non_dominated_sort_regular(F, epsilon=0, **kwargs):
    
    M = EpsDominator.calc_domination_matrix(F, epsilon=epsilon)

    fronts = []
    remaining = set(range(M.shape[0]))

    while len(remaining) > 0:

        front = []

        for i in remaining:

            is_dominated = False
            dominating = set()

            for j in front:
                rel = M[i, j]
                if rel == 1:
                    dominating.add(j)
                elif rel == -1:
                    is_dominated = True
                    break

            if is_dominated:
                continue
            else:
                front = [x for x in front if x not in dominating]
                front.append(i)

        [remaining.remove(e) for e in front]
        fronts.append(front)

    return fronts

