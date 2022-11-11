# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set


cdef extern from "math.h":
    double HUGE_VAL


def calc_spacing_distances(double[:, :] X):

    return c_calc_spacing_distances(X)


cdef c_calc_spacing_distances(double[:, :] X):

    cdef:
        int N, M, n, m
        cpp_set[int] calc_items
        double[:, :] D
        double[:] d
        double dij, _dijm

    N = X.shape[0]
    M = X.shape[1]

    # Initialize neighbors and distances
    _D = np.full((N, N), -1.0, dtype=np.double)
    dd = np.full((N,), HUGE_VAL, dtype=np.double)

    D = _D[:, :]
    d = dd[:]

    # Iterate over items to calculate
    for i in range(N):

        # Iterate over elements in X
        for j in range(N):

            # Calculate distance if elements satisfy rule
            if ((j != i) and (D[i, j] <= d[i])):

                # Calculate if different from -1
                if D[i, j] == -1.0:

                    # Squared distance
                    dij = 0
                    for m in range(M):
                        _dijm = X[j, m] - X[i, m]

                        if _dijm >= 0:
                            dij = dij + _dijm

                        else:
                            dij = dij - _dijm

                    # Fill values
                    D[i, j] = dij
                    D[j, i] = D[i, j]

                else:
                    dij = D[i, j]

                # Check is any should be replaced
                if (dij <= d[i]):
                    d[i] = dij

    return dd
