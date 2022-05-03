# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector

def iterate_sum(x):
    return c_iterate_sum(x)

cdef double c_iterate_sum(double[:] x):

    cdef:
        int n_points, i
        double a

    n_points = x.shape[0]

    a = 0.0

    for i in range(n_points):

        a = a + x[i]

    return a

def calc_mnn(X, calc_items=None):

    #Number of functions
    N, M = X.shape
    
    #Define eval if None
    if calc_items is None: calc_items = np.arange(N)
    
    #Upper limits for squared l2 distances
    _Xsum = np.sum(X, axis=1, keepdims=True)
    upper_square = (_Xsum - _Xsum.T)
    upper_square = upper_square * upper_square / M


cdef vector[int[:, :], int[:, :]] c_calc_mnn(
    double[:, :] X, int N, int M, vector[int] calc_items,
    double[:, :] upper_square
    ):

    cdef:
        int i, j, k
        double dij
        int[:, :] Mnn
        double[:, :] Dij
    
    #Initialize neighbors and distances
    Mnn = np.zeros((N, M), dtype=int)
    Dij = np.full((N, M), np.inf, dtype=float)
    
    #Iterate over items to calculate
    for i in calc_items:

        #Iterate over elements in X
        for j in range(N):
            
            #Calculate distance if elements satisfy rule
            if (upper_square[i, j] <= max(Dij[i])) and (j != i):
                
                #Squared distance
                dij = 0
                for k in range(M):
                    dij = dij + (X[j, k] - X[i, k]) * (X[j, k] - X[i, k])
                
                #Iterate over current values
                for k in range(M):
                    
                    #Check is any should be replaced
                    if dij <= Dij[i, k]:
                        
                        #Replace higher values
                        Dij[i, k + 1:] = Dij[i, k:-1]
                        Mnn[i, k + 1:] = Mnn[i, k:-1]
                        
                        #Replace current value
                        Dij[i, k] = dij
                        Mnn[i, k] = j
                        
                        break
            
            else:
                s = s + 1
                
    print(s)
    
    return [Dij, Mnn]





