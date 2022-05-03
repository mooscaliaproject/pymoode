import numpy as np
import cython

def calc_mnn(X: cython.double[:, :], calc_items=None) ->\
    cython.ArrayType[cython.double[:, :], cython.int[:, :]]:
    
    #Define eval if None
    if calc_items is None: calc_items = np.arange(X.shape[0])
    
    N: cython.int
    M: cython.int
    i: cython.int
    j: cython.int
    k: cython.int
    dij: cython.double
    Mnn: cython.int[:, :]
    Dij: cython.double[:, :]
    calc_items: cython.vector[int]
    upper_square: cython.double[:, :]

    #Number of functions
    N, M = X.shape
    
    #Upper limits for squared l2 distances
    _Xsum = np.sum(X, axis=1, keepdims=True)
    upper_square = (_Xsum - _Xsum.T)
    upper_square = upper_square * upper_square / M
    
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
                pass
    
    return [Dij, Mnn]

