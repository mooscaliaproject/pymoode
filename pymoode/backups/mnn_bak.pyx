# distutils: language = c++
# cython: language_level=2, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set as cpp_set


cdef extern from "math.h":
    double HUGE_VAL


def calc_mnn(double[:, :] X, int n_remove=0):

    cdef:
        int N, M, n
        cpp_set[int] extremes
        vector[int] extremes_min, extremes_max

    N = X.shape[0]
    M = X.shape[1]

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M

    extremes_min = c_get_argmin(X)
    extremes_max = c_get_argmax(X)

    extremes = cpp_set[int]()

    for n in extremes_min:
        extremes.insert(n)

    for n in extremes_max:
        extremes.insert(n)
    
    X = c_normalize_array(X, extremes_max, extremes_min)

    return c_calc_mnn(X, n_remove, N, M, extremes)


def calc_2nn(double[:, :] X, int n_remove=0):

    cdef:
        int N, M, n
        cpp_set[int] extremes
        vector[int] extremes_min, extremes_max

    N = X.shape[0]
    M = X.shape[1]

    if n_remove <= (N - M):
        if n_remove < 0:
            n_remove = 0
        else:
            pass
    else:
        n_remove = N - M

    extremes_min = c_get_argmin(X)
    extremes_max = c_get_argmax(X)

    extremes = cpp_set[int]()

    for n in extremes_min:
        extremes.insert(n)

    for n in extremes_max:
        extremes.insert(n)

    X = c_normalize_array(X, extremes_max, extremes_min)
    
    M = 2

    return c_calc_mnn(X, n_remove, N, M, extremes)


cdef c_calc_mnn(double[:, :] X, int n_remove, int N, int M, cpp_set[int] extremes):

    cdef:
        int n, n_removed, k
        cpp_set[int] calc_items
        cpp_set[int] H
        double[:, :] Dnn, Dij, upper_square
        double[:] d
        int[:, :] Mnn
    
    #Define items to calculate distances
    calc_items = cpp_set[int]()
    for n in range(N):
        calc_items.insert(n)
    for n in extremes:
        calc_items.erase(n)
    
    #Define remaining items to evaluate
    H = cpp_set[int]()
    for n in range(N):
        H.insert(n)
    
    #Upper limits for squared l2 distances
    _Xsum = np.sum(X, axis=1, keepdims=True)
    _upper_square = (_Xsum - _Xsum.T) * (_Xsum - _Xsum.T) / M
    upper_square = _upper_square[:, :]

    #Initialize
    n_removed = 0

    #Initialize neighbors and distances
    _Mnn = np.full((N, M), 0, dtype=np.intc)
    _Dnn = np.full((N, M), HUGE_VAL, dtype=np.double)
    _Dij = np.full((N, N), -1.0, dtype=np.double)
    dd = np.full((N,), HUGE_VAL, dtype=np.double)

    Mnn = _Mnn[:, :]
    Dnn = _Dnn[:, :]
    Dij = _Dij[:, :]
    d = dd[:]

    #Fill in neighbors and distance matrix
    c_calc_mnn_iter(
            X,
            Mnn,
            Dnn,
            Dij,
            N, M,
            calc_items,
            H,
            upper_square
        )

    #Obtain distance metrics
    c_calc_d(d, Dnn, calc_items, M)

    #While n_remove not acheived
    while n_removed < n_remove:

        #Obtain element to drop
        k = c_get_drop(d, H)
        H.erase(k)

        #Update index
        n_removed = n_removed + 1
        if n_removed == n_remove:
            break

        else:

            #Get items to be recalculated
            calc_items = c_get_calc_items(Mnn, Dnn, H, k, M)
            for n in extremes:
                calc_items.erase(n)
            
            #Fill in neighbors and distance matrix
            c_calc_mnn_iter(
                    X,
                    Mnn,
                    Dnn,
                    Dij,
                    N, M,
                    calc_items,
                    H,
                    upper_square
                )

            #Obtain distance metrics
            c_calc_d(d, Dnn, calc_items, M)

    return dd


cdef c_calc_mnn_iter(
    double[:, :] X,
    int[:, :] Mnn,
    double[:, :] Dnn,
    double[:, :] Dij,
    int N, int M,
    cpp_set[int] calc_items,
    cpp_set[int] H,
    double[:, :] upper_square
    ):

    cdef:
        int i, j, m, MM
        double dij
    
    #MM might be different from M in 2NN
    MM = X.shape[1]
    
    #Iterate over items to calculate
    for i in calc_items:

        #Iterate over elements in X
        for j in H:
            
            #Calculate distance if elements satisfy rule
            if ((j != i)
                and (upper_square[i, j] <= Dnn[i, M-1])
                and (Dij[i, j] <= Dnn[i, M-1])):
                
                #Calculate if different from -1
                if Dij[i, j] == -1.0:

                    #Squared distance
                    dij = 0
                    for m in range(MM):
                        dij = dij + (X[j, m] - X[i, m]) * (X[j, m] - X[i, m])
                    
                    #Fill values
                    Dij[i, j] = dij
                    Dij[j, i] = Dij[i, j]

                else:
                    dij = Dij[i, j]
                
                #Iterate over current values
                for m in range(M):
                    
                    #Check is any should be replaced
                    if (dij <= Dnn[i, m]):
                        
                        #Replace higher values
                        Dnn[i, m + 1:] = Dnn[i, m:-1]
                        Mnn[i, m + 1:] = Mnn[i, m:-1]
                        
                        #Replace current value
                        Dnn[i, m] = dij
                        Mnn[i, m] = j
                        
                        break


#Calculate crowding metric
cdef c_calc_d(double[:] d, double[:, :] Dnn, cpp_set[int] calc_items, int M):

    cdef:
        int i, m
    
    for i in calc_items:

        d[i] = 1
        for m in range(M):
            d[i] = d[i] * Dnn[i, m]


#Returns indexes of items to be recalculated after removal
cdef cpp_set[int] c_get_calc_items(
    int[:, :] Mnn,
    double[:, :] Dnn,
    cpp_set[int] H,
    int k, int M):

    cdef:
        int i, m
        cpp_set[int] calc_items
    
    calc_items = cpp_set[int]()

    for i in H:

        for m in range(M):

            if Mnn[i, m] == k:

                Dnn[i, :] = HUGE_VAL
                calc_items.insert(i)
    
    return calc_items


#Returns elements to remove based on crowding metric d and heap of remaining elements H
cdef int c_get_drop(double[:] d, cpp_set[int] H):

    cdef:
        int i, min_i
        double min_d

    min_d = HUGE_VAL
    min_i = 0

    for i in H:

        if d[i] <= min_d:
            min_d = d[i]
            min_i = i
    
    return min_i


#Returns vector of positions of minimum values along axis 0 of a 2d memoryview
cdef vector[int] c_get_argmin(double[:, :] X):

    cdef:
        int N, M, min_i, n, m
        double min_val
        vector[int] indexes
    
    N = X.shape[0]
    M = X.shape[1]

    indexes = vector[int]()
    
    for m in range(M):

        min_i = 0
        min_val = X[0, m]

        for n in range(N):

            if X[n, m] < min_val:

                min_i = n
                min_val = X[n, m]
        
        indexes.push_back(min_i)
    
    return indexes


#Returns vector of positions of maximum values along axis 0 of a 2d memoryview
cdef vector[int] c_get_argmax(double[:, :] X):

    cdef:
        int N, M, max_i, n, m
        double max_val
        vector[int] indexes
    
    N = X.shape[0]
    M = X.shape[1]

    indexes = vector[int]()
    
    for m in range(M):

        max_i = 0
        max_val = X[0, m]

        for n in range(N):

            if X[n, m] > max_val:

                max_i = n
                max_val = X[n, m]
        
        indexes.push_back(max_i)
    
    return indexes


#Performs normalization of a 2d memoryview
cdef double[:, :] c_normalize_array(double[:, :] X, vector[int] extremes_max, vector[int] extremes_min):

    cdef:
        int N = X.shape[0]
        int M = X.shape[1]
        int n, m, l, u
        double l_val, u_val, diff_val
        vector[double] min_vals, max_vals
    
    min_vals = vector[double]()
    max_vals = vector[double]()

    m = 0
    for u in extremes_max:
        u_val = X[u, m]
        max_vals.push_back(u_val)
        m = m + 1
    
    m = 0
    for l in extremes_min:
        l_val = X[l, m]
        min_vals.push_back(l_val)
        m = m + 1
    
    for m in range(M):

        diff_val = max_vals[m] - min_vals[m]
        if diff_val == 0.0:
            diff_val = 1.0

        for n in range(N):

            X[n, m] = (X[n, m] - min_vals[m]) / diff_val
    
    return X