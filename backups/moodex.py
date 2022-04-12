"""
Modified differential evolution crossover
"""


import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem


_scale_vector = np.arcsin(1)
_small_number = np.finfo(float).eps

def de_mutation(X, F, SA=0.5, dither=None, jitter=True, fnorm=None, refpoint=1.0, gamma=1e-4,
                return_differentials=True):
    
    #Probability
    if (not SA is None) and (fnorm is None):
        print("WARNING: did not pass a normalized function arg")
    
    n_parents, n_matings, n_var = X.shape
    assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"
    
    #Separate F into Flow and Fhi
    if hasattr(F, "__iter__"):
        Fl, Fh = F
        
    else:
        Fl, Fh = F, F
        dither = None
    
    # make sure F is a one-dimensional vector
    Fl = np.ones(n_matings) * Fl
    Fh = np.ones(n_matings) * Fh

    #Build the pairs for the differentials
    pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

    #The differentials from each pair subtraction
    diffs = np.zeros((n_matings, n_var))
    
    #No self-adaptive strategy
    if SA is None:
        
        #For each difference
        for i, j in pairs:
        
            #Obtain F randomized in range
            F = randomize_scale_factor(Fl, Fh, n_matings, dither=dither)
            
            #New difference vector
            diff = F[:, None] * (X[i] - X[j])
            
            #Randomize each individual component
            if jitter:
                diff = diff * (1 + gamma * (np.random.random((n_matings, n_var)) - 0.5))
            
            #Add the difference to the first vector
            diffs = diffs + diff
    
    #Using self-adaptive strategy
    else:
        
        #For each difference
        for i, j in pairs:
        
            #Obtain F randomized in range
            F_pred = randomize_scale_factor(Fl, Fh, n_matings, dither=dither)
            
            #Get relative positions
            dist_i = fnorm[i] - refpoint
            dist_j = fnorm[j] - refpoint
            ratio_f = get_relative_positions(dist_i, dist_j)
            
            #Impact of ratio in F range
            F_init = Fl + (Fh - Fl)*ratio_f
            
            #Obtain F
            F = F_init.copy() + (F_pred - F_init)*ratio_f
            
            #Restore some random F values
            rand_mask = np.random.random(n_matings) > SA
            F[rand_mask] = F_pred[rand_mask]
            
            #New difference vector
            diff = F[:, None] * (X[i] - X[j])
            
            #Randomize each individual component
            if jitter:
                diff = diff * (1 + gamma * (np.random.random((n_matings, n_var)) - 0.5))
            
            #Add the difference to the first vector
            diffs = diffs + diff

    # now add the differentials to the first parent
    Xp = X[0] + diffs

    if return_differentials:
        return Xp, diffs
    else:
        return Xp

def randomize_scale_factor(Fl, Fh, n_matings, dither="vector"):
    
    if dither == "vector":
        F = (Fl + np.random.random(n_matings) * (Fh - Fl))
    elif dither == "scalar":
        F = Fl + np.random.random() * (Fh - Fl)
    else:
        F = (Fl + Fh)/2
    
    return F

def get_relative_positions(dist_i, dist_j):
    
    #Vectors norm
    Dist_i = np.linalg.norm(dist_i, axis=1)
    Dist_j = np.linalg.norm(dist_j, axis=1)
    
    #Possible clip, but likely to be unecessary
    Dist_i = np.clip(Dist_i, _small_number, 1)
    Dist_j = np.clip(Dist_j, _small_number, 1)
    
    #Compute angular position as ratio from scalar product
    ratio_f = np.absolute((dist_i*dist_j).sum(axis=1))/(Dist_i*Dist_j)
    
    #Possible clip, but likely to be unecessary
    ratio_f = np.clip(ratio_f, _small_number, 1 - _small_number)
    
    #Cumpute additional ratio term as Zhang et al. (2021) doi: 10.1016/j.asoc.2021.107317
    max_dist = np.max(np.vstack((Dist_i, Dist_j)), axis=0)
    min_dist = np.min(np.vstack((Dist_i, Dist_j)), axis=0)
    ratio_f = ratio_f*min_dist/max_dist
    
    return ratio_f


class MOODEX(Crossover):

    def __init__(self,
                 CR=0.9,
                 F=None,
                 SA=None,
                 variant="bin",
                 dither="vector",
                 jitter=False,
                 gamma=1e-4,
                 refpoint=1.0,
                 n_diffs=1,
                 n_iter=1,
                 at_least_once=True,
                 **kwargs):
        """
        Multiobjective DE crossover class implemented in NSGA-II-DE.

        Args:
            CR (float, optional): Crossover parameter. Defaults to 0.8.
            F (float, tuple, or None, optional): Mutation parameter. If using self-adaptative strategy, \
                must be either a tuple of None. If None, becomes (0, 1). Defaults to None.
            saF (float or None, optional): Self-adaptative probability. If None, strategy is avoided. Defaults to None.
            variant (str, optional): Either "bin" or "exp". Defaults to "bin".
            dither (str, optional): Type of dither operation. "vector" is strogly recommended. Defaults to "vector".
            jitter (bool, optional): Either or not to use jitter. Defaults to False.
            refpoint (float, array-like, or None, optional): Reference point in self-adaptative strategy. \
                If None, it is set to 1.0. For convex fronts, 1.0 is recommended, while for concave, 0.0 is recommended. \
                    Defaults to 1.0.
            n_diffs (int, optional): Number of pertubation vectors. Defaults to 1.
            n_iter (int, optional): Number of iterations to assure bounds are not violated. Defaults to 1.
            at_least_once (bool, optional): Selection parameter. Defaults to True.
            only_mutants (bool, optional): Either or not to previously remove duplicates. Defaults to False.
        """
        if F is None:
            F = (0, 1)
            
        super().__init__(2 + 2 * n_diffs, 1, **kwargs)
        
        self.n_diffs = n_diffs
        self.F = F
        self.CR = CR
        self.SA = SA
        self.variant = variant
        self.at_least_once = at_least_once
        self.dither = dither
        self.jitter = jitter
        self.gamma = gamma
        self.refpoint = refpoint
        self.n_iter = n_iter

    def do(self, problem, pop, parents, **kwargs):

        #Get all X values for each parent in DEX
        X = pop.get("X")[parents.T].copy()
        assert len(X.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

        #About Xr
        n_parents, n_matings, n_var = X.shape
        
        #Target vector of index i
        Xp = X[0]
        
        #Randomly selected vectors to mutation
        Xr = X[1:]

        #A mask over matings that need to be repeated
        m = np.arange(n_matings)

        # if the user provides directly an F value to use
        F = self.F if self.F is not None else (0, 1)

        #Obtain normalized function values if necessary
        if self.SA is None:
            fnorm = np.full((n_matings, 1), None)[parents.T[1:]].copy()
        else:
            fnorm = normalize_fun(pop.get("F"))[parents.T[1:]].copy()
            
        V, diffs = de_mutation(Xr[:, m], F, SA=self.SA, dither=self.dither, jitter=self.jitter,
                               fnorm=fnorm, refpoint=self.refpoint, gamma=self.gamma,
                               return_differentials=True)

        #If the problem has boundaries to be considered
        if problem.has_bounds():

            for k in range(self.n_iter):
                
                #Find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, V)
                
                if len(m) > 0:

                    rand_mult = np.random.random(len(m))
                    diffs[m] = rand_mult[:, None] * diffs[m]

                    #Re-perform mutation
                    V[m] = Xr[0, m] + diffs[m]
                    
            #If still infeasible choose a random value between base vector and bound
            V = bounce_back(V, Xr[0], *problem.bounds())

        #Binomial crossover
        if self.variant == "bin":
            M = mut_binomial(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        #Exponential crossover
        elif self.variant == "exp":
            M = mut_exp(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        else:
            raise Exception(f"Unknown variant: {self.variant}")

        #Add mutated elements in corresponding main parent
        Xp[M] = V[M]

        off = Population.new("X", Xp)
        
        return off

def bounce_back(Xp, Xb, xl, xu):
    
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + np.random.random(len(i)) * (Xb[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - Xb[i, j])

    return Xp

"""def bounce_back(Xp, Xb, xl, xu):
    
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + np.random.random(len(i)) * (Xb[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - Xb[i, j])

    return Xp"""

def normalize_fun(fun):
    
    fmin = fun.min(axis=0)
    fmax = fun.max(axis=0)
    den = fmax - fmin
    
    return (fun - fmin)/den
