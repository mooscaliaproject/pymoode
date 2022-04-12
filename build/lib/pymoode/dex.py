import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.core.population import Population
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.crossover.expx import mut_exp
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem

_scale_vector = np.arcsin(1)
_small_number = np.finfo(float).eps


# =========================================================================================================
# Implementation
# =========================================================================================================


def get_relative_positions(dist_i, dist_j):
    
    #Vectors norm
    Dist_i = np.linalg.norm(dist_i, axis=1)
    Dist_j = np.linalg.norm(dist_j, axis=1)
    
    #Possible clip, but likely to be unecessary
    Dist_i = np.clip(Dist_i, _small_number, 1)
    Dist_j = np.clip(Dist_j, _small_number, 1)
    
    #Compute angular position as ratio from scalar product
    ratio_f = np.absolute((dist_i * dist_j).sum(axis=1))/(Dist_i * Dist_j)
    
    #Possible clip, but likely to be unecessary
    ratio_f = np.clip(ratio_f, _small_number, 1 - _small_number)
    
    #Cumpute additional ratio term as Zhang et al. (2021) doi: 10.1016/j.asoc.2021.107317
    max_dist = np.max(np.vstack((Dist_i, Dist_j)), axis=0)
    min_dist = np.min(np.vstack((Dist_i, Dist_j)), axis=0)
    ratio_f = ratio_f*min_dist/max_dist
    
    return ratio_f


class DEM:
    
    def __init__(self,
                 F=None,
                 gamma=1e-4,
                 SA=None,
                 refpoint=1.0,
                 rnd_iter=0,
                 **kwargs):

        #Default value for F
        if F is None:
            F = (0.0, 1.0)
        
        #Define which method will be used to generate F values
        if hasattr(F, "__iter__"):
            self.scale_factor = self._randomize_scale_factor
            self.rand_mult_repair = self._dither_rand_mult_repair
        else:
            self.scale_factor = self._scalar_scale_factor
            self.rand_mult_repair = self._rand_mult_repair
            SA = None
        
        #Define which strategy of rotation will be used
        if gamma is None:
            self.get_diff = self._diff_simple
        else:
            self.get_diff = self._diff_jitter
        
        #Define base functions based on self-adaptive strategy
        if SA is None:
            self.get_fnorm = self._avoid_fnorm
            self.get_diffs = self._simple_diffs
        else:
            self.get_fnorm = self._get_fnorm
            self.get_diffs = self._adaptive_diffs
            
        self.F = F
        self.gamma = gamma
        self.SA = SA
        self.refpoint = refpoint
        self.rnd_iter = rnd_iter
        
    def do(self, problem, pop, parents, **kwargs):

        #Get all X values for mutation parents
        Xr = pop.get("X")[parents.T].copy()
        assert len(Xr.shape) == 3, "Please provide a three-dimensional matrix n_parents x pop_size x n_vars."

        #Obtain normalized function values if necessary
        self.get_fnorm(pop, parents)
        
        #Create mutation vectors
        V, diffs = self.de_mutation(Xr, return_differentials=True)

        #If the problem has boundaries to be considered
        if problem.has_bounds():

            #Iterations in repair by multiplying difference vectors
            for _ in range(self.rnd_iter):
                
                #Find the individuals which are still infeasible
                m = is_out_of_bounds_by_problem(problem, V)
                
                if len(m) > 0:

                    rand_mult = self.rand_mult_repair(m)
                    diffs[m] = rand_mult[:, None] * diffs[m]

                    #Re-perform mutation using a individual-oriented bounce-back to preserve directions
                    V[m] = Xr[0, m] + diffs[m]
                    #V[m] = np.where((V[m] < problem.xl) | (V[m] > problem.xu), Xr[0, m] + diffs[m], V[m])
                    
            #If still infeasible perform a feature-oriented bounce back
            V = bounce_back(V, Xr[0], *problem.bounds())
        
        return Population.new("X", V)
    
    def de_mutation(self, Xr, return_differentials=True):
    
        #Probability
        if (not self.SA is None) and (self.fnorm is None):
            print("WARNING: did not pass a normalized function arg")
        
        n_parents, n_matings, n_var = Xr.shape
        assert n_parents % 2 == 1, "For the differential an odd number of values need to be provided"

        #Build the pairs for the differentials
        pairs = (np.arange(n_parents - 1) + 1).reshape(-1, 2)

        #The differentials from each pair subtraction
        diffs = self.get_diffs(Xr, pairs, n_matings, n_var)

        #Add the difference vectors to the base vector
        Xp = Xr[0] + diffs

        if return_differentials:
            return Xp, diffs
        else:
            return Xp
    
    def _get_fnorm(self, pop, parents):
        self.fnorm = normalize_fun(pop.get("F"))[parents.T].copy()

    def _avoid_fnorm(self, pop, parents):
        self.fnorm = None
    
    def _randomize_scale_factor(self, n_matings):
         return (self.F[0] + np.random.random(n_matings) * (self.F[1] - self.F[0]))
     
    def _scalar_scale_factor(self, n_matings):
         return np.full(n_matings, self.F)
     
    def _diff_jitter(self, F, Xi, Xj, n_matings, n_var):
        F = F[:, None] * (1 + self.gamma * (np.random.random((n_matings, n_var)) - 0.5))
        return F * (Xi - Xj)
    
    def _diff_simple(self, F, Xi, Xj, n_matings, n_var):
        return F[:, None] * (Xi - Xj)
    
    def _simple_diffs(self, Xr, pairs, n_matings, n_var):
        
        #The differentials from each pair subtraction
        diffs = np.zeros((n_matings, n_var))
        
        #For each difference
        for i, j in pairs:
        
            #Obtain F randomized in range
            F = self.scale_factor(n_matings)
            
            #New difference vector
            diff = self.get_diff(F, Xr[i], Xr[j], n_matings, n_var)
            
            #Add the difference to the first vector
            diffs = diffs + diff
        
        return diffs

    def _adaptive_diffs(self, Xr, pairs, n_matings, n_var):
        
        #The differentials from each pair subtraction
        diffs = np.zeros((n_matings, n_var))
        
        #For each difference
        for i, j in pairs:
        
            #Obtain F randomized in range
            F_pred = self.scale_factor(n_matings)
            
            #Get relative positions
            dist_i = self.fnorm[i] - self.refpoint
            dist_j = self.fnorm[j] - self.refpoint
            ratio_f = get_relative_positions(dist_i, dist_j)
            
            #Impact of ratio in F range
            F_init = self.F[0] + (self.F[1] - self.F[0]) * ratio_f
            
            #Obtain F
            F = F_init.copy() + (F_pred - F_init) * ratio_f
            
            #Restore some random F values
            rand_mask = np.random.random(n_matings) > self.SA
            F[rand_mask] = F_pred[rand_mask]
            
            #New difference vector
            diff = self.get_diff(F, Xr[i], Xr[j], n_matings, n_var)
            
            #Add the difference to the first vector
            diffs = diffs + diff
        
        return diffs
    
    def _dither_rand_mult_repair(self, m):
        return self.F[0] + (self.F[1] - self.F[0]) * np.random.random(len(m))

    def _rand_mult_repair(self, m):
        return np.random.random(len(m))
        
    
class DEX(Crossover):
    
    def __init__(self,
                 variant="bin",
                 CR=0.9,
                 F=None,
                 gamma=1e-4,
                 SA=None,
                 refpoint=1.0,
                 n_diffs=1,
                 rnd_iter=0,
                 at_least_once=True,
                 **kwargs):
        
        #Default value for F
        if F is None:
            F = (0.0, 1.0)
        
        #Create instace for mutation
        self.dem = DEM(F=F,
                       gamma=gamma,
                       SA=SA,
                       refpoint=refpoint,
                       rnd_iter=rnd_iter,
                       **kwargs)
    
        self.CR = CR
        self.variant = variant
        self.at_least_once = at_least_once
        
        super().__init__(2 + 2 * n_diffs, 1, **kwargs)
    
    def do(self, problem, pop, parents, **kwargs):
        
        #Get target vectors
        X = pop.get("X")[parents[:, 0]]
        
        #About Xi
        n_matings, n_var = X.shape
        
        #Obtain mutants
        mutants = self.dem.do(problem, pop, parents[:, 1:], **kwargs)
        
        #Obtain V
        V = mutants.get("X")
        
        #Binomial crossover
        if self.variant == "bin":
            M = mut_binomial(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        #Exponential crossover
        elif self.variant == "exp":
            M = mut_exp(n_matings, n_var, self.CR, at_least_once=self.at_least_once)
        else:
            raise Exception(f"Unknown variant: {self.variant}")

        #Add mutated elements in corresponding main parent
        X[M] = V[M]

        off = Population.new("X", X)
        
        return off


def bounce_back(Xp, Xb, xl, xu):
    """Repair strategy

    Args:
        Xp (2d array like): Original vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + np.random.random(len(i)) * (Xb[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - np.random.random(len(i)) * (XU[i, j] - Xb[i, j])

    return Xp

def squared_bounce_back(Xp, Xb, xl, xu):
    """Repair strategy

    Args:
        Xp (2d array like): Original vectors including violations.
        Xb (2d array like): Reference vectors for repair in feasible space.
        xl (1d array like): Lower-bounds.
        xu (1d array like): Upper-bounds.

    Returns:
        2d array like: Repaired vectors.
    """
    
    XL = xl[None, :].repeat(len(Xp), axis=0)
    XU = xu[None, :].repeat(len(Xp), axis=0)

    i, j = np.where(Xp < XL)
    if len(i) > 0:
        Xp[i, j] = XL[i, j] + np.square(np.random.random(len(i))) * (Xb[i, j] - XL[i, j])

    i, j = np.where(Xp > XU)
    if len(i) > 0:
        Xp[i, j] = XU[i, j] - np.square(np.random.random(len(i))) * (XU[i, j] - Xb[i, j])

    return Xp

def normalize_fun(fun):
    
    fmin = fun.min(axis=0)
    fmax = fun.max(axis=0)
    den = fmax - fmin
    
    return (fun - fmin)/den
