import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoode.algorithms import NSDER
from pymoode.algorithms import GDE3
from pymoode.survival import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

import numpy as np

def test_many():
    
    np.random.seed(3)
    assert abs(np.random.rand() - 0.5507979025745755) <= 1e-8
    
    problem = get_problem("dtlz2")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    
    NGEN = 250
    POPSIZE = 136
    SEED = 3
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.2, F=(0.0, 1.0), gamma=1e-4,
                survival=RankAndCrowding(crowding_func="mnn"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=True,
                        verbose=True)
    
    igd_gde3 = igd.do(res_gde3.F)
    assert abs(igd_gde3 - 0.04033600998468917) <= 1e-8
    
    nsder = NSDER(ref_dirs=ref_dirs, pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 1.0), gamma=1e-4)
    
    res_nsder = minimize(problem,
                        nsder,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=True,
                        verbose=True)
    
    igd_nsder = igd.do(res_nsder.F)
    assert abs(igd_nsder - 0.00292746167097894) <= 1e-8
    
    
    
    