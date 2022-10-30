import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoode.algorithms import NSDER
from pymoode.algorithms import GDE3
from pymoode.survival import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

import numpy as np

@pytest.mark.parametrize('selection', ["rand", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
@pytest.mark.parametrize('crowding_func', ["mnn", "2nn"])
def test_many_run(selection, crossover, crowding_func):
    
    problem = get_problem("dtlz2")
    
    NGEN = 150
    POPSIZE = 136
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant=f"DE/{selection}/1/{crossover}", CR=0.2, F=(0.0, 1.0), gamma=1e-4,
                survival=RankAndCrowding(crowding_func=crowding_func))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(res_gde3.opt) > 0
    

def test_many_perf():
    
    np.random.seed(3)
    assert abs(np.random.rand() - 0.5507979025745755) <= 1e-8
    
    problem = get_problem("dtlz2")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=15)
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    
    NGEN = 150
    POPSIZE = 136
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.2, F=(0.0, 1.0), gamma=1e-4,
                survival=RankAndCrowding(crowding_func="mnn"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3 = igd.do(res_gde3.F)
    assert abs(igd_gde3 - 0.04011488503871424) <= 1e-8
    
    nsder = NSDER(ref_dirs=ref_dirs, pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 1.0), gamma=1e-4)
    
    res_nsder = minimize(problem,
                        nsder,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsder = igd.do(res_nsder.F)
    assert abs(igd_nsder - 0.004877000918527632) <= 1e-8
    
    
    
    