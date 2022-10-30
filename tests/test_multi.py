import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoode.algorithms import GDE3, NSDE
from pymoode.survival import RankAndCrowding


@pytest.mark.parametrize('crowding_func', ["mnn", "2nn", "cd", "pcd", "ce"])
def test_multi_run(crowding_func):
    
    problem = get_problem("truss2d")

    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), repair="bounce-back",
                survival=RankAndCrowding(crowding_func=crowding_func))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert len(res_gde3.opt) > 0
    

def test_multi_perf():
    
    problem = get_problem("truss2d")
    igd = IGD(pf=problem.pareto_front(), zero_to_one=True)
    
    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), repair="bounce-back",
                survival=RankAndCrowding(crowding_func="cd"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3 = igd.do(res_gde3.F)
    assert abs(igd_gde3 - 0.005859828655308572) <= 1e-8
    
    gde3p = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))

    res_gde3p = minimize(problem,
                        gde3p,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_gde3p = igd.do(res_gde3p.F)
    assert abs(igd_gde3p - 0.004744463013355145) <= 1e-8
    
    nsde = NSDE(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))
        
    res_nsde = minimize(problem,
                        nsde,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    igd_nsde = igd.do(res_nsde.F)
    assert abs(igd_nsde - 0.004562068055351625) <= 1e-8