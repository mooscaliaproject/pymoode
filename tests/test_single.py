import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoode.algorithms import GDE3
from pymoode.algorithms import DE


def test_single():
    problem = get_problem("rastrigin")
    
    NGEN = 100
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=CR, F=F)

    res_de = minimize(problem,
                    de,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=True,
                    verbose=True)

    assert len(res_de.opt) > 0
    assert res_de.F <= 1e-6
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=CR, F=F)
    
    res_gde3 = minimize(problem,
                    gde3,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=True,
                    verbose=True)
    
    assert res_gde3.F <= 1e-6


