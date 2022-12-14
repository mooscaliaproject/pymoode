import pytest

import numpy as np
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoode.performance import SpacingIndicator
from pymoode.algorithms import GDE3
from pymoode.survival import RankAndCrowding


def test_spacing_values():
    
    F = np.array([
        [1.2, 7.8],
        [2.8, 5.1],
        [4.0, 2.8],
        [7.0, 2.2],
        [8.4, 1.2]
    ])
    
    assert abs(SpacingIndicator().do(F) - 0.73) <= 1e-3
    
    F = np.array([
        [1.0, 7.5],
        [1.1, 5.5],
        [2.0, 5.0],
        [3.0, 4.0],
        [4.0, 2.8],
        [5.5, 2.5],
        [6.8, 2.0],
        [8.4, 1.2]
    ])

    assert abs(SpacingIndicator().do(F) - 0.316) <= 1e-3


def test_spacing_zero_to_one():
    
    problem = get_problem("truss2d")
    sp_from_pf = SpacingIndicator(pf=problem.pareto_front(), zero_to_one=True)
    sp_from_ref_points = SpacingIndicator(zero_to_one=True, nadir=problem.nadir_point(), ideal=problem.ideal_point())
    sp_no_scale = SpacingIndicator(zero_to_one=False)

    NGEN = 250
    POPSIZE = 100
    SEED = 5
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9), de_repair="bounce-back",
                survival=RankAndCrowding(crowding_func="pcd"))

    res_gde3 = minimize(problem,
                        gde3,
                        ('n_gen', NGEN),
                        seed=SEED,
                        save_history=False,
                        verbose=False)
    
    assert sp_from_pf.do(res_gde3.F) > 0
    assert sp_from_ref_points.do(res_gde3.F) > 0
    assert sp_no_scale.do(res_gde3.F) > 0
    assert abs(sp_from_pf.do(res_gde3.F) - sp_from_ref_points.do(res_gde3.F)) <= 1e-6