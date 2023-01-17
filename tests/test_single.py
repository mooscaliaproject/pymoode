import pytest

from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.operators.mutation.pm import PM
from pymoode.algorithms import GDE3
from pymoode.algorithms import DE


@pytest.mark.parametrize('selection', ["rand", "best", "current-to-best", "current-to-rand", "ranked"])
@pytest.mark.parametrize('crossover', ["bin", "exp"])
@pytest.mark.parametrize('de_repair', ["bounce-back", "midway", "rand-init", "to-bounds"])
def test_de_run(selection, crossover, de_repair):
    problem = get_problem("rastrigin")
    
    NGEN = 50
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant=f"DE/{selection}/1/{crossover}", CR=CR, F=F, de_repair=de_repair)

    res_de = minimize(problem,
                    de,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=False,
                    verbose=False)

    assert len(res_de.opt) > 0


def test_de_pm_run():
    
    problem = get_problem("rastrigin")
    
    NGEN = 50
    POPSIZE = 20
    SEED = 3
    
    #DE Parameters
    CR = 0.5
    F = (0.3, 1.0)

    de = DE(pop_size=POPSIZE, variant=f"DE/rand/1/bin", CR=CR, F=F)

    res_de = minimize(problem,
                      de,
                      ('n_gen', NGEN),
                      seed=SEED,
                      save_history=False,
                      verbose=False)
    
    depm = DE(pop_size=POPSIZE, variant=f"DE/rand/1/bin", CR=CR, F=F, genetic_mutation=PM())

    res_pm = minimize(problem,
                      depm,
                      ('n_gen', NGEN),
                      seed=SEED,
                      save_history=False,
                      verbose=False)

    assert len(res_pm.opt) > 0
    assert len(sum(res_de.pop.get("F") - res_pm.pop.get("F"))) >= 1e-6
    

def test_de_perf():
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
                    save_history=False,
                    verbose=False)

    assert len(res_de.opt) > 0
    assert res_de.F <= 1e-6
    
    gde3 = GDE3(pop_size=POPSIZE, variant="DE/rand/1/bin", CR=CR, F=F)
    
    res_gde3 = minimize(problem,
                    gde3,
                    ('n_gen', NGEN),
                    seed=SEED,
                    save_history=False,
                    verbose=False)
    
    assert res_gde3.F <= 1e-6

