# pymoode
A Python framework for Differential Evolution using [pymoo](https://github.com/anyoptimization/pymoo) (Blank & Deb, 2020).

## Contents
[Install](#install) / [Algorithms](#algorithms) / [Survival Operators](#survival-operators) / [Crowding Metrics](#crowding-metrics) / [Usage](#usage) / [Citation](#citation) / [References](#references) / [Contact](#contact) / [Acknowledgements](#acknowledgements)

## Install
First, make sure you have a Python 3 environment installed.

From PyPi:
```
pip install pymoode
```

From the current version on github:
```
pip install -e git+https://github.com/mooscalia/pymoode#egg=pymoode
```

## Algorithms
- **DE**: Differential Evolution for single-objective problems proposed by Storn & Price (1997). Other features later implemented are also present, such as dither, jitter, selection variants, and crossover strategies. For details see Price et al. (2005).
- **NSDE**: Non-dominated Sorting Differential Evolution, a multi-objective algorithm that combines DE mutation and crossover operators to NSGA-II (Deb et al., 2002) survival.
- **GDE3**: Generalized Differential Evolution 3, a multi-objective algorithm that combines DE mutation and crossover operators to NSGA-II survival with a hybrid type survival strategy. In this algorithm, individuals might be removed in a one-to-one comparison before truncating the population by the multi-objective survival operator. It was proposed by Kukkonen, S. & Lampinen, J. (2005).
- **NSDE-R**: Non-dominated Sorting Differential Evolution based on Reference directions (Reddy & Dulikravich, 2019). It is an algorithm for many-objective problems that works as an extension of NSDE using NSGA-III (Deb & Jain, 2014) survival strategy.

## Survival Operators
- **RankSurvival**: Flexible structure to implement NSGA-II rank and crowding survival with different options for crowding metric and elimination of individuals.
- **ConstrainedRankSurvival**: A survival operator based on rank and crowding with a special constraint handling approach proposed by Kukkonen, S. & Lampinen, J. (2005).

## Crowding Metrics
- **Crowding Distance** (*'cd'*): Proposed by Deb et al. (2002) in NSGA-II. Imported from *pymoo*.
- **Crowding Entropy** (*'ce'*): Proposed by Wang et al. (2010) in MOSADE.
- ***M*-Nearest Neighbors** (*'mnn'*): Proposed by Kukkonen & Deb (2006) in an extension of GDE3 to many-objective problems.
- **2-Nearest Neighbors** (*'2nn'*): Also proposed by Kukkonen & Deb (2006), it is a variant of M-Nearest Neighbors in which the number of neighbors is two.

## Usage
For more examples, see the example notebooks [single](https://github.com/mooscaliaproject/pymoode/blob/main/notebooks/EXAMPLE_SOO.ipynb), [multi](https://github.com/mooscaliaproject/pymoode/blob/main/notebooks/EXAMPLE_MULTI.ipynb), [many](https://github.com/mooscaliaproject/pymoode/blob/main/notebooks/EXAMPLE_MANY.ipynb) objective problems, and a [complete tutorial](https://github.com/mooscaliaproject/pymoode/blob/main/notebooks/tutorial.ipynb)

```python
import matplotlib.pyplot as plt
from pymoo.factory import get_problem
from pymoo.util.plotting import plot
from pymoo.optimize import minimize
from pymoode.nsde import NSDE

problem = get_problem("tnk")
```

```python
gde3 = GDE3(pop_size=50, variant="DE/rand/1/bin", CR=0.5, F=(0.0, 0.9))
    
res = minimize(problem, nsde, ('n_gen', 200), save_history=True, verbose=True)
```

```python
fig, ax = plt.subplots(figsize=[6, 5], dpi=100)
ax.scatter(pf[:, 0], pf[:, 1], color="navy", label="True Front")
ax.scatter(res.F[:, 0], res.F[:, 1], color="firebrick", label="GDE3")
ax.set_ylabel("$f_2$")
ax.set_xlabel("$f_1$")
ax.legend()
fig.tight_layout()
plt.show()
```

![tnk_nsde](https://github.com/mooscaliaproject/pymoode/raw/b02d9d46e8d9558af670a2c80eec9689c49d79cb/images/tnk_gde3.png)

## Citation
Please cite this library via its current ResearchGate file:

Leite, B., 2022. pymoode: Differential Evolution in Python. doi:10.13140/RG.2.2.12935.27043

## References
Blank, J. & Deb, K., 2020. pymoo: Multi-Objective Optimization in Python. IEEE Access, Volume 8, pp. 89497-89509.

Deb, K. & Jain, H., 2014. An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Transactions on Evolutionary Computation, 18(4), pp. 577???601.

Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. A. M. T., 2002. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), pp. 182-197.

Kukkonen, S. & Deb, K., 2006. A fast and effective method for pruning of non-dominated solutions in many-objective problems. In: Parallel problem solving from nature-PPSN IX. Berlin: Springer, pp. 553-562.

Kukkonen, S. & Lampinen, J., 2005. GDE3: The third evolution step of generalized differential evolution. 2005 IEEE congress on evolutionary computation, Volume 1, pp. 443-450.

Reddy, S. R. & Dulikravich, G. S., 2019. Many-objective differential evolution optimization based on reference points: NSDE-R. Struct. Multidisc. Optim., Volume 60, pp. 1455-1473.

Price, K. V., Storn, R. M. & Lampinen, J. A., 2005. Differential Evolution: A Practical Approach to Global Optimization. 1st ed. Springer: Berlin.

Storn, R. & Price, K., 1997. Differential evolution???a simple and efficient heuristic for global optimization over continuous spaces. J. Glob. Optim., 11(4), pp. 341-359.

Wang, Y.-N., Wu, L.-H. & Yuan, X.-F., 2010. Multi-objective self-adaptive differential evolution with elitist archive and crowding entropy-based diversity measure. Soft Comput., 14(3), pp. 193-209.

## Contact
e-mail: bruscalia12@gmail.com

## Acknowledgements
To Julian Blank, who created the amazing structure of pymoo, making such a project possible.

To Esly F. da Costa Junior, who made it possible all along, trusted in me from the start, and guided me through the path of modeling and optimization.
