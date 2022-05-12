_de_params = """pop_size (int, optional): Population size. Defaults to 100.
    sampling (Sampling, optional): Sampling strategy of pymoo. Defaults to LHS().
    variant (str, optional): Differential evolution strategy. Must be a string in the format:
        "DE/selection/n/crossover", in which, n in an integer of number of difference vectors, and crossover is either "bin" or "exp".
        Selection variants are:
            - "ranked"
            - "rand"
            - "best"
            - "current-to-best"
            - "current-to-best"
            - "current-to-rand"
            - "rand-to-best"
        The selection strategy "ranked" might be helpful to improve convergence speed without much harm to diversity.
        Defaults to "DE/rand/1/bin"
    CR (float, optional): Crossover parameter. Defined in the range [0, 1]
        To reinforce mutation, use higher values. To control convergence speed, use lower values.
    F (iterable of float or float, optional): Scale factor or mutation parameter. Defined in the range (0, 2]
        To reinforce exploration, use higher lower bounds; for exploitation, use lower values.
    gamma (float, optional): Jitter deviation parameter. Should be in the range (0, 2). Defaults to 1e-4.
    SA (float, optional): Probability of using self-adaptive scale factor. Defaults to None.
    pm (Mutation, optional): Pymoo's mutation operators after crossover. Defaults to NoMutation().
    reapair (Repair, optional): Repair of mutant vectors. Is either callable or one of:
        "bounce-back"
        "midway"
        "rand-init"
        "to-bounds"
        If callable, has the form fun(X, Xb, xl, xu) in which X contains mutated vectors including violations, Xb contains reference vectors for repair in feasible space, xl is a 1d vector of lower bounds, and xu a 1d vector of upper bounds.
        Defaults to "bounce-back".
    survival (Survival, optional): Pymoo's survival strategy.
        Defaults to RankSurvival() with bulk removal ("full") and crowding distances ("cd").
        In GDE3, the survival strategy is applied after a one-to-one comparison between child vector and corresponding parent when both are non-dominated by the other.
"""

_de = """Single-objective Differential Evolution proposed by Storn and Price (1997).

Storn, R. & Price, K., 1997. Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. J. Glob. Optim., 11(4), pp. 341-359.

Args:
"""

_gde3 = """GDE3 is an extension of DE to multi-objective problems using a mixed type survival strategy.
It is implemented in this version with the same constraint handling strategy of NSGA-II by default.
We recommend using it to problems with many local fronts in which it is necessary to avoid premature convergence. In this context, low CR values (~ 0.3) are highly recommended.
For problems that demand high CR values (~0.9), NSDE is more recommended.
For many-objective problems, try using NSDER or RankSurvival with 'mnn' crowding metric.

Kukkonen, S. & Lampinen, J., 2005. GDE3: The third evolution step of generalized differential evolution. 2005 IEEE congress on evolutionary computation, Volume 1, pp. 443-450.

Args:
"""

_nsde = """NSDE is an algorithm that combines that combines NSGA-II sorting and survival strategies to DE mutation and crossover following the implementation by Leite et al. (2022) with a self-adaptative mutation (scale factor) F parameter as in SA-NSDE.
When using low CR values (~0.3), try using GDE3.
For many-objective problems, try using NSDER or RankSurvival with 'mnn' crowding metric.

Leite, B., Costa, A. O. S. & Costa Junior, E. F., 2022. A self-adaptive multi-objective differential evolution algorithm applied to the styrene reactor optimization. Available at SSRN: https://ssrn.com/abstract=4081771, or http://dx.doi.org/10.2139/ssrn.4081771.

Args:
"""

_nsder = """NSDE-R is an extension of NSDE to many-objective problems (Reddy & Dulikravich, 2019) using NSGA-III survival.
        
S. R. Reddy and G. S. Dulikravich, "Many-objective differential evolution optimization based on reference points: NSDE-R," Struct. Multidisc. Optim., vol. 60, pp. 1455-1473, 2019.

Args:

    ref_dirs (array like): The reference direction that should be used during the optimization.
        Each row represents a reference line and each column a variable.
"""