Algorithms
==========

The general overview of algorithms is presented in this section, although for more usage details we 
suggest referring to `usage page <Usage/index.ipynb>`_.


DE
--

Differential Evolution for single-objective problems proposed by
Storn & Price [:cite:label:`de_article`].
Other features later implemented are also present, such as dither, jitter, selection variants, 
and crossover strategies. For details see Price et al. [:cite:label:`de_book`].


GDE3
----

Generalized Differential Evolution 3, a multi-objective algorithm that combines 
DE mutation and crossover operators to NSGA-II [:cite:label:`nsga2`] survival with a hybrid type survival strategy.
In this algorithm, individuals might be removed in a one-to-one comparison before truncating
the population by the multi-objective survival operator. It was proposed by 
Kukkonen, S. & Lampinen, J. [:cite:label:`gde3`].
Variants with M-Nearest Neighbors and 2-Nearest Neighbors survival [:cite:label:`gde3many`]
are also available.


NSDE
----

Non-dominated Sorting Differential Evolution, a multi-objective algorithm that combines
DE mutation and crossover operators to NSGA-II [:cite:label:`nsga2`] survival.


NSDE-R
------

Non-dominated Sorting Differential Evolution based on Reference directions [:cite:label:`nsder`].
It is an algorithm for many-objective problems that works as an extension of NSDE
using NSGA-III [:cite:label:`nsga3-part1`] [:cite:label:`nsga3-part2`] survival strategy.