.. pymoode documentation master file, created by
   sphinx-quickstart on Tue Jan  3 02:07:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://github.com/mooscaliaproject/pymoode/raw/main/images/logo_header_pymoode.png
    :alt: header
    :align: center
    :target: https://github.com/mooscaliaproject/pymoode


pymoode: Differential Evolution in Python
=========================================

A Python framework for Differential Evolution using `pymoo <https://pymoo.org/>`_ [:cite:label:`pymoo`].


Install
-------

First, make sure you have a Python 3 environment installed.

From PyPi:

.. code:: bash

    pip install pymoode


From the current version on github:

.. code:: bash

    pip install -e git+https://github.com/mooscalia/pymoode#egg=pymoode


New features
------------

This package was written as an extension of pymoo, providing some additional
features for DE algorithms and survival operators. One might refer to the sections
:doc:`Algorithms <algorithms>`, `Survival <Theory/Survival.ipynb>`_ and 
`Rank and Crowding <Theory/Rank-and-Crowding.ipynb>`_ for more details.

For instance, these solutions for the DTLZ2 problem were obtained using GDE3 with the M-Nearest Neighbors crowding metric.

.. image:: https://github.com/mooscaliaproject/pymoode/raw/main/images/gde3mnn_example.gif
    :alt: header
    :align: center
    :width: 500
    :target: https://github.com/mooscaliaproject/pymoode


.. toctree::
   :maxdepth: 1
   :caption: Usage

   Usage/Complete-tutorial.ipynb
   Usage/Single-objective.ipynb
   Usage/Multi-objective.ipynb
   Usage/Many-objective.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms


.. toctree::
   :maxdepth: 1
   :caption: Theory

   Theory/DE.ipynb
   Theory/Survival.ipynb
   Theory/Rank-and-Crowding.ipynb


.. toctree::
   :maxdepth: 3
   :caption: Package

   modules


Citation
========

This package was developed as part of an academic optimization project [:cite:label:`pymoodestyrene`], as well as pymoo [:cite:label:`pymoo`].
Please, if you use it for research purposes, cite it accordingly:

`Blank, J. & Deb, K., 2020. pymoo: Multi-Objective Optimization in Python. IEEE Access, Volume 8, pp. 89497-89509. doi:10.1109/ACCESS.2020.2990567. <https://doi.org/10.1109/ACCESS.2020.2990567>`_

`Leite, B., Costa, A. O. S., Costa, E. F., 2023. Multi-objective optimization of adiabatic styrene reactors using Generalized Differential Evolution 3 (GDE3). Chem. Eng. Sci., Volume 265, Article 118196. doi:10.1016/j.ces.2022.118196. <https://doi.org/10.1016/j.ces.2022.118196>`_


References
==========

To the complete reference list, please refer to :doc:`this page <references>`. 
