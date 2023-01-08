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

A Python framework for Differential Evolution using `pymoo <https://pymoo.org/>`_ (Blank & Deb, 2020).


Install
-------

First, make sure you have a Python 3 environment installed.

From PyPi:

.. code:: bash

    pip install pymoode


From the current version on github:

.. code:: bash

    pip install -e git+https://github.com/mooscalia/pymoode#egg=pymoode


.. toctree::
   :maxdepth: 1
   :caption: Usage:

   Usage/Complete tutorial.ipynb
   Usage/Single-objective.ipynb
   Usage/Multi-objective.ipynb
   Usage/Many-objective.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Theory:

   Theory/DE.ipynb


.. toctree::
   :maxdepth: 3
   :caption: Package:

   modules


Citation
========

This package was developed as part of an academic optimization project, as well as pymoo (Blank & Deb, 2020). Please, if you use it for research purposes, cite it accordingly:

`Blank, J. & Deb, K., 2020. pymoo: Multi-Objective Optimization in Python. IEEE Access, Volume 8, pp. 89497-89509. doi:10.1109/ACCESS.2020.2990567. <https://doi.org/10.1109/ACCESS.2020.2990567>`_

`Leite, B., Costa, A. O. S., Costa, E. F., 2023. Multi-objective optimization of adiabatic styrene reactors using Generalized Differential Evolution 3 (GDE3). Chem. Eng. Sci., Volume 265, Article 118196. doi:10.1016/j.ces.2022.118196. <https://doi.org/10.1016/j.ces.2022.118196>`_
