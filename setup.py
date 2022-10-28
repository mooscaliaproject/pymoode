import os
import sys
from setuptools import find_packages, setup, Extension

import numpy as np

# ---------------------------------------------------------------------------------------------------------
# ADD MARKED HTML FILES FOR CYTHON
# ---------------------------------------------------------------------------------------------------------

mark_cython = False
if "--mark_cython" in sys.argv:
    mark_cython = True
    sys.argv.remove("--mark_cython")

if mark_cython:
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True

# ---------------------------------------------------------------------------------------------------------
# HANDLING CYTHON FILES
# ---------------------------------------------------------------------------------------------------------

do_cythonize = False
if "--cythonize" in sys.argv:
    do_cythonize = True
    sys.argv.remove("--cythonize")

ext_modules = []
cython_folder = os.path.join("pymoode", "cython")
cython_files = os.listdir(cython_folder)

# if the pyx files should be translated and then compiled
if do_cythonize:
    from Cython.Build import cythonize
    ext_modules = cythonize(["pymoode/cython/*.pyx"])

# otherwise use the existing pyx files - normal case during pip installation
else:
    # find all cpp files in czthon folder
    cpp_files = [f for f in cython_files if f.endswith(".cpp")]

    # add for each file an extension object to be compiled
    for source in cpp_files:
        ext = Extension("pymoode.cython.%s" % source[:-4], [os.path.join(cython_folder, source)])
        ext_modules.append(ext)

if len(ext_modules) == 0:
    print('*' * 75)
    print("WARNING: No modules for compilation available.")
    print("Cython .pyx files are available at the github package repository https://github.com/mooscaliaproject/pymoode")
    print("To compile .pyx files, run:")
    print("python setup.py build_ext --inplace --cythonize")
    print("And then:")
    print("python setup.py install")
    print('*' * 75)
    print("If no compilation occurs, .py files will be used instead, which provide the same results but with worse computational time.")
    print('*' * 75)
else:
    print('*' * 75)
    print("External cython modules found.")
    print("To verify compilation success run:")
    print("from pymoode.survival._metrics import IS_COMPILED")
    print("This variable will be True to mark compilation success;")
    print("If no compilation occurs, .py files will be used instead, which provide the same results but with worse computational time.")
    print('*' * 75)
    
# ---------------------------------------------------------------------------------------------------------
# GENERAL
# ---------------------------------------------------------------------------------------------------------

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
BASE_PACKAGE = 'pymoode'

base_kwargs = dict(
    name = 'pymoode',
    packages = [BASE_PACKAGE] + [f"{BASE_PACKAGE}." + e for e in find_packages(where=BASE_PACKAGE)],
    version = '0.2.2.dev1',
    license='Apache License 2.0',
    description = 'A Python optimization package using Differential Evolution.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Bruno Scalia C. F. Leite',
    author_email = 'mooscaliaproject@gmail.com',
    url = 'https://github.com/mooscaliaproject/pymoode',
    download_url = 'https://github.com/mooscaliaproject/pymoode',
    include_dirs = np.get_include(),
    keywords = ['Multi-objective optimization',
                'GDE3',
                'NSDE',
                'NSDE-R',
                'NSGA-II',
                'Differential Evolution',
                'Genetic Algorithm',
                'Crowding Distances',
                'Evolutionary Algorithms',
                'Evolutionary Optimization'],
    install_requires=[
            'numpy>=1.19.*',
            'pymoo==0.6.*',
            'scipy>=1.7.*',
            'future',
        ],
)

compiled_kwargs = base_kwargs.copy()
compiled_kwargs["ext_modules"] = ext_modules

try:
    setup(**compiled_kwargs)
    print('*' * 75)
    print("External cython modules found.")
    print("To verify compilation success run:")
    print("from pymoode.survival._metrics import IS_COMPILED")
    print('*' * 75)
except:
    print('*' * 75)
    print("Running setup with cython compilation failed.")
    print("Attempt to a pure Python setup.")
    print("If no compilation occurs, .py files will be used instead, which provide the same results but with worse computational time.")
    print('*' * 75)
    setup(**base_kwargs)
