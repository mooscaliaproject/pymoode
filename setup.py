import argparse
import os
import sys
from distutils.errors import CompileError

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# -----------------------------------------------------------------------------
# BASE CONFIGURATION
# -----------------------------------------------------------------------------

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

BASE_PACKAGE = 'pymoode'

base_kwargs = dict(  # noqa: C408
    name='pymoode',
    packages=[BASE_PACKAGE]
    + [f'{BASE_PACKAGE}.' + e for e in find_packages(where=BASE_PACKAGE)],
    version='0.3.1.dev1',
    license='Apache License 2.0',
    description='A Python optimization package using Differential Evolution.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bruno Scalia C. F. Leite',
    author_email='mooscaliaproject@gmail.com',
    url='https://github.com/mooscaliaproject/pymoode',
    download_url='https://github.com/mooscaliaproject/pymoode',
    keywords=[
        'Differential Evolution',
        'Multi-objective optimization',
        'GDE3',
        'NSDE',
        'NSDE-R',
        'NSGA-II',
        'Genetic Algorithm',
        'Crowding Distances',
        'Evolutionary Algorithms',
        'Evolutionary Optimization',
        'Metaheuristics',
    ],
    install_requires=[
        'numpy>=1.19.0',
        'pymoo==0.6.1',
        'scipy>=1.7.0',
        'future',
    ],
)

# -----------------------------------------------------------------------------
# ARGPARSER
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    '--nopyx',
    dest='nopyx',
    action='store_true',
    help='Whether the pyx files shall be considered at all.',
)
parser.add_argument(
    '--nocython',
    dest='nocython',
    action='store_true',
    help='Whether pyx files shall be cythonized.',
)
parser.add_argument(
    '--nolibs',
    dest='nolibs',
    action='store_true',
    help='Whether the libraries should be compiled.',
)
parser.add_argument(
    '--markcython',
    dest='markcython',
    action='store_true',
    help='Whether to mark the html cython files.',
)

params, _ = parser.parse_known_args()
sys.argv = [arg for arg in sys.argv if arg.lstrip('-') not in params]

# -----------------------------------------------------------------------------
# ADD MARKED HTML FILES FOR CYTHON
# -----------------------------------------------------------------------------

if params.markcython:
    import Cython.Compiler.Options

    Cython.Compiler.Options.annotate = True

# -----------------------------------------------------------------------------
# CLASS TO BUILD EXTENSIONS
# -----------------------------------------------------------------------------


# exception that is thrown when the build fails
class CompilingFailed(Exception):
    pass


# try to compile, if not possible throw exception
def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        def run(self):
            try:
                build_ext.run(self)
            except BaseException as e:
                raise CompilingFailed() from e

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except BaseException as e:
                raise CompilingFailed() from e

    return WrappedBuildExt


# -----------------------------------------------------------------------------
# HANDLING CYTHON FILES
# -----------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.realpath(__file__))

if params.nopyx:
    ext_modules = []

else:
    try:
        if params.nocython:
            path = os.path.join(ROOT, 'pymoode', 'cython')
            pyx = [
                os.path.join(path, f)
                for f in os.listdir()
                if f.endswith('.pyx')
            ]
            ext_modules = [
                Extension(f'pymoode.cython.{source[:-4]}', [source])
                for source in pyx
            ]
        else:
            try:
                from Cython.Build import cythonize

                ext_modules = cythonize('pymoode/cython/*.pyx')
            except ImportError:
                print('*' * 75)
                print('No Cython package found to convert .pyx files.')
                print(
                    'If no compilation occurs, .py files will be used instead,'
                    ' which provide the same results but with'
                    ' worse computational time.'
                )
                print('*' * 75)
                ext_modules = []
    except Exception as e:
        print('*' * 75)
        print('Problems compiling .pyx files.')
        print(f'Exception: {type(e)} - {e}')
        print(
            'If no compilation occurs, .py files will be used instead,'
            ' which provide the same results but '
            'with worse computational time.'
        )
        print('*' * 75)
        ext_modules = []

if not params.nolibs:
    if len(ext_modules) > 0:
        base_kwargs['ext_modules'] = ext_modules

        try:
            import numpy as np

            base_kwargs['include_dirs'] = [np.get_include()]

        except BaseException as e:
            raise CompileError(
                'NumPy libraries must be installed for compiled extensions!'
                ' Speedups are not enabled.'
            ) from e

        base_kwargs['cmdclass'] = dict(  # noqa: C408
            build_ext=construct_build_ext(build_ext)
        )

    else:
        print('*' * 75)
        print('External cython modules found.')
        print('To verify compilation success run:')
        print(
            'from pymoode.survival.rank_and_crowding.metrics'
            ' import IS_COMPILED'
        )
        print('This variable will be True to mark compilation success;')
        print(
            'If no compilation occurs, .py files will be used instead,'
            ' which provide the same results but'
            ' with worse computational time.'
        )
        print('*' * 75)

# -----------------------------------------------------------------------------
# RUN SETUP
# -----------------------------------------------------------------------------

compiled_kwargs = base_kwargs.copy()
compiled_kwargs['ext_modules'] = ext_modules

try:
    setup(**compiled_kwargs)
    print('*' * 75)
    print('Installation successful at the first attempt.')
    print('To verify compilation success run:')
    print('from pymoode.survival.rank_and_crowding.metrics import IS_COMPILED')
    print('*' * 75)
except Exception as e:
    print('*' * 75)
    print(f'Exception: {type(e)} - {e}')
    print('Running setup with cython compilation failed.')
    print('Attempt to a pure Python setup.')
    print(
        'If no compilation occurs, .py files will be used instead,'
        ' which provide the same results but with worse computational time.'
    )
    print('*' * 75)
    setup(**base_kwargs)
