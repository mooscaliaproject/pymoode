import distutils.core
import Cython.Build
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("knn.pyx"))