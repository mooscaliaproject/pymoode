# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pymoode'
copyright = '2023, Bruno Scalia C. F. Leite'
author = 'Bruno Scalia C. F. Leite'
release = '0.2.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
]

templates_path = ['_templates']
exclude_patterns = []
autoclass_content = 'both'

intersphinx_mapping = {
    'python': ('http://docs.python.org/2', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', "http://docs.scipy.org/doc/numpy/objects.inv"),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('http://matplotlib.sourceforge.net', None),
    'pymoo': ('http://pymoo.org', None)
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
