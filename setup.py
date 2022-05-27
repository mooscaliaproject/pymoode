from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  name = 'pymoode',
  packages = ['pymoode'],
  version = '0.1.7',
  license='Apache License 2.0',
  description = 'A Python optimization package using Differential Evolution.',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Bruno Scalia C. F. Leite',
  author_email = 'mooscaliaproject@gmail.com',
  url = 'https://github.com/mooscaliaproject/pymoode',
  download_url = 'https://github.com/mooscaliaproject/pymoode',
  keywords = ['Multi-objective optimization',
              'NSGA-II',
              'GDE3',
              'NSDE',
              'SA-NSDE',
              'NSDE-R',
              'Differential Evolution',
              'Genetic Algorithm',
              'Evolutionary Algorithms',
              'Evolutionary optimization'],
  install_requires=[
          'numpy>=1.19.*',
          'pymoo==0.5.*',
          'scikit-learn>=1.0.*',
          'scipy>=1.7.*',
      ],
)