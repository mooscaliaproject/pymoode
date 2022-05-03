from setuptools import setup

setup(
  name = 'pymoode',
  packages = ['pymoode'],
  version = '0.1.0',
  license='MIT',
  description = 'A Python optimization package using Differential Evolution.',
  author = 'Bruno Scalia C. F. Leite',
  author_email = 'mooscaliaproject@gmail.com',
  url = 'https://github.com/mooscaliaproject/pymoode',
  download_url = 'https://github.com/mooscaliaproject/pymoode',
  keywords = ['Multi-objective optimization',
              'NSGA-II',
              'GDE3',
              'NSDE',
              'Differential Evolution',
              'Genetic Algorithm',
              'Evolutionary Algorithms',
              'Evolutionary optimization'],
  install_requires=[
          'numpy==1.22.*',
          'pymoo==0.5.*',
          'scikit-learn==1.0.*',
          'scipy==1.8.*'
      ],
)