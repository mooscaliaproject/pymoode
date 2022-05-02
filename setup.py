from setuptools import setup

setup(
  name = 'pymoode',
  packages = ['pymoode'],
  version = '0.1.0',
  license='MIT',
  description = 'A multi-objective optimization package using differential evolution.',
  author = 'Bruno Scalia C. F. Leite',
  author_email = 'mooscaliaproject@gmail.com',
  url = 'https://github.com/mooscaliaproject/pymoode',
  download_url = 'https://github.com/mooscaliaproject/pymoode',
  keywords = ['Multi-objective optimization',
              'NSGA-II',
              'Differential Evolution',
              'Genetic Algorithm',
              'Evolutionary optimization'],
  install_requires=[
          'numpy==1.22.*',
          'pymoo==0.5.*',
      ],
)