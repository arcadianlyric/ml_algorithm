Need Python 3.7 and jupyter with the following environment.yml:
name: project
channels:
  - defaults
dependencies:
  - python=3.7.4
  - scikit-learn
  - mlrose
  - numpy
  - scipy
  - pandas
  - matplotlib==3.1.0
  - seaborn
  - pytest
  - pip
  - pip:
    - delayed-assert

To reproduce results: 
Part1: python ro_part1.py
Part2: python ro_part2.py