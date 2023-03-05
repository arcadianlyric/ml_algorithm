### FourPeaks
import numpy as np 
import pandas as pd
import mlrose_hiive
from time import process_time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
from utility import *

def by_algo(max_attempts, max_iters, maximize, max_val, length, algo, fitness_fn, _range):
    # ProblemSize
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)

    for length in _range:
        ins.problem_size(length)
    ins.plot_problem_size(_range)

    # max_iters
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    length = 50
    ins.problem_size(length)
    ins.plot_iteration()

    # RHC
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    restarts_len= 10
    ins.rhc_hypterparameter(restarts_len)

    # MIMIC
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    range_keep_pct, range_pop_size = [0.1,0.5], [100,200,400]
    ins.mimic_hypterparameter(range_keep_pct, range_pop_size)

    # SA
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length,  fitness_fn, algo)
    ins.sa_hypterparameter()

    # GA
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    range_mutation_prob, range_pop_size = [0.1,0.5], [100,200,400]
    ins.ga_hypterparameter(range_mutation_prob, range_pop_size)


if __name__ == "__main__":
    # algo='FourPeaks'
    # fitness_fn = mlrose_hiive.FourPeaks(t_pct=0.1)
    # max_attempts, max_iters, maximize, max_val, length = 100,1000,True,2,10
    # _range = range(5,75,5)
    # by_algo(max_attempts, max_iters, maximize, max_val, length, algo, fitness_fn, _range)

    # algo='FlipFlop'
    # _range = range(5,75,5)
    # fitness_fn = mlrose_hiive.FlipFlop()
    # max_attempts, max_iters, maximize, max_val, length = 100,1000,True,2,10
    # by_algo(max_attempts, max_iters, maximize, max_val, length, algo, fitness_fn, _range)

    algo='ContinuousPeaks'
    _range = range(5,100,5)
    fitness_fn = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    max_attempts, max_iters, maximize, max_val, length = 100,1000,True,2,10
    by_algo(max_attempts, max_iters, maximize, max_val, length, algo, fitness_fn, _range)
