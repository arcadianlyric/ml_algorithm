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

def by_algo(fitness_fn, algo):
    # ProblemSize
    ins = random_optimization(fitness_fn, algo)
    _range = range(5,15,5)
    for length in _range:
        ins.problem_size(length)
    ins.plot_problem_size(_range)

    # max_iters
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    length = 50
    ins.problem_size(length)
    ins.plot_iteration()

    # GA
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    range_mutation_prob, range_pop_size = [0.1,0.3], [100,200,500]
    ins.ga_hypterparameter(range_mutation_prob, range_pop_size)

    # MIMIC
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    range_keep_pct, range_pop_size = [0.1,0.3], [100,200,500]
    ins.mimic_hypterparameter(range_keep_pct, range_pop_size)

    # RHC
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo)
    restarts_len= 20
    ins.rhc_hypterparameter(restarts_len)

    # SA
    ins = random_optimization(max_attempts, max_iters, maximize, max_val, length,  fitness_fn, algo)
    ins.sa_hypterparameter()

    # FourPeaks t_pct
    if algo=='FourPeaks':
        for t in [0.1,0.3,0.5]:
            ins = random_optimization(max_attempts, max_iters, maximize, max_val, length,  fitness_fn, algo)
            curve=[]
            fitness_fn = mlrose_hiive.FourPeaks(t_pct=t)
            length=10
            ins.problem_size(length)
            ins.plot_t_pct(t)

    

if __name__ == "__main__":
    algo='FourPeaks'
    fitness_fn = mlrose_hiive.FourPeaks(t_pct = 0.1)
    max_attempts, max_iters, maximize, max_val, length = 100,1000,True,2,10
    by_algo(max_attempts, max_iters, maximize, max_val, length, algo, fitness_fn)