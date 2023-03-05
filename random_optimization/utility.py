import numpy as np 
import pandas as pd
import mlrose_hiive
from sklearn import preprocessing
from time import process_time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

class random_optimization(object):
    def __init__(self, max_attempts, max_iters, maximize, max_val, length, fitness_fn, algo):
        self.time_sa, self.time_ga, self.time_rhc, self.time_mimic = [],[],[],[] 
        self.fitness_sa, self.fitness_ga, self.fitness_rhc, self.fitness_mimic = [],[],[],[] 
        self.curve_sa, self.curve_ga, self.curve_rhc, self.curve_mimic = [],[],[],[] 
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.maximize=maximize
        self.max_val=max_val # so DiscreteOpt output (0,1)
        self.length=length
        self.fitness_fn=fitness_fn
        self.algo=algo

        
    def problem_size(self, length):
        init_state = np.array([i for i in range(length)])
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=self.fitness_fn, maximize=self.maximize, max_val=self.max_val)
        problem.set_mimic_fast_mode(True)
        schedule = mlrose_hiive.ExpDecay()
        max_attempts = self.max_attempts
        max_iters = self.max_iters
        
        def for_each_algo(func):
            time_start = process_time()
            best_state, best_fitness, curve_fitness= func
            time_end = process_time()
            time_cost = time_end-time_start
            # print(best_fitness)
            # print(time_cost)
            return best_fitness, time_cost, curve_fitness

        func = mlrose_hiive.random_hill_climb(problem, max_attempts=self.max_attempts, max_iters=self.max_iters, init_state=init_state, curve=True)
        best_fitness, time_cost, curve_fitness = for_each_algo(func)
        self.time_rhc.append(time_cost)
        self.fitness_rhc.append(best_fitness)
        self.curve_rhc.append(curve_fitness)
        
        func = mlrose_hiive.simulated_annealing(problem, schedule=schedule, max_attempts=self.max_attempts, max_iters=self.max_iters, init_state=init_state, curve=True)
        best_fitness, time_cost, curve_fitness = for_each_algo(func)
        self.time_sa.append(time_cost)
        self.fitness_sa.append(best_fitness)
        self.curve_sa.append(curve_fitness)

        func = mlrose_hiive.genetic_alg(problem, max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True)
        best_fitness, time_cost, curve_fitness = for_each_algo(func)
        self.time_ga.append(time_cost)
        self.fitness_ga.append(best_fitness)
        self.curve_ga.append(curve_fitness)
        
        func = mlrose_hiive.mimic(problem, max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True)
        best_fitness, time_cost, curve_fitness = for_each_algo(func)
        self.time_mimic.append(time_cost)
        self.fitness_mimic.append(best_fitness)
        self.curve_mimic.append(curve_fitness)
        
    def rhc_hypterparameter(self, restarts_len):
        init_state = np.random.randint(2, size=self.length)
        problem = mlrose_hiive.DiscreteOpt(length=self.length, fitness_fn=self.fitness_fn, maximize=self.maximize, max_val=self.max_val)
        problem.set_mimic_fast_mode(True)
        max_attempts = self.max_attempts
        max_iters = self.max_iters
        restarts = range(0,restarts_len,2)
        curve=[]
        for i in restarts:
            best_state, best_fitness, curve_ = mlrose_hiive.random_hill_climb(problem, restarts=i, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, curve=True)
            curve.append(curve_)
            
        plt.figure()
        for i in range(len(restarts)):
            plt.plot(curve[i][:,0], label='restarts_'+str((list(restarts))[i]))
        title="RHC_hypterparameter_"+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
        
    def sa_hypterparameter(self):
        init_state = np.random.randint(2, size=self.length)
        problem = mlrose_hiive.DiscreteOpt(length=self.length, fitness_fn=self.fitness_fn, maximize=self.maximize, max_val=self.max_val)
        problem.set_mimic_fast_mode(True)

        best_state, best_fitness, curve_ExpDecay = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts=self.max_attempts, max_iters=self.max_iters, init_state=init_state, curve=True)
        best_state, best_fitness, curve_GeomDecay = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.GeomDecay(), max_attempts=self.max_attempts, max_iters=self.max_iters, init_state=init_state, curve=True)
        best_state, best_fitness, curve_ArithDecay = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ArithDecay(), max_attempts=self.max_attempts, max_iters=self.max_iters, init_state=init_state, curve=True)
        plt.figure()
        plt.plot(curve_ExpDecay[:,0], label='ExpDecay')
        plt.plot(curve_GeomDecay[:,0], label='GeomDecay')
        plt.plot(curve_ArithDecay[:,0], label='ArithDecay')
        title="SA_hypterparameter_"+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
        
    def plot_problem_size(self, _range):
        plt.figure()
        plt.plot(_range, self.fitness_rhc, label='RHC')
        plt.plot(_range, self.fitness_sa, label='SA')
        plt.plot(_range, self.fitness_ga, label='GA')
        plt.plot(_range, self.fitness_mimic, label='MIMIC')
        title="Finess_ProblemSize_"+self.algo
        plt.title(title)
        plt.xlabel("ProblemSize")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
        plt.figure()
        plt.plot(_range, self.time_rhc, label='RHC')
        plt.plot(_range, self.time_sa, label='SA')
        plt.plot(_range, self.time_ga, label='GA')
        plt.plot(_range, self.time_mimic, label='MIMIC')
        title="TimeCost_ProblemSize_"+self.algo
        plt.title(title)
        plt.xlabel("ProblemSize")
        plt.ylabel("TimeCost")
        plt.legend()
        plt.savefig(title+'.png')
        
    def plot_iteration(self):
        plt.figure()
        plt.plot(self.curve_rhc[0][:,0], label='RHC')
        plt.plot(self.curve_sa[0][:,0], label='SA')
        plt.plot(self.curve_ga[0][:,0], label='GA')
        plt.plot(self.curve_mimic[0][:,0], label='MIMIC')
        title="Finess_Iteration_"+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
                  
        
    def ga_hypterparameter(self, range_mutation_prob, range_pop_size):
        problem = mlrose_hiive.DiscreteOpt(length=self.length, fitness_fn=self.fitness_fn, maximize=self.maximize, max_val=self.max_val)
        problem.set_mimic_fast_mode(True)
        curve = []
        for i in range_mutation_prob:
            for j in range_pop_size:
                best_state, best_fitness, curve_ = mlrose_hiive.genetic_alg(problem, mutation_prob=i, pop_size=j, max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True)
                curve.append((curve_,(i,j)))
        
        plt.figure()
        for i in range(len(curve)):
            plt.plot(curve[i][0][:,0], label='mutation_prob'+str(curve[i][1][0])+'_pop_size'+str(curve[i][1][1]))
        title="GA_hypterparameter_"+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
          
    def mimic_hypterparameter(self, range_keep_pct, range_pop_size):
        problem = mlrose_hiive.DiscreteOpt(length=self.length, fitness_fn=self.fitness_fn, maximize=self.maximize, max_val=self.max_val)
        problem.set_mimic_fast_mode(True)
        curve = []
        for i in range_keep_pct:
            for j in range_pop_size:
                best_state, best_fitness, curve_ = mlrose_hiive.mimic(problem, keep_pct=i, pop_size=j, max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True)
                curve.append((curve_,(i,j)))
        
        plt.figure()
        for i in range(len(curve)):
            plt.plot(curve[i][0][:,0], label='keep_pct'+str(curve[i][1][0])+'_pop_size'+str(curve[i][1][1]))
        title="MIMIC_hypterparameter_"+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
                   
    def plot_t_pct(self, t):
        plt.figure()
        plt.plot(self.curve_rhc[0][:,0], label='RHC')
        plt.plot(self.curve_sa[0][:,0], label='SA')
        plt.plot(self.curve_ga[0][:,0], label='GA')
        plt.plot(self.curve_mimic[0][:,0], label='MIMIC')
        title="Finess_t_pct_"+str(t)+'_'+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')
        
    def plot_max_weight_pct(self, max_weight_pct):
        plt.figure()
        plt.plot(self.curve_rhc[0][:,0], label='RHC')
        plt.plot(self.curve_sa[0][:,0], label='SA')
        plt.plot(self.curve_ga[0][:,0], label='GA')
        plt.plot(self.curve_mimic[0][:,0], label='MIMIC')
        title="Finess_max_weight_pct_"+str(max_weight_pct)+'_'+self.algo
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(title+'.png')