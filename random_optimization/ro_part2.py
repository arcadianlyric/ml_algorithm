import numpy as np 
import pandas as pd
import mlrose_hiive
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import time
import pickle

class task2(object):
	def __init__(self, learning_rates, param, algo):
		# data from A1
		data = pd.read_csv('data/breast_cancer.csv')
		dict_type = {'M':1, 'B':0}
		data.diagnosis=data.diagnosis.map(dict_type)
		y = data.diagnosis.values
		x = data.drop(['id','diagnosis'], axis=1)
		x = preprocessing.scale(x)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
		self.train_accuracy = np.zeros((len(learning_rates),len(param)))
		self.val_accuracy = np.zeros((len(learning_rates),len(param)))
		self.val_accuracy_best = 0.0
		self._best_idx_1 = 0
		self._best_idx_2 = 0
		self.test_accuracy = np.zeros((len(learning_rates),len(param)))
		self.time_cost = np.zeros((len(learning_rates),len(param)))
		train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
		self.train_x =train_x
		self.val_x = val_x
		self.train_y = train_y
		self.val_y = val_y
		self.test_x = test_x
		self.test_y = test_y
		self.learning_rates = learning_rates
		self.param = param
		self.algo=algo

	def for_each(self, func1, func2):
		self.clf_nn_best = func1

		for idx_1, learning_rate in enumerate(self.learning_rates):
			for idx_2, val in enumerate(self.param):
				clf_nn = func2(learning_rate, val)

				start = time.time()
				clf_nn.fit(self.train_x, self.train_y)
				end = time.time()
				_time_cost = end - start

				train_pred_y = clf_nn.predict(self.train_x)
				train_accuracy_y = accuracy_score(self.train_y, train_pred_y)
				self.train_accuracy[idx_1][idx_2] = train_accuracy_y
				val_pred_y = clf_nn.predict(self.val_x)
				val_accuracy_y = accuracy_score(self.val_y, val_pred_y)
				self.val_accuracy[idx_1][idx_2]= val_accuracy_y
				test_pred_y = clf_nn.predict(self.test_x)
				test_accuracy_y = accuracy_score(self.test_y, test_pred_y)
				self.test_accuracy[idx_1][idx_2] = test_accuracy_y
				self.time_cost[idx_1][idx_2]= _time_cost

				if val_accuracy_y > self.val_accuracy_best:
					self.clf_nn_best = clf_nn
					self._best_idx_1 = idx_1
					self._best_idx_2 = idx_2
					self.val_accuracy_best = val_accuracy_y
					print("learning_rate=", learning_rate)
					print(str(val), idx_2)
					print("time=", _time_cost)

				# print("iteration done {idx_1},{idx_2}".format(idx_1=learning_rates[idx_1], idx_2=param[idx_2]))
		return 

	def plot_nn(self):
		plt.figure()
		plt.plot(self.clf_nn_best.fitness_curve[:,0])
		plt.xlabel('num_iteration')
		plt.ylabel('train_loss')
		plt.grid()
		plt.title('train_loss_num_iteration_best_{}'.format(self.algo))
		plt.savefig('train_loss_num_iteration_best_{}.png'.format(self.algo))

		# test_pred_y = self.clf_nn_best.predict(self.test_x)
		# _confusion_matrix = confusion_matrix(self.test_y, test_pred_y)

		print("time_cost=", self.time_cost)
		print("test_accuracy=", self.test_accuracy[self._best_idx_1][self._best_idx_2])
		# print("confusion_matrix=", _confusion_matrix)

	def plot_bp(self):
		plt.figure()
		plt.plot(-self.clf_nn_best.fitness_curve)
		plt.xlabel('num_iteration')
		plt.ylabel('train_loss')
		plt.grid()
		plt.title('train_loss_num_iteration_best_{}'.format(self.algo))
		plt.savefig('train_loss_num_iteration_best_{}.png'.format(self.algo))

if __name__ == "__main__":
	learning_rates = [0.00001, 0.0001, 0.01, 0.1]
	restarts = [2, 4, 6, 8]
	schedules = [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
	pop_size = [25, 50, 100, 200]

	max_iters =1000
	max_attempts = 100
	hidden_nodes = [8]
	activation ='relu'

	# Backpropagation
	algo='BP'
	algorithm='gradient_descent'
	param=[1]
	ins =task2(learning_rates, param, algo)
	func1 = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[0], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	def func2(learning_rate,param):
		clf_nn = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
		return clf_nn	
	ins.for_each(func1, func2)
	ins.plot_bp()
	_bp =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best_idx_1, ins._best_idx_2]
	with open('_bp', 'wb') as fp:
		pickle.dump(_bp, fp)

	algo='RHC'
	algorithm='random_hill_climb'
	ins =task2(learning_rates, restarts, algo)
	func1 = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[0], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	def func2(learning_rate, restarts):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, restarts=restarts,  learning_rate=learning_rate, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
		return clf_nn	
	ins.for_each(func1, func2)
	ins.plot_nn()
	_rhc =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best_idx_1, ins._best_idx_2]
	with open('_rhc', 'wb') as fp:
		pickle.dump(_rhc, fp)

	algo='SA'
	algorithm='simulated_annealing'
	ins =task2(learning_rates, schedules, algo)
	func1 = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[0], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	def func2(learning_rate, schedule):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, schedule=schedule, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
		return clf_nn	
	ins.for_each(func1, func2)
	ins.plot_nn()
	_sa =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best_idx_1, ins._best_idx_2]
	with open('_sa', 'wb') as fp:
		pickle.dump(_sa, fp)

	algo='GA'
	algorithm='genetic_alg'
	max_iters=500
	ins =task2(learning_rates, pop_size, algo)
	func1 = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[0], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	def func2(learning_rate, pop_size):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, pop_size=pop_size, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
		return clf_nn	
	ins.for_each(func1, func2)
	ins.plot_nn()
	_ga =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best_idx_1, ins._best_idx_2]
	with open('_ga', 'wb') as fp:
		pickle.dump(_ga, fp)

	def feature_plot(feature_i):
		titles = ['time', 'test_accuracy', 'train_accuracy', 'validation_accuracy']
		title=titles[feature_i]
		ylims=[(0.0,10.0),(0.9,1.0),(0.9,1.0),(0.9,1.0)]
		plt.figure()
		plt.ylim(ylims[i])
		plt.bar(['RHC', 'SA', 'GA', 'Backpropagation'], [_rhc[i][_rhc[4]][_rhc[5]], _sa[i][_sa[4]][_sa[5]], _ga[i][_ga[4]][_ga[5]], _bp[i][_bp[4]][_bp[5]]])
		plt.xlabel("algorithm")
		plt.ylabel(title)
		plt.title(title)
		plt.savefig('nn_best_performance_{}.png'.format(title))

	for i in range(4):
		feature_plot(i)


	### learning_curve
	def learning_curve(algo, func):
		data = pd.read_csv('data/breast_cancer.csv')
		dict_type = {'M':1, 'B':0}
		data.diagnosis=data.diagnosis.map(dict_type)
		y = data.diagnosis.values
		x = data.drop(['id','diagnosis'], axis=1)
		X = preprocessing.scale(x)
	
		test_sizes = [0.9,0.8,0.6,0.4,0.2,0.1]
		train_acc_ = []
		test_acc_ = []
		train_sizes = [1 - test_size for test_size in test_sizes]

		for test_size in test_sizes:
			train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42)
			clf_nn = func
			clf_nn.fit(train_x, train_y)
			train_pred_y_ = clf_nn.predict(train_x)
			train_accuracy_y_ = accuracy_score(train_y, train_pred_y_)
			train_acc_.append(train_accuracy_y_)
			test_pred_y_ = clf_nn.predict(test_x)
			test_accuracy_y_ = accuracy_score(test_y, test_pred_y_)
			test_acc_.append(test_accuracy_y_)

		plt.figure()
		plt.plot(train_sizes, train_acc_, label='train')
		plt.plot(train_sizes, test_acc_, label='test')
		plt.xlabel('percentage_train')
		plt.ylabel('accuracy_score')
		plt.grid()
		plt.legend()
		plt.title('nn_learning_curve_{}'.format(algo))
		plt.savefig('nn_learning_curve_{}.png'.format(algo))

	algo='RHC'
	algorithm='random_hill_climb'
	max_iters =1000
	max_attempts = 100
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_ga[4]], restarts=restarts[_ga[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	learning_curve(algo, func)

	algo='SA'
	algorithm='simulated_annealing'
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=_sa[4], schedule=schedules[_sa[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	learning_curve(algo, func)

	algo='BP'
	algorithm='gradient_descent'
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_ga[4]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	learning_curve(algo, func)

	algo='GA'
	algorithm='genetic_alg'
	max_iters =500
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_ga[4]], pop_size=pop_size[_ga[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, bias=True, is_classifier=True, early_stopping=True)
	learning_curve(algo, func)