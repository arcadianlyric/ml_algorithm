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
		self._best__row = 0
		self._best__col = 0
		self.curve = []
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
		self.clf_nn_best=''
		

	def for_each(self, func2):

		for _row, learning_rate in enumerate(self.learning_rates):
			for _col, val in enumerate(self.param):
				clf_nn = func2(learning_rate, val)
				if self.clf_nn_best=='':
					self.clf_nn_best =clf_nn

				start = time.time()
				clf_nn.fit(self.train_x, self.train_y)
				end = time.time()
				_time_cost = end - start

				train_pred_y = clf_nn.predict(self.train_x)
				self.train_accuracy[_row][_col] = accuracy_score(self.train_y, train_pred_y)
				val_pred_y = clf_nn.predict(self.val_x)
				val_accuracy_y = accuracy_score(self.val_y, val_pred_y)
				self.val_accuracy[_row][_col]= val_accuracy_y
				test_pred_y = clf_nn.predict(self.test_x)
				self.test_accuracy[_row][_col] = accuracy_score(self.test_y, test_pred_y)
				self.time_cost[_row][_col]= _time_cost
				if self.algo=='BP':
					self.curve.append(clf_nn.fitness_curve)
				else:
					self.curve.append(clf_nn.fitness_curve[:,0])

				if val_accuracy_y > self.val_accuracy_best:
					self.clf_nn_best = clf_nn
					self._best__row = _row
					self._best__col = _col
					self.val_accuracy_best = val_accuracy_y

		return 

	def plot_nn(self):
		if self.algo=='BP':
			plt.figure()
			plt.plot(-self.clf_nn_best.fitness_curve)
			plt.xlabel('num_iteration')
			plt.ylabel('train_loss')
			plt.grid()
			plt.title('train_loss_num_iteration_best_{}'.format(self.algo))
			plt.savefig('train_loss_num_iteration_best_{}.png'.format(self.algo))
		else:
			plt.figure()
			plt.plot(self.clf_nn_best.fitness_curve[:,0])
			plt.xlabel('num_iteration')
			plt.ylabel('train_loss')
			plt.grid()
			plt.title('train_loss_num_iteration_best_{}'.format(self.algo))
			plt.savefig('train_loss_num_iteration_best_{}.png'.format(self.algo))

			plt.figure()
			print(len(self.curve))
			for i in range(len(self.curve)):
				plt.plot(self.curve[i])
			plt.xlabel('iteration')
			plt.ylabel('train_loss')
			plt.grid()
			plt.title('train_loss_{}'.format(self.algo))
			plt.savefig('train_loss_{}.png'.format(self.algo))

		print("time_cost=", self.time_cost[self._best__row][self._best__col])
		print("test_accuracy=", self.test_accuracy[self._best__row][self._best__col])


if __name__ == "__main__":
	schedules = [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
	learning_rates = [0.0001, 0.01, 0.1, 1.0]
	# pop_size = [25, 50, 100, 200]
	mutation_prob = [0.1,0.2,0.5]
	restarts = [3,5,7,9]

	max_iters =1000
	max_attempts = 100
	hidden_nodes = [12]
	activation ='relu'

	# Backpropagation
	algo='BP'
	algorithm='gradient_descent'
	param=[1]
	ins =task2(learning_rates, param, algo)
	def func2(learning_rate,param):
		clf_nn = mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
		return clf_nn	
	ins.for_each(func2)
	ins.plot_nn()
	_bp =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best__row, ins._best__col]
	with open('ins_bp', 'wb') as fp:
		pickle.dump(ins, fp)

	algo='RHC'
	algorithm='random_hill_climb'
	ins =task2(learning_rates, restarts, algo)
	def func2(learning_rate, restarts):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, restarts=restarts,  learning_rate=learning_rate, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
		return clf_nn	
	ins.for_each(func2)
	ins.plot_nn()
	_rhc =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best__row, ins._best__col]
	with open('ins_rhc', 'wb') as fp:
		pickle.dump(ins, fp)

	algo='SA'
	algorithm='simulated_annealing'
	ins =task2(learning_rates, schedules, algo)
	def func2(learning_rate, schedule):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, schedule=schedule, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
		return clf_nn	
	ins.for_each(func2)
	ins.plot_nn()
	_sa =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best__row, ins._best__col]
	with open('ins_sa', 'wb') as fp:
		pickle.dump(ins, fp)

	algo='GA'
	algorithm='genetic_alg'
	# max_iters=500
	ins =task2(learning_rates, mutation_prob, algo)
	def func2(learning_rate, mutation_prob):
		clf_nn=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rate, mutation_prob=mutation_prob, max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
		return clf_nn	
	ins.for_each(func2)
	ins.plot_nn()
	_ga =[ins.time_cost, ins.test_accuracy, ins.train_accuracy, ins.val_accuracy, ins._best__row, ins._best__col]
	with open('ins_ga', 'wb') as fp:
		pickle.dump(ins, fp)

	def feature_plot(feature_i):
		titles = ['time', 'test_accuracy', 'train_accuracy', 'validation_accuracy']
		title=titles[feature_i]
		ylims=[(0.0,70.0),(0.9,1.0),(0.9,1.0),(0.9,1.0)]
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
	
		test_range = np.linspace(0.01, 0.99, 10)
		train_range = [1 - test_size for test_size in test_range]
		train_acc_, test_acc_ = [], []

		for test_size in test_range:
			train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42)
			clf_nn = func
			clf_nn.fit(train_x, train_y)
			train_pred_y_ = clf_nn.predict(train_x)
			train_acc_.append(accuracy_score(train_y, train_pred_y_))
			test_pred_y_ = clf_nn.predict(test_x)
			test_accuracy_y_ = accuracy_score(test_y, test_pred_y_)
			test_acc_.append(test_accuracy_y_)

		plt.figure()
		plt.plot(train_range, train_acc_, label='train')
		plt.plot(train_range, test_acc_, label='test')
		plt.xlabel('percentage_train')
		plt.ylabel('accuracy_score')
		plt.grid()
		plt.legend()
		plt.title('nn_learning_curve_{}'.format(algo))
		plt.savefig('nn_learning_curve_{}.png'.format(algo))

	algo='RHC'
	algorithm='random_hill_climb'
	max_iters =2000
	max_attempts = 100
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_rhc[4]], restarts=restarts[_rhc[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
	learning_curve(algo, func)

	algo='SA'
	algorithm='simulated_annealing'
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_sa[4]], schedule=schedules[_sa[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
	learning_curve(algo, func)

	algo='BP'
	algorithm='gradient_descent'
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_bp[4]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
	learning_curve(algo, func)

	algo='GA'
	algorithm='genetic_alg'
	# max_iters =500
	func=mlrose_hiive.NeuralNetwork(algorithm=algorithm, learning_rate=learning_rates[_ga[4]], mutation_prob=mutation_prob[_ga[5]], max_attempts=max_attempts, random_state=42, curve=True, hidden_nodes=hidden_nodes, activation=activation, max_iters=max_iters, early_stopping=True)
	learning_curve(algo, func)