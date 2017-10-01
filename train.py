from neural_networ import MLP_NeuralNetwork
import numpy as np
myNetwork = MLP_NeuralNetwork(7, 12, 1, iterations = 100, learning_rate = 0.1, momentum = 0.5, rate_decay = 0.001)

def load_data():
	training_data = []
	with(open('elastic_multi1_no_heads.txt','r')) as data:
		i = 0
		for line in data:
			if(i != 0):
				point = line.split()
				training_data.append([point[0:6], point[-1]])
			i += 1	

	return training_data

def load_data2():
	data = np.loadtxt('elastic_multi1_no_heads.txt')
		
	out = []

	data_normed = data / data.max(axis=0)
	y_max = data.max(axis=0)[-1]
	for i in range(data_normed.shape[0]):
		fart = list((data_normed[i,0:-1].tolist(), [data_normed[i,-1]])) # don't mind this variable name
		out.append(fart)

	return out, y_max

my_data, y_max = load_data2()

myNetwork.train(my_data)

myNetwork.test(my_data, y_max)