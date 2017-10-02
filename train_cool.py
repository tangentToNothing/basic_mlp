from neural_networ import MLP_NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def load_data2():
	data = np.loadtxt('elastic_multi1_no_heads.txt')
		
	out = []

	data_normed = data / data.max(axis=0)
	y_max = data.max(axis=0)[-1]
	for i in range(data_normed.shape[0]):
		fart = list((data_normed[i,0:-1].tolist(), [data_normed[i,-1]])) # don't mind this variable name
		out.append(fart)

	return out, y_max

def test_networks_on_lr(learning_rates, my_data, y_max):
	errors_and_lr = []
	for rate in learning_rates:
		myNetwork = MLP_NeuralNetwork(7, 12, 1, iterations = 50, learning_rate = rate, momentum = 0.5, rate_decay = 0.001)
		error_over_time = myNetwork.train(my_data)
		errors_and_lr.append({'learning_rate': rate, 'errors': error_over_time})
	return(errors_and_lr)

def test_networks_on_hidden(hidden_neural_network_size, my_data, y_max):
	errors_and_hidden = []
	for hidden_layer in hidden_neural_network_size:
		myNetwork = MLP_NeuralNetwork(7, hidden_layer, 1, iterations = 50, learning_rate = 0.1, momentum = 0.5, rate_decay = 0.001)
		error_over_time = myNetwork.train(my_data)
		errors_and_hidden.append({'hidden_size': hidden_layer, 'errors': error_over_time})
	return(errors_and_hidden)

def test_networks_on_decay(rate_decay, my_data, y_max):
	errors_and_decay = []
	for decay in rate_decay:
		myNetwork = MLP_NeuralNetwork(7, 12, 1, iterations = 50, learning_rate = 0.1, momentum = 0.5, rate_decay = rate_decay)
		error_over_time = myNetwork.train(my_data)
		errors_and_decay.append({'decay_rate': decay, 'errors': error_over_time})
	return(errors_and_decay)

my_data, y_max = load_data2()

learning_rates = [0.5, 0.1, 0.01]
epochs = range(0,50)

learning_data = test_networks_on_lr(learning_rates, my_data, y_max)

plt.figure(0)
plt.plot(epochs, learning_data[0]['errors'], 'r', epochs, learning_data[1]['errors'], 'b', epochs, learning_data[2]['errors'], 'g')
plt.ylabel('error')
plt.xlabel('iterations')
plt.title('Varying Learning Rates')

hidden_sizes = [16, 14, 12]
hidden_data = test_networks_on_hidden(hidden_sizes, my_data, y_max)

plt.figure(1)
plt.plot(epochs, hidden_data[0]['errors'], 'r', epochs, hidden_data[1]['errors'], 'b', epochs, hidden_data[2]['errors'], 'g')
plt.ylabel('error')
plt.xlabel('iterations')
plt.title('Varying Hidden Layer Sizes')

decay_opts = [0.01, 0.001, 0.001]
decay_data = test_networks_on_decay(decay_opts, my_data, y_max)

plt.figure(2)
plt.plot(epochs, decay_data[0]['errors'], 'r', epochs, decay_data[1]['errors'], 'b', epochs, decay_data[2]['errors'], 'g')
plt.ylabel('error')
plt.xlabel('iterations')
plt.title('Varying Decay Amounts')

plt.figure(0)
plt.show()

plt.figure(1)
plt.show()

plt.figure(2)
plt.show()


