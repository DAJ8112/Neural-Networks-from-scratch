# ReLU Activation function code added here

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()



X = [[1, 2, 3, 2.5], # this could be the raw actual data and the first row is the first sample . No of features/attributes per sample here = 4
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)

"""
ReLU simple execution
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
if i > 0:
	output.append(i)
elif i <= 0:
	output.append(0)
	OR
	output.append(max(0, i ))

print(output)
"""

class Layer_Dense:

	def __init__(self, n_inputs, n_neurons): # n_inputs basically would be the no of features/attributes per sample of data and n_neurons would be our choice
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #Multiplied by 0.10 to keep values smaller
		self.biases = np.zeros((1, n_neurons))
	
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights)+ self.biases


class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

