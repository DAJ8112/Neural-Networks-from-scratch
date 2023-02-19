""" Softmax Activation 
Softmax to work with a batch of inputs rather than just a vector.
Will use it next in the part adding to the existing p6 """


import math
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
				[8.9, -1.81, 0.2],
				[1.41, 1.051, 0.026]]

E = math.e

exp_values = np.exp(layer_outputs)

# print(np.sum(layer_outputs, axis=1, keepdims=True)) # axis=1 -> to get sum of rows || keepdims=True -> keep the dimensions same


norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
# print(sum(norm_values))