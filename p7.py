# Softmax Activation 

import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = []
for output in layer_outputs:
	exp_values.append(E**output)

# OR using numpy 

exp_values_np = np.exp(layer_outputs)

print(exp_values)
print(exp_values_np)

# Normalization

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
	norm_values.append(value/norm_base)

# OR using numpy 

norm_values_np = exp_values_np/np.sum(exp_values)

print(f"Using math {norm_values} with a sum of {sum(norm_values)}")
print(f"Using numpy {norm_values_np} with a sum of {sum(norm_values_np)}")