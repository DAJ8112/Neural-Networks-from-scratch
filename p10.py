""" Loss using Categorial Cross Entropy """

import math 

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
		 math.log(softmax_output[1])*target_output[1] + 
		 math.log(softmax_output[2])*target_output[2])

print(loss)

loss = -math.log(softmax_output[0])
print(loss)

print(-math.log(0.7)) #higher confidence of our neural network and because the corresponding target class is correct the loss would be less compared to the below one.
print(-math.log(0.5))
