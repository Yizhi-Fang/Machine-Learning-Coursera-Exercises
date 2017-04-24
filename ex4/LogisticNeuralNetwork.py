#!usr/bin/env python3

"""Logistic regression with neural network."""

from function_all import *
import scipy.io as sio


# Test gradient.
check_nn_gradient()

# Load data and setup.
data = sio.loadmat("./ex4/ex4data1.mat")
x = data["X"]
y = data["y"]
input_layer = 400
hidden_layer = 25
num_labels = 10
lamb = 1
args = (input_layer, hidden_layer, num_labels, x, y, lamb)

# Test cost function.
weights = sio.loadmat("./ex4/ex4weights.mat")
theta = np.append(weights["Theta1"], weights["Theta2"])
print("Cost for lambda "
      "= {:d} is {:f}.".format(lamb, cost_function(theta, *args)))
print("It should be 0.287629 (lambda = 0) and 0.383770 (lambda = 1).")

# Display data.
sel = np.random.permutation(x)
display_data(sel[:100, :])

# Train data.
num = hidden_layer*(input_layer+1) + num_labels*(hidden_layer+1)
init_theta = random_initialize(num)
theta = fmincg(cost_function, gradient, init_theta, args=args, maxiter=50)

# Predict.
network = NeuralNetwork(input_layer, hidden_layer, num_labels)
h = network.predict(x, theta)
p = h.argmax(axis=1) + 1 # find index of max per row
p = p.reshape(y.shape)
accuracy = np.mean(p == y)
print("Accuracy for this training set is {:f}".format(accuracy))
