#!usr/bin/env python3

"""Logistic regression with regulation.

This script uses default fmin (in stead of gradient descent)
to fit logistic regression with regulations to avoid overfitting
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def map_features(x1, x2):
    x = []
    for i in range(6):
        for j in range(6):
            x.append(x1**i * x2**j)
    x = np.asarray(x).T
    return x

# Load data.
col_names = ["test1", "test2", "result"]
data = pd.read_table("./ex2/ex2data2.txt", sep=",", names=col_names)
x1 = np.asarray(data["test1"])
x2 = np.asarray(data["test2"])
y = np.asarray(data["result"])

# Define cost function.
x = map_features(x1, x2)
y = y.reshape(len(y), 1)
lamb = 1  # Regulation parameter.
m, n = np.shape(x)
args = (x, y, lamb, m, n) # Define parameters.


def cost_function(theta, *args):
    x, y, lamb, m, n = args
    theta = theta.reshape(n, 1)
    h = sigmoid(x @ theta)
    cost = (-(y.T@np.log(h) + (1-y).T@np.log(1-h)
           - 0.5*lamb*np.sum(theta[1:, 0:1]**2)) / m)
    return float(cost)


def gradient(theta, *args):
    x, y, lamb, m, n = args
    theta = theta.reshape(n, 1)
    h = sigmoid(x @ theta)
    grad = np.zeros(n)
    grad[0] = x[:, 0:1].T@(h-y)/m
    temp = (x[:, 1:].T@(h - y) + lamb*theta[1:, 0:1]) / m # Returns 2D array.
    grad[1:] = temp.reshape(n-1,)
    return grad

# Predict data.
init_theta = np.zeros(n) # All fmin seems to ONLY take 1D as input.
result = minimize(cost_function, init_theta,
                  args=args,
                  method="BFGS",
                  jac=gradient)
theta = result.x.reshape(n, 1)

predict = x @ theta
predict[predict>=0.5] = 1
predict[predict<0.5] = 0

accuracy = np.mean(predict==y)
print("Accuracy for train sets is {}".format(accuracy))

# Plot data and boundary.
fig, ax = plt.subplots(figsize=(9, 6))
# Recall that y is 2D now while x1 and x2 are still 1D.
pos = y.reshape(len(y),)==1
neg = y.reshape(len(y),)==0
l1, = ax.plot(x1[pos], x2[pos], "b+", markersize=10, label="Positive")
l2, = ax.plot(x1[neg], x2[neg], "yo", markersize=10, label="Negative")

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
boundary = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        boundary[i, j] = map_features(u[i], v[j]) @ theta

cs = ax.contour(u, v, boundary.T, colors="g", linewidths=2, levels=[0])
line = plt.Line2D((0, 0), (0, 0), color="g", linewidth=2)

plt.legend([l1, l2, line],
           ["Positive", "Negative", "Decision boundary"],
           loc="upper right")
plt.setp(ax,
         title="Data visualization",
         xlabel="Test 1",
         ylabel="Test 2")
plt.show()
