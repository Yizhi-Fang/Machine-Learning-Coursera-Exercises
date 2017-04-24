#!usr/bin/env python3

"""Linear Regression for multi-variables.

This script uses Gradient Descent to predict house price given the area
and number of bedrooms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(x):
    """Normalize x for gradient descent.

    Returns:
        x_norm: Normalized x.
        mu: Feature mean of x.
        sigma: Feature standard deviation of x.
    """
    x_norm = x
    mu = np.mean(x, axis=0) # compute mean in columns
    sigma = np.std(x, axis=0)
    for j in range(np.shape(x)[1]):
        x_norm[:, j] = (x[:, j] - mu[j]) / sigma[j]
    return [x_norm, mu, sigma]


def gradient_descent(x, y, alpha, iters):
    """Calculate cost function of gradient descent.

    Args:
        x: Training samples.
        y: Labels.
        alpha: Learning rate of gradient descent.
        iters: Number of iterations.

    Returns:
        theta: Optimal theta.
        cost_hist: List of cost functions in each iteration.
    """
    m = len(y)
    theta = np.asarray([[0] for i in range(np.shape(x)[1])])
    cost_hist = []
    for i in range(iters):
        theta = theta - alpha*x.T@(x@theta - y)/m
        cost = 0.5 * np.sum((x@theta - y)**2) / m
        cost_hist.append(cost)
    return [theta, cost_hist]

# Read data.
col_names = ["area", "br", "price"]
data = pd.read_table("./ex1/ex1data2.txt",
                     sep=",",
                     names=col_names,
                     dtype="float")

x = np.asarray(data[["area", "br"]])
y = np.asarray(data["price"])
y = y.reshape(len(y), 1)  # Original y is 1D array.

# Gradient descent.
alpha = 0.01
iters = 400

x, mu, sigma = normalize(x)
x = np.insert(x, 0, 1, axis=1)  # Insert a column with 1s at the beginning.
theta, cost_hist = gradient_descent(x, y, alpha, iters)

# Predict new house.
sample = [1650, 3]
for j in range(len(sample)):
    sample[j] = (sample[j] - mu[j])/sigma[j]
sample = np.insert(sample, 0, 1)
price = float(sample@theta)

print("The predicted price for a house of 1650 "
      "ft square and 3 bedrooms is ${}".format(price))

# Plot cost function as function of iterations.
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(range(iters), cost_hist, linewidth=2, color="red")
plt.setp(ax,
         title="Cost function history",
         xlabel="Iterations", ylabel="Cost function")
plt.show()

# Play with different alpha values (optional).
col_names = ["area", "br", "price"]
data = pd.read_table("../ex1/ex1data2.txt",
                     sep=",",
                     names=col_names,
                     dtype="float")

x = np.asarray(data[["area", "br"]])
y = np.asarray(data["price"])
y = y.reshape(len(y), 1) # Original y is 1D array.

x, mu, sigma = normalize(x)
x = np.insert(x, 0, 1, axis=1) # Insert a column with 1s at the beginning.

fig, ax = plt.subplots(figsize=(9, 6))
for c, alpha in zip(np.linspace(0, 1, 5), [0.001, 0.003, 0.01, 0.03, 0.1]):
    cost_hist = gradient_descent(x, y, alpha, iters=400)[1]
    ax.plot(range(400), cost_hist,
            linewidth=2,
            color=plt.cm.rainbow(c),
            label="alpha={}".format(alpha))
plt.setp(ax,
         title="Gradient Descent with different alpha",
         xlabel="Iterations",
         ylabel="Cost function")
plt.legend(loc="upper right")
plt.show()
