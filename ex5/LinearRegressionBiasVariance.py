#!usr/bin/env python3

"""Play bias and variance with linear regression."""

from function_all import *
import scipy.io as sio


# Load data.
data = sio.loadmat("./ex5/ex5data1.mat")
x = data["X"]
y = data["y"]
xval = data["Xval"]
yval = data["yval"]
xtest = data["Xtest"]
ytest = data["ytest"]

# Linear regression.
lamb = 0.0
init_theta = np.zeros((x.shape[1] + 1, 1))
args = (x, y, lamb)
theta = fmincg(cost_function, gradient, init_theta, args=args, maxiter=10)
predict = np.insert(x, 0, 1, axis=1) @ theta

error_train, error_val = compute_error(x, y, xval, yval, lamb)

display_data(x, y, predict, lamb)
display_error(error_train, error_val, lamb)

# Polynomial regression.
x_poly = poly_features(x[:, 0], 8)
x_poly, mu, sigma = feature_normalize(x_poly)

x_poly_val = poly_features(xval[:, 0], 8)
x_poly_val = (x_poly_val - mu) / sigma

lamb = 0.0
init_theta = np.zeros((x_poly.shape[1] + 1, 1))
args = (x_poly, y, lamb)
theta = fmincg(cost_function, gradient, init_theta, args=args, maxiter=10)
predict = np.insert(x_poly, 0, 1, axis=1) @ theta

error_train, error_val = compute_error(x_poly, y, x_poly_val, yval, lamb)

display_data(x, y, predict, lamb)
display_error(error_train, error_val, lamb)
