#!usr/bin/env python3

"""Anomaly Detection and Recommender System."""

import scipy.io as sio
from function_all import *


# 2D anomaly detection.
data = sio.loadmat("./ex8/ex8data1.mat")
x = data["X"]
xval = data["Xval"]
yval = data["yval"]

p = multivariate_gaussian(x)
epsilon = select_threshold(xval, yval)
outliers = x[p < epsilon, :]
ax = display_fit(x)
ax.scatter(outliers[:, 0], outliers[:, 1],
           marker="o",
           s=60,
           linewidths=2,
           facecolors="none", edgecolors="r")
plt.show()

# High dimension anomaly detection.
data = sio.loadmat("./ex8/ex8data2.mat")
x = data["X"]
xval = data["Xval"]
yval = data["yval"]

p = multivariate_gaussian(x)
epsilon = select_threshold(xval, yval)
outliers = x[p < epsilon, :]
print("The best epsilon found is "
      "{:.2e} and {:d} anomalies found".format(epsilon, len(outliers)))

# Recommender System.
# Load movie rating dataset.
data = sio.loadmat("./ex8/ex8_movies.mat")
y = data["Y"]
r = data["R"]

# Load pre-assigned movie parameters.
data = sio.loadmat("./ex8/ex8_movieParams.mat")
x = data["X"]
theta = data["Theta"]

# Reduce dimensions to run faster and test cost function and gradient.
num_users = 4
num_movies = 5
num_features = 3

x = x[:num_movies, :num_features]
theta = theta[:num_users, :num_features]
y = y[:num_movies, :num_users]
r = r[:num_movies, :num_users]

num = num_movies*num_features + num_users*num_features
x_stack = np.append(x, theta).reshape(num, 1)
lamb = 1.5

args = (num_users, num_movies, num_features, y, r, lamb)
check_rs_gradient(x_stack, *args)

# Create a new user and rate some movies.
data = sio.loadmat("./ex8/ex8_movies.mat")
y = data["Y"]
r = data["R"]

my_ratings = np.zeros((y.shape[0], 1))
my_ratings[0] = 4
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[97] = 2
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# Add the new user to data matrix.
y = np.concatenate((y, my_ratings), axis=1)
r = np.concatenate((r, my_ratings != 0), axis=1)

# Normalizing ratings.
y_mean, y_norm = normalize_ratings(y, r)

# Initialize all parameters.
num_movies, num_users = y.shape
num_features = 10
lamb = 10

args = (num_users, num_movies, num_features, y, r, lamb)
num = num_movies*num_features + num_users*num_features
x0 = random_initialize(num)

# Predict my ratings for all movies.
x_stack = fmincg(cost_function, gradient, x0, args=args, maxiter=100)
x = x_stack[:num_movies * num_features].reshape(num_movies, num_features)
theta = x_stack[num_movies * num_features:].reshape(num_users, num_features)
my_preds = x @ theta[-1, :].T + y_mean
indices = np.argsort(my_preds)[::-1]  # Descending predictions.
reco = my_preds[indices]

# Print out top 10 movies.
print("Top 10 recommendations for you:")
for i in range(10):
    movie_name = movie_id_search(indices[i] + 1)
    print("Predicting rating {:.1f} for movie".format(reco[i]), movie_name)
