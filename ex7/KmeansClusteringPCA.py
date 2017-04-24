#!usr/bin/env python3

"""K-means clustering and Principle Component Analysis."""

from function_all import *
import scipy.io as sio


# K-means clustering.
data = sio.loadmat("./ex7/ex7data2.mat")
x = data["X"]

init_centroids = randon_init_centroids(x, k=3)
centroids, idx = run_kmeans(x, init_centroids, plot=True)

# Image compression with K-means.
data = sio.loadmat("./ex7/bird_small.mat")
img = data["A"]
m = img.shape[0] * img.shape[1]
x = img.reshape((m, img.shape[2]))

init_centroids = randon_init_centroids(x, k=16)
centroids, idx = run_kmeans(x, init_centroids)
x_compressed = centroids[idx - 1, :]
img_compressed = x_compressed.reshape(img.shape)

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(img_compressed)
plt.show()

# PCA 2D-1D.
data = sio.loadmat("./ex7/ex7data1.mat")
x = data["X"]
x_norm, mu, sigma = feature_normalize(x)

x_rec = pca(x_norm, 1)
display_projection(x_norm, x_rec)

# PCA on faces.
data = sio.loadmat("./ex7/ex7faces.mat")
x = data["X"]
x_norm, mu, sigma = feature_normalize(x)

display_data(x[:100, :])
x_rec = pca(x_norm, 100)
display_data(x_rec[:100, :])
