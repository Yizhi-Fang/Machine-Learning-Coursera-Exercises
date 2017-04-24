"""All functions are here.

A. K-means cluster.
B. Principle Component Analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


# K-means cluster.
def randon_init_centroids(x, k):
    x_random = np.random.permutation(x)
    return x_random[:k, :]


def find_closest_centroids(x, centroids):
    """Find centroids for each sample.

    Assign each sample the closest centroid (1:k) index found.

    Returns:
        idx: A list of centroid index for each sample.
    """
    m = len(x)
    k = len(centroids)
    # Cost matrix, each row represents a sample
    # and each column represents one class.
    cost = np.zeros((m, k))
    for i in range(k):
        cost[:, i] = np.sum((x - centroids[i, :]) ** 2, axis=1)  # Sum per row.
    idx = np.argmin(cost, axis=1) + 1  # Find min per row.
    return idx


def compute_centroids(x, idx):
    """Calculate each class mean.

    Calculate the mean position of current class and mark it as a new centroid.

    Returns:
        centroids: A new calculated centroids.
    """
    n = x.shape[1]
    k = np.max(idx)
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i+1)[0]
        centroids[i, :] = np.mean(x[indices], axis=0)  # Mean per column.
    return centroids


def plot_progress(fig, ax, iter, x, idx, centroids, centroids_old):
    """Plot K-means progress per iteration."""
    k = len(centroids)
    for i, c in zip(range(k), np.linspace(0, 1, k)):
        indices = np.where(idx == i+1)[0]
        ax.scatter(x[indices, 0], x[indices, 1], s=20, c=plt.cm.rainbow(c))
        ax.scatter(centroids[i, 0], centroids[i, 1],
                   marker="x",
                   s=40,
                   c="k",
                   linewidths=2)
        line = plt.Line2D((centroids_old[i, 0], centroids[i, 0]),
                          (centroids_old[i, 1], centroids[i, 1]),
                          lw=1, c="k")
        plt.gca().add_line(line)
        fig.canvas.draw()  # To update the figure every iteration.
    # To update the title every iteration.
    fig.canvas.set_window_title("Iteration number {:d}".format(iter + 1))
    plt.show()


def run_kmeans(x, init_centroids, max_iter=10, plot=False):
    centroids = init_centroids
    centroids_old = centroids

    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 6))

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids)
        if plot:
            plot_progress(fig, ax, i, x, idx, centroids, centroids_old)
            input("Please press Enter to continue...")
        centroids_old = centroids
        centroids = compute_centroids(x, idx)

    return centroids, idx


# Principle Component Analysis.
def feature_normalize(x):
    """Normalize features.

    Returns:
        x_norm: Normalized training samples.
        mu: Feature mean.
        sigma: Feature standard deviation.
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma


def pca(x, k):
    """Apply Principle Component Analysis on x.

    Somehow the default PCA function doesn't work well
    so I use Singular Value Decomposition instead.

    Returns:
        x_rec: Recovered x (dimension reduced)
    """
    # Compute covariance matrix sigma.
    m = len(x)
    sigma = x.T @ x / m
    u, s, v = la.svd(sigma)
    z = x @ u[:, :k]
    x_rec = z @ u[:, :k].T
    return x_rec

def display_projection(x, x_rec):
    """Display original and dimension reduced data."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[:, 0], x[:, 1], s=20, label="Original")
    ax.scatter(x_rec[:, 0], x_rec[:, 1],
               c="r",
               label="Recovered X (dimension reduced)")
    for m in range(len(x)):
        line = plt.Line2D((x[m, 0], x_rec[m, 0]),
                          (x[m, 1], x_rec[m, 1]),
                          ls="--", c="k")
        plt.gca().add_line(line)
    plt.legend(loc="upper left")
    plt.setp(ax, xlim=[-3, 3], ylim=[-3, 3])
    plt.show()

def display_data(x):
    """Display face dataset."""
    m, n = x.shape
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n/example_width)
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))

    pad = 1  # Between image padding.
    display_array = -np.ones((pad+display_rows*(example_height+pad),
                              pad+display_cols*(example_width+pad)))

    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex == m:
                break
            max_val = np.max(np.abs(x[curr_ex, :]))
            row_start = pad + j*(example_height+pad)
            col_start = pad + i*(example_width+pad)
            display_array[row_start:row_start+example_height,
                          col_start:col_start+example_width] = (
                x[curr_ex, :].reshape(example_height, example_width) / max_val)
            curr_ex += 1
        if curr_ex == m:
            break

    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(display_array.T, extent=[0, 10, 0, 10], cmap=plt.cm.plasma)
    plt.show()
