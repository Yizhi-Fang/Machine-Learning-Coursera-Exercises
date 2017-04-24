"""All functions are here.

A. Anomaly Detection.
B. Recommender System.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings
import re


# Anomaly Detection.
def multivariate_gaussian(x):
    """Compute multi-variable gaussian probability."""
    mu = np.mean(x, axis=0)
    sigma2 = np.std(x, axis=0) ** 2  # Means sigma**2.
    # Compute variance matrix.
    sigma2_matrix = np.diag(sigma2)
    # Dimension of data.
    k = len(mu)
    x_new = x - mu
    # Multi-variate gaussian probability (1D).
    p = ((2*np.pi)**(-k/2) * la.det(sigma2_matrix)**(-0.5)
         * np.exp(-0.5 * np.sum(x_new@la.inv(sigma2_matrix)*x_new, axis=1)))
    return p


def display_fit(x):
    """Display original data and gaussian fit.

    Returns:
        ax: The axis to plot for further plotting.
    """
    u = np.linspace(0, 30, 60)
    x1, x2 = np.meshgrid(u, u)
    x1_stack = x1.reshape(x1.shape[0]*x1.shape[1], 1)
    x2_stack = x2.reshape(x2.shape[0]*x2.shape[1], 1)
    x_stack = np.concatenate((x1_stack, x2_stack), axis=1)
    p = multivariate_gaussian(x_stack)
    p = p.reshape(x1.shape)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[:, 0], x[:, 1], marker="x", s=20)
    ax.contour(x1, x2, p)
    plt.setp(ax, xlabel="Latency (ms)", ylabel="Throughput (mb/s)")
    plt.show()

    return ax


def select_threshold(xval, yval):
    """Find out best threshold epsilon for anomaly detection."""
    pval = multivariate_gaussian(xval)
    epsilons = np.linspace(np.min(pval), np.max(pval), 1000)
    f_best = 0
    epsilon_best = 0

    for epsilon in epsilons:
        preds = pval < epsilon
        preds = preds.reshape(len(yval), 1)
        tp = np.sum((preds == 1) & (yval == 1))
        fp = np.sum((preds == 1) & (yval == 0))
        fn = np.sum((preds == 0) & (yval == 1))
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)
        f1 = 2 * prec * rec / (prec+rec)
        if f1 > f_best:
            f_best = f1
            epsilon_best = epsilon

    return epsilon_best


# Recommender System.
def cost_function(x_stack, *args):
    """Compute cost function for recommender system.

    Args:
        x_stack: Stacked column vector of x and theta.
        *args: Other arguments,
               num_users, num_movies, num_features, y, r, lamb
    """
    num_users, num_movies, num_features, y, r, lamb = args
    x = x_stack[:num_movies * num_features].reshape(num_movies, num_features)
    theta = x_stack[num_movies * num_features:].reshape(num_users, num_features)

    cost = (0.5*np.sum((x@theta.T - y)**2 * r)
            + 0.5*lamb*np.sum(theta**2)
            + 0.5*lamb*np.sum(x**2))

    return cost


def gradient(x_stack, *args):
    """Compute gradient for recommender system.

    Args:
        x_stack: Stacked column vector of x and theta.
        *args: Other arguments,
               num_users, num_movies, num_features, y, r, lamb
    """
    num_users, num_movies, num_features, y, r, lamb = args
    x = x_stack[:num_movies * num_features].reshape(num_movies, num_features)
    theta = x_stack[num_movies * num_features:].reshape(num_users, num_features)

    x_grad = (x@theta.T - y)*r@theta + lamb*x
    theta_grad = ((x@theta.T - y)*r).T@x + lamb*theta
    grad = np.append(x_grad, theta_grad).reshape(len(x_stack), 1)
    return grad


def numerical_gradient(x, *args):
    e = 1e-4
    perturb = np.zeros(x.shape)
    num_grad = np.zeros(x.shape)

    for i in range(len(x)):
        perturb[i] = e
        cost1 = cost_function(x + perturb, *args)
        cost2 = cost_function(x - perturb, *args)
        num_grad[i] = 0.5 * (cost1-cost2) / e
        perturb[i] = 0

    return num_grad


def check_rs_gradient(x, *args):
    lamb = args[-1]
    J = cost_function(x, *args)
    print("The cost function with lambda = {:.1f} is {:.2f}.".format(lamb, J))
    grad = gradient(x, *args)
    num_grad = numerical_gradient(x, *args)
    diff = la.norm(num_grad-grad) / la.norm(num_grad+grad)
    print("The relative difference should be small (less than 1e-9).\n"
          "Relative difference with lambda "
          "= {:.1f} is {:.2e}.".format(lamb, diff))


def random_initialize(num):
    np.random.seed(45466)
    return np.array([np.random.randn() for i in range(num)]).reshape(num, 1)


def normalize_ratings(y, r):
    """Normalize rating matrix y.

    Compute mean ratings of each movie only when r = 1

    Returns:
        y_mean: Mean of each movie.
        y_norm: Normalized y.
    """
    y_mean = np.zeros((len(y),))
    y_norm = np.zeros(y.shape)
    for i in range(len(y)):
        indices = r[i, :] == 1
        y_mean[i] = np.mean(y[i, indices])
        y_norm[i, indices] = y[i, indices] - y_mean[i]
    return y_mean, y_norm


def movie_id_search(id):
    """Search movie id and return movie name."""
    movie_list = []
    with open("./ex8/movie_ids.txt", "r", encoding = "ISO-8859-1") as f:
        for i, line in enumerate(f, 1):
            line = re.sub("^[0-9]+\s", "", line)
            movie_list.append(line.rstrip())
    movie_name = movie_list[id-1]
    return movie_name


def fmincg(f, jac, x0, args=(), alpha=0.25, beta=0.5, maxiter=50, tol=1e-9):
    """Minimize function with backtracking.

    See more details: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method

    Args:
        f: Function to be minimized.
        jac: Gradient of function.
        x0: Initial value/vector.
        args: All other parameters in function.
        alpha: Constants that I don't know...
        beta: Another constants that I don't know...
        maxiter: Max number of iterations.
        tol: Tolerance between optimal and initial value/vector.

    Returns:
        x: Optimal value/vector x.
    """
    warnings.filterwarnings("ignore")

    # Initialization.
    dx_prev = -jac(x0, *args)
    s_prev = dx_prev
    x = x0

    print("Minimizing function...")

    for i in range(maxiter):
        dx = -jac(x, *args)
        if dx.T@dx < tol:
            print("Terminated because tolerance criterion is reached...")
            return x

        # Polak-Ribiere formula
        beta_pr = dx.T @ (dx-dx_prev) / (dx_prev.T@dx_prev)
        # Positive beta_pr indicates moving to min.
        beta_pr = np.max((0, beta_pr))
        # Search direction.
        s = dx + beta_pr*s_prev

        # Backtracking (https://en.wikipedia.org/wiki/Backtracking).
        t = 1.0
        cost = f(x, *args)
        grad = jac(x, *args)
        cost_new = f(x+t*s, *args)
        alpha_grad = alpha * grad.T @ s
        # f(x+t*s) - f(x) = t*s*jac(x) where s*jac(x) is alpha_grad.
        while cost_new > cost + t*alpha_grad:
            t = beta * t
            cost_new = f(x+t*s, *args)
        # Throw out all big values of t*s.

        # Search right side.
        t_right = 2 * t
        t_temp = t
        while t_right - t > 1e-4:
            cost_right = f(x+t_right*s, *args)
            if cost_right > cost_new:
                t_right = (t+t_right) / 2
            else:
                t = t_right
                t_right = 2 * t
                cost_new = cost_right

        # Search left side.
        if t == t_temp:
            t_left = t / 2
            while t - t_left > 1e-4:
                cost_left = f(x+t_left*s, *args)
                if cost_left > cost_new:
                    t_left = (t+t_left) / 2
                else:
                    t = t_left
                    t_left = t / 2
                    cost_new = cost_left

        x = x + t*s
        s_prev = s
        dx_prev = dx

        # To replace old line with new lines, add end="\r".
        print("Iteration {:d} | Cost: {:f}".format(i+1, cost_new), end="\r")

    print("\nProgram completed.")

    return x
