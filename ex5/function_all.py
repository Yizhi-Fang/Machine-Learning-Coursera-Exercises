"""All functions are here."""

import numpy as np
import warnings
import matplotlib.pyplot as plt


def cost_function(theta, *args):
    x, y, lamb = args
    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    cost = 0.5*np.sum((x@theta-y)**2)/m + 0.5*lamb*np.sum(theta[1:]**2)/m
    return cost


def gradient(theta, *args):
    x, y, lamb = args
    m = x.shape[0]
    x = np.insert(x, 0, 1, axis=1)
    grad = np.zeros(theta.shape)
    grad[0] = x[:, :1].T @ (x@theta-y) / m
    grad[1:] = x[:, 1:].T@(x@theta-y)/m + lamb*theta[1:]/m
    return grad


def poly_features(x, p):
    """Create a polynomial feature matrix up to p-th power of x.
    """
    x = np.zeros((x.shape[0], p))
    for i in range(p):
        x[:, i] = x**(i+1)
    return x


def feature_normalize(x):
    """Normalize all features.

    Returns:
        x_norm: Normalized training sets.
        mu: Feature mean.
        sigma: Feature standard deviation.
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma


def display_data(x, y, predict, lamb):
    # Sort X and y ascending.
    indices = np.argsort(x, axis=0).reshape(len(x), )

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, marker="x", s=20, c="r", label="Data")
    ax.plot(x[indices], predict[indices],
            ls="--",
            lw=2,
            c="b",
            label="Prediction")
    ax.legend(loc="upper left")
    plt.setp(ax,
             xlabel="Change in water level",
             ylabel="Water flowing of the dam",
             title="Data and prediction for lambda = {:.1f}".format(lamb))
    plt.show()


def compute_error(x, y, xval, yval, lamb):
    """Compute error in training and validation sets."""
    error_train = []
    error_val = []
    init_theta = np.zeros((x.shape[1] + 1, 1))

    for m in range(15):
        xtrain = x[:m + 1, :]
        ytrain = y[:m + 1]
        args = (xtrain, ytrain, lamb)
        theta = fmincg(cost_function, gradient, init_theta,
                       args=args,
                       maxiter=10)
        error_train.append(cost_function(theta, *args))
        args2 = (xval, yval, 0)
        error_val.append(cost_function(theta, *args2))

    return error_train, error_val


def display_error(error_train, error_val, lamb):
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(15), error_train, lw=2, c="b", label="Train")
    ax.plot(range(15), error_val, lw=2, c="g", label="Cross validation")
    ax.legend(loc="upper right")
    plt.setp(ax,
             xlabel="Number of training samples",
             ylabel="Error",
             title="Learning curve of polynomial "
                   "regression for lambda = {:.1f}".format(lamb))
    plt.show()


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
