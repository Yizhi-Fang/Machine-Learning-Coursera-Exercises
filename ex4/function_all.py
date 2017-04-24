"""All functions are here.

Calculate cost function of neural network with forward
propagation and gradient with backpropagation.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import warnings


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))


class NeuralNetwork(object):
    """
    Attributes:
        input_layer: input player i.e. features not including x0
        hidden_layer: hidden layer assuming there's only one hidden layer
        num_labels: output layer/number of labels
    """
    def __init__(self, input_layer, hidden_layer, num_labels):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.num_labels = num_labels

    def compute_theta1(self, theta):
        """Restore Theta1 from theta."""
        theta1 = theta[:self.hidden_layer*(self.input_layer+1)]
        theta1 = theta1.reshape(self.hidden_layer, self.input_layer+1)
        return theta1

    def compute_theta2(self, theta):
        """Restore Theta2 from theta."""
        theta2 = theta[self.hidden_layer*(self.input_layer+1):]
        theta2 = theta2.reshape(self.num_labels, self.hidden_layer+1)
        return theta2

    def predict(self, x, theta):
        """Compute predictions by forward propagation."""
        theta1 = self.compute_theta1(theta)
        theta2 = self.compute_theta2(theta)
        if x.shape[1] == self.input_layer:
            x = np.insert(x, 0, 1, axis=1)  # m x input_layer+1
        elif x.shape[1] == self.input_layer + 1:
            pass
        else:
            raise IndexError("X dimension wrong!")
        a2 = sigmoid(x @ theta1.T)
        a2 = np.insert(a2, 0, 1, axis=1)  # m x hidden_layer+1
        h = sigmoid(a2@theta2.T)  # m x num_labels
        return h

    def reshape_y(self, y):
        """Reshape y.

        Args:
            y: Labels, originally m x 1 vector.

        Returns:
            New y in m x num_labels matrix.
        """
        m = len(y)
        temp = np.zeros((m, self.num_labels))
        for i in range(m):
            temp[i, y[i]-1] = 1
        return temp  # m x num_labels


def cost_function(theta, *args):
    """Calculate cost function with forward propagation."""
    input_layer, hidden_layer, num_labels, x, y, lamb = args
    network = NeuralNetwork(input_layer, hidden_layer, num_labels)

    theta1 = network.compute_theta1(theta)  # hidden_layer x input_layer+1
    theta2 = network.compute_theta2(theta)  # num_labels x hidden_layer+1
    if x.shape[1] == input_layer:
        x = np.insert(x, 0, 1, axis=1)  # m x input_layer+1
    elif x.shape[1] == input_layer + 1:
        pass
    else:
        raise IndexError("X dimension wrong!")
    h = network.predict(x, theta)  # m x num_labels
    y = network.reshape_y(y)  # m x num_labels
    m = len(y)

    # Add diagonal elements of (h@y.T) to calculate cost
    temp = -(np.log(h)@y.T + np.log(1-h)@(1-y).T) / m
    cost = (np.sum(temp.diagonal())
            + 0.5*lamb*np.sum(theta1[:, 1:]**2)/m
            + 0.5*lamb*np.sum(theta2[:, 1:]**2)/m)

    return cost


def gradient(theta, *args):
    """Calculate gradient with backpropagation.

    Args:
        theta: Theta to be optimized.

    Returns:
        All theta (theta1 and theta2) folded in a vector.
    """
    input_layer, hidden_layer, num_labels, x, y, lamb = args
    network = NeuralNetwork(input_layer, hidden_layer, num_labels)

    theta1 = network.compute_theta1(theta)  # hidden_layer x input_layer+1
    theta2 = network.compute_theta2(theta)  # num_labels x hidden_layer+1
    if x.shape[1] == input_layer:
        x = np.insert(x, 0, 1, axis=1)  # m x input_layer+1
    elif x.shape[1] == input_layer + 1:
        pass
    else:
        raise IndexError("X dimension wrong!")
    h = network.predict(x, theta) # m x num_labels
    y = network.reshape_y(y) # m x num_labels
    m = len(y)

    delta3 = h - y
    delta2 = (delta3 @ theta2[:, 1:]
              * sigmoid_gradient(x @ theta1.T))  # m x hidden_Layer

    a2 = sigmoid(x@theta1.T)
    a2 = np.insert(a2, 0, 1, axis=1)  # m x hidden_layer+1

    grad1 = delta2.T @ x  # hidden_layer x input_layer+1
    grad2 = delta3.T @ a2  # num_labels x hidden_layer+1

    theta1_grad = np.zeros(np.shape(theta1))
    theta2_grad = np.zeros(np.shape(theta2))

    theta1_grad[:, :1] = grad1[:, :1] / m
    theta1_grad[:, 1:] = grad1[:, 1:]/m + lamb*theta1[:, 1:]/m
    theta2_grad[:, :1] = grad2[:, :1] / m
    theta2_grad[:, 1:] = grad2[:, 1:]/m + lamb*theta2[:, 1:]/m

    return np.append(theta1_grad, theta2_grad).reshape(len(theta), 1)


def random_initialize(num):
    """Randomly initialization."""
    np.random.seed(45466)
    return np.array([np.random.randn() for i in range(num)]).reshape(num, 1)


def numerical_gradient(theta, *args):
    """Calculate numerical gradient."""
    e = 1e-4
    perturb = np.zeros((len(theta), 1))
    numgrad = np.zeros((len(theta), 1))
    for i in range(len(theta)):
        perturb[i] = e
        cost1 = cost_function(theta + perturb, *args)
        cost2 = cost_function(theta - perturb, *args)
        numgrad[i] = 0.5 * (cost1-cost2) / e
        perturb[i] = 0
    return numgrad


def check_nn_gradient():
    """Check Neural Network gradient.

    Create a small neural network and check if the backpropagation works.
    """
    input_layer = 3
    hidden_layer = 5
    num_labels = 3
    m = 5
    lamb = 1

    num = hidden_layer*(input_layer+1) + num_labels*(hidden_layer+1)
    theta = random_initialize(num)
    x = random_initialize(m * input_layer).reshape(m, input_layer)
    y = np.mod(range(m), num_labels) + 1

    args = (input_layer, hidden_layer, num_labels, x, y, lamb)
    numgrad = numerical_gradient(theta, *args)
    grad = gradient(theta, *args)
    diff = la.norm(numgrad-grad) / la.norm(numgrad+grad)

    print("If you backpropagation is corrent,\n"
          "then the relative difference should be small (less than 1e-9).\n"
          "Relative difference: {:.2e}".format(diff))


def display_data(x):
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
    ax.imshow(display_array.T, extent=[0, 10, 0, 10], cmap=cm.Greys)
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
