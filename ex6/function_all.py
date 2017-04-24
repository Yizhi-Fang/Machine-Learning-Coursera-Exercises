"""All functions are here.

Apply Support Vector Machine to multiple data including email classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer


# Support Vector Machine.
def display_data(x, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[pos, 0], x[pos, 1], marker="+", s=20, c="k")
    ax.scatter(x[neg, 0], x[neg, 1], marker="o", s=20, c="y")
    plt.show()
    return plt.gca()


def display_db_linear(ax, x, svc_linear, C):
    """Display linear decision boundary at current axe.

    Args:
        ax: Current axis.
        x: Training sets.
        svc_linear: Linear regression with support vector machine.
        C: Coefficient in support vector machine,
           ~ 1/lamb where lamb is regulation parameter.
    """
    x1 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50)
    x2 = (-(svc_linear.intercept_ + svc_linear.coef_[0, 0]*x1)
          / svc_linear.coef_[0, 1])

    ax.plot(x1, x2, lw=2, c="g")
    plt.setp(ax,
             title="SVM decision boundary with C = {:.2f}".format(C))
    plt.show()


def display_db(ax, x, svc, C, gamma):
    """Display non-linear decision boundary at current axe

    Args:
        ax: Current axis.
        x: Training sets.
        svc_linear: Linear regression with support vector machine.
        C: Coefficient in support vector machine,
           ~ 1/lamb where lamb is regulation parameter.
        gamma: Coefficient in support vector machine,
               ~ 1/2sigma**2 where sigma is standard deviation.
    """
    x1 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 50)
    x2 = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 50)
    xx1, xx2 = np.meshgrid(x1, x2)
    P = np.zeros(xx1.shape)

    for i, v in enumerate(x2):
        for j, u in enumerate(x1):
            P[i, j] = svc.decision_function([u, v])

    ax.contour(xx1, xx2, P, linewidths=2, colors="g", levels=[0])
    plt.setp(ax,
             title="SVM decision boundary with C "
                   "= {:.2f}, gamma = {:.2f}".format(C, gamma))
    plt.show()


def data3_params(x, y, xval, yval):
    """Find out optimal parameter C and gamma in Support Vector Machine.

    Train samples and test accuracy on validation samples
    to optimize C and gamma for data3.
    """
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # To return the smallest possible gamma
    # since bigger gamma is more likely to overfit.
    gammas = [0.5 / c**2 for c in Cs[::-1]]
    accuracy = []
    for c in Cs:
        for gamma in gammas:
            svc = SVC(kernel="rbf", C=c, gamma=gamma).fit(x, y)
            predicts = svc.predict(xval).reshape(yval.shape)
            predicts[predicts > 0.5] = 1
            predicts[predicts < 0.5] = 0
            accuracy.append(np.mean(predicts == yval))

    index = np.argmax(accuracy)
    C_opt = Cs[index//8]
    gamma_opt = gammas[index%8]
    return C_opt, gamma_opt


# Spam email SVM.
def process_email(content):
    """Pre-process email.

    Process email to remove non-word characters.
    """
    # Lower-casing.
    content = content.lower()

    # Stripping html.
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    pattern = "<[^<>]+>"
    content = re.sub(pattern, " ", content)

    # Normalizing urls.
    # Look for strings starting with http:// or https://
    pattern = "(http|https)://[^\s]*"
    content = re.sub(pattern, "httpaddr", content)

    # Normalizing email address.
    # Look for strings with @ in the middle
    pattern = "[^\s]+@[^\s]+"
    content = re.sub(pattern, "emailaddr", content)

    # Normalizing numbers.
    # Look for one or more characters between 0-9
    pattern = "[0-9]+"
    content = re.sub(pattern, "number", content)

    # Normalizing $ sign.
    pattern = "[$]+"
    content = re.sub(pattern, "dollar", content)

    # Removal of non-word.
    pattern = "[^a-z]+"
    content = re.sub(pattern, " ", content)

    # Word stemming.
    stemmer = PorterStemmer()
    content = [stemmer.stem(word) for word in content.split(" ")]
    content = " ".join(content)

    return content


def match_vocab_list(content):
    """Code email with corresponding numbers.

    Convert email contents to corresponding numbers
    according to stored vocabulary list.

    Returns:
        word_indices: A list of indices indicating words in emails.
    """
    vocab_list = pd.read_table("./ex6/vocab.txt", sep="\t", header=None)
    word_indices = []
    for word in content.split(" "):
        for index in range(len(vocab_list)):
            if word == vocab_list.iloc[index][1]:
                word_indices.append(vocab_list.iloc[index][0])
    return word_indices


def email_features(word_indices):
    """Create feature vector for each email.

    Convert each email to a feature array where xi = 1
    if i-th word of vocabulary list appears or 0 if not present.
    """
    vocab_list = pd.read_table("./ex6/vocab.txt", sep="\t", header=None)
    x = np.zeros((len(vocab_list), 1))
    for n in word_indices:
        x[n-1] = 1
    return x
