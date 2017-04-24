#!usr/bin/env python3

"""Logistic regression with Support Vector Machine."""

import scipy.io as sio
from function_all import *
from sklearn.svm import SVC
import warnings


warnings.filterwarnings("ignore")

# Linear kernel.
data = sio.loadmat("./ex6/ex6data1.mat")
x = data["X"]
y = data["y"]

C = 1
svc_linear = SVC(kernel="linear", C=C).fit(x, y)
ax = display_data(x, y)
display_db_linear(ax, x, svc_linear, C)

# Gaussian kernel.
data = sio.loadmat("./ex6/ex6data2.mat")
x = data["X"]
y = data["y"]

C = 1
gamma = 50
svc = SVC(kernel="rbf", C=C, gamma=gamma).fit(x, y)
ax = display_data(x, y)
display_db(ax, x, svc, C, gamma)

# Gaussian kernel and training params.
data = sio.loadmat("./ex6/ex6data3.mat")
x = data["X"]
y = data["y"]
xval = data["Xval"]
yval = data["yval"]

C, gamma = data3_params(x, y, xval, yval)
svc = SVC(kernel="rbf", C=C, gamma=gamma).fit(x, y)
ax = display_data(x, y)
display_db(ax, x, svc, C, gamma)

# Spam classification.
# Process email sample 1.
with open("./ex6/emailSample1.txt", "r") as f:
    content = f.read()

content = process_email(content)
word_indices = match_vocab_list(content)
x1 = email_features(word_indices)

# Load preprocessed training and testing samples.
train = sio.loadmat("./ex6/spamTrain.mat")
test = sio.loadmat("./ex6/spamTest.mat")
x = train["X"]
y = train["y"]
xtest = test["Xtest"]
ytest = test["ytest"]

C = 0.1
svc_linear = SVC(kernel="linear", C=C).fit(x, y)
# Training accuracy.
predicts = svc_linear.predict(x).reshape(y.shape)
predicts[predicts > 0.5] = 1
predicts[predicts < 0.5] = 0
acc_train = np.mean(predicts == y)
# Testing accuracy.
predicts = svc_linear.predict(xtest).reshape(ytest.shape)
predicts[predicts > 0.5] = 1
predicts[predicts < 0.5] = 0
acc_test = np.mean(predicts == ytest)
print("The spam classification has "
      "training accuracy is {:.1f}% and "
      "testing accuracy is {:.1f}%".format(100 * acc_train, 100 * acc_test))
