import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedShuffleSplit
data = load_breast_cancer()

X = data.data
y = data.target


clf = linear_model.LogisticRegression()


ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,
                  test_size=0.5)

train_index, test_index = next(ss.split(X, y))
clf.fit(X[train_index], y[train_index])

clf.score(X[test_index], y[test_index])


mnist = fetch_mldata('MNIST original')

mnist.COL_NAMES

mnist.DESCR

mnist.data.shape

mnist.target

# MNISTの場合，60000が学習，10000がテスト，と決まっている
# http://yann.lecun.com/exdb/mnist/

X_train = mnist.data[:60000]
X_test = mnist.data[60000:70000]

y_train = mnist.target[:60000]
y_test = mnist.target[60000:70000]

clf

# clf.fit(X_train, y_train) # たぶんデータが多すぎて1時間たっても終わらない


data = load_breast_cancer()

X = data.data
y = data.target

clf = linear_model.LogisticRegression()

ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,
                  test_size=0.5)

# trainとtestを分割：hold-out
train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


clf.fit(X_train, y_train)
clf.score(X_test, y_test)

ss = ShuffleSplit(n_splits=10,
                  train_size=0.5,
                  test_size=0.5)

for train_index, test_index in ss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

ss = ShuffleSplit(n_splits=1,
                  train_size=0.95,
                  test_size=0.05,
                  random_state=3)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

print(np.unique(y,       return_counts=True))
print(np.unique(y,       return_counts=True)[1] / y.size)
print(np.unique(y_train, return_counts=True)[1] / y_train.size)
print(np.unique(y_test,  return_counts=True)[1] / y_test.size)


ss = StratifiedShuffleSplit(n_splits=1,
                            train_size=0.95,
                            test_size=0.05,
                            random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

print(np.unique(y,       return_counts=True))
print(np.unique(y,       return_counts=True)[1] / y.size)
print(np.unique(y_train, return_counts=True)[1] / y_train.size)
print(np.unique(y_test,  return_counts=True)[1] / y_test.size)
