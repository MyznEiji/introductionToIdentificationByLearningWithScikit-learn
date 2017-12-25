import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


data = load_breast_cancer()


data.feature_names
df = pd.DataFrame(data.data[:, 0:10],
                  columns=data.feature_names[0:10])
scatter_matrix(df, figsize=(10, 10))

df = pd.DataFrame(data.data[:, 6:8],
                  columns=data.feature_names[6:8])
scatter_matrix(df, figsize=(3, 3))

df = pd.DataFrame(data.data[:, [0, 2]],
                  columns=data.feature_names[[0, 2]])
scatter_matrix(df, figsize=(3, 3))

X = data.data[:, [0, 2]]
y = data.target
names = data.feature_names[[0, 2]]

X.shape, y.shape

plt.scatter(X[:, 0], X[:, 1])
plt.xlim(0, 180)
plt.ylim(20, 200)
plt.xlabel(names[0])
plt.ylabel(names[1])

pca = PCA()

pca.fit(X)

X_new = pca.transform(X)

plt.scatter(X_new[:, 0], X_new[:, 1])
plt.ylim(-60, 120)

pca.explained_variance_

pca.explained_variance_ / pca.explained_variance_.sum()

pca.explained_variance_ratio_

X = data.data[:, [6, 7]]
y = data.target
names = data.feature_names[[6, 7]]

plt.scatter(X[:, 0], X[:, 1])
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.xlabel(names[0])
plt.ylabel(names[1])

pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.xlim(-0.1, 0.4)
plt.ylim(-0.25, 0.25)

pca.explained_variance_

pca.explained_variance_ratio_  # 寄与率

# eigで検算
m = X.mean(axis=0)

Xp = (X - m)

C = Xp.transpose().dot(Xp)

w, _ = np.linalg.eig(C)

w

w / w.sum()  # 寄与率

# data全体でやる

X = data.data
y = data.target

ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


pca.fit(X_train)

plt.plot(pca.explained_variance_ratio_)

plt.plot(np.add.accumulate(pca.explained_variance_ratio_))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


clf = linear_model.LogisticRegression()

clf.fit(X_train_pca, y_train)
clf.score(X_test_pca, y_test)

clf.fit(X_train_pca[:, 0:1], y_train)
clf.score(X_test_pca[:, 0:1], y_test)

clf.fit(X_train_pca[:, 0:3], y_train)
clf.score(X_test_pca[:, 0:3], y_test)

scores = []
i_range = range(1, 31)

for i in i_range:

    clf.fit(X_train_pca[:, 0:i], y_train)
    scores.append(clf.score(X_test_pca[:, 0:i],
                            y_test))

scores = np.array(scores)

plt.plot(i_range, scores)
plt.ylim(0.7, 1)

clf.fit(X_train_pca[:, 0:2], y_train)
clf.score(X_test_pca[:, 0:2], y_test)


polf = PolynomialFeatures(degree=2)

polf.fit(X_train)

X_train_poly = polf.transform(X_train)
X_test_poly = polf.transform(X_test)

X_train.shape, X_train_poly.shape

X_test.shape, X_test_poly.shape

clf.fit(X_train_poly, y_train)
clf.score(X_test_poly, y_test)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

pca.fit(X_train_poly)

plt.plot(np.add.accumulate(pca.explained_variance_ratio_))

scores = []

n_features = X_train_poly.shape[1]
i_range = range(1, n_features, 10)

X_train_poly_pca = pca.transform(X_train_poly)
X_test_poly_pca = pca.transform(X_test_poly)

for i in i_range:

    clf.fit(X_train_poly_pca[:, 0:i], y_train)

    scores.append(clf.score(X_test_poly_pca[:, 0:i],
                            y_test))

scores = np.array(scores)

plt.plot(i_range, scores)
plt.title("max {0:.4f} at {1}".format(scores.max(),
                                      i_range[np.argmax(scores)]))

for d in [2, 3, 4]:
    print("d=", d)

    polf = PolynomialFeatures(degree=d)
    polf.fit(X_train)
    X_train_poly = polf.transform(X_train)
    X_test_poly = polf.transform(X_test)

    pca.fit(X_train_poly)
    X_train_poly_pca = pca.transform(X_train_poly)
    X_test_poly_pca = pca.transform(X_test_poly)

    scores = []
    n_features = min(500, X_train_poly.shape[1])
    i_range = range(1, n_features, 10)

    print("max dimension: ", X_train_poly.shape[1])

    print("i=", end="")
    for i in i_range:
        print(i, end=",")
        clf.fit(X_train_poly_pca[:, 0:i], y_train)
        scores.append(clf.score(X_test_poly_pca[:, 0:i], y_test))
    print("")

    scores = np.array(scores)
    plt.plot(i_range, scores, label="d={0}".format(d))

plt.legend()
