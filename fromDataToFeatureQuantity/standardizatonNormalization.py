import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
data = load_breast_cancer()

data.feature_names

print(data.DESCR)
data.feature_names[4], data.feature_names[3]

plt.scatter(data.data[:, 3], data.data[:, 4])
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])

plt.scatter(data.data[:, 3], data.data[:, 4])
plt.xlim(0, 3000)
plt.ylim(0, 3000)
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])
plt.show()

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

"""
Standardization
"""

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)

X_train_scale.mean(axis=0), X_train_scale.std(axis=0)

X_train.mean(axis=0), X_train.std(axis=0)

X_test_scale = scaler.transform(X_test)

X_test_scale.mean(axis=0), X_test_scale.std(axis=0)

plt.scatter(X_train_scale[:, 3],
            X_train_scale[:, 4], c='blue',
            label="train")
plt.scatter(X_test_scale[:, 3],
            X_test_scale[:, 4],  c='red',
            label="test")
plt.xlabel(data.feature_names[3] + " (standardised)")
plt.ylabel(data.feature_names[4] + " (standardised)")
plt.legend(loc="best")
plt.show()

clf = linear_model.LogisticRegression()


clf.fit(X_train_scale, y_train)
print(clf.score(X_test_scale, y_test))

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

"""
range scaling
"""

mmscaler = MinMaxScaler([-1, 1])

mmscaler.fit(X_train)
X_train_mms = mmscaler.transform(X_train)
X_test_mms = mmscaler.transform(X_test)

X_train_mms.max(axis=0), X_train_mms.min(axis=0)

X_train.max(axis=0), X_train.min(axis=0)

plt.scatter(X_train_mms[:, 3],
            X_train_mms[:, 4], c='blue',
            label="train")
plt.scatter(X_test_mms[:, 3],
            X_test_mms[:, 4],  c='red',
            label="test")
plt.xlabel(data.feature_names[3] + " (scaled)")
plt.ylabel(data.feature_names[4] + " (scaled)")
plt.legend(loc="best")
plt.show()


clf.fit(X_train_mms, y_train)
print(clf.score(X_test_mms, y_test))

"""
Normalization
"""


normalizer = Normalizer()

# normalizer.fit(X_train)

X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)

np.linalg.norm(X_train, axis=1)[:20]

np.linalg.norm(X_train_norm, axis=1)[:20]

clf.fit(X_train_norm, y_train)
print(clf.score(X_test_norm, y_test))

plt.scatter(X_train_norm[:, 3],
            X_train_norm[:, 4], c='blue',
            label="train")
plt.scatter(X_test_norm[:, 3],
            X_test_norm[:, 4],  c='red',
            label="test")
plt.xlabel(data.feature_names[3] + " (normalized)")
plt.ylabel(data.feature_names[4] + " (normalized)")
plt.legend(loc="best")
plt.show()
for norm in ['l2', 'l1', 'max']:
    normalizer = Normalizer(norm=norm)
    normalizer.fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    clf.fit(X_train_norm, y_train)
    print(norm, clf.score(X_test_norm, y_test))

"""
Normalization
"""

plt.scatter(data.data[:, 6], data.data[:, 7])

X = data.data[:, [6, 7]]
y = data.target
plt.scatter(X[:, 0], X[:, 1])
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.show()


pca = PCA()
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.xlim(-0.1, 0.4)
plt.ylim(-0.25, 0.25)
plt.show()
X_new.mean(axis=0), X_new.std(axis=0)

pca = PCA(whiten=True)
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.xlim(-4, 10)
plt.ylim(-4, 10)
plt.show()
X_new.mean(axis=0), X_new.std(axis=0)

"""
ZCA Whitening
"""

X = np.random.uniform(low=-1, high=1, size=(1000, 2)) * (2, 1)
y = 2 * X[:, 0] + X[:, 1]
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
angle = np.pi / 4

R = np.array([[np.sin(angle), -np.cos(angle)],
              [np.cos(angle), np.sin(angle)]])
R

X_rot = X.dot(R)
plt.scatter(X_rot[:, 0], X_rot[:, 1], c=y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

X = X_rot

pca = PCA(whiten=False)
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("PCA")
plt.show()

pca = PCA(whiten=True)
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("PCA Whitening")
plt.show()

X_new2 = X_new.dot(pca.components_)
plt.scatter(X_new2[:, 0], X_new2[:, 1], c=y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.title("ZCA Whitening")
plt.show()
pca.components_
