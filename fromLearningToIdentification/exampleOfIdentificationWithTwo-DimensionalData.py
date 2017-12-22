import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model

with open('2D_example.csv') as f:
    print(f.read())

data = np.loadtxt("2D_example.csv", delimiter=",")
type(data)
y = data[:, 0].astype(int)
y
X = data[:, 1:3]
X
X[:, 0]
X[:, 1]

plt.set_cmap(plt.cm.Paired)  # 色設定
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k")

# 境界線を引く関数の定義


def plotBoundary(X, clf, mesh=True, boundary=True, n_neighbors=1):

    # plot range
    x_min = min(X[:, 0])
    x_max = max(X[:, 0])
    y_min = min(X[:, 1])
    y_max = max(X[:, 1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]  # make a grid

    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])  # evaluate the value

    Z = Z.reshape(XX.shape)  # just reshape

    if mesh:
        # paint in 2 colors, if Z > 0 or not
        plt.pcolormesh(XX, YY, Z, zorder=-10)

    if boundary:
        plt.contour(XX, YY, Z,
                    colors='k', linestyles='-', levels=[0])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


# オブジェクト作成
clf = neighbors.KNeighborsClassifier(n_neighbors=1)

clf.fit(X, y)  # 学習

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

plotBoundary(X, clf)  # 境界線の描画


# オブジェクト作成
clf = linear_model.LogisticRegression()
clf.fit(X, y)  # 学習

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

plotBoundary(X, clf)  # 境界線の描画


# オブジェクト作成
clf = svm.SVC(kernel='linear')
clf.fit(X, y)  # 学習

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

plotBoundary(X, clf)  # 境界線の描画


# オブジェクト作成
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)  # 学習

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

plotBoundary(X, clf)  # 境界線の描画
