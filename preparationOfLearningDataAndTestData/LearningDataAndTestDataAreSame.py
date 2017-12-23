import numpy as np
import matplotlib.pylab as plt
from sklearn import neighbors

N = 500
X = np.random.uniform(low=0, high=1, size=[N, 2])
X
y = np.random.choice([0, 1], size=N)
y

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Paired, edgecolors='k')


clf = neighbors.KNeighborsClassifier(n_neighbors=1)

X_train = X
X_test = X

y_train = y
y_test = y

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

X_test2 = np.random.uniform(low=0, high=1, size=[N, 2])
y_test2 = np.random.choice([0, 1], size=N)

clf.score(X_test2, y_test2)
