import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data = load_breast_cancer()

X = data.data
y = data.target

clf = linear_model.LogisticRegression()


ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


scores = cross_val_score(clf,
                         X_train, y_train,
                         cv=10)  # StratifiedKFold


scores.mean()

C_range_exp = np.linspace(start=-15, stop=20, num=36)
C_range = 10 ** C_range_exp
print(C_range)

all_scores_mean = []
all_scores_std = []

for C in C_range:
    clf.C = C
    scores = cross_val_score(clf,
                             X_train, y_train,
                             cv=10)  # StratifiedKFold

    all_scores_mean.append(scores.mean())
    all_scores_std.append(scores.std())

all_scores_mean = np.array(all_scores_mean)
all_scores_std = np.array(all_scores_std)


plt.plot(C_range_exp, all_scores_mean)

plt.errorbar(C_range_exp,
             all_scores_mean,
             yerr=all_scores_std)

plt.ylim(0, 1)
plt.ylabel('accurary')
plt.xlabel('$\log_{10}$(C)')
plt.title('Accuracy for different values of C')

all_scores_mean.max()

max_index = np.argmax(all_scores_mean)

C_range_exp[max_index]

clf.C = 10 ** C_range_exp[max_index]

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
