import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score


data = load_breast_cancer()

X = data.data
y = data.target

clf = linear_model.LogisticRegression()

loocv = LeaveOneOut()

train_index, test_index = next(loocv.split(X, y))  # １つだけ

y.size, train_index.size, test_index.size  # サイズを見てみる

scores = cross_val_score(clf,
                         X, y,
                         cv=loocv)  # LeaveOneOut

scores.mean() * 100, scores.std() * 100, scores.size

loocv = LeavePOut(2)
# scores = cross_val_score(clf, X, y, cv=loocv) # LeavePOut 終わらない！ n_C_2オーダー
# scores.mean(), scores.std(), scores.size

group = np.array(list(range(50)) * 12)
group = np.sort(group[:y.size])
group.size

group

loocv = LeaveOneGroupOut()

for train_index, test_index in loocv.split(X, y, group):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


scores = cross_val_score(clf,
                         X, y,
                         groups=group,
                         cv=loocv)  # LeaveOneGroupOut

scores.mean(), scores.std(), scores.size
