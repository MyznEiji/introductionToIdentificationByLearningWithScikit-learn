# モジュールの準備
import numpy as np

# ガンのデータを準備
from sklearn.datasets import load_breast_cancer
# 線形モデルを準備
from sklearn import linear_model


# dataにデータをセット
data = load_breast_cancer()

X = data.data  # .dataにデータが入っている

X.shape  # データ個数×特徴数

X[0]  # 最初のデータ（30次元ベクトル）


# 特徴の意味

y = data.target  # .targetにラベルが入っている

y.shape  # データ個数

y

y[0]

data.target_names  # ラベルの意味

print()  # データの詳細な記述



# 識別器を作成
clf = linear_model.LogisticRegression()

clf

n_samples = X.shape[0]  # データの個数
n_train = n_samples // 2  # 半分のデータを学習
n_test = n_samples - n_train  # テストデータ数

# 0,1,...,n_train-1：最初の半分
hoge = range(10)
hoge
train_index = range(0, n_train)
train_index
# n_train,n_train+1,...,n_samples-1：残りの半分
test_index = range(n_train, n_samples)
test_index
# 確認してみる
np.array(train_index), np.array(test_index)

X_train = X[train_index]  # 学習データ
X_test = X[test_index]  # テストデータ

y_train = y[train_index]  # 学習データのラベル
y_test = y[test_index]  # テストデータのラベル

# 上とまったく同じ
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

clf.fit(X_train, y_train)  # 識別器の学習
print(clf.score(X_train, y_train))  # 学習データの精度
print(clf.score(X_test, y_test))  # テストデータの精度

clf.predict(X_test)  # テストデータの識別


wrong = 0
for i, j in zip(clf.predict(X_test), y_test):
    if i == j:
        print(i, j)
    else:
        print(i, j, " Wrong!")  # 不正解
        wrong += 1

print("{0} / {1} = {2}".format(wrong,
                               n_test,
                               1 - wrong / n_test))
