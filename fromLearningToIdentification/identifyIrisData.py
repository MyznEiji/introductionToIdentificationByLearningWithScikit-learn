# numpyの準備
import numpy as np
# あやめのデータを準備
from sklearn.datasets import load_iris
# 線形モデルを準備
from sklearn import linear_model
# ランダムにシャッフルして，学習・テストに分割するモジュール
from sklearn.model_selection import ShuffleSplit
# dataにデータをセット
data = load_iris()

X = data.data  # .dataにデータが入っている
dir(data)
X.shape  # データ個数×特徴数
X[0]  # 最初のデータ（4次元ベクトル）
data.feature_names  # 特徴の意味
y = data.target  # .targetにラベルが入っている
y.shape  # データ個数
y
y[0]  # 最初のデータのラベル
data.target_names  # ラベルの意味
print(data.DESCR)  # データの詳細な記述

# 識別器を作成
clf = linear_model.LogisticRegression()
clf
n_samples = X.shape[0]  # データの個数
n_train = n_samples // 2  # 半分のデータを学習
n_test = n_samples - n_train  # テストデータ数
# 0,1,...,n_train-1：最初の半分を学習データのインデックスに
train_index = range(0, n_train)

# n_train,n_train+1,...,n_samples-1：残りの半分をテストデータのインデックスに
test_index = range(n_train, n_samples)
np.array(train_index), np.array(test_index)  # 確認してみる

X_train, X_test = X[train_index], X[test_index]  # 学習データ，テストデータ
y_train, y_test = y[train_index], y[test_index]  # 学習データのラベル，テストデータのラベル

clf.fit(X_train, y_train)  # 識別器の学習
print(clf.score(X_train, y_train))  # 学習データの精度

print(clf.score(X_test, y_test))  # テストデータの精度

clf.predict(X_test), y_test  # テストデータの識別

wrong = 0
for i, j in zip(clf.predict(X_test), y_test):
    if i == j:
        print(i, j)
    else:
        print(i, j, " Wrong!")
        wrong += 1

print("{0} / {1} = {2}".format(wrong,
                               n_test,
                               1 - wrong / n_test))


y_train, y_test  # 学習ラベルとテストラベルを確認してみる


ss = ShuffleSplit(n_splits=1,      # 分割を1個生成
                  train_size=0.5,  # 学習は半分
                  test_size=0.5,  # テストも半分
                  random_state=0)  # 乱数種（再現用）

# 学習データとテストデータのインデックスを作成
train_index, test_index = next(ss.split(X))

list(train_index), list(test_index)  # 確認してみる

X_train, X_test = X[train_index], X[test_index]  # 学習データ，テストデータ
y_train, y_test = y[train_index], y[test_index]  # 学習データのラベル，テストデータのラベル

y_train, y_test  # 学習ラベルとテストラベルを確認してみる

clf.fit(X_train, y_train)  # 識別器の学習

print(clf.score(X_train, y_train))  # 学習データの精度

print(clf.score(X_test, y_test))  # テストデータの精度

clf.predict(X_test), y_test  # テストデータの識別


wrong = 0
for i, j in zip(clf.predict(X_test), y_test):
    if i == j:
        print(i, j)
    else:
        print(i, j, " Wrong!")
        wrong += 1

print("{0} / {1} = {2}".format(wrong,
                               n_test,
                               1 - wrong / n_test))
