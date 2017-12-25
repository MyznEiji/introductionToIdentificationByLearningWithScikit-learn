# numpyの準備
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer


# csvファイルの表示
with open('2D_example_dame.csv') as f:
    print(f.read())

# dataにデータをセット
data = np.loadtxt("2D_example_dame.csv", delimiter=",")

y = data[:, 0].astype(int)  # 1列目がラベル．整数に変換

y

X = data[:, 1:3]  # 2,3列目がデータ

X

X[:, 0]  # Xの1列目（csvファイルの2列目）
X[:, 1]  # Xの2列目（csvファイルの3列目）

# matplotlibの準備

plt.set_cmap(plt.cm.Paired)  # 色設定

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k')  # 2次元散布図でプロット

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k')  # 2次元散布図でプロット
plt.xlim(-10, 10)

plt.ylim(-10, 10)
~np.isnan(X)
~np.isnan(X[:, 0])

~np.isnan(X[:, 1])

~np.isnan(X[:, 1]) & ~np.isnan(X[:, 0])

# X1にはNaNがない
X1 = X[~np.isnan(X[:, 1]) & ~np.isnan(X[:, 0])]
y1 = y[~np.isnan(X[:, 1]) & ~np.isnan(X[:, 0])]

X1, X1.shape
    y1, y1.shape

(abs(X1[:, 0]) < 10), (abs(X1[:, 1]) < 10)  # 外れ値対策

X2 = X1[(abs(X1[:, 0]) < 10) & (abs(X1[:, 1]) < 10)]
y2 = y1[(abs(X1[:, 0]) < 10) & (abs(X1[:, 1]) < 10)]

X2, X2.shape

plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=50, edgecolors='k')  # 2次元散布図でプロット


missing_value_to_mean = Imputer()

missing_value_to_mean.fit(X)

X

X_new = missing_value_to_mean.transform(X)
X_new

plt.scatter(X_new[:, 0], X_new[:, 1], c=y, s=50, edgecolors='k')  # 2次元散布図でプロット

missing_value_to_median = Imputer(strategy='median')
missing_value_to_median.fit(X)
X_new2 = missing_value_to_median.transform(X)
X_new2

plt.scatter(X_new2[:, 0], X_new2[:, 1], c=y,
            s=50, edgecolors='k')  # 2次元散布図でプロット
