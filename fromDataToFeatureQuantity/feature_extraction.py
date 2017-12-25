import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np


# データのダウンロード
urllib.request.urlretrieve(
    "http://www.gutenberg.org/files/11/11-0.txt", "allice.txt")


with open('allice.txt', 'r', encoding='UTF-8') as f:
    print(f.read()[710:1400])

txt_vec = CountVectorizer(input='filename')

txt_vec.fit(['allice.txt'])

txt_vec.get_feature_names()[100:120]

len(txt_vec.get_feature_names())

allice_vec = txt_vec.transform(['allice.txt'])

allice_vec

allice_vec.shape

allice_vec = allice_vec.toarray()

allice_vec[0, 100:120]

for word, count in zip(txt_vec.get_feature_names()[100:120],
                       allice_vec[0, 100:120]):
    print(word, count)


china = load_sample_image('china.jpg')

plt.imshow(china)

china.shape

histR = plt.hist(china[:, :, 0].ravel(), bins=10)
plt.show()
histG = plt.hist(china[:, :, 1].ravel(), bins=10)
plt.show()
histB = plt.hist(china[:, :, 2].ravel(), bins=10)
plt.show()


histRGBcat = np.hstack((histR[0], histG[0], histB[0]))

plt.bar(range(len(histRGBcat)), histRGBcat)

histRGBcat_l1 = histRGBcat / (china.shape[0] * china.shape[1])
plt.bar(range(len(histRGBcat_l1)), histRGBcat_l1)
