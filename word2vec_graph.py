#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import word2vec
# load the word2vec model
model = word2vec.load('corpusWord2Vec.bin')
rawWordVec=model.vectors

# reduce the dimension of word vector
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

# show some word(center word) and it's similar words
index1,metrics1 = model.cosine(u'识别')
index2,metrics2 = model.cosine(u'显示卡')
index3,metrics3 = model.cosine(u'关机')

# add the index of center word
index01=np.where(model.vocab==u'识别')
index02=np.where(model.vocab==u'显示卡')
index03=np.where(model.vocab==u'关机')

index1=np.append(index1,index01)
index2=np.append(index2,index03)
index3=np.append(index3,index03)

# plot the result
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
fig = plt.figure()
ax = fig.add_subplot(111)

for i in index1:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='r')

for i in index2:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='b')

for i in index3:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='g')


ax.axis([-1,1,-0.5,1])
plt.show()