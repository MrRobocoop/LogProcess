##!/usr/bin/env python
## coding=utf-8
import jieba
import xlrd
import word2vec

fileSegWordDonePath ='corpusSegDone.txt'

data = xlrd.open_workbook('log_infos.xls')  # 打开xls文件
table1 = data.sheets()[0] # 打开第一张表

num_cols = table1.col_values(8)

err_info = []

errs = num_cols[1:]

for curr_index in range(len(errs)):
    curr_err = errs[curr_index].encode("utf-8")
    if curr_err != '' and curr_err != '故障描述' :
        err_info.append(curr_err)


fileTrainSeg = []
for i in range(len(err_info)):
    fileTrainSeg.append([' '.join(list(jieba.cut(err_info[i], cut_all=False)))])
    if i % 1000 == 0 :
        print i

#print fileTrainSeg

with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n')

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

from sklearn.feature_extraction.text import TfidfVectorizer
#import codecs
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# cp = codecs.open('corpusSegDone.txt','r','utf-8').read()
# List_row = cp.readlines()

list_source = []

for i in range(len(fileTrainSeg)):
    column_list = fileTrainSeg[i][0].decode('utf-8')  # 每一行split后是一个列表
    list_source.append(column_list)  # 加入list_source

#print list_source
with open("list_source.txt",'wb') as fW:
    for i in range(len(list_source)):
        fW.write(list_source[i].encode('utf-8'))
        fW.write('\n')
'''
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 

tfidf = TfidfVectorizer()
re = tfidf.fit_transform(corpus) 
print re
'''

vectorizer = TfidfVectorizer(stop_words=stpwrdlst, min_df=1)

list_tfidf = []

list_tfidf = vectorizer.fit_transform(list_source)

from sklearn.decomposition import SparsePCA

pca=SparsePCA(n_components=50)

final_tfidf = pca.fit_transform(list_tfidf.toarray())

print final_tfidf


'''
for j in range(len(list_source)):
    column_tfidf = vectorizer.fit_transform(list_source[j])
    list_tfidf.append(column_tfidf)
'''
'''
for i in range(len(list_source)):
    print list_source[i], list_tfidf[i]
'''
#array_tfidf = []
'''
with open("list_tfidf.txt",'wb') as fW:
    for i in range(len(final_tfidf)):
        fW.write(list_source[i].encode('utf-8'))
        fW.write(str(final_tfidf[i]).encode('utf-8'))
        fW.write('\n')
'''
"""
wordlist = vectorizer.get_feature_names()#获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = list_tfidf.toarray()
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print "-------第",i,"段文本的词语tf-idf权重------"
    for j in range(len(wordlist)):
        print wordlist[j],weightlist[i][j]
"""

from sklearn.cluster import KMeans

clf = KMeans(n_clusters=20)
s = clf.fit(final_tfidf)
print s

# 20个中心点
print(clf.cluster_centers_)

# 每个样本所属的簇
print(clf.labels_)
i = 1
while i <= len(clf.labels_):
    print i, clf.labels_[i - 1]
    i = i + 1

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
print(clf.inertia_)

for i in range(len(list_source)):

    print list_source[i], clf.labels_[i]

j=0
with open("result.txt",'wb') as fW:
    while j <= 19:
        fW.write('label')
        fW.write(str(j).encode('utf-8'))
        fW.write('\n')
        for i in range(len(clf.labels_)):
            if clf.labels_[i] == j:
                fW.write(list_source[i].encode('utf-8'))
                fW.write(str(clf.labels_[i]).encode('utf-8'))
                fW.write('\n')
        j = j + 1