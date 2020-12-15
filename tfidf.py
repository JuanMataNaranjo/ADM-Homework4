# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:22:53 2020

@author: Zain
"""

from clearDataset import clear
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = clear("Dataset/Reviews.csv")

corpus = dataset['TextWithOutStopWords']
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names()


pipe = Pipeline([('count', TfidfVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
count = pipe['count'].transform(corpus).toarray()

tfid = pipe['tfid'].idf_

count = pd.DataFrame(count,columns=vocabulary)
count.columns = vocabulary

svd = TruncatedSVD(n_components=600,n_iter=7)
svd.fit(count)

cumSum = np.cumsum(svd.explained_variance_ratio_)
num = svd.explained_variance_ratio_
print(svd.explained_variance_ratio_.sum())

df = svd.fit(count).transform(count)
plt.plot(cumSum)
#sb.heatmap(pd.DataFrame(svd.components_,columns=vocabulary))


cluster = [1,2,3,4,5,6,7,8,9,10,11,12]
clusterScore = []
for i in cluster:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    clusterScore.append(kmeans.inertia_)
    
plt.plot(cluster,clusterScore)

print(svd.singular_values_)
