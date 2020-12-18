# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:36:21 2020

@author: Zain
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random as rd
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
nltk.download('punkt')
stemming = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
global dataset
words = set(nltk.corpus.words.words())

#Clearing Corpus from Stop words, Lemmatizing, Stemming
def identify_tokens(row):
    review = row['TextWithOutStopWords']
    tokens = nltk.word_tokenize(review)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def stem_list(row):
    my_list = row['TextWithOutStopWords']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

def lemmatize_text(row):
    return [lemmatizer.lemmatize(w) for w in row]

def englishDic(sentence):
    return " ".join(w for w in nltk.wordpunct_tokenize(sentence) if w.lower() in words or not w.isalpha())

def clear(dataset):
    dataset = pd.read_csv('Dataset/Reviews.csv',nrows=1000)
    dataset = dataset.iloc[:,[1,2,3,9]]
    dataset['TextWithOutStopWords'] = dataset['Text'].str.lower()
    #Tokenization
    dataset['TextWithOutStopWords'] = dataset.apply(identify_tokens,axis=1)
    #Stemming
    dataset['TextWithOutStopWords'] = dataset.apply(stem_list, axis=1)
    #Lemmenization
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lemmatize_text)
    #Removing Stop words
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    #Removing Non-English words
    #dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(englishDic)
    return dataset


# Random centroid Selection K-means
def getDistance(examples,Centroids,K,dataset):
    Dist=np.array([]).reshape(examples,0)
    for k in range(K):
        tempDist=np.sum((dataset-Centroids[:,k])**2,axis=1)
        Dist=np.c_[Dist,tempDist]
    C=np.argmin(Dist,axis=1)+1
    return C,Dist

def randomCent(K,features,dataset,examples):
    randomPoint = np.array([]).reshape(features,0) 
    for i in range(K):
        randomPoint=np.c_[randomPoint,rd.choice(dataset)]
    return randomPoint

def evaluationScore(K,Centroids,final):
    score = 0
    for k in range(K):
        score += np.sum((final[k+1]-Centroids[:,k])**2)
    return score

def kmeanss(dataset,K):
    examples = len(dataset)
    features = len(dataset.T)
    randomPoint = randomCent(K,features,dataset,examples)
    final={}
    for i in range(100):
         
          C,Dist = getDistance(examples,randomPoint,K,dataset)
          
          for k in range(K):
              tempDist=np.sum((dataset-randomPoint[:,k])**2,axis=1)
              Dist=np.c_[Dist,tempDist]
          C = np.argmin(Dist,axis=1)+1
         
          Y={}
          for f in range(K):
              Y[f+1]=np.array([]).reshape(features,0)
          for g in range(examples):
              Y[C[g]]=np.c_[Y[C[g]],dataset[g]]
         
          for h in range(K):
              Y[h+1]=Y[h+1].T
        
          for l in range(K):
              with np.errstate(divide='ignore', invalid='ignore'):
                  randomPoint[:,l]=np.mean(Y[l+1],axis=0)
          final=Y
          
    score = evaluationScore(K,randomPoint,final)

    return final,score


#Word-cloud finding similar
def findSimilerWords(word,vocabulary,count):
    cosineScore = cosine_similarity(count)
    word_vect = cosineScore[vocabulary.index(word)]
    print("Word: ",word,"\n")
    topScoreWords = word_vect.argsort()
    for i in range(len(topScoreWords)):
        print("Word",vocabulary[topScoreWords[i]] ,"is same as ",word,"\n")
        if i == 20: break