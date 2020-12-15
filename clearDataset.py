# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:32:37 2020

@author: Zain
"""

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
    dataset = pd.read_csv('Dataset/Reviews.csv',nrows=200000)
    dataset = dataset.iloc[:,[1,2,3,9]]
    dataset['TextWithOutStopWords'] = dataset['Text'].str.lower()
    #Tokenization
    dataset['TextWithOutStopWords'] = dataset.apply(identify_tokens,axis=1)
    #Stemming
    #dataset['TextWithOutStopWords'] = dataset.apply(stem_list, axis=1)
    #Lemmenization
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lemmatize_text)
    #Removing Stop words
    dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    #Removing Non-English words
    #dataset['TextWithOutStopWords'] = dataset['TextWithOutStopWords'].apply(englishDic)
    return dataset





