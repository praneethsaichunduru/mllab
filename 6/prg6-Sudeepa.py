import os
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import fetch_20newsgroups
import sklearn.datasets as skd
import numpy as np
categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twen_trai=fetch_20newsgroups(data_home='home/master/Desktop/ml/6th/20news-bydate_py3.pkz',subset='train')
twen_test=fetch_20newsgroups(data_home='home/master/Desktop/ml/6th/20news-bydate_py3.pkz',subset='test')
print(len(twen_trai.data))
print(len(twen_test.data))
print(twen_trai.target_names)

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_train_tf=count_vect.fit_transform(twen_trai.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_trans=TfidfTransformer()
X_train_tfidf=tfidf_trans.fit_transform(X_train_tf)
X_train_tfidf.shape

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
mod=MultinomialNB()
mod.fit(X_train_tfidf,twen_trai.target)
X_test_tf=count_vect.transform(twen_test.data)
X_test_tfidf=tfidf_trans.transform(X_test_tf)
predicted=mod.predict(X_test_tfidf)
print(accuracy_score(twen_test.target,predicted))
print(classification_report(twen_test.target,predicted,target_names=twen_test.target_names))
print(confusion_matrix(twen_test.target,predicted))
