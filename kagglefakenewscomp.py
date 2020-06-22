# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:24:46 2020

@author: Jacob
"""
####Import dependencies
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from pandas import DataFrame








###Import Data
dat=pd.read_csv(r'C:\Users\Jacob\Documents\PythonScripts\20200617\\train.csv')


####Create training and test data sets
x_train,x_test,y_train,y_test=train_test_split(dat['text'].apply(lambda x: np.str_(x)), dat['label'], test_size=0.0001, random_state=7)

##Create TF-IDF Vectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))

##Use TF-IDF Vectorizer to fit and transform training set, and transform test set 
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

####Create passive aggressive classifier for tfidf transformed data
pac=PassiveAggressiveClassifier(max_iter=10000)
pac.fit(tfidf_train,y_train)

####Use tfidf vectorizer to predict test set with PAC
pac_y_pred=pac.predict(tfidf_test)
pac_score=accuracy_score(y_test,pac_y_pred)
print(f'Passive Aggressive Classifier TF-IDF Accuracy: {round(pac_score*100,2)}%')

###Apply TF-IDF vectorizer, passive aggressive classifier model to test data
tes=pd.read_csv(r'C:\Users\Jacob\Documents\PythonScripts\20200617\\test.csv')

testrain= tes['text'].apply(lambda x: np.str_(x))

tfidf_ctes=tfidf_vectorizer.transform(testrain)

tfidfpac_y_tes=pac.predict(tfidf_ctes)

###Make dataframe for submission .csv
tesid = tes['id']

submission = pd.DataFrame(tesid, columns=["id"])
submissionpred = pd.DataFrame(tfidfpac_y_tes, columns=["label"])
submission = submission.join(submissionpred)

submission.to_csv(r'C:\Users\Jacob\Documents\PythonScripts\20200621\\20200622_JBW_FakeNews3.csv', index=False)













