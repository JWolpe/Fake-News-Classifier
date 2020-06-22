# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:13:27 2020

@author: Jacob
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
import sklearn.metrics as metrics
from sklearn.linear_model import SGDClassifier
from pandas import DataFrame
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')








##############
##############Vectorize data using bag of words and tfidf#########


dat=pd.read_csv(r'C:\Users\Jacob\Documents\PythonScripts\20200617\\train.csv')

##Convert dataframe to floats
#dat['id'] = dat['id'].astype(float)
#dat['label'] = dat['label'].astype(float)

pd.set_option('display.max_columns', 7)
dat.head()
dat.shape

##label vector
labels=dat.label
labels.head()

####Create training and test data sets
x_train,x_test,y_train,y_test=train_test_split(dat['text'].apply(lambda x: np.str_(x)), labels, test_size=0.2, random_state=7)

##Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

##Fit and train count vectorizer
count_train = count_vectorizer.fit_transform(x_train)
count_test = count_vectorizer.transform(x_test)

##TFIDF Vectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

####Fit to tfidf data and transform test and training sets to normalized tfidf vector
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

####Hashing Vectorizer
hash_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)

###Fit hashing vectorizer to data and transform both test and training sets
hash_train = hash_vectorizer.fit_transform(x_train)
hash_test = hash_vectorizer.transform(x_test)


###Feature names
#print(tfidf_vectorizer.get_feature_names()[-10:])
#print(count_vectorizer.get_feature_names()[:10])


############################
############################
############################
#####Classifier Model Comparison using tfidf, count and hashing vectorizers####









####Create passive aggressive classifier for tfidf transformed data
pactf=PassiveAggressiveClassifier(max_iter=1000)
pactf.fit(tfidf_train,y_train)

####Use tfidf vectorizer to predict test set with PAC
pactf_y_pred=pactf.predict(tfidf_test)
pactf_score=accuracy_score(y_test,pactf_y_pred)
print(f'Passive Aggressive Classifier TF-IDF Accuracy: {round(pactf_score*100,2)}%')
pactf_cm = metrics.confusion_matrix(y_test, pactf_y_pred, labels=[1, 0])
plot_confusion_matrix(pactf_cm, classes=['FAKE', 'REAL'], title = "Passive Aggressive Classifier TF-IDF")
scores = []
scores.append(pactf_score)    

####Create passive aggressive classifier for count transformed data
pacbw=PassiveAggressiveClassifier(max_iter=1000)
pacbw.fit(count_train,y_train)

####Use count vectorizer to predict test set with passive aggressive classifier
pacbw_y_pred=pacbw.predict(count_test)
pacbw_score=accuracy_score(y_test,pacbw_y_pred)
print(f'Passive Aggressive Classifier Bag of Words Accuracy: {round(pacbw_score*100,2)}%')
pacbw_cm = metrics.confusion_matrix(y_test, pacbw_y_pred, labels=[1, 0])
plot_confusion_matrix(pacbw_cm, classes=['FAKE', 'REAL'], title = "Passive Aggressive Classifier Bag of Words")
scores.append(pacbw_score) 

####Create passive aggressive classifier for hashing transformed data
pachs=PassiveAggressiveClassifier(max_iter=1000)
pachs.fit(hash_train,y_train)

####Use hashing vectorizer to predict test set with passive aggressive classifier
pachs_y_pred=pachs.predict(hash_test)
pachs_score=accuracy_score(y_test,pachs_y_pred)
print(f'Passive Aggressive Classifier Hashing Accuracy: {round(pachs_score*100,2)}%')
pachs_cm = metrics.confusion_matrix(y_test, pachs_y_pred, labels=[1, 0])
plot_confusion_matrix(pachs_cm, classes=['FAKE', 'REAL'], title = "Passive Aggressive Classifier Hashing")
scores.append(pachs_score)


###Multinomial naive bayes classification for all three vectorizers
mnbtf = MultinomialNB() 
mnbbw = MultinomialNB() 
mnbhs = MultinomialNB()

###tfidf MNNB predictions
mnbtf.fit(tfidf_train, y_train)
mnbtfpred = mnbtf.predict(tfidf_test)
mnbtf_score = accuracy_score(y_test, mnbtfpred)
print(f'Multinomial Naive Bayes TF-IDF Accuracy: {round(mnbtf_score*100,2)}%')
mnbtf_cm = metrics.confusion_matrix(y_test, mnbtfpred, labels=[1, 0])
plot_confusion_matrix(mnbtf_cm, classes=['FAKE', 'REAL'], title = "Multinomial Naive Bayes TF-IDF")
scores.append(mnbtf_score)

###Bag of words MNNB Predictions
mnbbw.fit(count_train, y_train)
mnbbwpredc = mnbbw.predict(count_test)
mnbbw_score = metrics.accuracy_score(y_test, mnbbwpredc)
print(f'Multinomial Naive Bayes Bag of Words Accuracy: {round(mnbbw_score*100,2)}%')
mnbbwcm = metrics.confusion_matrix(y_test, mnbbwpredc, labels=[1,0])
plot_confusion_matrix(mnbbwcm, classes=['FAKE', 'REAL'], title = "Multinomial Naive Bayes Bag of Words")
scores.append(mnbbw_score)

###Hashing MNNB Predictions
mnbhs.fit(hash_train, y_train)
mnbhspred = mnbhs.predict(hash_test)
mnbhs_score = metrics.accuracy_score(y_test, mnbhspred)
print(f'Multinomial Naive Bayes Hashing Accuracy: {round(mnbhs_score*100,2)}%')
mnbhscm = metrics.confusion_matrix(y_test, mnbhspred, labels=[1,0])
plot_confusion_matrix(mnbhscm, classes=['FAKE', 'REAL'], title = "Multinomial Naive Bayes Hashing")
scores.append(mnbhs_score)

######################################################################################
#Complement Naive Bayes
cnbtf = ComplementNB() 
cnbbw = ComplementNB() 
cnbhs = ComplementNB()

###tfidf CNB predictions
cnbtf.fit(tfidf_train, y_train)
cnbtfpred = cnbtf.predict(tfidf_test)
cnbtf_score = accuracy_score(y_test, cnbtfpred)
print(f'Complement Naive Bayes TF-IDF Accuracy: {round(cnbtf_score*100,2)}%')
cnbtf_cm = metrics.confusion_matrix(y_test, cnbtfpred, labels=[1, 0])
plot_confusion_matrix(cnbtf_cm, classes=['FAKE', 'REAL'], title = "Complement Naive Bayes TF-IDF")
scores.append(cnbtf_score)

###Bag of words CNB Predictions
cnbbw.fit(count_train, y_train)
cnbbwpredc = cnbbw.predict(count_test)
cnbbw_score = metrics.accuracy_score(y_test, cnbbwpredc)
print(f'Complement Naive Bayes Bag of Words Accuracy: {round(cnbbw_score*100,2)}%')
cnbbwcm = metrics.confusion_matrix(y_test, cnbbwpredc, labels=[1,0])
plot_confusion_matrix(cnbbwcm, classes=['FAKE', 'REAL'], title = "Complement Naive Bayes Bag of Words")
scores.append(cnbbw_score)

###Hashing CNNB Predictions
cnbhs.fit(hash_train, y_train)
cnbhspred = cnbhs.predict(hash_test)
cnbhs_score = metrics.accuracy_score(y_test, cnbhspred)
print(f'Complement Naive Bayes Hashing Accuracy: {round(cnbhs_score*100,2)}%')
cnbhscm = metrics.confusion_matrix(y_test, cnbhspred, labels=[1,0])
plot_confusion_matrix(cnbhscm, classes=['FAKE', 'REAL'], title = "Complement Naive Bayes Hashing")
scores.append(cnbhs_score)

####Create stochastic gradient descent classifier for tfidf data
sgdtf=SGDClassifier(max_iter=1000)
sgdtf.fit(tfidf_train,y_train)

####Use tfidf model to predict test set with SGD
sgdtf_y_pred=sgdtf.predict(tfidf_test)
sgdtf_score=accuracy_score(y_test,sgdtf_y_pred)
print(f'Stochastic Gradient Descent TF-IDF Accuracy: {round(sgdtf_score*100,2)}%')
sgdtf_cm = metrics.confusion_matrix(y_test, sgdtf_y_pred, labels=[1, 0])
plot_confusion_matrix(sgdtf_cm, classes=['FAKE', 'REAL'], title = "Stochastic Gradient Descent TF-IDF")
scores.append(sgdtf_score)

####Create stochastic gradient descent classifier for BOW data
sgdbw=SGDClassifier(max_iter=1000)
sgdbw.fit(count_train,y_train)

####Use BW model to predict test set with SGD
sgdbw_y_pred=sgdbw.predict(count_test)
sgdbw_score=accuracy_score(y_test,sgdbw_y_pred)
print(f'Stochastic Gradient Descent Bag of Words Accuracy: {round(sgdbw_score*100,2)}%')
sgdbw_cm = metrics.confusion_matrix(y_test, sgdbw_y_pred, labels=[1, 0])
plot_confusion_matrix(sgdbw_cm, classes=['FAKE', 'REAL'], title = "Stochastic Gradient Descent Bag of Words")
scores.append(sgdbw_score)

####Create stochastic gradient descent classifier for Hashing data
sgdhs=SGDClassifier(max_iter=1000)
sgdhs.fit(hash_train,y_train)

####Use Hashing vectorized data to predict test set with SGD
sgdhs_y_pred=sgdhs.predict(hash_test)
sgdhs_score=accuracy_score(y_test,sgdhs_y_pred)
print(f'Stochastic Gradient Descent Hashing Accuracy: {round(sgdhs_score*100,2)}%')
sgdhs_cm = metrics.confusion_matrix(y_test, sgdhs_y_pred, labels=[1, 0])
plot_confusion_matrix(sgdhs_cm, classes=['FAKE', 'REAL'], title = "Stochastic Gradient Descent Hashing")
scores.append(sgdhs_score)
###

###Make comparison table
vecandclass = ["Passive Aggressive Classifier TF-IDF", "Passive Aggressive Classifier Bag of Words", "Passive Aggressive Classifier Hashing", "Multinomial Naive Bayes TF-IDF", "Multinomial Naive Bayes Bag of Words", "Multinomial Naive Bayes Hashing", "Complement Naive Bayes TF-IDF",  "Complement Naive Bayes Bag of Words", "Complement Naive Bayes Hashing",  "Stochastic Gradient Descent TF-IDF", "Stochastic Gradient Descent Bag of Words", "Stochastic Gradient Descent Hashing"]
comparedf = pd.DataFrame(vecandclass, columns = ["Analysis"])
scoress = [i * 100 for i in scores]
scoress = DataFrame (scoress,columns=['Accuracy'])
comparedf = comparedf.join(scoress)
print(comparedf)

###Plot Bar Graph of Comparison
comparedf.plot.bar(x='Analysis', y='Accuracy', rot=75, figsize=(26,12), legend=False, ylim=[75,100])

#



##############################################################################
#####Parameter tuning for Passive Aggressive Classifier TF-IDF
pactfpt=PassiveAggressiveClassifier(max_iter=1000)

##Step Size tuning

for C in np.arange(0,2.1,.1):
    pa_classifier = PassiveAggressiveClassifier(C=C)
    pa_classifier.fit(tfidf_train, y_train)
    predpt = pa_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, predpt)
    print("C: {:.2f} Score: {:.5f}".format(C, score))
####Step size has little impact


##Iteration tuning
scorept = []
iterpt = []
for max_iter in np.arange(1000,10200,200):
    pa_classifier = PassiveAggressiveClassifier(max_iter=max_iter)
    pa_classifier.fit(tfidf_train, y_train)
    predpt = pa_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, predpt)
    scorept.append(score)
    iterpt.append(max_iter)
    print("max_iter: {:.2f} Score: {:.5f}".format(max_iter, score))

##Plot Iterations vs. Accuracy
itercompare = pd.DataFrame(iterpt, columns = ["Iterations"])
scorept = [i * 100 for i in scorept]
scorept = DataFrame (scorept,columns=['Accuracy'])
itercompare = itercompare.join(scorept)

iterplots = plt.figure()
plt.scatter(itercompare["Iterations"],itercompare["Accuracy"])
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.title('Accuracy vs. Iterations', fontsize=20)
z = np.polyfit(itercompare["Iterations"],itercompare["Accuracy"], 1)
p = np.poly1d(z)
plt.plot(itercompare["Iterations"],p(itercompare["Iterations"]),"r--")

###Iteration number does not give better results above noise, nor does C

##tol(stop function) tuning
for tol in np.arange(0,1e-3,1e-4):
    pa_classifier = PassiveAggressiveClassifier(tol=tol)
    pa_classifier.fit(tfidf_train, y_train)
    predpt = pa_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, predpt)
    print("tol: {:.5f} Score: {:.5f}".format(tol, score))

####tol tuning does not alter accuracy above noise

###Extract feature names- words related to real or fake news
#feature_names = tfidf_vectorizer.get_feature_names()

##Real Features
#sorted(zip(pactfpt.coef_[0], feature_names), reverse=True)[:20]


##Fake features
#sorted(zip(pactfpt.coef_[0], feature_names))[:20]


########################################################################################################
##Use 2 ngrams
tfidf_vectorizergram=TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))

tfidf_traingram=tfidf_vectorizergram.fit_transform(x_train) 
tfidf_testgram=tfidf_vectorizergram.transform(x_test)


####Create passive aggressive classifier for tfidfgram transformed data
pactfgram=PassiveAggressiveClassifier(max_iter=1000)
pactfgram.fit(tfidf_traingram,y_train)

####Use tfidf vectorizer to predict test set with PAC
pactfgram_y_pred=pactfgram.predict(tfidf_testgram)
pactfgram_score=accuracy_score(y_test,pactfgram_y_pred)
print(f'Passive Aggressive Classifier TF-IDF (2ngrams) Accuracy: {round(pactfgram_score*100,2)}%')
finalvecclass = []
finalvecclass.append("Passive Aggressive Classifier TF-IDF")
finalscore = []
finalscore.append(pactfgram_score*100)

###max_df tuning
for max_df in np.arange(0.5,1,0.1):
    tfidfpt_vectorizergram=TfidfVectorizer(stop_words='english', max_df=max_df, ngram_range=(1, 2))
    tfidfpt_traingram=tfidfpt_vectorizergram.fit_transform(x_train) 
    tfidfpt_testgram=tfidfpt_vectorizergram.transform(x_test)
    pa_classifier = PassiveAggressiveClassifier()
    pa_classifier.fit(tfidfpt_traingram, y_train)
    predpt = pa_classifier.predict(tfidfpt_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("max_df: {:.2f} Score: {:.5f}".format(max_df, score*100))
###0.7 was best, long load time

###sublinear_tf test
tfidff_vectorizergram=TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2), sublinear_tf=True)

tfidff_traingram=tfidff_vectorizergram.fit_transform(x_train) 
tfidff_testgram=tfidff_vectorizergram.transform(x_test)

####Create passive aggressive classifier for tfidffgram transformed data
pactffgram=PassiveAggressiveClassifier(max_iter=1000)
pactffgram.fit(tfidff_traingram,y_train)

####Use tfidff vectorizer to predict test set with PAC
pactffgram_y_pred=pactffgram.predict(tfidff_testgram)
pactffgram_score=accuracy_score(y_test,pactffgram_y_pred)
print(f'Passive Aggressive Classifier TF-IDF (sublinear TF) Accuracy: {round(pactffgram_score*100,2)}%')


##Get 2ngram feature names
#feature_namesgram = tfidf_vectorizergram.get_feature_names()

##Real Features
#sorted(zip(pactfpt.coef_[0], feature_namesgram), reverse=True)[:20]

##Fake features
#sorted(zip(pactfpt.coef_[0], feature_namesgram))[:20]

####Loss funtion comparison with 2ngrams and passive aggressive tf-idf
lossfxn = ['hinge', 'squared_hinge']
for idx, loss in enumerate(lossfxn):
    pa_classifier = PassiveAggressiveClassifier(loss=loss)
    pa_classifier.fit(tfidf_traingram, y_train)
    predpt = pa_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print(loss, "loss Score: {:.5f}".format(score*100))
###After 3 runs, hinge is best


###############################################################################################################
##Use 3 ngrams
tfidf_vectorizergramm=TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 3))

tfidf_traingramm=tfidf_vectorizergramm.fit_transform(x_train) 
tfidf_testgramm=tfidf_vectorizergramm.transform(x_test)

####Create passive aggressive classifier for tfidfgram transformed data
pactfgramm=PassiveAggressiveClassifier(max_iter=1000)
pactfgramm.fit(tfidf_traingramm,y_train)

####Use tfidf vectorizer to predict test set with PAC
pactfgramm_y_pred=pactfgramm.predict(tfidf_testgramm)
pactfgramm_score=accuracy_score(y_test,pactfgramm_y_pred)
print(f'Passive Aggressive Classifier TF-IDF (3ngrams) Accuracy: {round(pactfgramm_score*100,2)}%')

##Get 3ngram feature names
# = tfidf_vectorizergramm.get_feature_names()

##Real Features
#sorted(zip(pactfpt.coef_[0], feature_namesgramm), reverse=True)[:20]

##Fake features
#sorted(zip(pactfpt.coef_[0], feature_namesgramm))[:20]

#############################################################################################################
######################Best results for tf-idf PAC is 2ngrams

###################################################################################################################
############Parameter tuning for SGD tf-idf

##Loss Function tuning
lossfxn = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive']
for idx, loss in enumerate(lossfxn):
    SGD_classifier = SGDClassifier(loss=loss)
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print(loss, "loss Score: {:.5f}".format(score*100))
###After 3 runs, epsilon_insensitive scored highest in all 3

###Penalty term tuning
penaltyfxn = ['l2', 'l1', 'elasticnet']
for idx, penalty in enumerate(penaltyfxn):
    SGD_classifier = SGDClassifier(penalty=penalty, loss='epsilon_insensitive')
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print(penalty, "penalty term Score: {:.5f}".format(score*100))
###After 3 runs, l2 (default) scored highest in all 3

###Alpha optimization
for alpha in np.arange(0.00001,0.001,0.0001):
    SGD_classifier = SGDClassifier(alpha=alpha, loss='epsilon_insensitive')
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("alpha: {:.5f} Score: {:.5f}".format(alpha, score*100))
####Default value is best

####max_iter optimization
for max_iter in np.arange(1000,10200,200):
    SGD_classifier = SGDClassifier(max_iter=max_iter, loss='epsilon_insensitive')
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("max_iter: {:.5f} Score: {:.5f}".format(max_iter, score*100))
###Increasing max_iter does not seem to increase accuracy

###epsilon tuning
for epsilon in np.arange(0.01,1.0,0.1):
    SGD_classifier = SGDClassifier(epsilon=epsilon, loss='epsilon_insensitive')
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("epsilon: {:.5f} Score: {:.5f}".format(epsilon, score*100))
##After 3 runs, 0.61 is best on avg.

###learning rate optimization
learningratefxn = ['constant', 'optimal', 'invscaling', 'adaptive']
for idx, learning_rate in enumerate(learningratefxn):
    SGD_classifier = SGDClassifier(learning_rate=learning_rate, loss='epsilon_insensitive', epsilon=0.61, eta0=0.15)
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print(learning_rate, "Learning Rate Score: {:.5f}".format(score*100))
###Adaptive performed nearly as well as optimal-> tune eta0 values for adaptive to see range

###Adaptive+eta0 tuning
for eta0 in np.arange(0.01,1.0,0.1):
    SGD_classifier = SGDClassifier(eta0=eta0, loss='epsilon_insensitive', epsilon=0.61, learning_rate='adaptive')
    SGD_classifier.fit(tfidf_traingram, y_train)
    predpt = SGD_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("eta0: {:.5f} Score: {:.5f}".format(eta0, score*100))
###After 3 runs, 0.51 was best on avg, at 97.2+

####Tuned SGD TFIDF Score (adaptive)
SGD_classifier = SGDClassifier(eta0=0.51, loss='epsilon_insensitive', epsilon=0.61, learning_rate='adaptive')
SGD_classifier.fit(tfidf_traingram, y_train)
predpt = SGD_classifier.predict(tfidf_testgram)
SGDa_score = metrics.accuracy_score(y_test, predpt)
print("Stochastic Gradient Descent TF-IDF Score: {:.5f}".format(SGDa_score*100))
finalvecclass.append("Stochastic Gradient Descent(Adaptive) TF-IDF")
finalscore.append(SGDa_score*100)

####Tuned SGD TFIDF Score (optimal)
SGDo_classifier = SGDClassifier(eta0=0.51, loss='epsilon_insensitive', epsilon=0.61)
SGDo_classifier.fit(tfidf_traingram, y_train)
predpt = SGDo_classifier.predict(tfidf_testgram)
SGDo_score = metrics.accuracy_score(y_test, predpt)
print("Stochastic Gradient Descent TF-IDF Score: {:.5f}".format(SGDo_score*100))
finalvecclass.append("Stochastic Gradient Descent(Optimal) TF-IDF")
finalscore.append(SGDo_score*100)



###################################################################################################################
############Parameter tuning for Passive Aggressive Classifier, Hashing Vectorizer
####Hashing Vectorizer 2ngrams
hashpt_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, ngram_range=(1,2))

###Fit hashing vectorizer to data and transform both test and training sets
hashpt_train = hashpt_vectorizer.fit_transform(x_train)
hashpt_test = hashpt_vectorizer.transform(x_test)

####Create passive aggressive classifier for hashing parameter tuning
pachspt=PassiveAggressiveClassifier(max_iter=1000)
pachspt.fit(hashpt_train,y_train)

###Test vs 1ngram result (96.08%)
pachspt_y_pred=pachspt.predict(hashpt_test)
pachspt_score=accuracy_score(y_test,pachspt_y_pred)
print(f'Passive Aggressive Classifier Hashing 2ngram Accuracy: {round(pachspt_score*100,2)}%')
####Accuracy from 2ngrams ~97.00% after several runs
finalvecclass.append("Passive Aggressive Classifier Hashing")
finalscore.append(pachspt_score*100)


##################################################################################################################
####Hashing Vectorizer 3ngrams
hashptt_vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, ngram_range=(1,3))

###Fit hashing vectorizer to data and transform both test and training sets
hashptt_train = hashptt_vectorizer.fit_transform(x_train)
hashptt_test = hashptt_vectorizer.transform(x_test)

####Create passive aggressive classifier for hashing parameter tuning
pachsptt=PassiveAggressiveClassifier(max_iter=1000)
pachsptt.fit(hashptt_train,y_train)

###Test vs 1ngram result (96.08%)
pachsptt_y_pred=pachsptt.predict(hashptt_test)
pachsptt_score=accuracy_score(y_test,pachsptt_y_pred)
print(f'Passive Aggressive Classifier Hashing 3ngram Accuracy: {round(pachsptt_score*100,2)}%')
#####3ngram accuracy is slightly lower (96.85, 96.92, 97.04) or on par with 2ngrams


##################################################################################################################
####Passive aggressive classifier tuning with Hashing vectorizer
lossfxn = ['hinge', 'squared_hinge']
for idx, loss in enumerate(lossfxn):
    pah_classifier = PassiveAggressiveClassifier(loss=loss)
    pah_classifier.fit(hashpt_train, y_train)
    predpt = pah_classifier.predict(hashpt_test)
    score = metrics.accuracy_score(y_test, predpt)
    print(loss, "loss Score: {:.5f}".format(score*100))
###Hinge was best in 2/3

####C parameter tuning
for C in np.arange(0,2.1,.1):
    pah_classifier = PassiveAggressiveClassifier(C=C)
    pah_classifier.fit(hashpt_train, y_train)
    predpt = pah_classifier.predict(hashpt_test)
    score = metrics.accuracy_score(y_test, predpt)
    print("C: {:.2f} Score: {:.5f}".format(C, score*100))
####Results appear to be random


####Iteration tuning
for max_iter in np.arange(1000,10200,200):
    pah_classifier = PassiveAggressiveClassifier(max_iter=max_iter)
    pah_classifier.fit(hashpt_train, y_train)
    predpt = pah_classifier.predict(hashpt_test)
    score = metrics.accuracy_score(y_test, predpt)
    print("max_iter: {:.2f} Score: {:.5f}".format(max_iter, score))
####Results not above noise

##########################################################################################################
####Use tfidf vectorizer to predict test set with Perceptron
####Create perceptron classifier for tfidf data
pcttf=Perceptron(max_iter=1000)
pcttf.fit(tfidf_traingram,y_train)


pcttf_y_pred=pcttf.predict(tfidf_testgram)
pcttf_score=accuracy_score(y_test,pcttf_y_pred)
print(f'Perceptron TF-IDF Accuracy: {round(pcttf_score*100,2)}%')


###Perceptron Penalty term tuning
penaltyfxn = ['l2', 'l1', 'elasticnet']
for idx, penalty in enumerate(penaltyfxn):
    pct_classifier = Perceptron(penalty=penalty)
    pct_classifier.fit(tfidf_traingram, y_train)
    predpt = pct_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print(penalty, "Perceptron penalty term Score: {:.5f}".format(score*100))
###None (default) is clearly best

###eta0 perceptron tuning
for eta0 in np.arange(0.1,2.0,0.1):
    pct_classifier = Perceptron(eta0=eta0)
    pct_classifier.fit(tfidf_traingram, y_train)
    predpt = pct_classifier.predict(tfidf_testgram)
    score = metrics.accuracy_score(y_test, predpt)
    print("eta0: {:.5f} Score: {:.5f}".format(eta0, score*100))
###changes not above noise


###########################################Final Comparison
###Make comparison table
finalcomparedf = pd.DataFrame(finalvecclass, columns = ["Analysis"])
finalscores = DataFrame (finalscore,columns=['Accuracy'])
finalcomparedf = finalcomparedf.join(finalscores)
print(finalcomparedf)

###Plot Bar Graph of Comparison
finalcomparedf.plot.bar(x='Analysis', y='Accuracy', rot=75, figsize=(26,12), legend=False, ylim=[75,100])









