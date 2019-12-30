# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:57:49 2019

@author: fakab
"""

import os, sys
import pickle
import json
from sklearn.naive_bayes import MultinomialNB as nb
# from sklearn.linear_model import SGDClassifier as svm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#%%

cwd = os.getcwd() #Current working directory

#Read training data
f = open(os.path.join(cwd, r'training.pkl'), 'rb')
(X_data, X_label) = pickle.load(f)
f.close()


#Read test data
f = open(os.path.join(cwd, r'testing.pkl'), 'rb')
(Y_data, Y_label) = pickle.load(f)
f.close()

train = []
trainLabel = []


#%%

label2no = {u'support':0, u'query':1, u'deny':2, u'comment':3}

#Convert list of lists to nd array (Required for NB Training)
for key in X_label.keys():
	train.append(X_data[key])
	trainLabel.append(label2no[X_label[key]])

train = np.array(train)
trainLabel = np.array(trainLabel)
min1 = train.min()
# print (min1)
for i in range(len(train)):
	for j in range(len(train[i])):
		train[i][j] = train[i][j] + abs(min1) 
    
#%%
#Naive Bayes Classifier Training        
nb_clf = nb().fit(train, trainLabel.transpose())

#%%
test = []
testLabel = []

for key in Y_label.keys():
	test.append(Y_data[key])
	testLabel.append(label2no[Y_label[key]])

test = np.array(test)
testLabel = np.array(testLabel)
min1 = test.min()
for i in range(len(test)):
	for j in range(len(test[i])):
		test[i][j] += min1 



#%%
predicted=nb_clf.predict(test)
#%%
print("Classification accuracy: ", accuracy_score(testLabel, predicted))
#%%
print("Confusion matrix: ", confusion_matrix(testLabel, predicted))
target_names = ['support', 'query', 'deny', 'comment']
print(classification_report(testLabel, predicted, target_names=target_names))
#Accuracy
print(np.mean(predicted == testLabel)*100)

#%%
#AN ENSEMBLE (WITH VOTING) OF LOGISTIC, GAUSSIAN NB, RANDOM FOREST
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],  voting='hard')

eclf = eclf.fit(train, trainLabel.transpose())


#%%
#AN ENSEMBLE (WITH BAGGING) OF KNN
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
x = bagging.fit(train, trainLabel.transpose())

#accuracy 0.5891325071496664


#%%
#AN ENSEMBLE (WITH VOTING) OF LOGISTIC, GAUSSIAN NB, RANDOM FOREST
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
clf3 = SVC(kernel='rbf', probability=True).fit(train, trainLabel.transpose())
nb_clf = nb().fit(train, trainLabel.transpose())
eclf = VotingClassifier(estimators=[('nb', nb_clf), ('svm', clf3)],  voting='hard')

for clf, label in zip([nb_clf, clf3, eclf], ['NB', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, train, trainLabel.transpose(), scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
eclf = eclf.fit(train, trainLabel.transpose())
#acc  0.7254528122020972

#Accuracy: 0.64 (+/- 0.00) [NB]
#Accuracy: 0.69 (+/- 0.01) [SVM]
#Accuracy: 0.69 (+/- 0.01) [Ensemble]

#%%
#AN ENSEMBLE (WITH RANDOM FOREST) 
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train, trainLabel.transpose())

# acc 0.3517635843660629

#AN ENSEMBLE(WITH ADABOOST WITH DECISION TREE)
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=100)
clf.fit(train, trainLabel.transpose())
scores = cross_val_score(clf, train, trainLabel.transpose(), cv=5)
scores.mean()

#acc 0.7416587225929456

# base_estimator=RandomForestClassifier(n_estimators=10) acc 0.6687041057183608

#%%
#AN ENSEMBLE(WITH VOTING) OF LOGISTIC, GAUSSIAN NB, RANDOM FOREST
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier



clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = nb()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2, 5, 1])

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, train, trainLabel.transpose(), scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
clf.fit(train, trainLabel.transpose())

#0.09532888465204957
#Accuracy: 0.70 (+/- 0.01) [Logistic Regression]
#Accuracy: 0.70 (+/- 0.01) [Random Forest]
#Accuracy: 0.64 (+/- 0.00) [naive Bayes]
#Accuracy: 0.70 (+/- 0.00) [Ensemble]

#%%

#AN ENSEMBLE(WITH VOTING) OF DECISION TREE, KNN, SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier


clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2, 1, 2])

for clf, label in zip([clf1, clf2, clf3, eclf], ['DT', 'KNN', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, train, trainLabel.transpose(), scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

clf1 = clf1.fit(train, trainLabel.transpose())
clf2 = clf2.fit(train, trainLabel.transpose())
clf3 = clf3.fit(train, trainLabel.transpose())
eclf = eclf.fit(train, trainLabel.transpose())

#Accuracy: 0.69 (+/- 0.01) [DT]
#Accuracy: 0.65 (+/- 0.02) [KNN]
#Accuracy: 0.69 (+/- 0.01) [SVM]
#Accuracy: 0.70 (+/- 0.01) [Ensemble]


