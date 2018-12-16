#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:06:21 2018

@author: pratikeshsingh
"""

import pandas as pd
import numpy as np
#from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
pd.set_option('display.max_columns',None)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
name=["ID number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
  
df = pd.read_csv(url,names=name)
df.isnull().sum()
df = df.replace(['?'],np.nan)
df.head()
df["Bare Nuclei"].fillna(df["Bare Nuclei"].mode()[0],inplace=True)

df["Bare Nuclei"]=df["Bare Nuclei"].astype(int)
df.dtypes

X = df.values[:,1:-1]
Y = df.values[:,-1]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=10)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


classifier = (LogisticRegression()) 
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc


y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)
new_y_pred_prob=np.round(y_pred_prob,2)
print(new_y_pred_prob)

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.32:
        y_pred_class.append(4)
    else:
        y_pred_class.append(2)
        
cfm = confusion_matrix(Y_test,y_pred_class)
print(cfm)

print(classification_report(Y_test,y_pred_class))

acc = accuracy_score(Y_test,y_pred_class)
acc

for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1]>a,4,2)
    cfm = confusion_matrix(Y_test, predict_mine)
    total_err = cfm[0,1]+cfm[1,0]
    print("Errors at threshold ",a, ":", total_err, ", type 2 error :", cfm[1,0], ", type 1 error: ", cfm[0,1])


estimators = [] 
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
print(estimators)

ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
print(Y_pred)
cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc