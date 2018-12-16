#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:52:19 2018

@author: pratikeshsingh
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)
network_data = pd.read_csv("/Users/pratikeshsingh/Desktop/python/OBS_Network_data.csv", header = None, delimiter = " *, *", engine='python')
network_data.head()
network_data.shape
network_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]
network_data.isnull().sum()
network_data_rev=pd.DataFrame.copy(network_data)
network_data_rev.head()
network_data_rev=network_data_rev.drop('Packet Size_Byte', axis=1)
network_data_rev.shape
colname=['Node','Full_Bandwidth','Node Status','Class']

from sklearn import preprocessing 
le={} 
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    network_data_rev[x]=le[x].fit_transform(network_data_rev[x])

X = network_data_rev.values[:,:-1]
Y = network_data_rev.values[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)

Y=Y.astype(int) 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=10)


from sklearn.tree import DecisionTreeClassifier
model_DecisionTree = (DecisionTreeClassifier()) 
model_DecisionTree.fit(X_train,Y_train) 



Y_pred = model_DecisionTree.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc

y_pred_prob = model_DecisionTree.predict_proba(X_test)
print(y_pred_prob)



classifier=(DecisionTreeClassifier())

from sklearn import cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10) 
print(kfold_cv)

kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
print(kfold_cv_result.mean())

from sklearn.linear_model import LogisticRegression
classifier = (LogisticRegression())

from sklearn import svm
svc_model = svm.SVC(kernel='rbf',C=3.0,gamma=1)

svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)

acc = accuracy_score(Y_test,Y_pred)
acc
#hence decision tree gave best output without any tuning required


from sklearn import tree
with open("/Users/pratikeshsingh/Desktop/python/model_DecisionTree.txt","w") as f: #w means open the file in write mode
    f=tree.export_graphviz(model_DecisionTree,out_file=f)  #write all the steps which the object will perform at the back end


#Ensemble Modelling using ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

Y_pred = model_DecisionTree.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc


#Ensemble learning using RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

model=(RandomForestClassifier(501))
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

Y_pred = model_DecisionTree.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc



#ensemble Learning using AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost = (AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=40))

model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc

#ensemble Learning using GradientBoost
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting = GradientBoostingClassifier(n_estimators=120)

model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)


cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)

print(classification_report(Y_test,Y_pred))

acc = accuracy_score(Y_test,Y_pred)
acc


#Ensemble Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

estimators = [] #this takes the list of all the models that we will be running
#model1=LogisticRegression()
#estimators.append(('log',model1)) #votingclassifier requires 2 parameters to be passed, one is the name and the second is the actual data. the first name can be any name and the second parameter has to be the data
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
