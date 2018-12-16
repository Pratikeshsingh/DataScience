
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# # Read data from csv

# In[5]:


pd.set_option('display.max_columns', None)


# In[6]:


from pandas import read_csv
credit_df = pd.read_csv('credit_data.txt',  delimiter='\t', low_memory=False, parse_dates=["issue_d"])


# In[7]:


credit_df.head()


# In[9]:


credit_df.info()


# In[10]:


credit_df.describe(include = [np.number]) #numerical value describe


# # Preprocessing

# ## NULL values

# #### Let's see how many NULL values we have in our data!

# In[11]:


credit_df.isnull().sum()


# In[12]:


#filling value to 0 for these two columns
for col in ('revol_util', 'collections_12_mths_ex_med'):
    credit_df[col] = credit_df[col].fillna(0)


# #### There are columns where most of values are NULLs. So remove those columns where more than 1% of the rows for that column contain a null value.

# In[13]:


cleaned_data = credit_df[[label for label in credit_df if credit_df[label].isnull().sum() <= 0.01 * credit_df.shape[0]]]


# ##### Let's see how it looks like now!

# In[14]:


cleaned_data.head()


# In[15]:


cleaned_data.isnull().sum()


# In[16]:


cleaned_data.shape[0] / credit_df.shape[0]


# # Remove useless columns

# - As we can see, two first columns contain randomly generated numbers, which are some identifiers. 
# - Column"zip_code" is redundant with the "addr_state" and only 3 digits of 5 digit code zip are visible.
# - Column sub_grade is reduntant to columns "grade" and "int_rate".
# - Column "title"  requires a lot of processing to become useful.

# In[17]:


cleaned_data = cleaned_data.drop(["id", "member_id", "sub_grade", "zip_code", "title"], axis=1)


# ### Columns with only one value

# In[18]:


for label in list(cleaned_data):
    if len(cleaned_data[label].unique()) < 5:
        print(cleaned_data[label].value_counts())
        print("\n")


# - We can see that feature "pymnt_plan" has only two possible values: "n" and "y", but with only 10 occurrences of "y" (less than 1%), so definitely it is insignificant. 
# - The same with "application_type" feature: value "joint" has 0,05% frequency. 
# - On the other hand, feature "policy_code" has only one possible value, so it's absolutely useless for us.

# In[19]:


cleaned_data = cleaned_data.drop(["pymnt_plan", "policy_code", "application_type"], axis=1)


# ### Categorical features

# In[20]:


cleaned_data.select_dtypes(include=["object"]).head()


# - "initial_list_status" has only 2 possible values, so we can map it to 1/0 feature.
# - "term" is a numerical feature, but we have to delete "months" from this.
# - "last_pymnt_amnt" is definitely numerical, so the parsing is enough.

# In[21]:


cleaned_data["initial_list_status"] = cleaned_data["initial_list_status"].map({"f": 1, "w": 0})
cleaned_data["last_pymnt_amnt"] = cleaned_data["last_pymnt_amnt"].astype("float")
cleaned_data['term'].replace(regex=True,inplace=True,to_replace=r'months',value=r'')


# ## Datetime features

# In[22]:


cleaned_data.select_dtypes(include=["datetime"]).head()


# ### Remvoing highly correlated or un-correlated data

# In[23]:


plt.figure(figsize=(40,20))
sns.set_context("paper", font_scale=1.5)
sns.heatmap(cleaned_data.corr(), vmax=.8, square=True, annot=True, fmt='.2f')


# In[25]:


cleaned_data = cleaned_data.drop(["funded_amnt", "funded_amnt_inv", "out_prncp_inv",
                                 "total_pymnt_inv", "total_rec_prncp", "collections_12_mths_ex_med"], axis=1)


# #### Removing datetime features as it is not required while model building

# In[26]:


cleaned_data = cleaned_data.drop(["earliest_cr_line", "last_credit_pull_d"], axis=1)


# #### Found some columns are not helping while building model building, so removing them.

# In[27]:


cleaned_data = cleaned_data.drop(['total_pymnt','total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
                 'last_pymnt_amnt', 'acc_now_delinq', 'out_prncp', 'initial_list_status'], axis=1)


# In[28]:


cleaned_data.isnull().sum()


# ### Dividing data into Train and Test

# In[29]:


#train data creation using issue_d ( June 2007 - May 2015 )
train_data=cleaned_data[cleaned_data['issue_d']<='May-2015']
train_data.shape


# In[30]:


#Test data creation using issue_d ( June 2015 - Dec 2015 )
test_data=cleaned_data[cleaned_data['issue_d']>'May-2015']
test_data.shape


# ### Formatting of the Categorical columns

# In[31]:


colname=['term','grade','home_ownership','verification_status','purpose','addr_state']
colname


# In[32]:


from sklearn import preprocessing

le={}

for label in colname:
    le[label]=preprocessing.LabelEncoder()
    
for label in colname:
    train_data[label]=le[label].fit_transform(train_data.__getattr__(label))

train_data.head()


# In[33]:


from sklearn import preprocessing

le={}

for label in colname:
    le[label]=preprocessing.LabelEncoder()
    
for label in colname:
    test_data[label]=le[label].fit_transform(test_data.__getattr__(label))

test_data.head()


# ### Removing issue_d column as it is not required while column building

# In[34]:


train_data = train_data.drop(["issue_d"], axis=1)
test_data = test_data.drop(["issue_d"], axis=1)


# ### Selecting dependent and independent variables for train and test data

# In[35]:


x_train=train_data.values[:, 1:-1]
y_train=train_data.values[:,-1]

x_test=test_data.values[:, 1:-1]
y_test=test_data.values[:,-1]


# ### Transformation of final data

# In[36]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
print(x_train)

scaler.fit(x_test)
x_test=scaler.transform(x_test)
print(x_test)


# # Model Building

# # Logistic Regression

# In[153]:


from sklearn.linear_model import LogisticRegression

#create a model
classifier=(LogisticRegression(random_state=0))

#Training the model - Fitting training data into model
classifier.fit(x_train, y_train)

#Test the model
y_pred=classifier.predict(x_test)
print(list(zip(y_test, y_pred)))


# In[154]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# # Random Forest

# In[155]:


#predicting using the Random forest Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=(RandomForestClassifier(100))
#fit the model on the data and predict the values
model_RandomForest=model_RandomForest.fit(x_train, y_train)

y_pred=model_RandomForest.predict(x_test)


# In[156]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# # Bagging using ExtraTreesClassifier

# In[157]:


#predicting using the Bagging Classifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))
#fit the model on the data and predict the values
model=model.fit(x_train, y_train)

y_pred=model.predict(x_test)


# In[158]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# # Running Decision Tree Model

# In[163]:


DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                      max_features=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort=True, random_state=None,
                      splitter='best')


# In[164]:


#predicting using Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(x_train, y_train)

#fit the model in the data and predict the values
y_pred=model_DecisionTree.predict(x_test)
print(y_pred)
print(list(zip(y_test, y_pred)))


# In[165]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# # Boosting

# In[166]:


#predicting using the AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100))
#fit the model on the data and predict the values
model_AdaBoost=model_AdaBoost.fit(x_train, y_train)

y_pred=model_AdaBoost.predict(x_test)


# In[167]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# # SVM

# In[ ]:


from sklearn import svm
svc_model=svm.SVC(kernel='rbf', C=1.0, gamma=0.1)
#from sklearn.linear_model import LogisticRegression
#svc_model=logisticRegression()
svc_model.fit(x_train, y_train)
y_pred=svc_model.predict(x_test)
print(list(y_pred))


# In[ ]:


from sklearn import svm
svc_model=svm.SVC(kernel='rbf', C=1.0, gamma=0.1)
#from sklearn.linear_model import LogisticRegression
#svc_model=logisticRegression()
svc_model.fit(x_train, y_train)
y_pred=svc_model.predict(x_test)
print(list(y_pred))

