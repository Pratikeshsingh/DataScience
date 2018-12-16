
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[2]:


# read data into a DataFrame
data = pd.read_csv('/Users/pratikeshsingh/Desktop/python/Advertising.csv', index_col=0)
data.head()


# In[3]:


print(data.dtypes)
print(data.shape)
print(data.describe())


# In[4]:


# create X and y
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
Y = data.sales


# In[5]:


import seaborn as sns
sns.pairplot(data,x_vars=["TV","radio","newspaper"],y_vars="sales",kind='reg')


# In[6]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
print(X)


# In[7]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=10)  


# In[8]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)

# print intercept and coefficients
print (lm.intercept_)
print (lm.coef_)


# In[9]:


# pair the feature names with the coefficients
print(list(zip(feature_cols, lm.coef_)))


# In[10]:


X1=50
X2=50
X3=50
y_pred=3.41064158861+(0.04303172 *X1)+(0.19352212*X2)+(-0.00386729*X3)
print(y_pred)
#y_pred_new = 14.122070736568478+(3.68520775*X1)+(2.86599405 *X2)+ (-0.08401343*X3)
#print(y_pred_new)
# In[11]:


Y_pred=lm.predict(X_test)
print(Y_pred)


# In[12]:


from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2score=r2_score(Y_test,Y_pred)
print(r2score)

rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)


# In[13]:


print(min(Y))
print(max(Y))      


# In[14]:


import statsmodels.formula.api as sm

# create a fitted model with all three features
lm_model = sm.ols(formula='sales ~ TV + radio + newspaper', data=data).fit()

# print the coefficients
print(lm_model.params)
print(lm_model.summary())


# In[58]:


Y_pred=lm_model.predict()

from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

r2score=r2_score(data['sales'],Y_pred)
print(r2score)

rmse=np.sqrt(mean_squared_error(data['sales'],Y_pred))
print(rmse)


# In[52]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
ind_df=data.iloc[:,:-1]

vif_df = pd.DataFrame()
vif_df["features"] = ind_df.columns
vif_df["VIF Factor"] = [vif(ind_df.values, i) for i in range(ind_df.shape[1])]
vif_df.round(2)


# In[15]:


import statsmodels.formula.api as sm

# create a fitted model with two features
lm_model = sm.ols(formula='sales ~ TV + radio ', data=data).fit()

# print the coefficients
print(lm_model.params)
print(lm_model.summary())


# In[55]:


import seaborn as sns

corr_df=ind_df.corr(method="pearson")
print(corr_df)

sns.heatmap(corr_df,vmax=1.0,vmin=-1.0)


# In[54]:


ind_df=data.iloc[:,:-1]
sns.set(color_codes=True)
#sns.distplot(ind_df['newspaper'])
sns.distplot(X[:,2])

