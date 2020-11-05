#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


# In[25]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\Insurance claims.csv')


# In[26]:


df.head()


# In[27]:


df.columns


# In[28]:


df.info()


# In[29]:


df.describe()


# In[30]:


df['EmploymentStatus'].value_counts()


# In[31]:


df.shape


# In[32]:


df.head()


# In[33]:


df.drop(['Customer'],axis=1,inplace=True)


# In[34]:


df.head()


# In[35]:


df['Vehicle Size'].unique()


# In[36]:


sns.countplot(df['Vehicle Size'],hue=df['Education'])


# In[38]:


sns.countplot(df['Vehicle Size'],hue=df['Coverage'])


# In[37]:


sns.countplot(df['Vehicle Size'],hue=df['Claim Reason'])


# In[39]:


sns.countplot(df['Vehicle Size'],hue=df['Policy'])


# In[40]:


sns.countplot(df['Vehicle Size'],hue=df['Policy Type'])


# In[41]:


sns.countplot(df['Policy Type'],hue=df['Coverage'])


# In[42]:


df.head()


# In[45]:


df['Effective To Date']=pd.to_datetime(df['Effective To Date'])
df['Effective To Date']=df['Effective To Date'].map(dt.datetime.toordinal)


# In[46]:


df.head()


# In[47]:


df.info()


# In[58]:


catcols=['Coverage','Response','Education','EmploymentStatus','Gender','Policy Type','Policy','Claim Reason','Sales Channel','Vehicle Size','Marital Status','Vehicle Class']


# In[59]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[60]:


for x in catcols:
    df[x] = le.fit_transform(df[x])


# In[61]:


df.head()


# In[63]:


df.drop(['State Code'],axis=1,inplace=True)


# In[64]:


df.info()


# In[65]:


df.drop(['Location Code'],axis=1,inplace=True)


# In[66]:


df.info()


# In[78]:


df.drop(['Country'],axis=1,inplace=True)


# In[79]:


le.fit_transform(df['State'])


# In[81]:


df['State']=le.fit_transform(df['State'])


# In[82]:


df.head()


# In[83]:


df.info()


# In[84]:


# splitting the data into dependent and independent variables

x = df.drop('Claim Amount', axis = 1)
y = df['Claim Amount']


# In[85]:


print(x.shape)
print(y.shape)


# In[86]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[87]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,y_train)
predict=LR.predict(X_test)


# In[88]:


from sklearn import metrics
print('Meanabserror:', metrics.mean_absolute_error(y_test, predict))
print('Meansquareerror:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[89]:


plt.scatter(x=y_test,y=predict)


# In[90]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(X_train,y_train)
pred=RF.predict(X_test)


# In[91]:


print('Meanabserror', metrics.mean_absolute_error(y_test, pred))
print('Meansqerror:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[98]:


sns.distplot((y_test-pred),bins=5)


# In[93]:


#The random forest regressor shows a better curve so as a result this model is suitable for the data


# In[ ]:




