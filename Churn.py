#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


df = pd.read_excel(r'C:\Users\karti\OneDrive\Desktop\Analytics work\Churn.xlsx')


# In[25]:


df.head()


# In[26]:


df.describe()


# In[27]:


df.info()


# In[28]:


df.isnull().sum()


# In[29]:


sns.heatmap(df.isnull())


# In[30]:


for x in df.columns:
    print(x)
    print (df[x].unique())


# In[31]:


df.info()


# In[39]:


catcols = df.select_dtypes(exclude=["number","bool_","float_"])


# In[33]:


df['churn'] = df['churn'].astype('int64')
df.head()


# In[34]:


df['churn'].value_counts()


# In[35]:


df.drop(columns=['phone number'],axis=1,inplace=True)


# In[36]:


df.head()


# In[40]:


catcols.head()


# In[49]:


df.groupby('churn')['customer service calls'].mean()
#People who are able to achieve a better churn make more customer service calls


# In[50]:


plt.hist(df['total day minutes'], bins = 100) 


# In[52]:


sns.boxplot(x = 'churn', 
            y = 'customer service calls', 
            data = df,                  
            hue = "international plan") 


# In[53]:


df.columns


# In[55]:


X = df.iloc[:, 0:19].values 
y = df.iloc[:, 19].values


# In[56]:


from sklearn.preprocessing import LabelEncoder 
  


# In[57]:


labelencoder_X_1 = LabelEncoder() 
X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3]) 
  
labelencoder_X_2 = LabelEncoder() 
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4]) 


# In[59]:


X.shape


# In[61]:


y.shape


# In[65]:


y


# In[66]:


Xstate = pd.get_dummies(X[:, 0], drop_first = True) 


# In[67]:


X = pd.DataFrame(X)


# In[68]:


X.head()


# In[69]:


X = X.drop([0], axis = 1) 


# In[73]:


frames = [Xstate, X] 
final = pd.concat(frames, axis = 1, ignore_index = True) 
  


# In[74]:


X = final


# In[75]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size = 0.2,  
                                                    random_state = 0) 


# In[76]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 


# # RANDOM FOREST CLASSSIFIER

# In[77]:


from sklearn.ensemble import RandomForestClassifier 
RF = RandomForestClassifier() 
RF.fit(X_train, y_train)


# In[78]:


y_pred = RF.predict(X_test)


# In[79]:


from sklearn.metrics import accuracy_score 
  


# In[80]:


accuracy_score(y_test, y_pred)


# In[81]:


from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 


# # LOGISTIC REGRESSION

# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[85]:


LR=LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_train,y_train)
predlr=LR.predict(X_test)
print(accuracy_score(y_test,predlr))
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# # svc

# In[86]:


from sklearn.svm import SVC


# In[87]:


SV=SVC()
SV.fit(X_train,y_train)
SV.score(X_train,y_train)
predsv=SV.predict(X_test)
print(accuracy_score(y_test,predsv))
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# # It can be observed that the decision tree algorithm gives us the highest accuracy of nearly 95 percent so we will be going with that model

# In[ ]:




