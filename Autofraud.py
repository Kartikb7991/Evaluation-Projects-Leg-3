#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[157]:


df=pd.read_csv('C:\\Users\\karti\\OneDrive\\Desktop\\Analytics work\Automobilefraud.csv')


# In[158]:


df.head()


# In[159]:


df.shape


# In[160]:


df.info()


# In[161]:


df.describe()


# In[162]:


df.isnull().sum()


# In[163]:


df.drop(['_c39','policy_number'],axis=1,inplace=True)


# In[164]:


df.head()


# In[165]:


df.loc[:, 'incident_date'] = pd.to_datetime(df.incident_date)
df.loc[:, 'policy_bind_date'] = pd.to_datetime(df.policy_bind_date)


# In[166]:


df= df.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)


# In[167]:


df['fraud_reported'].value_counts().plot(kind='bar')


# In[168]:


df['incident_state'].value_counts()


# In[169]:


pd.crosstab(df['age'], df['fraud_reported'])


# In[170]:


df.total_claim_amount.min()


# In[171]:


df.total_claim_amount.max()


# In[172]:


pd.crosstab(df.insured_sex, df.fraud_reported).plot(kind = 'bar', figsize = (10,6))


# In[173]:


df.policy_annual_premium.min()


# In[174]:


df.policy_annual_premium.max()


# In[175]:


df.columns


# In[176]:


df.drop(columns = ['policy_bind_date','policy_csl','insured_zip','insured_education_level','insured_occupation', 'insured_occupation','insured_hobbies','insured_relationship', 'incident_date', 'incident_state', 'incident_city', 'incident_type', 'authorities_contacted', 'property_damage', 'injury_claim', 'property_claim', 'vehicle_claim'], axis = 1, inplace = True)


# In[177]:


df.head()


# In[178]:


onehotenc = df[['policy_state','insured_sex', 'collision_type', 'incident_severity', 'police_report_available']]


# In[179]:


from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(sparse = False)


# In[180]:


onehottransformed = OHE.fit_transform(onehotenc)


# In[181]:


df.head()


# In[182]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[183]:


catcols = ['incident_location','policy_state',"collision_type","auto_make",'auto_model',"fraud_reported",'insured_sex']
for x in catcols:
    df[x] = le.fit_transform(df[x])


# In[184]:


df.head()


# In[185]:


df.drop(['police_report_available'],axis=1,inplace=True)


# In[186]:


df.head()


# In[187]:


df.columns.all


# In[188]:


df.info()


# In[189]:


df['incident_severity'].value_counts()


# In[190]:


catcols2=['incident_location','incident_severity']
for x in catcols2:
    df[x] = le.fit_transform(df[x])


# In[191]:


df.info()


# In[199]:


X = df.drop(columns=['fraud_reported'], axis=1)
y = df['fraud_reported']


# In[ ]:





# In[200]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:





# In[201]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[202]:


LR=LogisticRegression()
LR.fit(X_train,y_train)
LR.score(X_train,y_train)
predlr=LR.predict(X_test)
print(accuracy_score(y_test,predlr))
print(confusion_matrix(y_test,predlr))
print(classification_report(y_test,predlr))


# In[203]:


SV=SVC()
SV.fit(X_train,y_train)
SV.score(X_train,y_train)
predsv=SV.predict(X_test)
print(accuracy_score(y_test,predsv))
print(confusion_matrix(y_test,predsv))
print(classification_report(y_test,predsv))


# In[204]:


DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
DT.score(X_train,y_train)
preddt=DT.predict(X_test)
print(accuracy_score(y_test,preddt))
print(confusion_matrix(y_test,preddt))
print(classification_report(y_test,preddt))


# In[ ]:





# In[ ]:





# In[ ]:




