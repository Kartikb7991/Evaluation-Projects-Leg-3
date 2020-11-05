#!/usr/bin/env python
# coding: utf-8

# In[709]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVR


# In[710]:


Ftest = pd.read_excel(r'C:\Users\karti\OneDrive\Desktop\Analytics work\Test_set.xlsx')
Ftrain=pd.read_excel(r'C:\Users\karti\OneDrive\Desktop\Analytics work\Data_Train.xlsx')


# In[ ]:





# In[711]:


Ftrain.head()


# In[712]:


Ftest.head()


# In[713]:



Ftrain.shape,Ftest.shape


# In[714]:


Ftrain.info()


# In[715]:


Ftrain.columns


# In[716]:


#Null values in a training dataset
Ftrain.isnull().sum()


# In[717]:


Ftrain.dropna(inplace=True)


# In[718]:


Ftrain.isnull().sum()


# In[719]:


Ftest.isnull().sum()


# In[ ]:





# In[720]:


Ftrain.isnull().sum()


# In[721]:


Ftrain['Airline'].value_counts()


# In[722]:


plt.figure(figsize=(35,18))
sns.countplot(Ftrain['Airline'])


# In[723]:


Ftrain['Destination'].value_counts()


# In[724]:


sns.countplot(Ftrain['Destination'])


# In[725]:


pd.crosstab(flight_train['Airline'],Ftrain['Total_Stops'])


# In[726]:


#extracting the month from the flight date
Ftrain['Date_of_Journey'] = pd.to_datetime(Ftrain['Date_of_Journey'])
Ftrain['Month'] =Ftrain['Date_of_Journey'].dt.month
Ftrain['Day'] = Ftrain['Date_of_Journey'].dt.day


# In[727]:


Ftrain.sort_values('Date_of_Journey', inplace = True)


# In[728]:


sns.countplot(x = 'Month', data = Ftrain)
plt.xlabel('Month')
plt.ylabel('Count of flights')


# In[729]:


plt.figure(figsize = (40, 20))
plt.title('Count of flights with different Airlines')
sns.countplot(x = 'Airline', data = Ftrain)
plt.xlabel('Airline')
plt.ylabel('Count of flights')


# In[730]:


Ftrain.head()


# In[731]:


Ftest["Destination"].value_counts()


# In[732]:


import datetime as dt


# In[733]:


Ftrain["Dayofjourney"] =pd.to_datetime(Ftrain["Date_of_Journey"]).dt.day
Ftrain["Monthofjourney"] =pd.to_datetime(Ftrain["Date_of_Journey"]).dt.month


# In[734]:


Ftrain.sample(5)


# In[735]:


Ftrain.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[736]:


Ftrain["Hourofdep"] = pd.to_datetime(Ftrain["Dep_Time"]).dt.hour
Ftrain["Minofdep"] = pd.to_datetime(Ftrain["Dep_Time"]).dt.minute
Ftrain["Hourofarrival"] = pd.to_datetime(Ftrain["Arrival_Time"]).dt.hour
Ftrain["Minofarrival"] = pd.to_datetime(Ftrain["Arrival_Time"]).dt.minute


# In[737]:


Ftrain.drop(["Dep_Time"], axis = 1, inplace = True)
Ftrain.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[738]:


Ftrain.head()


# In[739]:


#This was a new code and experience for me dealing with time variable so i have taken reference from youtube and for loops

duration = list(Ftrain["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:   
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]       

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0])) 
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))  


# In[740]:


Ftrain["Duration_hours"] = duration_hours
Ftrain["Duration_mins"] =duration_mins


# In[741]:


Ftrain.head()


# In[742]:


Ftrain.drop('Duration',axis=1,inplace=True)


# In[743]:


Ftrain.head()


# In[744]:


Ftrain['Airline'].value_counts()


# In[745]:


def convertingstopstonumeric(X):
    if X == '4 stops':
        return 4
    elif X == '3 stops':
        return 3
    elif X == '2 stops':
        return 2
    elif X == '1 stop':
        return 1
    elif X == 'non stop':
        return 0


# In[746]:



Ftrain['Total_Stops'] = Ftrain['Total_Stops'].map(convertingstopstonumeric)


# In[747]:


Ftrain.head()


# In[748]:


#Additional info has mostly no info values and route and total stops are correlated so both columns are being dropped

Ftrain.drop(["Route","Additional_Info"],axis=1, inplace=True)


# In[682]:


Ftrain.head()


# In[749]:


Ftrain.isnull().sum()


# In[751]:


Ftrain['Total_Stops'].fillna(Ftrain['Total_Stops'].mean(),inplace=True)


# In[752]:


Ftrain.isnull().sum()


# In[753]:


Airline = Ftrain[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first =True)


# In[754]:


Source =Ftrain[["Source"]]

Source =pd.get_dummies(Source, drop_first=True)


# In[755]:


Destination =Ftrain[["Destination"]]

Destination =pd.get_dummies(Destination, drop_first=True)


# In[756]:


Ftrain=pd.concat([Ftrain,Airline,Source,Destination], axis=1)


# In[757]:


Ftrain.head()


# In[758]:


Ftrain.drop(["Airline","Source", "Destination"], axis=1, inplace=True)


# In[759]:


Ftrain.head()


# # WORKING ON TEST DATA

# In[760]:


print("Test data Info")
print("-"*75)
print(Ftest.info())

print()
print()

print("Null values :")
print("-"*75)
Ftest.dropna(inplace = True)
print(Ftest.isnull().sum())


# In[761]:



# Date_of_Journey
Ftest["Journey_day"] = pd.to_datetime(Ftest.Date_of_Journey, format="%d/%m/%Y").dt.day
Ftest["Journey_month"] = pd.to_datetime(Ftest["Date_of_Journey"], format = "%d/%m/%Y").dt.month
Ftest.drop(["Date_of_Journey"], axis = 1, inplace = True)

Ftest["Dep_hour"] = pd.to_datetime(Ftest["Dep_Time"]).dt.hour
Ftest["Dep_min"] = pd.to_datetime(Ftest["Dep_Time"]).dt.minute
Ftest.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
Ftest["Arrival_hour"] = pd.to_datetime(Ftest.Arrival_Time).dt.hour
Ftest["Arrival_min"] = pd.to_datetime(Ftest.Arrival_Time).dt.minute
Ftest.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[762]:


duration = list(Ftest["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
Ftest["Duration_hours"] = duration_hours
Ftest["Duration_mins"] = duration_mins
Ftest.drop(["Duration"], axis = 1, inplace = True)

    


# In[763]:


print("Airline")
print("-"*75)
print(Ftest["Airline"].value_counts())
Airline = pd.get_dummies(Ftest["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(Ftest["Source"].value_counts())
Source = pd.get_dummies(Ftest["Source"], drop_first= True)


# In[764]:


print("Destination")
print("-"*75)
print(Ftest["Destination"].value_counts())
Destination = pd.get_dummies(Ftest["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
Ftest.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
Ftest.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[765]:



Ftest = pd.concat([Ftest, Airline, Source, Destination], axis = 1)

Ftest.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", Ftest.shape)


# In[766]:


X= Ftrain.drop(['Price'],axis=1)
y=Ftrain['Price']


# In[767]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[768]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[ ]:





# In[769]:


X.head()


# In[770]:


X.shape


# In[771]:


y.isnull().sum()


# In[707]:





# In[772]:


X.isnull().sum()


# # Random Forest Regressor

# In[773]:


from sklearn.ensemble import RandomForestRegressor
RGR = RandomForestRegressor()
RGR.fit(X_train, y_train)


# In[775]:


y_pred = RGR.predict(X_test)


# In[776]:


RGR.score(X_train, y_train)


# In[777]:


RGR.score(X_test, y_test)


# In[778]:


sns.distplot(y_test-y_pred)
plt.show()


# In[779]:


from sklearn import metrics


# In[780]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Linear Regression

# In[784]:


from sklearn.linear_model import LinearRegression


# In[785]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[786]:


Y_Pred = reg.predict(X_test)
Y_Pred


# In[787]:


mse = mean_squared_error(y_test, Y_Pred)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, Y_Pred))


# # Between the two , random forest would be a better model for the data
