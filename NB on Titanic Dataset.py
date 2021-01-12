#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classifier on Titanic Dataset

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\titanic.csv")

df.head(10)


# In[38]:


sns.countplot('Survived',hue='Survived',data=df)
#obs : no of passenger not survived is more.


# In[39]:


#Obs : 
#male passengers not survived more from total count
#Female passengers  survived more from total count

sns.countplot('Gender',hue='Gender',data=df)


# In[40]:


df.columns


# In[41]:


df.drop(['Cabin','Embarked','Parch','SibSp','Name','Ticket'],axis=1,inplace=True)
df


# In[42]:


X = df.iloc[: , :5]
y = df['Survived']


# In[43]:


X


# In[44]:


y


# In[45]:


split_gender=pd.get_dummies(X.Gender)


# In[46]:


split_gender


# In[47]:


X= pd.concat([X,split_gender],axis = 'columns')


# In[48]:


X


# In[49]:


X.drop(['male','Gender'],axis=1,inplace=True)


# In[50]:


X


# In[51]:


X.info()


# In[52]:


X.isnull()


# In[53]:


X.isnull().any()


# In[54]:


X.Age.mean()


# In[55]:


X.Age=X.Age.fillna(X.Age.mean())
print('Missing value is filled..!! ')


# In[56]:


X.isnull().any()


# In[57]:


X


# In[58]:


# Split Dataset into training and testing

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.3,random_state = 3)


# In[59]:


print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)

print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[60]:


from sklearn.naive_bayes import GaussianNB

NB_Model=GaussianNB()
NB_Model.fit(X_train,y_train)

print("Model trained Successfully..!!")


# In[62]:


y_predict=NB_Model.predict(X_test)

y_predict


# In[64]:


#Accurracy
from sklearn import metrics
acc=metrics.accuracy_score(y_predict,y_test)
print("NB_Accurracy is:",acc*100)


# In[65]:


X.columns


# In[68]:


PassengerId=int(input("PassengerId :"))
Pclass = int(input("Pclass : "))
Age  = float(input("Age : "))
Fare = float(input("Fare : "))
female = int(input("Female :"))


# In[69]:


NB_Model.predict([[PassengerId,Pclass,Age,Fare,female]])


# # Log Reg for Titanic 

# In[70]:


from sklearn.linear_model import LogisticRegression
Log_model = LogisticRegression()
Log_model.fit(X_train,y_train)
print("Log_Model Trained Sucessfully...!!")


# In[71]:


y_pred = Log_model.predict(X_test)
y_pred


# In[72]:


#Accucracy
from sklearn import metrics
acc = metrics.accuracy_score(y_pred,y_test)
print("NB_model Acc is : ",acc*100)

