#!/usr/bin/env python
# coding: utf-8

# Muhammad Usman Malik
# Task1 titanci data analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#load data from csv to dataframe in pandas
titanic_data =pd.read_csv('tested.csv')


# In[3]:


#printing the rows for preview
titanic_data.head()


# In[4]:


# number of rows and columns count
titanic_data.shape


# In[6]:


# getting basice information of the dataset
titanic_data.info()


# In[7]:


# count the number of missing values
titanic_data.isnull().sum()


# In[8]:


# dealing with the missing values
titanic_data = titanic_data.drop(columns='Cabin',axis =1)


# In[9]:


# replacing the missing values in age column with mean values
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[10]:


titanic_data=titanic_data.dropna()


# In[11]:


# check the missing values
titanic_data.isnull().sum()


# Data Analysis

# In[12]:


# preview of statical measures about the data
titanic_data.describe()


# In[13]:


# finding the number of people survived
titanic_data['Survived'].value_counts()


# Data visualization

# In[14]:


sns.set()


# In[15]:


# making count plot for survivde column
sns.countplot(x='Survived', data = titanic_data)


# In[16]:


titanic_data['Sex'].value_counts()


# In[17]:


# countplot on the basis of geneder wise survivde
sns.countplot(x='Sex', data = titanic_data)


# In[ ]:


# gender base number of surviver
sns.countplot(x='Sex', hue='Survived', data=titanic_data)


# In[ ]:


# making count plot for Passengerclass column
sns.countplot(x='Pclass', data = titanic_data)


# In[18]:


sns.countplot(x='Pclass', hue='Survived' , data = titanic_data)


# Encoding catagorical columns

# In[19]:


titanic_data['Sex'].value_counts()


# In[20]:


titanic_data['Embarked'].value_counts()


# In[21]:


# converting catagorical columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'C':0,'S':1,'Q':2}},inplace=True)


# In[22]:


titanic_data.head()


# seprating features& target

# In[23]:


X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y= titanic_data['Survived']


# In[ ]:


print(X)


# In[ ]:


print(y)


# Spliting the data into test and train data

# In[29]:


X_train,X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, random_state=2)


# In[30]:


print(X.shape, X_train.shape, X_test.shape)


# 
# **Model Training**
# 
# 

# Logistic Regression

# In[31]:


model = LogisticRegression()


# In[32]:


# training the model with the train data
model.fit(X_train,Y_train)


# MODEL EVALUATION

# In[28]:


# ACCURACY SCORE
X_train_prediction = model.predict(X_train)


# In[ ]:


print(X_train_prediction)


# In[33]:


training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print('Accuracy Score of data :',training_data_accuracy)


# In[34]:


# ACCURACY SCORE on test data
X_test_prediction = model.predict(X_test)


# In[35]:


print(X_test_prediction)


# In[36]:


test_data_acccuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of Test data :',test_data_acccuracy)

