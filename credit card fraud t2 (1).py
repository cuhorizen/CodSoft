#!/usr/bin/env python
# coding: utf-8
Muhammad Usman Malik 
TASK3 Credit card fraud detection 
# In[31]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[32]:


credit_card_data = pd.read_csv('creditcard.csv')


# In[33]:


credit_card_data.head()


# In[34]:


credit_card_data.tail()


# In[35]:


credit_card_data.info()


# In[36]:


# Dealing with missing values if exist
credit_card_data.isnull().sum()


# In[37]:


# classifying legit and fraud transactions
credit_card_data['Class'].value_counts()


# # this is unbalanced dataset if we train our model on this it will not findout the fraud transaction 

# In[38]:


# seperating the data for analysis
legit =credit_card_data[credit_card_data.Class ==0]
fraud =credit_card_data[credit_card_data.Class ==1]


# In[39]:


print(legit.shape)
print(fraud.shape)


# In[40]:


# statical measures of the data
legit.Amount.describe()


# In[41]:


fraud.Amount.describe()


# In[42]:


# compare the vlues for both transaction
credit_card_data.groupby('Class').mean()


# #  Dealing with unbalanced (under sampling)

# In[43]:


# i am using under sampling
# i am bulding a sample dataset containing similar distribution of normal transaction and Fraudulent transaction 
legit_sample=legit.sample(n=492)


# In[44]:


#we will concatinate these data frame to make a balance dataset
new_dataset =pd.concat([legit_sample,fraud],axis=0)


# In[45]:


new_dataset.head()


# In[46]:


new_dataset.tail()


# In[47]:


new_dataset['Class'].value_counts()


# In[48]:


new_dataset.groupby('Class').mean()


# # splitting the data

# In[49]:


x= new_dataset.drop(columns='Class',axis=1)
y= new_dataset['Class']


# In[50]:


print(x)


# In[51]:


print(y)


# In[52]:


#spliting data into test and train data
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)


# In[53]:


print(x.shape,x_train.shape,x_test.shape)


# #  model training

# In[55]:


model = LogisticRegression()


# In[57]:


model.fit(x_train,y_train)


# #  evaluation

# In[59]:


# accuracy score
x_train_prediction =model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[60]:


print('Accuracy of model on training data :',training_data_accuracy)


# In[61]:


x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[63]:


print('Accuracy of model on test data :',test_data_accuracy)

