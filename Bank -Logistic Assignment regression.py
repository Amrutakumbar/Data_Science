#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # import the data set

# In[2]:


bank=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 6. Logistic Regression\\bank-full.csv")


# # EDA

# In[3]:


bank.head()


# In[4]:


bank.info()


# In[5]:


bank.isnull().sum()


# In[6]:


bank.isnull().values.any()


# In[7]:


bank.describe()


# In[8]:


bank.nunique()


# In[9]:


bank.columns


# In[10]:


bank_new=bank.columns.drop(['job', 'marital', 'education', 'default','housing','loan', 'contact', 'day', 'month','pdays', 'previous', 'poutcome'])
bank_new


# # Visualisation

# In[11]:


columns=['age', 'balance', 'duration', 'campaign', 'y']
column=bank[columns]


# In[12]:


#crosstab
pd.crosstab(column.age,column.y).plot(kind='line')


# In[13]:


#boxplot
column['y']=column.y.map({'no':0,'yes':1})


# In[14]:


column.boxplot(column='age',by='y')


# # Splitting the data into train and test

# In[15]:


input=['age','balance','duration','campaign']
output=['y']
x=column[input]
y=column[output]


# # Model fitting

# In[16]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x,y)


# # Calculate coeffient and intercepts
# 

# In[17]:


classifier.coef_


# In[18]:


classifier.predict_proba(x)


# # predicting the x data set

# In[19]:


y_pred=classifier.predict(x)
y_pred


# # Confusion matrix

# In[20]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix


# In[21]:


((39342+854)/(39342+580+854+4435))*100


# In[22]:


#heatmap


# In[23]:


sns.heatmap(confusion_matrix, annot=True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# + from above heatmap we concluded that the client has subscribed a term deposite
# + accuracy=88.90%(good model)

# In[ ]:




