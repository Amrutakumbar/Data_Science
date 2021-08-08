#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # load the dataset

# In[2]:


fraud=pd.read_csv('C:\\Users\\DELL\\Downloads\\Assignment 11. Decision Tree\\Fraud_check.csv')
fraud


# In[3]:


#create dummy variables for categorical


# In[4]:


Fraud=pd.get_dummies(fraud,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)
Fraud


# In[5]:


#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for risky and good
Fraud["Tax"] = pd.cut(Fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
Fraud


# In[6]:


#creating dummies for the Tax
Fraud = pd.get_dummies(Fraud,columns = ["Tax"],drop_first=True)


# # standardisation function

# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_fraud_df = scaler.fit_transform(Fraud.iloc[:,0:8])
scaled_fraud_df


# In[8]:


#converting Tax column into category.


# In[9]:


Fraud['Taxable.Income']=Fraud['Taxable.Income'].astype('category')


# # splitting the data

# In[10]:


x=Fraud.iloc[:,1:8]
y=Fraud.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=0)


# # Decision Tree- C5.0 (ENTROPY)

# In[11]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree


# In[12]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[13]:


#PLot the decision tree
tree.plot_tree(model);


# In[16]:


#prdiction


# In[17]:


y_pred=model.predict(x_test)
y_pred


# In[18]:


#accuracy


# In[19]:


model.score(x_test,y_test)


# # Decision Tree-CART(classification)

# In[20]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[21]:


model_gini.fit(x_train, y_train)


# In[22]:


#Prediction and computing the accuracy
y_pred=model.predict(x_test)
np.mean(y_pred==y_test)


# # Decision Tree- CART (Regression)

# In[23]:


from sklearn.tree import  DecisionTreeRegressor
from sklearn import tree


# In[24]:


model = DecisionTreeRegressor()
model.fit(x_train, y_train)


# In[25]:


#PLot the decision tree
tree.plot_tree(model);


# In[26]:


#prdiction


# In[27]:


y_pred=model.predict(x_test)


# In[28]:


#accuracy


# In[29]:


model.score(x_test,y_test)


# In[ ]:




