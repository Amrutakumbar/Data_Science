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


company=pd.read_csv('C:\\Users\\DELL\\Downloads\\Assignment 11. Decision Tree\\Company_Data.csv')
company.head()


# In[3]:


company.drop(columns=['CompPrice','Income','Price','Age','Education'],inplace=True)


# In[4]:


#Creating dummy vairables for ['ShelveLoc','US','Urban'] dropping first dummy variable
Company=pd.get_dummies(company,columns=['ShelveLoc','US','Urban'], drop_first=True)
Company


# In[5]:


#Creating new cols sale and dividing 'sales' cols on the basis of [2,9,15] for yes and no
Company["Sale"] = pd.cut(Company["Sales"], bins = [2,9,15], labels = ["No", "Yes"])
Company


# In[6]:


#creating dummies for the sales
Company = pd.get_dummies(Company,columns = ["Sale"],drop_first=True)


# ## standardisation function

# In[7]:



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_company_df = scaler.fit_transform(Company.iloc[:,0:8])
scaled_company_df


# In[8]:


#converting sales column into category.


# In[9]:


Company['Sales'] = Company['Sales'].astype('category')


# # splitting the data

# In[10]:


x=Company.iloc[:,0:7]
y=Company.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


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


# In[14]:


fn=['Advertising','Population','ShelveLoc_Good','ShelveLoc_Medium','US_Yes','Urban_YES']
cn=['0', '1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[15]:


#prdiction


# In[16]:


y_pred=model.predict(x_test)
y_pred


# In[17]:


#accuracy


# In[18]:


model.score(x_test,y_test)


# # Decision Tree-CART(classification)

# In[19]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[20]:


model_gini.fit(x_train, y_train)


# In[21]:


#Prediction and computing the accuracy
y_pred=model.predict(x_test)
np.mean(y_pred==y_test)


# In[ ]:





# # Decision Tree- CART (Regression)

# In[22]:


from sklearn.tree import  DecisionTreeRegressor
from sklearn import tree


# In[23]:


model = DecisionTreeRegressor()
model.fit(x_train, y_train)


# In[24]:


#prdiction


# In[25]:


y_pred=model.predict(x_test)
y_pred


# In[26]:


#accuracy


# In[27]:


model.score(x_test,y_test)


# In[ ]:





# In[ ]:




