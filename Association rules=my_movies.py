#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # load the dataset

# In[2]:


movies=pd.read_csv('C:\\Users\\DELL\\Downloads\\Assignment 9. Association Rules\\my_movies.csv')
movies.head()


# In[3]:


movies.columns


# In[4]:


movies.isnull()


# In[5]:


df=pd.get_dummies(movies)
df.head()


# In[6]:


df.describe()


# # apriori and association rules

# In[7]:


from mlxtend.frequent_patterns import apriori,association_rules


# ## When min_support is 0.1

# In[8]:


freq_items = apriori(df, min_support=0.1, use_colnames=True)
freq_items.head()


# In[9]:


rules = association_rules(freq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# In[10]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:





# In[ ]:





# ## When min_support is 0.2

# In[11]:


freq_items = apriori(df, min_support=0.2, use_colnames=True, verbose=1)
freq_items.head()


# In[12]:


rules = association_rules(freq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# In[13]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


#  + when min support be 0.2, the model will be good because support and confidence morethan the min support at 0.1.
#  so chance of watching movie is more

# In[ ]:




