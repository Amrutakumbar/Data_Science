#!/usr/bin/env python
# coding: utf-8

# ## import the libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset

# In[2]:


book=pd.read_csv('C:\\Users\\DELL\\Downloads\\Assignment 9. Association Rules\\book.csv')
book.head()


# ## EDA

# In[3]:


book.isnull()


# In[4]:


book.info()


# In[5]:


book.columns


# # apriori and association rules

# In[6]:


from mlxtend.frequent_patterns import apriori,association_rules


# ### When min_support is 0.1

# In[7]:


freq_items = apriori(book, min_support=0.1, use_colnames=True, verbose=1)
freq_items.head()


# In[8]:


rules = association_rules(freq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# In[9]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


#  + from above association rule we come to know that -when min_support is 0.1,the confidence for italcook and cookbook is higher than other books
# support=11%
# confidence=100%
# lift=2.3
# so 100% of confidence of buying items italcook with cookbook with minimum lift=2 and support 11%

# In[ ]:





# ### When min_support is 0.2

# In[10]:


freq_items = apriori(book, min_support=0.2, use_colnames=True, verbose=1)
freq_items.head()


# In[11]:


rules = association_rules(freq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# In[12]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


#  + from above association rule we come to know that -when min_support is 0.2,the confidence for childbooks and cookbooks is higher than other books
# support=25% 
# confidence=59-60% 
# lift=1.4
# so less confidence of buying items childbook with cookbook with minimum lift=1 and support 25%

#  + therefore minimun support shoulb be 0.1 than the 0.2

# In[ ]:




