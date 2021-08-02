#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[1]:


import pandas as pd
import numpy as np


# # load the dataset

# In[2]:


book=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 10. Recommendation systems\\book1.csv")
book


# # EDA

# In[3]:


book.info()


# In[4]:


#Count of duplicated rows
book[book.duplicated()].shape


# In[5]:


#Print the duplicated rows
book[book.duplicated()]


# In[6]:


book1=book.drop_duplicates()
book1


# In[7]:


book1.isnull().sum() #null values


# In[8]:


#number of unique users in the dataset
len(book1.user_id.unique())


# In[9]:


len(book1.Book_title.unique())


# In[10]:


book.shape


# In[11]:


book.columns


# In[12]:


book.mean


# In[13]:


# lets make a pivot table in order to make rows are users and columns are movies. And values are rating
Book = book.pivot_table(index = ["user_id"],columns = ["Book_title"],values = ["Book_rating"]).reset_index(drop=True)
Book.head(10)


# In[14]:


Book1=Book.fillna(0,inplace=True)
Book1


# # visualisation
# 

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#average of ratings of books
book_mean=book.groupby('Book_title')['Book_rating'].mean().sort_values(ascending=False).head()
book_mean


# In[17]:


#counts of ratings of books
book_count=book.groupby('Book_title')['Book_rating'].count().sort_values(ascending=False).head()
book_count


# In[18]:


#histogram


# In[19]:


sns.distplot(book['Book_rating'].dropna(), kde=False)


# # Calculating Cosine Similarity between Users

# In[20]:


from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


user_sim = 1 - pairwise_distances( Book.values,metric='cosine')
user_sim


# In[22]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df


# In[23]:


#Set the index and column names to user ids 
user_sim_df.index = book1.user_id.unique()
user_sim_df.columns = book1.user_id.unique()


# In[24]:


user_sim_df.iloc[0:5, 0:5]


# In[25]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[26]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[27]:


book1[(book1['user_id']==276729) | (book1['user_id']==276726)]


# In[28]:


user_1=book1[book1['user_id']==276729]


# In[29]:


user_2=book1[book1['user_id']==276798]


# In[30]:


user_2.Book_title


# In[31]:


user_1.Book_title


# In[32]:


pd.merge(user_1,user_2,on='Book_title',how='outer')


# In[ ]:




