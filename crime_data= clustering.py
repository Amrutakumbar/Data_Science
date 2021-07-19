#!/usr/bin/env python
# coding: utf-8

# # 1]Hierachical Clustering

# ## import the libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset

# In[2]:


crime=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\crime_data.csv")
crime.head()


# In[3]:


columns=['Murder', 'Assault','Rape','UrbanPop']
crime1=crime[columns]


# In[4]:


input=['Murder', 'Assault','Rape']
output=['UrbanPop']
x=crime1[input]
y=crime1[output]


# ## Dendrogram

# In[5]:


from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch


# In[6]:


plt.title("Dendrograms")
dendrogram = sch.dendrogram(sch.linkage(crime1, method='ward'))


# ## Clustering

# In[7]:


from sklearn.cluster import AgglomerativeClustering


hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
hc.fit(x)
hc.fit_predict(x)


# ## Predict

# In[8]:


y_predict=hc.fit_predict(crime1[['UrbanPop']])
y_predict


# # ACCURACY

# In[9]:


import sklearn.metrics as sm

sm.accuracy_score(y_predict,hc.fit_predict(x))


# In[ ]:





# In[ ]:





# # 2]KMeans

# ## import the libraries

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset

# In[11]:


crime=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\crime_data.csv")
crime.head()


# In[12]:


columns=['Murder', 'Assault','Rape','UrbanPop']
crime1=crime[columns]


# In[13]:


input=['Murder', 'Assault','Rape']
output=['UrbanPop']
x=crime1[input]
y=crime1[output]


# ## Sum of square error

# In[14]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_crime_df = scaler.fit_transform(crime.iloc[:,1:4])


# In[15]:


from sklearn.cluster import KMeans


# In[16]:


K_rng= range(1,11)
sse=[]
for K in K_rng:
    km=KMeans(n_clusters=K,random_state=0)
    km.fit(scaled_crime_df)
    sse.append(km.inertia_)
    
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# ## Clusters

# In[17]:


km=KMeans(n_clusters=4)
km
km.fit(x)
km.labels_


# ## Predict

# In[18]:


y_predict=km.fit_predict(crime1[['UrbanPop']])
y_predict


# In[ ]:





# In[ ]:





# # 3]DBScan

# ## import the libraries
# 

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset
# 

# In[20]:


crime=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\crime_data.csv")
crime.head()


# In[21]:


crime_c=crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime_c


# In[22]:


columns=['Murder', 'Assault','Rape','UrbanPop']
crime1=crime[columns]


# In[23]:


input=['Murder', 'Assault','Rape']
output=['UrbanPop']
x=crime1[input]
y=crime1[output]


# In[24]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_crime_df = scaler.fit_transform(crime.iloc[:,1:4])


# ## model fitting

# In[25]:


from sklearn.cluster import DBSCAN


# In[26]:


model=DBSCAN(eps=0.25,min_samples=10)
model.fit(x)


# In[27]:


model.labels_


# ## cluster

# In[28]:


cl=pd.DataFrame(model.labels_,columns=['cluster'])
cl.head()


# In[29]:


pd.concat([crime,cl],axis=1).head()


# In[ ]:




