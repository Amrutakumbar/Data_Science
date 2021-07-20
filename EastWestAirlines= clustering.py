#!/usr/bin/env python
# coding: utf-8

# # 1]Hierachical Clustering

# ## import the libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# ## load the dataset

# In[2]:


airline=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\EastWestAirlines.csv")
airline.head()


# In[3]:


airline.columns


# In[4]:


columns=['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll',
        'Award?']
airlines=airline[columns]


# In[5]:


input=['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12','Days_since_enroll']
output=['Award?']
x=airlines[input]
y=airlines[output]


# ## Dendrogram

# In[6]:


from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch


# In[7]:


plt.title("Dendrograms")
dendrogram = sch.dendrogram(sch.linkage(airlines, method='ward'))


# ## Clustering

# In[8]:


from sklearn.cluster import AgglomerativeClustering


hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
hc.fit(x)
hc.fit_predict(x)


# ## Predict

# In[9]:


y_predict=hc.fit_predict(airlines[['Award?']])
y_predict


# ## ACCURACY

# In[10]:


import sklearn.metrics as sm

sm.accuracy_score(y_predict,hc.fit_predict(x))


# + good accuracy i.e 77%

# In[ ]:





# # 2]KMeans

# ## import the libraries

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset

# In[12]:


airline=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\EastWestAirlines.csv")
airline.head()


# ## Sum of square error
# 

# In[13]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airline_df = scaler.fit_transform(airline.iloc[:,1:11])


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


K_rng= range(1,11)
sse=[]
for K in K_rng:
    km=KMeans(n_clusters=K,random_state=0)
    km.fit(scaled_airline_df)
    sse.append(km.inertia_)
    
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# ## Clusters

# In[16]:


km=KMeans(n_clusters=5)
km
km.fit(x)
km.labels_


# ## Predict

# In[17]:


y_predict=km.fit_predict(airline[['Award?']])
y_predict


# In[ ]:





# In[ ]:





# ## 3]DBScan

# ## import the libraries

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the dataset

# In[19]:


airline=pd.read_csv("C:\\Users\\DELL\\Downloads\\Assignment 7.Clustering\\EastWestAirlines.csv")
airline.head()


# In[20]:


airline.drop(['ID#'],axis=1,inplace=True)


# In[21]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_airline_df = scaler.fit_transform(airline.iloc[:,1:11])


# ## model fitting

# In[22]:


from sklearn.cluster import DBSCAN


# In[23]:


model=DBSCAN(eps=0.20,min_samples=15)
model.fit(x)


# In[24]:


model.labels_


# ## cluster

# In[25]:


cl=pd.DataFrame(model.labels_,columns=['cluster'])
cl.head()


# In[26]:


pd.concat([airlines,cl],axis=1).head()


# In[ ]:




