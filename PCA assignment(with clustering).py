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


wine=pd.read_csv('C:\\Users\\DELL\\Downloads\\Assignment 8.PCA\\wine.csv')
wine.head()


# In[3]:


wine.isnull()


# In[4]:


wine.values


# In[5]:


wine.info()


# In[6]:


wine.describe()


# In[7]:


wine.nunique()


# In[8]:


columns=['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline']
Wine=wine[columns]


# In[9]:


input=['Type', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline']
output=['Alcohol']
x=Wine[input]
y=Wine[output]


# ## Dendrogram

# In[10]:


from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch


# In[11]:


plt.title("Dendrograms")
dendrogram = sch.dendrogram(sch.linkage(Wine, method='ward'))


# ## Clustering

# In[12]:


from sklearn.cluster import AgglomerativeClustering


hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
hc.fit(x)
hc.fit_predict(x)


# ## Predict

# In[13]:


y_predict=hc.fit_predict(Wine[['Alcohol']])
y_predict


# ## ACCURACY

# In[14]:


import sklearn.metrics as sm

sm.accuracy_score(y_predict,hc.fit_predict(x))


# In[ ]:





# In[ ]:





# ## 2]KMeans

# In[15]:


input=['Type', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline']
output=['Alcohol']
x=Wine[input]
y=Wine[output]


# ## Standardization

# In[16]:


# Normalization function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_wine_df = scaler.fit_transform(wine.iloc[:,:])


# ## Sum of square error

# In[17]:


from sklearn.cluster import KMeans


# In[18]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_wine_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#  + from elbow method, easily we can observe no. clusters should be 3

# ## Clusters

# In[19]:


km=KMeans(n_clusters=3)
km
km.fit(x)
km.labels_


# ## Predict

# In[20]:


y_predict=hc.fit_predict(Wine[['Alcohol']])
y_predict


# In[ ]:





# # 3]PCA

# In[21]:


#import the libraries
#load the data


# In[22]:


wine.keys()


# In[23]:


wine.shape


# In[24]:


x.shape


# In[25]:


y.shape


# ## Scaling

# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


scaler=StandardScaler()
scaler.fit(wine)
scaled_data=scaler.transform(wine)


# # create PCA

# In[28]:


from sklearn.decomposition import PCA


# In[29]:


pca = PCA()
pca_values = pca.fit_transform(scaled_data)


# ## Visualisation

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


plt.figure(figsize=(8,6))
plt.scatter(pca_values[0:1],pca_values[1:2],cmap="plasma")
plt.xlabel('PC1')
plt.ylabel('PC2')


# ## Varience

# In[32]:


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# In[33]:


# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1


# In[34]:


pca.components_


# In[35]:


# Variance plot for PCA components obtained 
plt.plot(var1,color="red")


# In[36]:


import seaborn as sns


# In[37]:


#heatmap
Wine=pd.DataFrame(pca.components_)
sns.heatmap(Wine,cmap='plasma')


# In[ ]:




