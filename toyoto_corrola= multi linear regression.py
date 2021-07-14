#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


#read the data
toyoto_corrola=pd.read_csv('C:\\Users\\DELL\\Downloads\\toyoto_corrola.csv')
toyoto_corrola


# In[3]:


toyoto_corrola.head()


# In[4]:


toyoto_corrola.info()


# In[5]:


#missing value
toyoto_corrola.isna().sum()


# In[6]:


#correlation
toyoto_corrola.corr()


# In[7]:


#Format the plot background and scatter plots for all the variables
sns.set_style(style='darkgrid')
sns.pairplot(toyoto_corrola)


# In[8]:


#Build model
import statsmodels.formula.api as smf 
model=smf.ols('Price~KM+HP+Doors+Cylinders+Gears+Weight',data=toyoto_corrola).fit()


# In[9]:


#coefficients
model.params


# In[10]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# In[11]:


#R squared values
(model.rsquared,model.rsquared_adj)


# In[12]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[13]:


list(np.where(model.resid>10))


# In[14]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[15]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[16]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[17]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyoto_corrola)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[18]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# In[19]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[20]:


k = toyoto_corrola.shape[1]
n = toyoto_corrola.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[21]:


#from the above plot,it is evident that data point 956 and 991 are the influencers


# In[22]:


toyoto_corrola[toyoto_corrola.index.isin([956,991])]


# In[23]:


#see the difference between price and other values
toyoto_corrola.head()


# In[24]:


#IMPROVING THE MODEL


# In[25]:


#load new data
tc_new=pd.read_csv('C:\\Users\\DELL\\Downloads\\toyoto_corrola.csv')
tc_new


# In[26]:


tc=tc_new.drop(columns=['Id'])
tc


# In[27]:


#buliding new data
import statsmodels.formula.api as smf 
model=smf.ols('Price~KM+HP+Doors+Cylinders+Gears+Weight',data=tc).fit()


# In[30]:


finaldata=smf.ols('Price~KM+HP+Doors+Cylinders+Gears+Weight',data=tc).fit()


# In[31]:


finaldata.rsquared,finaldata.aic


# In[32]:


#prediction for new data
new_data=pd.DataFrame({"Age_08_04":24,"KM":17016,'HP':90,'Doors':3,'Cylinders':4,'Gears':5,'Weight':1165},index=[1])


# In[33]:


finaldata.predict(new_data)


# In[34]:


finaldata.predict(tc_new.iloc[0:5])


# In[36]:


pred_y=finaldata.predict(tc_new)
pred_y


# In[37]:


model.summary()


# In[ ]:




