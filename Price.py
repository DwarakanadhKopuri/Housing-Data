
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams


# In[2]:


from pandas.core import datetools


# In[3]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[4]:


print(boston.keys())


# In[5]:


print(boston.data.shape)


# In[6]:


print(boston.feature_names)


# In[7]:


print(boston.DESCR)


# In[9]:


bos = pd.DataFrame(boston.data)
print(bos.head())


# In[10]:


bos.columns = boston.feature_names
print(bos.head())


# In[12]:


print(boston.target.shape)


# In[13]:


bos['PRICE'] = boston.target
print(bos.head())


# In[14]:


print(bos.describe())


# In[15]:


X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[17]:


import sklearn.cross_validation


# In[18]:


X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


# In[20]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

