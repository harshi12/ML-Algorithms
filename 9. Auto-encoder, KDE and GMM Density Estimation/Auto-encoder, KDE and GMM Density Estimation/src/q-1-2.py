#!/usr/bin/env python
# coding: utf-8

# # Question 1-2

# ## Importing libraries

# In[1]:


from copy import deepcopy
import numpy as np
import pandas as pd
import math as mt
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

eps = np.finfo(float).eps


# ### Random initial position for K clusters centers

# In[2]:


def initialize(X, K, c):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    centroids = np.random.randn(K,c)*std + mean
    return centroids


# ### K-Means Alogo

# In[3]:


def K_means(K, X):
    n = X.shape[0]
    c = X.shape[1]
    centroids = initialize(X, K, c)
    centers_new = deepcopy(centroids) # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n,K))
    error = 1
    itr = 0
    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        itr += 1
        print("Iteration: ", itr)
        # Measure the distance to every center
        for i in range(K):
            distances[:,i] = np.linalg.norm(X - centers_new[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(K):
            centers_new[i] = np.mean(X[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
    return centers_new, clusters


# ### Piechart plotter

# In[4]:


def pieplotter(labels, sizes):
    plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='c', edgecolor='k')
    
    colors = ['lightcoral','gold', 'yellowgreen','lightskyblue', 'red','blue']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend( loc = 'best', labels=['%s, %1.1f %%' % (l, 100*s/sum(sizes)) for l, s in zip(labels, sizes)])
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ### Reading the data set

# In[5]:


dataSet = pd.read_csv("./../output_data/q-1-1_a_linear.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[6]:


K = 5


# In[7]:


centers_new, clusters = K_means(K,X)


# In[8]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True)  
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:





# In[ ]:





# ### Reading the data set

# In[9]:


dataSet = pd.read_csv("./../output_data/q-1-1_b_sigmoid.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[10]:


K = 5


# In[11]:


centers_new, clusters = K_means(K,X)


# In[12]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True)  
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:





# ### Reading the data set

# In[13]:


dataSet = pd.read_csv("./../output_data/q-1-1_b_ReLU.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[14]:


K = 5


# In[15]:


centers_new, clusters = K_means(K,X)


# In[16]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True)  
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:





# ### Reading the data set

# In[17]:


dataSet = pd.read_csv("./../output_data/q-1-1_b_tanh.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[18]:


K = 5


# In[19]:


centers_new, clusters = K_means(K,X)


# In[20]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True)  
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:




