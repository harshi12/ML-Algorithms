#!/usr/bin/env python
# coding: utf-8

# # Question 1-4

# ## Importing libraries

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch


# ### Ploting the clusters in of first two attributes

# In[2]:


# Visualising the clusters
def visualize(X, clusters):
    plt.scatter(X[clusters == 0, 0], X[clusters == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[clusters == 1, 0], X[clusters == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[clusters == 2, 0], X[clusters == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[clusters == 3, 0], X[clusters == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[clusters == 4, 0], X[clusters == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


# ### Dendrogram

# In[3]:


# Using the dendrogram to find the optimal number of clusters
def dendrogram(X):
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'single'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()


# ### Piechart plotter
# 

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


# Importing the dataset
dataSet = pd.read_csv("./../output_data/q-1-1_a_linear.csv")
Class = "xAttack"
X = dataSet.iloc[:,:-1].values


# In[6]:


X


# ### setting parameters

# In[7]:


K = 5


# In[8]:


# dendrogram(X)


# ### AgglomerativeClustering using sklearn

# In[9]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = K, affinity = 'euclidean', linkage = 'single')
clusters = hc.fit_predict(X)


# In[10]:


visualize(X,clusters)


# In[11]:


clusters


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





# In[ ]:





# ### Reading the data set

# In[13]:


# Importing the dataset
dataSet = pd.read_csv("./../output_data/q-1-1_b_sigmoid.csv")
Class = "xAttack"
X = dataSet.iloc[:,:-1].values


# In[14]:


X


# ### setting parameters

# In[15]:


K = 5


# In[16]:


# dendrogram(X)


# ### AgglomerativeClustering using sklearn

# In[17]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = K, affinity = 'euclidean', linkage = 'single')
clusters = hc.fit_predict(X)


# In[18]:


visualize(X,clusters)


# In[19]:


clusters


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





# ### Reading the data set

# In[21]:


# Importing the dataset
dataSet = pd.read_csv("./../output_data/q-1-1_b_ReLU.csv")
Class = "xAttack"
X = dataSet.iloc[:,:-1].values


# In[22]:


X


# ### setting parameters

# In[23]:


K = 5


# In[24]:


# dendrogram(X)


# ### AgglomerativeClustering using sklearn

# In[25]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = K, affinity = 'euclidean', linkage = 'single')
clusters = hc.fit_predict(X)


# In[26]:


visualize(X,clusters)


# In[27]:


clusters


# In[28]:


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

# In[29]:


# Importing the dataset
dataSet = pd.read_csv("./../output_data/q-1-1_b_tanh.csv")
Class = "xAttack"
X = dataSet.iloc[:,:-1].values


# In[30]:


X


# ### setting parameters

# In[31]:


K = 5


# In[32]:


# dendrogram(X)


# ### AgglomerativeClustering using sklearn

# In[33]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = K, affinity = 'euclidean', linkage = 'single')
clusters = hc.fit_predict(X)


# In[34]:


visualize(X,clusters)


# In[35]:


clusters


# In[36]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True) 
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:




