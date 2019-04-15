#!/usr/bin/env python
# coding: utf-8

# # Question 1-3
# 

# ## Importing libraries

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


# ### Plotter for ellipse

# In[2]:


from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width = height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    clusters = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=clusters, s=10, cmap='viridis')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10)
    
    w_factor = 0.4 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    return clusters


# ### Piechart plotter
# 

# In[3]:


def pieplotter(labels, sizes):
    plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='c', edgecolor='k')
    
    colors = ['lightcoral','gold', 'yellowgreen','lightskyblue', 'red','blue']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend( loc = 'best', labels=['%s, %1.1f %%' % (l, 100*s/sum(sizes)) for l, s in zip(labels, sizes)])
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ### Reading the data set

# In[4]:


dataSet = pd.read_csv("./../output_data/q-1-1_a_linear.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[5]:


K = 5


# ### GMM object creatition

# In[6]:


gmm = GaussianMixture(n_components = K, covariance_type='full', max_iter = 1000, random_state=42)
clusters = plot_gmm(gmm, X[:,0:2])


# ### Fitting the gmm object

# In[7]:


clusters = gmm.fit(X).predict(X)
clusters


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


# ### GMM object creatition

# In[11]:


gmm = GaussianMixture(n_components = K, covariance_type='full', max_iter = 1000, random_state=42)
clusters = plot_gmm(gmm, X[:,0:2])


# ### Fitting the gmm object

# In[12]:


clusters = gmm.fit(X).predict(X)
clusters


# In[13]:


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

# In[14]:


dataSet = pd.read_csv("./../output_data/q-1-1_b_ReLU.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[15]:


K = 5


# ### GMM object creatition

# In[16]:


gmm = GaussianMixture(n_components = K, covariance_type='full', max_iter = 1000, random_state=42)
clusters = plot_gmm(gmm, X[:,0:2])


# ### Fitting the gmm object

# In[17]:


clusters = gmm.fit(X).predict(X)
clusters


# In[18]:


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

# In[19]:


dataSet = pd.read_csv("./../output_data/q-1-1_b_tanh.csv")
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
# X
Y = dataSet[Class]


# ### setting parameters

# In[20]:


K = 5


# ### GMM object creatition

# In[21]:


gmm = GaussianMixture(n_components = K, covariance_type='full', max_iter = 1000, random_state=42)
clusters = plot_gmm(gmm, X[:,0:2])


# ### Fitting the gmm object

# In[22]:


clusters = gmm.fit(X).predict(X)
clusters


# In[23]:


max_sum = 0
for i in range(K):
    clValue,counts = np.unique(dataSet[Class][clusters == i],return_counts=True) 
    pieplotter(clValue, counts)
    print(i,clValue,counts)
    max_sum += max(counts)
purity = max_sum/dataSet.shape[0]
    
purity


# In[ ]:




