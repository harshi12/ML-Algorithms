#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture


# In[2]:


# load the data
digits = load_digits()
# digits


# In[3]:


def createDataSet(digits, n_comp):
   # project the 64-dimensional data to a lower dimension
    pca = PCA(n_components=n_comp, whiten=False)
    data = pca.fit_transform(digits.data) 
    df = pd.DataFrame(data)
    df["target"] = digits.target
    pd.DataFrame(df).to_csv('./../output_data/digits'+ str(n_comp) +'d.csv')
    return pca, data


# In[4]:


def plot_digits(data, row, col):
    fig, ax = plt.subplots(row, col, figsize=(10, 10))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap = 'binary')
        im.set_clim(0, 16)
plot_digits(digits.data, 10, 10)


# In[5]:


def get_mn(y):
    sq = mt.floor(np.sqrt(y))
    for i in range(sq, 1, -1):
        if y % i == 0:
            return int(y/i), i


# In[6]:


def KDE(X):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-1, 1,  num = 30)}
    grid = GridSearchCV(KernelDensity(), params, cv=7)
    grid.fit(X)

    print("best bandwidth: ",grid.best_estimator_.bandwidth)

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    return kde


# In[7]:


def GMM(X):
    interval = 10
    n_components = np.arange(1, 300, interval)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) 
              for n in n_components]
    BIC = [m.bic(X) for m in models]
    plt.plot(n_components, BIC, label='BIC')
    # plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    m = BIC.index(min(BIC))
    print("minimum BIC: " + str(m*interval))
    return models[m]
#     return "minimum BIC: " + str(BIC.index(min(BIC))*interval)


# In[8]:


def drawKDE(kde, pca, n_sample):
    data_new = kde.sample(n_sample, random_state=0)
    digits_new = pca.inverse_transform(data_new)
    
    m, n = get_mn(n_sample)
    plot_digits(digits_new, m, n)
    return digits_new


# In[9]:


def drawGMM(gmm, pca, n_sample):
    print(gmm.converged_)

    data_new, y_new = gmm.sample(n_sample)

    digits_new = pca.inverse_transform(data_new)
    m, n = get_mn(n_sample)
    plot_digits(digits_new, m, n)
    return digits_new


# In[10]:


pca15d, data15d = createDataSet(digits, 15)


# In[11]:


pca30d, data30d = createDataSet(digits, 30)


# In[12]:


pca41d, data41d = createDataSet(digits, 41)


# In[13]:


kde15d = KDE(data15d)
kde15d


# In[14]:


kde30d = KDE(data30d)
kde30d


# In[15]:


kde41d = KDE(data41d)
kde41d


# In[ ]:





# In[16]:


gmm15d = GMM(data15d)
gmm15d


# In[17]:


gmm30d = GMM(data30d)
gmm30d


# In[18]:


gmm41d = GMM(data41d)
gmm41d


# In[ ]:





# In[19]:


drawKDE(kde15d,pca15d,48)


# In[20]:


drawKDE(kde30d,pca30d,48)


# In[21]:


drawKDE(kde41d,pca41d,48)


# In[ ]:





# In[22]:


drawGMM(gmm15d, pca15d, 48)


# In[23]:


drawGMM(gmm30d, pca30d, 48)


# In[24]:


drawGMM(gmm41d, pca41d, 48)


# In[ ]:




