#!/usr/bin/env python
# coding: utf-8

# # Question 2

# ### Consider the House Price Prediction dataset (Dataset Link). Suppose you need to predict the Sale Price of a house and for the task you want to use a neural network with 3 hidden layers. Write a report on how you would modify your above neural network for such task with proper reasoning.

# <b>For the given data set we need to predict the 'SalePices' of a house which is a continous/real valued attribute, therefore it is a regression task of ML. For this purpose we need to perform following changes to the previous code:</b>
# <br>
# 
# * Data Preprocessing: The given data contains numerical as well as catagorical data so we need to distingush between the two and preprocess the data. The preprocessing involves replacing the NAN/missing values with desired number in case of numerical attribute and encoding the catagorical data by one hot endcoding scheam. This way we can treat the catagorical attribute as numerical attributes. The reason behind this is to make the effect of different values of a catagorical attribute i.e. only the values which are applicable to a perticular data point will be considered rest all will have zero effect on weight matrix
# <br>
# 
# * No need to encode the class label as it a regression problem.
# <br>
# 
# * Change in activation function: In the case of classification problem we used sofmax function on an encoded class labels for the last layer now we need to use the same function in the last output layer as we used in the internal layers.
# <br>
# 
# * As specified in the problem we need to have 3 layers which can we configured easily in the previous code
# <br>
# 
# * Using Mean Square Error for score: Prediction will be real valued closer to the actual value and not exactly the same so we will use Mean Square Error for evaluation of our model.
# 

# # PREPROCESSING DATA

# ## Importing libraries

# In[85]:


import numpy as np
import pandas as pd
import math as mt
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import itertools


# ## Utility Functions
# 

# In[86]:


def make_one_hot(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    
def make_na_median(df, name):
    df[name] = df[name].fillna(df[name].median())
    
def normalize(df, name):
    df[name] = (df[name] - df[name].mean()) / df[name].std()
    
def preprocessing(df):
    df.drop("Id", axis = 1, inplace = True)
    catColumns = df.select_dtypes(include = ['object'])
    numColumns = df.select_dtypes(exclude = ['object'])
    for col in catColumns:
        make_one_hot(df, col)
    for col in numColumns:
        make_na_median(df, col)
        normalize(df, col)
    if "SalePrice" in numColumns:
        classColData = df["SalePrice"]
        df.drop("SalePrice",axis = 1, inplace = True)
        df["SalePrice"] = classColData


# ### Path of different data sets

# In[87]:


trainFilePath = "./../input_data/house-prices-advanced-regression-techniques/train.csv"
testFilePath = "./../input_data/house-prices-advanced-regression-techniques/test.csv"
y_testPath = "./../input_data/house-prices-advanced-regression-techniques/sample_submission.csv"


# ### Reading csv to pandas data frame

# In[88]:


trainDataSet = pd.read_csv(trainFilePath)
testDataSet = pd.read_csv(testFilePath)
y_test = pd.read_csv(y_testPath)


# In[89]:


trainDataSet.describe()


# In[90]:


trainDataSet


# In[91]:


testDataSet


# In[92]:


preprocessing(trainDataSet)
preprocessing(testDataSet)
preprocessing(y_test)


# In[93]:


trainDataSet


# In[94]:


Class = "SalePrice"
columns = list(trainDataSet.columns)
X_train = trainDataSet[columns[:-1]].values
y_train = trainDataSet[[Class]].values
X_test = testDataSet.values

