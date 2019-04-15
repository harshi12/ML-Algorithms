#!/usr/bin/env python
# coding: utf-8

# # Question 1

# ### Train and validate your own n-layer Neural Network on the Apparel dataset to predict the class label of a given apparel. You are free to choose the hyper-parameters, training strategy to handle large number of training data (Hint: Batch Size) architecture - number of hidden layers, number of nodes in each hidden layer etc.

# ## Importing libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import math as mt
import codecs, json 
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# ## Utility Functions

# NeuralNetwork class

# In[2]:


class NeuralNetwork:
    def __init__(self,input_size,output_size, No_HL, h_LS , activation = "sigmoid"):
        self.input_LS = input_size
        self.hidden_LS = h_LS
        self.output_LS = output_size
        self.No_HLayers = No_HL 
        self.W = np.empty(self.No_HLayers + 1, dtype = object)
        self.B = np.empty(self.No_HLayers + 1, dtype = object)
        self.H_in = np.empty(self.No_HLayers + 1, dtype = object)
        self.H_out = np.empty(self.No_HLayers + 2, dtype = object)
        self.w_grad = np.empty(self.No_HLayers + 1, dtype = object)
        self.b_grad = np.empty(self.No_HLayers + 1, dtype = object)
        bound = np.sqrt(1./self.input_LS)
        self.W[0] = np.random.uniform(-bound,bound,(self.input_LS, self.hidden_LS))
        self.B[0] = np.random.uniform(-bound,bound,(1, self.hidden_LS))
        for i in range(1, len(self.W)-1):
            self.W[i] = np.random.uniform(-bound,bound,(self.hidden_LS, self.hidden_LS))
            self.B[i] = np.random.uniform(-bound,bound,(1, self.hidden_LS))  
        self.W[len(self.W)-1] = np.random.uniform(-bound,bound,(self.hidden_LS,self.output_LS))
        self.B[len(self.B)-1] = np.random.uniform(-bound,bound,(1, self.output_LS))
        self.activation = activation
        
        
    def save_metaData(self, fileName):
        data = {"no_of_hidden_layers":self.No_HLayers,"no_of_neurons_per_layer":self.hidden_LS,"activation":self.activation,"weights": self.W, "bais": self.B}
        np.save('./../output_data/'+fileName+'.npy',data)

    def load_metaData(self,fileName):
        try:
            data_loaded = np.load('./../output_data/'+fileName+'.npy')
            self.activation = data_loaded[()]["activation"] 
            self.W = data_loaded[()]["weights"] 
            self.B = data_loaded[()]["bais"]
        except:
            pass
        
    def ReLU(self, z):
        return z * (z > 0)

    def delta_ReLU(self, z):
        return 1. * (z > 0)
    
    def tanh(self, z):
        return np.tanh(z)

    def delta_tanh(self, z):
        return 1. - z * z
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def delta_sigmoid(self, z):
        return z * (1 - z)
    
    def softmax(self, y_hat):
        tmp = y_hat - y_hat.max(axis=1).reshape(-1, 1)
        exp_tmp = np.exp(tmp)
        return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)
    
    
    def forwardprop(self, X):
        self.H_out[0] = X
        for i in range(0,len(self.W)):
            self.H_in[i] = np.dot(self.H_out[i], self.W[i]) + self.B[i]
            if i != len(self.W) - 1:
                if self.activation == "sigmoid":
                    self.H_out[i + 1] = self.sigmoid(self.H_in[i])
                elif self.activation == "ReLU": 
                    self.H_out[i + 1] = self.ReLU(self.H_in[i])
                elif self.activation == "tanh": 
                    self.H_out[i + 1] = self.tanh(self.H_in[i])
            else:
                self.H_out[i + 1] = self.softmax(self.H_in[i])
        self.yHat = self.H_out[len(self.W)]
        return self.yHat

    def backprop(self, X, error, LR = 0.01):  
        m = X.shape[0]
        err2 = error
        for i in range(len(self.w_grad) - 1, 0, -1):
            self.w_grad[i] = LR * np.dot(self.H_out[i].T,err2)/m
            self.b_grad[i] = LR * np.average(err2, axis=0)
            err2 = np.dot(err2, self.W[i].T)
            if self.activation == "sigmoid":
                err2 *= self.delta_sigmoid(self.H_out[i])
            elif self.activation == "ReLU": 
                err2 *= self.delta_ReLU(self.H_out[i])
            elif self.activation == "tanh": 
                err2 *= self.delta_tanh(self.H_out[i]) 
        self.w_grad[0] = LR * np.dot(self.H_out[0].T, err2)/m
        self.b_grad[0] = LR * np.average(err2, axis=0)

    def grad_update(self):
        self.W += self.w_grad
        self.B += self.b_grad
        
    def fit(self, X, y, epoch = 10, LR = 0.01, cost_check = False, batch = False, batch_size = -1, savefile = False):
        cost = [] 
        if batch_size == -1:
            batch_size = X.shape[0]
            batch = False
        XList = []
        yList = []
        if batch:
            XList = [X[i : i + batch_size][:] for i in range(0,X.shape[0], batch_size)]
            yList = [y[i : i + batch_size][:] for i in range(0,y.shape[0], batch_size)]    
        else:
            XList.append(X)
            yList.append(y)
        
        for i in range(epoch):
            itr = 0
            y_hatList = []
            for XL,yL in zip(XList, yList):
                y_hat = self.forwardprop(XL)
                error = yL - y_hat
                self.backprop(XL, error, LR)
                self.grad_update()
                y_hatList += list(y_hat)
                print("epoch: ",i ,"iteration: ",itr)
                itr += 1
            if cost_check:
                y_hatList = np.clip(y_hatList, 0.00001, 0.99999)
                cost.append(-np.sum(y * np.log(y_hatList))/y.shape[0])
                print("cost[",i,"]: " , cost[i])
        if savefile:
            self.save_metaData(self.activation)
        return cost
    
    def predict(self, X, loadfile = False):
        if loadfile:
            self.load_metaData(self.activation) 
        y_hat = self.forwardprop(X)
        return np.argmax(y_hat, axis=1)
        
            


# ### Reading the Data Set
# 

# In[3]:


output_file = "./../input_data/Apparel/apparel-trainval.csv"
dataSet = pd.read_csv(output_file)
Class = "label"
columns = list(dataSet.columns)
X = dataSet[columns[1:]].values
y = dataSet[[Class]].values


# ### Spliting the data set for train and test data randomly

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ### OneHotEncoding the class labels

# In[5]:


ohe = OneHotEncoder(n_values=10)
y_train_enc = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()


# ### Neural Network object 
# #### NeuralNetwork(input_size, output_size, No of hidden layers, No of neurons per layer ,  activation function = "sigmoid")
#  * For best results use NeuralNetwork(input_size, output_size, No of hidden layers = 2, No of neurons per layer = 500 ,  activation function = "sigmoid")

# In[6]:


NN = NeuralNetwork(X_train.shape[1] , 10, 2, 500, activation = "sigmoid")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile=False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.1, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[7]:


costs = NN.fit(X_train, y_train_enc, 1000, 0.1, cost_check=True, batch = True, batch_size = 500)


# ### Predicting the test data

# In[8]:


y_hat = NN.predict(X_test)
y_hat


# In[9]:


y_test


# ### Accuracy for sigmoid

# In[10]:


accuracy_score(y_test, y_hat)


# ### Cost vs epochs

# In[11]:


plt.plot(costs)


# ### Neural Network object 
# #### NeuralNetwork(input_size, output_size, No of hidden layers, No of neurons per layer ,  activation function = "sigmoid")
#  * For best results use NeuralNetwork(input_size, output_size, No of hidden layers = 2, No of neurons per layer = 500 ,  activation function = "ReLU")

# In[12]:


NNReLU = NeuralNetwork(X_train.shape[1] , 10, 2, 500, activation = "ReLU")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile = False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.001, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[13]:


costs = NNReLU.fit(X_train, y_train_enc, 1000, 0.001, cost_check=True, batch = True, batch_size = 500)


# ### Predicting the test data

# In[14]:


y_hat = NNReLU.predict(X_test)
y_hat


# In[15]:


y_test


# ### Accuracy for ReLU

# In[16]:


accuracy_score(y_test, y_hat)


# ### Cost vs epochs

# In[17]:


plt.plot(costs)


# ### Neural Network object 
# #### NeuralNetwork(input_size, output_size, No of hidden layers, No of neurons per layer ,  activation function = "sigmoid")
#  * For best results use NeuralNetwork(input_size, output_size, No of hidden layers = 2, No of neurons per layer = 500 ,  activation function = "tanh")

# In[18]:


NNtanh = NeuralNetwork(X_train.shape[1] , 10, 2, 500, activation = "tanh")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1,savefile = False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.01, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[19]:


costs = NNtanh.fit(X_train, y_train_enc, 1000, 0.005, cost_check=True, batch = True, batch_size = 500)


# ### Predicting the test data

# In[20]:


y_hat = NNtanh.predict(X_test)
y_hat


# In[21]:


y_test


# ### Accuracy for tanh

# In[22]:


accuracy_score(y_test, y_hat)


# ### Cost vs epochs

# In[23]:


plt.plot(costs)


# ## TEST YOUR DATA HERE...
# 

# In[24]:


testDataSetPath = "./../input_data/Apparel/apparel-test.csv"


# In[50]:


testDataSet = pd.read_csv(testDataSetPath)


# In[51]:


y_pred= NNReLU.predict(testDataSet)
y_pred


# In[52]:


predDF = pd.DataFrame(data = y_pred)


# In[53]:


predictedDataSetPath = "./../output_data/2018201040_prediction.csv"
predDF.to_csv(predictedDataSetPath)


# In[ ]:




