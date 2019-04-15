#!/usr/bin/env python
# coding: utf-8

# # Question 1 Part-1

# ## Importing libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import math as mt
import codecs, json 
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ## Utility Functions

# NeuralNetwork class

# In[16]:


class NeuralNetwork:
    def __init__(self,input_size,output_size, node_list , activation = "sigmoid"):
        self.input_LS = input_size
        self.hidden_LS = node_list
        self.output_LS = output_size
        self.No_HLayers = len(node_list) 
        self.W = np.empty(self.No_HLayers + 1, dtype = object)
        self.B = np.empty(self.No_HLayers + 1, dtype = object)
        self.H_in = np.empty(self.No_HLayers + 1, dtype = object)
        self.H_out = np.empty(self.No_HLayers + 2, dtype = object)
        self.w_grad = np.empty(self.No_HLayers + 1, dtype = object)
        self.b_grad = np.empty(self.No_HLayers + 1, dtype = object)
        bound = np.sqrt(1./self.input_LS)
        self.W[0] = np.random.uniform(-bound,bound,(self.input_LS, self.hidden_LS[0]))
        self.B[0] = np.random.uniform(-bound,bound,(1, self.hidden_LS[0]))
        for i in range(1, len(self.W)-1):
            self.W[i] = np.random.uniform(-bound,bound,(self.hidden_LS[i-1], self.hidden_LS[i]))
            self.B[i] = np.random.uniform(-bound,bound,(1, self.hidden_LS[i]))  
        self.W[len(self.W)-1] = np.random.uniform(-bound,bound,(self.hidden_LS[len(self.W)-2],self.output_LS))
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
        return self.sigmoid(z) *(1-self.sigmoid (z))
    
    def linear(self, z):
        return z
    
    def delta_linear(self,z):
        return np.ones(z.shape)

    
    def softmax(self, y_hat):
        tmp = y_hat - y_hat.max(axis=1).reshape(-1, 1)
        exp_tmp = np.exp(tmp)
        return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)
    
    
    def forwardprop(self, X):
        self.H_out[0] = X
        for i in range(0,len(self.W)):
            self.H_in[i] = np.dot(self.H_out[i], self.W[i]) + self.B[i]
            if self.activation == "sigmoid":
                self.H_out[i + 1] = self.sigmoid(self.H_in[i])
            elif self.activation == "ReLU": 
                self.H_out[i + 1] = self.ReLU(self.H_in[i])
            elif self.activation == "tanh": 
                self.H_out[i + 1] = self.tanh(self.H_in[i])
            elif self.activation == "linear": 
                self.H_out[i + 1] = self.linear(self.H_in[i])
#             print("h_out[",i + 1,"].shape",self.H_out[i+1].shape)
#             print("h_out[",i + 1,"]",self.H_out[i+1])
#         print("h_out[",len(self.W)/2,"].shape",self.H_out[int(len(self.W)/2)].shape)
#         print("h_out[",len(self.W)/2,"]",self.H_out[int(len(self.W)/2)])
            
        self.yHat = self.H_out[len(self.W)]
        return self.yHat , self.H_out[int(len(self.hidden_LS)//2) + 1]

    def backprop(self, X, yHat, error, LR = 0.01):  
        m = X.shape[0]
        err2 = error
        if self.activation == "sigmoid":
            err2 *= self.delta_sigmoid(yHat)
        elif self.activation == "ReLU": 
            err2 *= self.delta_ReLU(yHat)
        elif self.activation == "tanh": 
            err2 *= self.delta_tanh(yHat)
        elif self.activation == "linear": 
            err2 *= self.delta_linear(yHat)
        for i in range(len(self.w_grad) - 1, -1, -1):
            self.w_grad[i] = LR * np.dot(self.H_out[i].T,err2)/m
            self.b_grad[i] = LR * np.average(err2, axis=0)
            err2 = np.dot(err2, self.W[i].T)
            if self.activation == "sigmoid":
                err2 *= self.delta_sigmoid(self.H_out[i])
            elif self.activation == "ReLU": 
                err2 *= self.delta_ReLU(self.H_out[i])
            elif self.activation == "tanh": 
                err2 *= self.delta_tanh(self.H_out[i]) 
            elif self.activation == "linear": 
                err2 *= self.delta_linear(self.H_out[i])
#         self.w_grad[0] = LR * np.dot(self.H_out[0].T, err2)/m
#         self.b_grad[0] = LR * np.average(err2, axis=0)

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
            res_list = np.empty([0, self.hidden_LS[int(len(self.hidden_LS)//2)]])
            for XL,yL in zip(XList, yList):
                y_hat , result = self.forwardprop(XL)
                error = yL - y_hat
                res_list = np.concatenate((res_list, result), 0)
                self.backprop(XL, y_hat, error, LR)
                self.grad_update()
                y_hatList += list(y_hat)
                print("epoch: ",i ,"iteration: ",itr)
                itr += 1
            if cost_check:
#                 y_hatList = np.clip(y_hatList, 0.00001, 0.99999)
                cost.append(mean_squared_error(y,y_hatList))
                print("cost[",i,"]: " , cost[i])
        if savefile:
            self.save_metaData(self.activation)
        return cost, res_list
    
    def predict(self, X, loadfile = False):
        if loadfile:
            self.load_metaData(self.activation) 
        y_hat = self.forwardprop(X)
        return np.argmax(y_hat, axis=1)
        
            


# In[17]:


def save_to_csv(res_list,filename):
    resDF = pd.DataFrame(res_list)
    resDF[Class] = y
    resDF.to_csv("./../output_data/"+filename+".csv")


# ### Reading the Data Set
# 

# In[18]:


output_file = "./../input_data/data.csv"
dataSet = pd.read_csv(output_file)
Class = "xAttack"
columns = list(dataSet.columns)
X = dataSet[columns[:-1]].values
y = dataSet[[Class]].values
X = StandardScaler().fit_transform(X)
# X, y


# In[ ]:





# In[104]:


node_list = [14]
NN = NeuralNetwork(X.shape[1] , X.shape[1], node_list, activation = "linear")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile=False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.1, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[105]:


costs, res_list = NN.fit(X, X, 100, 0.01, cost_check=True, batch = True, batch_size = 512)


# In[106]:


save_to_csv(res_list, "q-1-1_a_linear")


# ### Cost vs epochs

# In[107]:


plt.plot(costs)


# In[ ]:





# In[124]:


node_list = [25,18,14,18,25]
NN = NeuralNetwork(X.shape[1] , X.shape[1], node_list, activation = "sigmoid")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile=False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.1, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[125]:


costs, res_list = NN.fit(X, X, 100, 0.01, cost_check=True, batch = True, batch_size = 512)


# In[110]:


save_to_csv(res_list, "q-1-1_b_sigmoid")


# ### Cost vs epochs

# In[111]:


plt.plot(costs)


# In[ ]:





# In[120]:


node_list = [14]
NN = NeuralNetwork(X.shape[1] , X.shape[1], node_list, activation = "ReLU")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile=False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.1, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[121]:


costs, res_list = NN.fit(X, X, 100, 0.01, cost_check=True, batch = True, batch_size = 512)


# In[122]:


save_to_csv(res_list, "q-1-1_b_ReLU")


# ### Cost vs epochs

# In[123]:


plt.plot(costs)


# In[ ]:





# In[116]:


node_list = [25,18,14,18,25]
NN = NeuralNetwork(X.shape[1] , X.shape[1], node_list, activation = "tanh")


# ### Fitting the model
# #### NN.fit(Feature set, output class, epoch = 10, Learning Rate = 0.01, cost_check = False, batch = False, batch_size = -1, savefile=False)
# * For best results use NN.fit(Feature set, output class, epoch = 1000, Learning Rate = 0.1, cost_check=True, batch = True, batch_size = 500, savefile=True)

# In[117]:


costs, res_list = NN.fit(X, X, 100, 0.01, cost_check=True, batch = True, batch_size = 512)


# In[118]:


save_to_csv(res_list, "q-1-1_b_tanh")


# ### Cost vs epochs

# In[119]:


plt.plot(costs)


# In[ ]:




