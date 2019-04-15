#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler


# In[2]:


def plotter(days, predicted, actual):
    #Visualization
    fig = plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(days, predicted, color='red', label = 'Predicted Stock Price')
    plt.plot(days, actual, color='blue', label = 'Real Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    fig.autofmt_xdate()
    plt.legend()
    plt.show()


# In[3]:


def prepare_data(X, tmstmp):
    X_t=[]
    y_t=[]
    for i in range(tmstmp, X.shape[0]):
        X_t.append(X[i - tmstmp:i,:-1])
        y_t.append(X[i,-1])
    X_t, y_t = np.array(X_t), np.array(y_t)
    y_t = y_t.reshape(y_t.shape[0],1)
    return X_t, y_t


# In[4]:


def LSTM_algo(X_train, nHiddenLayers, cell, optimizer, loss):
    regressor=Sequential()

    regressor.add(LSTM(units = cell, return_sequences=True, input_shape = X_train.shape[1:]))
    regressor.add(Dropout(0.4))
    
    for i in range(nHiddenLayers - 1):
        regressor.add(LSTM(units=cell,return_sequences=True))
        regressor.add(Dropout(0.6))
    
    regressor.add(LSTM(units=cell))
    regressor.add(Dropout(0.4))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer='adam',loss='mean_squared_error')
    
    return regressor


# In[5]:


def combinator():
    for n in layerList:
        for cell in cells:
            for ts in timestamp:  
                print("Combination: ",(n, cell, ts))
                X_train, y_train = prepare_data(training, ts)

                regressor = LSTM_algo(X_train, n, cell, optimizer = 'adam', loss = 'mean_squared_error')

                regressor.fit(X_train,y_train,epochs = epochs,batch_size = 32)

                stacked_validation = np.vstack((training[len(training) - ts : , :],validation))

                X_test, y_test = prepare_data(stacked_validation, ts)

                predicted_y = regressor.predict(X_test)
                stacked_validation = stacked_validation[ts:,:]
                y_true = MMsc.inverse_transform(stacked_validation)[:,-1]
                for i in range(len(predicted_y)):
                    stacked_validation[i,-1] = predicted_y[i]
                y_predicted = MMsc.inverse_transform(stacked_validation)[:,-1]

                plotter(days, y_predicted, y_true)


# In[6]:


DataSet = pd.read_csv('./../input_data/GoogleStocks.csv', thousands = ',')
DataSet = DataSet.iloc[1:,:]
DataSet = DataSet.sort_index(axis=0 ,ascending=False)

DataSet['date'] = pd.to_datetime(DataSet['date'].astype(str),format='%Y/%m/%d')
train_data, test_data = train_test_split(DataSet, test_size = 0.20, shuffle=False)
days = np.array(test_data['date'], dtype="datetime64[ms]")
# DataSet


# In[7]:


DataSet['avg'] = DataSet[['high','low']].mean(axis = 1)
DataSet = DataSet[['volume','avg','open']]
# DataSet


# In[8]:


MMsc = MinMaxScaler(feature_range = (0, 1))
DataSet_scaled = MMsc.fit_transform(DataSet)


# In[9]:


training, validation = np.split(DataSet_scaled, [int(0.8*len(DataSet_scaled))])


# In[10]:


layerList = [2,3]
cells = [30,50,80]
timestamp = [20,50,75]
epochs = 100


# In[11]:


combinator()


# In[ ]:




