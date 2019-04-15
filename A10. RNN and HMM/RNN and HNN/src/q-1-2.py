#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import randint



# In[2]:


def feature_extraction(df):
    open_price = np.array(df['open'])
    avg_price = np.array(df['avg'])
    
    frac_avg = (open_price - avg_price) / avg_price
    return frac_avg.reshape(-1,1)


# In[3]:


def plotter(days, actual, predicted):
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


# In[16]:


def _get_most_probable_outcome(day_index, n_latency_days, _possible_outcomes, hmm):
    previous_data_start_index = max(0, day_index - n_latency_days)
    previous_data_end_index = max(0, day_index - 1)
#     print("index: ",previous_data_start_index, previous_data_end_index)
    previous_data = test_data.iloc[previous_data_start_index:previous_data_end_index]
    previous_data_features = feature_extraction(previous_data)
#     print("previous_data_features: ",previous_data_features)
    outcome_score = []
    for possible_outcome in _possible_outcomes:
        total_data = np.row_stack((previous_data_features, possible_outcome))
        outcome_score.append(hmm.score(total_data))
    most_probable_outcome = _possible_outcomes[np.argmax(outcome_score)]

    return most_probable_outcome

def predict_on_day(day_index, n_latency_days, possible_outcomes, hmm):
    predicted_frac_avg = _get_most_probable_outcome(day_index, n_latency_days, possible_outcomes, hmm)
    avg_price = test_data.iloc[day_index]['avg']
    return avg_price * (1 + predicted_frac_avg)

def predict_for_next_n_days(test_data, n_days, n_latency_days, hmm, n_steps_frac_avg = randint(20,80)):
    frac_avg_range = np.linspace(-0.1, 0.1, n_steps_frac_avg)
    possible_outcomes = np.array(list(itertools.product(frac_avg_range)))

    _test_data = test_data[0: n_days]
    days = np.array(_test_data['date'], dtype="datetime64[ms]")
    actual_open_prices = _test_data['open']
    predicted_open_prices = [predict_on_day(day_index, n_latency_days, possible_outcomes, hmm) for day_index in tqdm(range(n_days))]

    plotter(days, actual_open_prices, predicted_open_prices)
    
    return np.mean((np.array(predicted_open_prices) - np.array(actual_open_prices))**2)


# In[17]:


DataSet = pd.read_csv('./../input_data/GoogleStocks.csv', thousands = ',')
DataSet['avg'] = DataSet[['high','low']].mean(axis = 1)
DataSet = DataSet.iloc[1:,:]
DataSet['date'] = pd.to_datetime(DataSet['date'].astype(str),format='%Y/%m/%d')
DataSet = DataSet.sort_index(axis = 0, ascending = False)


# In[18]:


test_size = 0.20
train_data, test_data = train_test_split(DataSet, test_size = 0.20, shuffle=False)


# In[19]:


hiddenstates = [4,8,12]
timestamp = [20,50,75]


# In[20]:


def HMM_algo():
    for n in hiddenstates:
        for ts in timestamp:
            print("Combination: ",(n, ts))
            hmm = GaussianHMM(n_components = n)
            feature_vector = feature_extraction(train_data)
            hmm.fit(feature_vector)
            mse = predict_for_next_n_days(test_data, 150, ts, hmm)
            print(mse)


# In[21]:


HMM_algo()


# In[ ]:





# In[ ]:




