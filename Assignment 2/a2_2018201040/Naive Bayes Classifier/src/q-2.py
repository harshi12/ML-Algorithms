
# coding: utf-8

# # Question 2

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

eps = np.finfo(float).eps


# ## Utility Functions

# Graph Plotter

# In[ ]:


def plotter(label_x, label_y, title, x_axis, y_axis, mark='', colr = 'blue'):
    plt.figure(num=None, figsize=(6, 4), dpi=175, facecolor='w', edgecolor='k')
    # plotting the points  
    plt.plot(x_axis, y_axis, marker = mark, color = colr, label = 'Error rate') 
    # naming the x axis 
    plt.xlabel(label_x) 
    # naming the y axis 
    plt.ylabel(label_y) 

    # giving a title to my graph 
    plt.title(title) 
    plt.grid(True)
    # function to show the plot 
    plt.show()


# Mean and Standared Deviation of all numerical attributes

# In[ ]:


def mean_stddev(df):
    meanst_no = {}
    meanst_yes = {}
    for attr in numerical[:-1]:
        mean = df[attr][df[Class] == ClassList[0]].mean()
        stddev = df[attr][df[Class] == ClassList[0]].std()
        meanst_no[attr] = (mean, stddev)
        mean = df[attr][df[Class] == ClassList[1]].mean()
        stddev = df[attr][df[Class] == ClassList[1]].std()
        meanst_yes[attr] = (mean, stddev)
    return [meanst_no, meanst_yes] 


# Gaussian Distribution Function for numerical attributes

# In[ ]:


def gaussian_distribution(x, mean, stdev):
    E = mt.exp(-(mt.pow(x-mean,2)/(2*mt.pow(stdev,2))))
    px = (1 / (mt.sqrt(2*mt.pi) * stdev)) * E
    return px


# Distribution plotter

# In[ ]:


def plot_value_vs_freq(df, attribute):
    frequency = df[attribute].value_counts()
    frequency.sort_index(inplace=True)
    index_list = frequency.index.tolist()
    frequency_list = frequency.values.tolist()
    prob_list = []
    for i in index_list:
        prob_list.append(gaussian_distribution(i, df[attribute].mean() , df[attribute].std()))
    plotter("Frequency" ,attribute, 'Frequency Distribution', index_list, frequency_list, '.', 'blue')
    plotter("Gaussian Probability" ,attribute, 'Probability Distribution', index_list, prob_list, '+', 'green')
    print("==============================================================================================================")
    
  
    


# Distribution Finder Function

# In[ ]:


def distribution_finder(df):
    for attr in numerical[:-1]:
        plot_value_vs_freq(df, attr)


# Prior

# In[ ]:


def prior(df, row, attr):
    den = len(df)
    num = len(df[df[attr] == row[attr]])
    return num/(den + eps)


# Evidence

# In[ ]:


def evidence(df, row):
    den = len(df)
    num_no = len(df[df[Class] == ClassList[0]])
    num_yes = len(df[df[Class] == ClassList[1]])
    evidence_no = num_no/(den + eps)
    evidence_yes = num_yes/(den + eps)
    return evidence_no, evidence_yes


# Likelihood

# In[ ]:


def likelihood(df, row, attr):
    den = len(df[df[attr] == row[attr]])
    num_no = len(df[df[attr] == row[attr]][df[Class] == ClassList[0]])
    num_yes = len(df[df[attr] == row[attr]][df[Class] == ClassList[1]])
    likelihood_no = num_no/(den + eps)
    likelihood_yes = num_yes/(den + eps)    
    return likelihood_no, likelihood_yes


# Probability Function for calagorical attributes

# In[ ]:


def cat_probability(df, row, attribute):
    li_no, li_yes = likelihood(df, row, attribute)
    pr = prior(df, row, attribute)
    e_no, e_yes = evidence(df, row)
    prob_no = (li_no*pr)/(e_no + eps)
    prob_yes = (li_yes*pr)/(e_yes + eps)
    return np.log2(prob_no), np.log2(prob_yes)


# Probability Function for Numerical attributes

# In[ ]:


def num_probability(meanst_no_yes, row, attr):
    prob_no = gaussian_distribution(row[attr], meanst_no_yes[0][attr][0], meanst_no_yes[0][attr][1])
    prob_yes = gaussian_distribution(row[attr], meanst_no_yes[1][attr][0], meanst_no_yes[1][attr][1])
    return np.log2(prob_no), np.log2(prob_yes)


# Function for calculating accuracy 

# In[ ]:


def accuracy(true_positive , true_negative , false_positive, false_negative):
    return ((true_positive + true_negative)*100)/(true_positive + true_negative + false_positive + false_negative + eps)


# Validation Function for validating the data set that returns accuracy

# In[ ]:


def validation(trainingSet, validationSet):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    meanst_no_yes = mean_stddev(trainingSet)
    for index, row in validationSet.iterrows():
        prob_no = 0
        prob_yes = 0
        for attr in catagorical[:-1]:
            lprob_no, lprob_yes = cat_probability(trainingSet, row, attr)
            prob_no += lprob_no
            prob_yes += lprob_yes
        for attr in numerical[:-1]:
            lprob_no, lprob_yes = num_probability(meanst_no_yes, row, attr)
            prob_no += lprob_no
            prob_yes += lprob_yes
        e_no, e_yes = evidence(trainingSet, row)
        prob_no += np.log2(e_no)
        prob_yes += np.log2(e_yes)
        print(row)
        print("====================================================================")
        if prob_yes > prob_no:
            if row[Class] == ClassList[1]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if row[Class] == ClassList[0]:
                true_negative += 1
            else:
                false_negative += 1
    return true_positive , true_negative , false_positive, false_negative


# ### Reading the Data Set

# In[ ]:


randomDataSet = dataSet = pd.read_csv("./../input_data/LoanDataset/data.csv")
randomDataSet = dataSet.sample(frac=1).reset_index(drop=True)
Class = "output"
columns = ["id","age", "year_of_experience", "annual_income","zipcode","family_size","average_spending","education_level","mortgage_value",Class,"have_security_acc","have_cd_acc","use_ib","use_credit_card"]
randomDataSet.columns = columns
randomDataSet = randomDataSet[["age", "year_of_experience", "annual_income","zipcode","family_size","average_spending","education_level","mortgage_value",Class,"have_security_acc","have_cd_acc","use_ib","use_credit_card"]]
trainingSet, validationSet = np.split(randomDataSet, [int(0.8*len(randomDataSet))])
catagorical = ['have_security_acc','have_cd_acc','use_ib','use_credit_card',Class]
numerical = ['age','year_of_experience','annual_income','zipcode','family_size','average_spending','education_level','mortgage_value',Class]
clvalue,counts = np.unique(randomDataSet[Class],return_counts=True) 
ClassList = list(clvalue)


# ### Finding the Frequency Distribution of different attributes

# In[ ]:


distribution_finder(randomDataSet)


# ### Validating the data set

# In[ ]:


true_positive, true_negative, false_positive, false_negative = validation(trainingSet, validationSet)


# ### Calculating the accuracy for validation set

# In[ ]:


accuracy(true_positive , true_negative , false_positive, false_negative)


# #### TESTING function resturns list of output

# In[ ]:


def TESTING_algo(trainingSet, testingSet):
    res = []
    meanst_no_yes = mean_stddev(trainingSet)
    for index, row in testingSet.iterrows():
        prob_no = 0
        prob_yes = 0
        for attr in catagorical[:-1]:
            lprob_no, lprob_yes = cat_probability(trainingSet, row, attr)
            prob_no += lprob_no
            prob_yes += lprob_yes
        for attr in numerical[:-1]:
            lprob_no, lprob_yes = num_probability(meanst_no_yes, row, attr)
            prob_no += lprob_no
            prob_yes += lprob_yes
        e_no, e_yes = evidence(trainingSet, row)
        prob_no += np.log2(e_no)
        prob_yes += np.log2(e_yes)
        if prob_yes > prob_no:
            res.append(ClassList[1])
        else:
            res.append(ClassList[0])
    return res


# ### TEST YOUR DATASET HERE....
# 

# In[ ]:


def test(arr):
    length = len(arr)
    global columns
    if length == 2:
        testSetPath = arr[1]
        testingData = pd.read_csv(testSetPath, header=None)
        testingData.columns = columns
        res = TESTING_algo(trainingSet, testingData)
        testingData['RESULT'] = res
        testingData.to_csv("./../output_data/q-2_data_output.csv",index = False)
    return testingData


# In[ ]:


test(sys.argv)


# ## Observation:
# #### 1) The distribution was identified by plotting graphs for each of the column against the frequencies.
# #### 2) Except mortgage value, all the columns follow the Gaussian Distribution.
# #### 3) Since the mortgage column did not have the gaussian distribution, the accuracy is lesser than it should be.
