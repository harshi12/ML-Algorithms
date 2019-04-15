
# coding: utf-8

# # Question 1-2

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import math as mt
import sys
import matplotlib.pyplot as plt

eps = np.finfo(float).eps


# ## Utility Functions

# Euclidian Distance

# In[ ]:


def euclidianDistance(row1 , row2):
    distance = 0
    for i in range(0 , row2.shape[0] - 1):
        distance += np.square(row1[i]-row2[i])
    return np.sqrt(distance)


# Manhattan Distance

# In[ ]:


def manhattanDistance(row1 , row2):
    distance = 0
    for i in range(0 , row2.shape[0] - 1):
        distance += abs(row1[i]-row2[i])
    return np.sqrt(distance)


# Minkowski Distance

# In[ ]:


def minkowskiDistance(row1 , row2):
    distance = 0
    for i in range(0 , row2.shape[0] - 1):
        distance += (abs(row1[i]-row2[i]))**3
    return np.sqrt(distance)


# Distance Calculator for row with that in data frame

# In[ ]:


def distanceCalculator(df, row , distUsing = 'euclidian'):
    distance = []
    index = []
    dist = 0
    for i in range(0 ,len(df)):
        if distUsing == 'euclidian':
            dist = euclidianDistance(df.loc[i,:] , row)
        if distUsing == 'manhattan':
            dist = manhattanDistance(df.loc[i,:] , row)
        if distUsing == 'minkowski':
            dist = minkowskiDistance(df.loc[i,:] , row)
        distance.append(dist)
        index.append(i)
    return [distance, index]


# Sorted DF according to Distance

# In[ ]:


def sortedDistanceDF(df , distance_index):
    df['distance'] = distance_index[0]
    df['index'] = distance_index[1]
    df = df.sort_values(['distance','index'])
    return df


# Select first K rows

# In[ ]:


def firstK(df,k):
    return df.head(k)


# Majority element amongst top K rows

# In[ ]:


def majority(df):
    clValue,counts = np.unique(df[Class],return_counts=True)  
    return clValue[np.argmax(counts)]


# K-Nearest Neighbore Function

# In[ ]:


def KNN_algo(df, row, k, distUsing = 'euclidian'):
    distance_index = distanceCalculator(df, row, distUsing)
    df = sortedDistanceDF(df , distance_index)
    df = firstK(df,k)
    predict = majority(df);
    return predict


# Accuracy

# In[ ]:


def accuracy(confusion):
    correct = 0
    total = 0
    for i in range(3):
        for j in range(3):
            if i==j:
                correct += confusion[i][j]
            total += confusion[i][j]
    
    return 100*correct/(float(total))


# Scores Calculator

# In[ ]:


def scores(ClassValue, confusion):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for predict in range(0,3):
        for actual in range(0,3):
            if predict == ClassList.index(ClassValue) and actual == ClassList.index(ClassValue):
                true_positive += confusion[predict][actual]
            elif predict == ClassList.index(ClassValue) and actual != ClassList.index(ClassValue):
                false_positive += confusion[predict][actual]
            elif predict != ClassList.index(ClassValue) and actual == ClassList.index(ClassValue):
                false_negative += confusion[predict][actual]
            else:
                true_negative += confusion[predict][actual]
    return true_positive, true_negative, false_positive, false_negative


# Recall

# In[ ]:


def recall(true_positive , false_negative):
    return true_positive*100/(true_positive +  false_negative+ eps)


# Precision

# In[ ]:


def precision(true_positive , false_positive):
    return true_positive*100/(true_positive +  false_positive + eps)


# F1-score

# In[ ]:


def f1score(recall , prescision):
    return 2/(1/(float(recall)+eps)+1/(float(prescision)+eps))


# Graph Plotter

# In[ ]:


def plotter(x_axis, y_axis):
    plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
    # plotting the points  
    plt.plot(x_axis, y_axis) 
    # naming the x axis 
    plt.xlabel('K') 
    # naming the y axis 
    plt.ylabel('Accuracy') 

    # giving a title to my graph 
    plt.title('Accuracy-vs-K') 
    plt.grid(True)
    # function to show the plot 
    plt.show()


# Validation Function

# In[ ]:


def validation(trainingSet, validationSet, distUsing = 'euclidian'):
    listK = []
    listACC = []
    lastK = int(np.sqrt(len(trainingSet)))
    print("Validating data set using ",distUsing," distance ")
    for k in range(1 , lastK + 1):
        confusion = [[0,0,0],[0,0,0],[0,0,0]]
        for index, row in validationSet.iterrows():
            result = KNN_algo(trainingSet,row,k, distUsing)
            confusion[ClassList.index(result)][ClassList.index(row[Class])] += 1
        print("---------------------------------------------------------------------------------------------------")
        print("                                          FOR K = ", k)
        print("---------------------------------------------------------------------------------------------------")
        print("confusion matrix: ")
        for lst in confusion:
            for i in lst:
                print(i," ",end='')
            print("\n")
        print("----------------------------------------------------------------------")
        print(" ")
        acc = accuracy(confusion)
        print ("ACCURACY: " , acc)
        for ClassValue in ClassList:
            true_positive, true_negative, false_positive, false_negative = scores(ClassValue , confusion)
            rec = recall(true_positive , false_negative)
            pre = precision(true_positive , false_positive)
            f1 = f1score(rec, pre)
            print(" ")
            print("------------Class Value ",ClassValue,"Scores are------------")
            print(" ")
            print("RECALL: ", rec)
            print("PRECISION: ", pre)
            print("F1-score: ", f1)
        print(" ")
        print("--END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END-END--")
        print(" ")
        listK.append(k)
        listACC.append(acc)
    return listK, listACC

        


# ### Reading the Data Set Iris

# In[ ]:


dataSet = pd.read_csv("./../input_data/Iris/Iris.csv", header=None)
randomDataSet = dataSet.sample(frac=1).reset_index(drop=True)
Class = "class"
columns = ['sepal length','sepal width','petal length','petal width', Class]
randomDataSet.columns = columns
trainingSet, validationSet = np.split(randomDataSet, [int(0.8*len(randomDataSet))])
clvalue,counts = np.unique(randomDataSet[Class],return_counts=True) 
ClassList = list(clvalue)
# print(ClassList)
# print(ClassList.index('Iris-virginica'))


# ### Validating the Iris data set with default euclidian Distance

# In[ ]:


listK, listACC = validation(trainingSet, validationSet)


# In[ ]:


plotter(listK, listACC)


# ### Validating the Iris data set with Manhattan Distance

# In[ ]:


listK, listACC = validation(trainingSet, validationSet, 'manhattan')


# In[ ]:


plotter(listK, listACC)


# ### Validating Iris data set with Minkowski Distance

# In[ ]:


listK, listACC = validation(trainingSet, validationSet, 'minkowski')


# In[ ]:


plotter(listK, listACC)


# #### TESTING function resturns list of output

# In[ ]:


def TESTING_algo(trainingSet, testingSet, k, distUsing = 'euclidian'):
    res = []
    for index, row in testingSet.iterrows():
        res.append(KNN_algo(trainingSet,row,k, distUsing))
    return res


# ### TEST YOUR DATASET HERE....
# 

# In[ ]:


def test(arr, K):
    length = len(arr)
    global columns
    if length == 2:
        testSetPath = arr[1]
        testingData = pd.read_csv(testSetPath, header=None)
        testingData.columns = columns
        res = TESTING_algo(trainingSet, testingData, K)
        testingData['RESULT'] = res
        testingData.to_csv("./../output_data/q-1-2_Iris_output.csv",index = False)
    return testingData


# In[ ]:


test(sys.argv, 5)


# ## Observations:
# #### 1) A large variation in predictions take place for small value of K.
# #### 2) Also, Large model bias occurs if we set K to a large value.
# #### 3) So, K should be set to a value large enough to minimize the probability of misclassification and small enough (with respect to the number of cases in the example sample) so that the K nearest points are closer to the query point. 
# #### 4)  Less Accuracy is obtained in the dataset Robot1 because of the noise
# #### 5) The best value for k is 9 or 11 which is correct according to the thumb rule which says that maximum accuracy will be obtained till root of length of training data
# #### 6) The knn algorithm gave almost same plot in euclidean  as well as manhattan distance.
# #### 7) The plot for different k values are fluctuating a lot in case of minkowski
