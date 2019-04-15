
# coding: utf-8

# # Question 3
# 

# ## import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

eps = np.finfo(float).eps


# ## Utility Functions

# Line Plotter

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


# Scatter Plotter

# In[ ]:


def scatterplot(label_x, label_y, title, x_axis, y_axis, mark='.', colr = 'red'):
    plt.figure(num=None, figsize=(6, 4), dpi=175, facecolor='w', edgecolor='k')
    # plotting the y = 0 line  
    plt.axhline(0, color='black')
    # plotting points as a scatter plot 
    plt.scatter(x_axis, y_axis, label= "stars", color = colr, marker= mark, s=30) 
    # naming the x axis 
    plt.xlabel(label_x) 
    # naming the y axis 
    plt.ylabel(label_y) 
    # giving a title to my graph 
    plt.title(title) 
    plt.grid(True)
    # function to show the plot 
    plt.show()


# Normalising the data set to reduce range of values

# In[ ]:


def normalize(randomDataSet, columns):
    for attr in columns[:-1]:
        randomDataSet[attr] = (randomDataSet[attr] - randomDataSet[attr].min())/(randomDataSet[attr].max() - randomDataSet[attr].min())
    # randomDataSet = (randomDataSet - randomDataSet.mean())/randomDataSet.std()
    # randomDataSet.head()


# Hypothesis function that defines equation of line

# In[ ]:


def hypothesis_hx(theta, row):
    hx = 0
    n = len(theta)
    for i in range(0, n - 1):
        hx += theta[i]*row[i]
    return hx + theta[n - 1]


# Error between calculated value of hypothesis and actul value of y

# In[ ]:


def error(df, theta):
    error_list = []
    for index, row in df.iterrows():
        y = row[Class]
        hx = hypothesis_hx(theta, row)
        error_list.append(hx - y)
    return error_list


# Cost Function

# In[ ]:


def costFunction(df, theta):
    Jtheta = 0
    er = 0
    error_list = error(df, theta)
    m = len(error_list)
    for i in range(0, m):
        er += np.square(error_list[i])
    Jtheta = er/(2*m)
    return Jtheta , error_list


# Derivative function to calculate derivative of cost function

# In[ ]:


def Derivative(df, error_list, theta,j):
    m = len(error_list)
    n = len(theta)
    sum_error = 0
    for i in range(0, m):
        if j == n - 1:
            sum_error += error_list[i]
        else:
            sum_error += error_list[i]*df.iloc[i][j]
    return sum_error/(m)


# Gradient Decent Function

# In[ ]:


def gradientDecent(df, theta, error_list):
    n = len(theta)
    for j in range(0, n):
        theta[j] = theta[j] - alpha*Derivative(df, error_list, theta, j)
    return theta


# Model Fitting Function

# In[ ]:


# Fitting the Model
def modelFitting(trainingSet,iterate):
    global theta
    iteration_list = []
    cost_list = []
    before = 10
    for i in range(iterate):
        cost , error_list = costFunction(trainingSet, theta)
        theta = gradientDecent(trainingSet, theta, error_list)
        iteration_list.append(i)
        cost_list.append(cost)
        print(i,cost)
    return iteration_list,cost_list


# Valdidation Function for validating the data set for error calculation

# In[ ]:


def valdidation(validationSet,theta):
    predicted = []
    actual = []
    for index, row in validationSet.iterrows():
        pred = hypothesis_hx(theta, row)
        ac = row[Class]
        predicted.append(pred)
        actual.append(ac)
        print(pred)
        print(ac)
        print("-----------")
    return predicted , actual


# Mean square error

# In[ ]:


# Mean square error
def MSE(predicted , actual):
    return np.mean((np.array(actual) - np.array(predicted))**2)


# Mean absolute error

# In[ ]:


# Mean absolute error
def MAE(predicted , actual):
    return np.mean(np.abs((np.array(actual) - np.array(predicted))))


# Mean absolute percentage error function

# In[ ]:


# Mean absolute percentage error function
def MAPE(predicted , actual):
    return np.mean(np.abs((np.array(actual) - np.array(predicted))/ np.array(actual))) * 100


# Residual Plotter for verifying correctness of the regression line

# In[ ]:


def ResidualPlotter(validationSet, predicted , actual):
    residual = np.array(actual) - np.array(predicted)
    for attr in validationSet:
        scatterplot(attr,'Residual','Residual-vs-' + attr, validationSet[attr],residual)


# ### Reading the Data Set

# In[ ]:


randomDataSet = dataSet = pd.read_csv("./../input_data/AdmissionDataset/data.csv")
randomDataSet = dataSet.sample(frac = 1).reset_index(drop = True)
Class = "Chance of Admit "
columns = ['Serial No.' , 'GRE Score' , 'TOEFL Score' , 'University Rating' , 'SOP' , 'LOR' , 'CGPA' , 'Research' , Class]
randomDataSet.columns = columns
columns = columns[1:]
randomDataSet = randomDataSet[columns]


# Plot for Data Analysis and model prediction

# In[ ]:


sns.pairplot(dataSet)


# Heat map for corelating the attributes

# In[ ]:


sns.heatmap(dataSet.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)


# ### Normalize Data Set

# In[ ]:


normalize(randomDataSet, columns)


# ### Data Set spliting

# In[ ]:


trainingSet, validationSet = np.split(randomDataSet, [int(0.8*len(randomDataSet))])


# ### Setting parameters for our model

# In[ ]:


alpha = 0.01
iterate = 1000
theta = np.zeros([trainingSet.shape[1]])


# ### Training the Model

# In[ ]:


iteration_list,cost_list = modelFitting(trainingSet,iterate)


# ### Plotting the Cost vs Iterations

# In[ ]:


plotter('Iterations','Cost','Cost-vs-Iterations',iteration_list,cost_list)


# ### Validating the Model

# In[ ]:


predicted , actual = valdidation(validationSet,theta)


# ### Mean Square Error

# In[ ]:


MSE(predicted , actual)


# ### Mean Absolute Error

# In[ ]:


MAE(predicted , actual)


# ### Mean Absolute Percentage Error

# In[ ]:


MAPE(predicted , actual)


# ### Plotting the Residual 

# In[ ]:


ResidualPlotter(validationSet, predicted , actual)


# #### TESTING function resturns list of output

# In[ ]:


def TESTING_algo(testingSet):
    global theta
    predicted = []
    for index, row in testingSet.iterrows():
        predicted.append(hypothesis_hx(theta, row))
    return predicted


# ### TEST YOUR DATASET HERE....
# 

# In[ ]:


def test(arr):
    length = len(arr)
    global columns
    if length == 2:
        testSetPath = arr[1]
        testingData = pd.read_csv(testSetPath)
        res = TESTING_algo(testingData)
        testingData['RESULT'] = res
        testingData.to_csv("./../output_data/q-3_data_output.csv",index = False)
    return testingData


# In[ ]:


test(sys.argv)


# ### Observations
# #### 1) The error value ( which is calculated as the cost function in the code), becomes constant after 800 iterations.
# #### 2) Since the range of output label is very small, it makes the mean square error even smaller which is why the mean absolute error is greater than MSE.
# #### 3) The CGPA column  gives the best resdiual error distribution.
# 
