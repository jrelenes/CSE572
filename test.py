#!/usr/bin/env python
# coding: utf-8
#! /usr/bin/env python3
# In[92]:


import csv
from csv import reader
import datetime
import pickle
from sklearn.decomposition import PCA
import pickle_compat
import numpy as np

pickle_compat.patch()

#import pickle_compat
#pickle_compat.patch()


# In[93]:


def ReadXlsx():
    
    testArray = []
    with open('test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            testArray.append(row)
            
    return testArray


# In[94]:


test = ReadXlsx()


# In[95]:


testValues = []
for i in test:
    
    arr = []
    
    for j in i:
        
        arr.append(float(j))
        
    testValues.append(arr)
    
#print(testValues[0])
#print(testValues[-1])
#print(len(testValues))


# In[96]:


def Tester(array):
    
    predictedLabels = []
    array = np.array(array)
    X_test = array

    pca = PCA(n_components=0.95)# adjust yourself
    pca.fit(X_test)


    X_t_test = pca.transform(X_test)

    
    with open ('model_pickle', 'rb') as filename:
    	modelRetrieved = pickle.load(filename)
    	prediction = modelRetrieved.predict(X_t_test)
    	predictedLabels = prediction
    
    return predictedLabels
        


# In[97]:


labels = Tester(testValues)


# In[98]:


#print(labels)
np.savetxt('Results.csv', labels, delimiter=',')


# In[ ]:





# In[ ]:




