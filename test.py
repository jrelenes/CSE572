#!/usr/bin/env python
# coding: utf-8

# In[44]:


import csv
from csv import reader
import datetime
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pickle_compat
#pickle_compat.patch()
import pandas as pd
import pandas as df
import pickle_compat
pickle_compat.patch()
import pickle
from sklearn.neural_network import MLPClassifier




# In[45]:


def ReadXlsx():
    
    testArray = []
    with open('test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            #print(row)
            testArray.append(row)
            
    return testArray


# In[46]:


test = ReadXlsx()


# In[47]:


testValues = []
for i in test:
    
    arr = []
    
    for j in i:
        
        arr.append(float(j))
        
    testValues.append(arr)

# In[48]:


def Tester(array):
    
    predictedLabels = []
    array = np.array(array)
    X_test = array
    print(X_test.shape)


    pca = PCA(n_components=22)# adjust yourself
    pca.fit(X_test)

    X_t_test = pca.transform(X_test)

    
    with open ('model_pickle', 'rb') as filename:
        modelRetrieved = pickle.load(filename)
        prediction = modelRetrieved.predict(X_t_test)
        predictedLabels = prediction
    # f = open('model_pickle.p', 'wb')   # Pickle file is newly created where foo1.py is
    # modelRetrieved = pickle.load(f)          # dump data to f
    # f.close()
    # prediction = modelRetrieved.predict(X_t_test)
    # predictedLabels = prediction
    
    return predictedLabels
        


# In[49]:


labels = Tester(testValues)


# In[50]:


#print(labels)
np.savetxt('Results.csv', labels, delimiter=',')


# In[ ]:





# In[ ]:




