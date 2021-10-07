#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import pandas as df
import numpy
import math
import datetime
from datetime import datetime
from datetime import date
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from datetime import datetime, timedelta
import numpy as np
import pickle

#import matplotlib.pyplot as plt
#from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense

# In[20]:


def formatDateTimeCGM(CGMData):
    
    formattedArray = []
    
    for i in CGMData:
            
        upperYear = i[0][0][0]
        upperMonth = i[0][0][1]
        upperDay = i[0][0][2]

        upperHour = i[0][1][0]
        upperMinute = i[0][1][1]
        upperSecond = i[0][1][2]
        
        date_2 = str(upperDay) + '/'+ str(upperMonth) +'/'+ str(upperYear) + ' ' +str(upperHour)+':'+ str(upperMinute)+':'+ str(upperSecond)
        date_format_str = '%d/%m/%Y %H:%M:%S'
        start = datetime.strptime(date_2, date_format_str)
        formattedArray.append([start,i[1],i[2]])
        

    return formattedArray
        


# In[21]:


def ReadXlsx(CGM,Insulin):
    
    #creating the matrix with labels below
    filename1 = pd.read_excel(CGM)
    filename2 = pd.read_excel(Insulin)
    #print(filename2.Index[0])
    CGMData_pre = []
    InsulinData_pre = []

    eatingBehavior = filename1['Sensor Glucose (mg/dL)'] #.to_numpy()
    ISIG = filename1['ISIG Value'] #.to_numpy()
    CGMDate = filename1['Date'] #.to_numpy()
    CGMTime = filename1['Time'] #.to_numpy()


    Eats =  filename2['BWZ Carb Input (grams)'] #.to_numpy()
    InsulinDate = filename2['Date'] #.to_numpy()
    InsulinTime = filename2['Time'] #.to_numpy()
    
    InsulinTimeAfter = []
    for i in InsulinTime:
        arr = []
        arr.append(int(str(i).split(':',3)[0]))
        arr.append(int(str(i).split(':',3)[1]))
        arr.append(int(str(i).split(':',3)[2]))
        InsulinTimeAfter.append(arr)
    CGMTimeAfter = []
    for i in CGMTime:
        arr = []
        arr.append(int(str(i).split(':',3)[0]))
        arr.append(int(str(i).split(':',3)[1]))
        arr.append(int(str(i).split(':',3)[2]))
        CGMTimeAfter.append(arr)


    InsulinDateAfter = []
    for i in InsulinDate:
        arr = []
        var = pd.to_datetime(i)
        q= str(var).split()
        arr.append(int(q[0].split('-',3)[0]))
        arr.append(int(q[0].split('-',3)[1]))
        arr.append(int(q[0].split('-',3)[2]))
        InsulinDateAfter.append(arr)


    CGMDateAfter = []
    for i in CGMDate:
        arr = []
        var = pd.to_datetime(i)
        q= str(var).split()
        arr.append(int(q[0].split('-',3)[0]))
        arr.append(int(q[0].split('-',3)[1]))
        arr.append(int(q[0].split('-',3)[2]))
        CGMDateAfter.append(arr)
    
    
    DateandTime = []
    for i in range(0, len(InsulinDateAfter),1):
        arr = []
        arr.append(InsulinDateAfter[i])
        arr.append(InsulinTimeAfter[i])
        DateandTime.append(arr)


    CGMDateandTime = []
    for i in range(0, len(CGMDateAfter),1):
        arr = []
        arr.append(CGMDateAfter[i])
        arr.append(CGMTimeAfter[i])
        CGMDateandTime.append(arr)
    
  
    
    DateAndFood = []
    for i in range(0, len(DateandTime),1):
        arr = []
        arr.append(DateandTime[i])
        arr.append(Eats[i])
        DateAndFood.append(arr)

    DateAndFood = DateAndFood[::-1]


    CGMDateAndFood = []
    for i in range(0, len(CGMDateandTime),1):
        arr = []
        arr.append(CGMDateandTime[i])
        arr.append(eatingBehavior[i])
        arr.append(ISIG[i])
        CGMDateAndFood.append(arr)

    CGMDateAndFood = CGMDateAndFood[::-1]
    
    DataClean = []
    for i in DateAndFood:
        arr = []
        empty = math.isnan(i[1])
        if i[1] > 0 and empty == False:
            arr.append(i[0])
            arr.append(i[1])  
            DataClean.append(arr)
            
            
    CGMDateAndFood = formatDateTimeCGM(CGMDateAndFood)
            
    return DataClean, CGMDateAndFood


# In[22]:


def separateDate(Data,Data1,disregard,keep, mealData, noMealData):
    
    diff_in_hours = 0
        
    lowerYear = Data[0][0]
    lowerMonth = Data[0][1]
    lowerDay = Data[0][2]
    
        
    lowerHour = Data[1][0]
    lowerMinute = Data[1][1]
    lowerSecond = Data[1][2]
    
    upperYear = Data1[0][0]
    upperMonth = Data1[0][1]
    upperDay = Data1[0][2]
        
    upperHour = Data1[1][0]
    upperMinute = Data1[1][1]
    upperSecond = Data1[1][2]
        
    date_1 = str(lowerDay) + '/'+ str(lowerMonth) +'/'+ str(lowerYear) + ' ' +str(lowerHour)+':'+ str(lowerMinute)+':'+ str(lowerSecond)
    date_2 = str(upperDay) + '/'+ str(upperMonth) +'/'+ str(upperYear) + ' ' +str(upperHour)+':'+ str(upperMinute)+':'+ str(upperSecond)
    date_format_str = '%d/%m/%Y %H:%M:%S'
    
        
    start = datetime.strptime(date_1, date_format_str)
    end =   datetime.strptime(date_2, date_format_str)
    # Get interval between two timstamps as timedelta object
    diff = end - start
    # Get interval between two timstamps in hours
    diff_in_hours = diff.total_seconds() / 3600
    
    
    # The dates and times must be added and subtracted for these ranges
    # with the other file for creating regions of values above and below the
    # modfied EatingData datetime
    
    #rule 1 of meal
    #rule no meal rule
    #no meal time after the first two hours and 30 minutes before the next meal
   
        
    if diff_in_hours < 2:

            disregard.append(True)
    
    else:
        
            disregard.append(False)

            if diff_in_hours == 2:

                keep.append(True)
                
            else:
                
                keep.append(False)

    
    
    
    if (diff_in_hours > 2 and disregard[-2] == False) or keep[-2] == True:
        mealData.append(Data)
        
    if diff_in_hours > 2 and keep[-2] == False:
        noMealData.append(Data)
        
    if diff_in_hours == 2 and disregard[-2] == False:
        mealData.append(Data)

        


# In[169]:


def Trainer(array1, array0):
    
    array0 = np.array(array0)
    array1 = np.array(array1)

    noMeal = (array0 - np.mean(array0, axis=0))/np.std(array0,axis =0)
    meal = (array1 - np.mean(array1, axis=0))/np.std(array1,axis =0)

    X_train = np.concatenate((noMeal,meal), axis=0)

    # print(len(X_train))
    # print(X_train[0])
    # print(X_train[-1])

    #####################
    #np.savetxt('test.csv', X_train, delimiter=',')
    
    
    ######################

    labels_0_tr = np.array([0]*array0.shape[0])
    labels_1_tr = np.array([1]*array1.shape[0])

    
    #the testing classes are grouped under one vector
    y_train = np.concatenate((labels_0_tr,labels_1_tr), axis=0)
    

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.20)

    pca = PCA(n_components='mle')# adjust yourself
    pca.fit(X_train)
    pca.fit(X_test)
    

    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)

    clf = SVC(kernel='rbf', gamma=0.99)
    


   
    #0.96,1    ,1,     1,   1
    #0.53,0.42, 0.53,0.539,0.44

    #0.46,  0.46, 0.488,0.46,0.51  
    #0.429,0.53,0.492,0.54,0.47
    
    #0.57,0.60,0.61,0.64
    #0.46,0.51,0.65,0.62    
    clf.fit(X_t_train, y_train)
    #print('score', clf.score(X_t_test, y_test))
    #print('pred label', clf.predict(X_t_test))
        
    # with open ('model_pickle', 'wb') as filename:
    #     pickle.dump(clf,filename)
        
    f = open('model_pickle', 'wb')   # Pickle file is newly created where foo1.py is
    pickle.dump(clf, f)          # dump data to f
    f.close()
 
        
    meanTrainingAccuracy = 0
    meanTestingAccuracy = 0
        
    for i in range(0,10,1):
        meanTrainingAccuracy += clf.score(X_t_train, y_train)
        meanTestingAccuracy += clf.score(X_t_test, y_test)
            #print('pred label', modelRetrieved.predict(X_t_test))

    #print(meanTrainingAccuracy/10)
    #print(meanTestingAccuracy/10)


# In[170]:


def trimDates(Data,deltaHoursLower,deltaHoursUpper):
    
    dataShift = []
        
    for i in Data:
    
        lowerYear = i[0][0]
        lowerMonth = i[0][1]
        lowerDay = i[0][2]    

        lowerHour = i[1][0]
        lowerMinute = i[1][1]
        lowerSecond = i[1][2]

        date_1 = str(lowerDay) + '/'+ str(lowerMonth) +'/'+ str(lowerYear) + ' ' +str(lowerHour)+':'+ str(lowerMinute)+':'+ str(lowerSecond)
        date_format_str = '%d/%m/%Y %H:%M:%S'

        start = datetime.strptime(date_1, date_format_str)

        lowerTime = start + timedelta(hours=deltaHoursLower)
        upperTime = start + timedelta(hours=deltaHoursUpper)

        dataShift.append([lowerTime,upperTime])
        
    return dataShift


# In[171]:


def Features(featureSpace):
    
    featuresForEachMeal = []
    for i in featureSpace:
        
        array = []
        #print(max(i[1]) - min(i[1]))
        array.append( max(i[1]) - min(i[1]))
        indexMax = i[1].index(max(i[1]))
        indexMin = i[1].index(min(i[1]))
        higher = i[0][indexMax]
        lower = i[0][indexMin]
        
        diff = higher -lower
        diff_in_hours = diff.total_seconds() / 3600

        #print(i[0][index])
        array.append(diff_in_hours)
        
        signal = np.array(i[1], dtype=float)
        fourier = np.fft.fft(signal)
        n = signal.size
        sampleRate = 100
        timestep = 0.1/sampleRate
        freq = np.fft.fftfreq(n, d=timestep)
        
    
        f = np.array(i[1], dtype=float)
        x = np.arange(f.size)
        
        if len(freq) >= 4 and len(x) > 4:
            
            array.append(freq[1])
            array.append(freq[2])
            array.append(freq[3])
           
            slope = np.gradient(f,x)        
            
            if len(slope) >= 20:
                
                array.append(slope[0])
                array.append(slope[1])
                array.append(slope[2])
                array.append(slope[3])
                array.append(slope[4])
                array.append(slope[5])
                array.append(slope[6])
                array.append(slope[7])
                array.append(slope[8])
                array.append(slope[9])
                array.append(slope[10])
                array.append(slope[11])
                array.append(slope[12])
                array.append(slope[13])
                array.append(slope[14])
                array.append(slope[15])
                array.append(slope[16])
                array.append(slope[17])
                array.append(slope[18])

                
                featuresForEachMeal.append(array)
              
              
    return featuresForEachMeal
        
        


# In[172]:


def dataForFeatures(featureSpace, LabelRange):
    
    array = []
    
    for i in LabelRange:
        subarray = []

        lower = i[0]
        upper = i[1]
        
        subarrayTime = []
        subarrayGlucose = []
        subarrayISMG = []
        
        for j in featureSpace:
            
           
            glucose = math.isnan(j[1])
            ismg = math.isnan(j[2])
            
            if j[0] >= lower and j[0] <= upper and glucose == False and ismg == False:
                
                subarrayTime.append(j[0])
                subarrayGlucose.append(j[1])
                subarrayISMG.append(j[2])
        
        if len(subarrayGlucose) > 0 and len(subarrayISMG):
            
            subarray.append(subarrayTime)
            subarray.append(subarrayGlucose)
            subarray.append(subarrayISMG)    
            array.append(subarray)
                
    return array


# In[173]:


DataClean, CGMDateAndFood= ReadXlsx('CGMData670GPatient3.xlsx','InsulinAndMealIntake670GPatient3.xlsx')

#print(len(DataClean))
#print(DataClean[0])
#print(CGMDateAndFood[0])
#print(len(CGMDateAndFood))




# In[174]:
#print(DataClean[0], CGMDateAndFood[0])

#print(CGMDateAndFood[0])


# In[175]:


disregard = [False]
keep = [False]
mealData = []
noMealData = []


for i in range(0, len(DataClean), 1):
    
    if i < len(DataClean) - 1:
        separateDate(DataClean[i][0],DataClean[i+1][0],disregard,keep, mealData, noMealData)
            


# In[176]:


meals = trimDates(mealData,0,2)
noMeals = trimDates(noMealData,2,4)

#print(meals[0])
#print(len(meals))

# In[177]:


#print(CGMDateAndFood[0])
#print(meals[0])
#print(noMeals[0])


# In[178]:


#print(len(meals))
#print(len(noMeals))


# In[ ]:





# In[179]:


mealpreFeatureData = dataForFeatures(CGMDateAndFood, meals)
noMealpreFeatureData = dataForFeatures(CGMDateAndFood, noMeals)


# In[180]:


#print(mealpreFeatureData[0][1])
#print(len(noMealpreFeatureData))
#print(max(mealpreFeatureData[0][1]) - min(mealpreFeatureData[0][1]))


# In[ ]:





# In[181]:


meal = Features(mealpreFeatureData)
noMeal = Features(noMealpreFeatureData)

#print(meal[0])
#print(features[0])



# In[ ]:





# In[182]:


Trainer(meal,noMeal)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




