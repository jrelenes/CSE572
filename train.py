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
import pickle_compat
import csv

from sklearn.neural_network import MLPClassifier

pickle_compat.patch()
#import matplotlib.pyplot as plt
#from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense

# In[20]:
def readCSV(cgm, insulin):
    
    filename1 = open(cgm,'r')
    filename2 = open(insulin,'r')
    date_format_str = '%d/%m/%Y %H:%M:%S'

    
    CGMData = []
    
    for col in csv.DictReader(filename1):
        row = []
        row.append(col['Date'])
        row.append(col['Time'])
        row.append(col['Sensor Glucose (mg/dL)'])
        row.append(col['ISIG Value'])
        CGMData.append(row)
        
    CGMData = CGMData[::-1]
    
    CGMDataClean = []
    for i in CGMData:
        arr = []
        month = int(str(i[0]).split('/',3)[0])
        day = int(str(i[0]).split('/',3)[1])
        year = int(str(i[0]).split('/',3)[2])
        hour = int(str(i[1]).split(':',3)[0])
        minute = int(str(i[1]).split(':',3)[1])
        second = int(str(i[1]).split(':',3)[2])
        date = str(day) + '/'+ str(month) +'/'+ str(year) + ' ' +str(hour)+':'+ str(minute)+':'+ str(second)
        dateAndTime = datetime.strptime(date, date_format_str)
        
        arr.append(dateAndTime)
                
        if i[2].strip() and i[3].strip():
            arr.append(float(i[2]))
            arr.append(float(i[3]))
            CGMDataClean.append(arr)
            
    ##############################################
    glucose = []
    isgm = []
    dates = []
    for i in CGMDataClean:
        glucose.append(i[1])
        isgm.append(i[2])
        dates.append(i[0])
        
    cgm = np.array(glucose)
    isg = np.array(isgm)
    cgm1 = (cgm - np.mean(cgm, axis=0))/np.std(cgm,axis =0)
    isg1 = (isg - np.mean(isg, axis=0))/np.std(isg,axis =0)
    
    CGMDataClean1 = []
    for i in range(0, len(CGMDataClean),1):
        arr = []
        arr.append(dates[i])
        arr.append(cgm1[i])
        arr.append(isg1[i])
        CGMDataClean1.append(arr)
        
    
    
    
    ###############################################
    
    
    InsulinData = []
    
    for col in csv.DictReader(filename2):
        row = []
        row.append(col['Date'])
        row.append(col['Time'])
        row.append(col['BWZ Carb Input (grams)'])
        InsulinData.append(row)
        
    InsulinData = InsulinData[::-1]
    
    
    InsulinDataClean = []
    
    for i in InsulinData:
        row = []
        row2 = []
        date = []
        time = []
        
        month = int(str(i[0]).split('/',3)[0])
        day = int(str(i[0]).split('/',3)[1])
        year = int(str(i[0]).split('/',3)[2])
        hour = int(str(i[1]).split(':',3)[0])
        minute = int(str(i[1]).split(':',3)[1])
        second = int(str(i[1]).split(':',3)[2])
        #[[[2017, 9, 5], [13, 14, 52]], 38.0]
        date.append(year)
        date.append(month)
        date.append(day)
        
        time.append(hour)
        time.append(minute)
        time.append(second)
#         print(col,time)

 
        row.append(date)
        row.append(time)
        row2.append(row)
        
        if i[2].strip():
            
            if float(i[2]) > 0 :
                row2.append(float(i[2]))
                
                InsulinDataClean.append(row2)
        
            
            
    return InsulinDataClean, CGMDataClean
            
        

            

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




# In[169]:


def Trainer(array1, array12, array0, array02):
    
    
    array0 = np.array(array0)
    array02 = np.array(array02)

    array1 = np.array(array1)
    array12 = np.array(array12)

    
    # array0 = (array0 - np.mean(array0, axis=0))/np.std(array0,axis =0)
    # array1 = (array1 - np.mean(array1, axis=0))/np.std(array1,axis =0)
    
    # array02 = (array02 - np.mean(array02, axis=0))/np.std(array02,axis =0)
    # array12 = (array12 - np.mean(array12, axis=0))/np.std(array12,axis =0)
    ####################################3
    meal = np.concatenate((array1,array12), axis=0)
    noMeal = np.concatenate((array0,array02), axis=0)
    mealLabels = np.array([1]*meal.shape[0])
    noMealLabels = np.array([0]*noMeal.shape[0])
    X_trainMeal, X_testMeal, y_trainMeal, y_testMeal = train_test_split(meal, mealLabels, 
            train_size = None,test_size=.20, random_state=10, shuffle=True, stratify=None)

    X_trainNoMeal, X_testNoMeal, y_trainNoMeal, y_testNoMeal = train_test_split(noMeal, noMealLabels, 
            train_size = None,test_size=.20, random_state=10, shuffle=True, stratify=None)

    
    X_train = np.concatenate((X_trainMeal,X_trainNoMeal), axis=0)
    X_test = np.concatenate((X_testMeal,X_testNoMeal), axis=0)

    y_train = np.concatenate((y_trainMeal,y_trainNoMeal), axis=0)
    y_test = np.concatenate((y_testMeal,y_testNoMeal), axis=0)





    #################################    

    # X_train = np.concatenate((array0,array02,array1,array12), axis=0)


    # labels_0_tr = np.array([0]*array0.shape[0])
    # labels_1_tr = np.array([1]*array1.shape[0])
    # labels_02_tr = np.array([0]*array02.shape[0])
    # labels_12_tr = np.array([1]*array12.shape[0])

    # #the testing classes are grouped under one vector
    # y_train = np.concatenate((labels_0_tr,labels_02_tr,labels_1_tr,labels_12_tr), axis=0)
    
    
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
    #         train_size = 0.8,test_size=.20, random_state=10, shuffle=True, stratify=None)

    
    
    

    #X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train,axis =0)

    
    
    #np.savetxt('test.csv', X_train, delimiter=',')
    
    ######################

    #labels_0_tr = np.array([0]*allNoMeals.shape[0])
    #labels_1_tr = np.array([1]*allMeals.shape[0])

    
    #the testing classes are grouped under one vector
    #y_train = np.concatenate((labels_0_tr,labels_1_tr), axis=0)
    
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.20)

#     print(X_test.shape)
#     print(X_test[0])
#     print(X_test[-1])
    
    #22
    pca = PCA(n_components=19)# adjust yourself
    pca.fit(X_train)
    pca.fit(X_test)
    
    #switch to 20 and set gamma to scale


    X_t_train = pca.transform(X_train)

    X_t_test = pca.transform(X_test)

    #gamma = 'scale' = 22/30
    clf = SVC(kernel='rbf', gamma = 'scale')
    # clf = MLPClassifier(hidden_layer_sizes=(100), activation='relu', solver='adam', 
    #         alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, 
    #         power_t=0.5, max_iter=1800, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, 
    #         momentum=0.7, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
    #         beta_2=0.999, epsilon=1e-08, n_iter_no_change=30)
        
    clf.fit(X_t_train, y_train)
    
#     plot_decision_regions(X_t_train, y_train, clf=clf, legend=2)
#     plt.show()
    #print('score', clf.score(X_t_test, y_test))
    #print('pred label', clf.predict(X_t_test))
    # print('kkkkkkkkkkkkkkkkkkkkkk')
    with open ('model_pickle', 'wb') as filename:
        pickle.dump(clf,filename)
        
    # f = open('model_pickle.p', 'wb')   # Pickle file is newly created where foo1.py is
    # pickle.dump(clf, f, -1)          # dump data to f
    # f.close()
 
        
    meanTrainingAccuracy = 0
    meanTestingAccuracy = 0
        
    for i in range(0,10,1):
        meanTrainingAccuracy += clf.score(X_t_train, y_train)
        meanTestingAccuracy += clf.score(X_t_test, y_test)
            #print('pred label', modelRetrieved.predict(X_t_test))

    print(meanTrainingAccuracy/10)
    print(meanTestingAccuracy/10)


# In[170]:




# In[171]:
def calculatesSlopes(i):

    array = []

    if len(i[1]) >= 3:

            numbers = np.array(i[1])
            #n = 15
                    
            for j in range(0,len(i[1]),2):
                            
                    if j + 2 < len(i[1]):
                        
                        
                        diff_in_hours = i[0][j+2] - i[0][j]
                        diff_in_hours = diff_in_hours.total_seconds() / 3600
                        slope3 = (i[1][j] +i[1][j+2] - 2*i[1][j+1]) / diff_in_hours
                        array.append(slope3)
                        
                                
    return array



def Features(featureSpace):

    
    featuresForEachMeal = []
    for i in featureSpace:
        
        
        array = []
        #print(max(i[1]) - min(i[1]))
        array.append( max(i[1]) - min(i[1]))
        indexMax = i[1].index(max(i[1]))
        indexMin = i[1].index(min(i[1]))
        ################################
        array.append(max(i[1]))
        array.append(indexMax)
        array.append(min(i[1]))
        array.append(indexMin)

        ################################
        higher = i[0][indexMax]
        lower = i[0][indexMin]
        
        diff = higher -lower
        diff_in_hours = diff.total_seconds() / 3600

        #print(i[0][index])
        array.append(round(diff_in_hours,3))
        
        signal = np.array(i[1], dtype=float)
        fourier = np.fft.fft(signal)
        n = signal.size
        sampleRate = 100
        timestep = 0.1/sampleRate
        freq = np.fft.fftfreq(n, d=timestep)
        
    
        f = np.array(i[1], dtype=float)
        x = np.arange(f.size)
        
        x1 = []
        count = 0.083
        for h in f:
            x1.append(count)
            count += 0.083
        
        x1 = np.array(x1)
        #print(x1.size, f.size)
        #break
        #print(modifiedSlope.size)
        #print(x1.size)
        #print(f.size)
        
        gradients = []
        
        
        #################
        
        ###################
        
        if len(freq) >= 4 and len(x) > 4 and x1.size > 4:
            
            array.append(round(freq[1],3))
            array.append(round(freq[2],3))
            array.append(round(freq[3],3))
           
            modifiedSlope = np.gradient(f, x1)
            slope = np.diff(i[1])
            #print(slope)
            
            if len(modifiedSlope) >= 5:
                ################################
                n = 6
                
                #grad = calculatesSlopes(i)
                
               # grad = np.array(grad)                         
                         
                #slopes = (-slope).argsort()[:n]
#                 plt.plot(slopes,slope[slopes],color='green')
                zero_crossings = numpy.where(numpy.diff(np.signbit(modifiedSlope)))[0]
                
                amplitudes = []
                
                for j in zero_crossings:
                    
                    amplitude = abs(modifiedSlope[j])
                    amplitudes.append([j, amplitude])
                        

                sorted_multi_list = sorted(amplitudes, key=lambda x: x[1])
                
               
                if len(amplitudes) >= 5 and len(i[1]) > 0 and len(i[2]) > 0:
                    
                    #print(amplitudes)

                    #plt.plot(range(len(modifiedSlope)),modifiedSlope,label='x',color='red')
                    #plt.plot(zero_crossings,modifiedSlope[zero_crossings],color='blue')

                    #temp = [sorted_multi_list[-5][0],sorted_multi_list[-4][0],sorted_multi_list[-3][0],sorted_multi_list[-2][0],sorted_multi_list[-1][0]]
                    #temp1 = [modifiedSlope[sorted_multi_list[-5][0]],modifiedSlope[sorted_multi_list[-4][0]],modifiedSlope[sorted_multi_list[-3][0]],modifiedSlope[sorted_multi_list[-2][0]],modifiedSlope[sorted_multi_list[-1][0]]]
                    
                   
                    #plt.scatter(temp, temp1,color='black')


        
                    array.append(sorted_multi_list[-1][0])
                    array.append(modifiedSlope[sorted_multi_list[-1][0]])
                    array.append(sorted_multi_list[-2][0])
                    array.append(modifiedSlope[sorted_multi_list[-2][0]])
                    array.append(sorted_multi_list[-3][0])
                    array.append(modifiedSlope[sorted_multi_list[-3][0]])
                    
                    array.append(sorted_multi_list[-4][0])
                    array.append(modifiedSlope[sorted_multi_list[-4][0]])
                    array.append(sorted_multi_list[-5][0])
                    array.append(modifiedSlope[sorted_multi_list[-5][0]])
                    
                    
                    mean = int(np.mean(i[1]))
                    array.append(mean)
                    array.append(int(np.mean(i[2])))
                    array.append(int(np.std(i[1])))
                    array.append(int(np.std(i[2])))

                    if mean > 180:
                        array.append(1)
                    
                    if mean < 70:
                        array.append(-1)
                        
                    if mean >= 70 and mean <= 180:
                        array.append(0)

                    featuresForEachMeal.append(array)


                    #plt.show()
 
 



              
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


#DataClean, CGMDateAndFood= ReadXlsx('CGMData670GPatient3.xlsx','InsulinAndMealIntake670GPatient3.xlsx')
DataClean, CGMDateAndFood= readCSV('CGM_patient2.csv','Isulin_patient2.csv')
DataClean2, CGMDateAndFood2 = readCSV( 'CGMData.csv', 'InsulinData.csv')

CGMDateAndFood = CGMDateAndFood[0:10000]
CGMDateAndFood2 = CGMDateAndFood2[0:40000]
#print(len(DataClean))
#print(DataClean[0])
#print(CGMDateAndFood[0])
#print(len(CGMDateAndFood))




# In[174]:
#print(DataClean[0], CGMDateAndFood[0])

#print(CGMDateAndFood[0])


# In[175]:


def extraction(data):

    disregard = [False]
    
    keep = [False]
    mealData = []
    noMealData = []
    
    i = 1
    
    while i < len(data) - 1:
    
            
            #data[i][0]
            #data[i+1][0]
                
            
            Year = data[i][0][0][0]
            Month = data[i][0][0][1]
            Day = data[i][0][0][2]
            
            Hour = data[i][0][1][0]
            Minute = data[i][0][1][1]
            Second = data[i][0][1][2]
            
        
        
            lowerYear = data[i-1][0][0][0]
            lowerMonth = data[i-1][0][0][1]
            lowerDay = data[i-1][0][0][2]
            
            lowerHour = data[i-1][0][1][0]
            lowerMinute = data[i-1][0][1][1]
            lowerSecond = data[i-1][0][1][2]

            upperYear = data[i+1][0][0][0]
            upperMonth = data[i+1][0][0][1]
            upperDay = data[i+1][0][0][2]

            upperHour = data[i+1][0][1][0]
            upperMinute = data[i+1][0][1][1]
            upperSecond = data[i+1][0][1][2]

            
            date_1 = str(lowerDay) + '/'+ str(lowerMonth) +'/'+ str(lowerYear) + ' ' +str(lowerHour)+':'+ str(lowerMinute)+':'+ str(lowerSecond)
            date_2 = str(upperDay) + '/'+ str(upperMonth) +'/'+ str(upperYear) + ' ' +str(upperHour)+':'+ str(upperMinute)+':'+ str(upperSecond)
            date_3 = str(Day) + '/'+ str(Month) +'/'+ str(Year) + ' ' +str(Hour)+':'+ str(Minute)+':'+ str(Second)

            date_format_str = '%d/%m/%Y %H:%M:%S'

            
            lower = datetime.strptime(date_1, date_format_str)
            upper =   datetime.strptime(date_2, date_format_str)
            current =   datetime.strptime(date_3, date_format_str)

            
            # Get interval between two timstamps as timedelta object
            diffUpper = upper - current
            diffLower = current - lower

            # Get interval between two timstamps in hours
            diffUpper_in_hours = diffUpper.total_seconds() / 3600
            diffLower_in_hours = diffLower.total_seconds() / 3600

                        
#             if diffLower_in_hours < 0.5:
                

            #rule 1
            #might need recursion
            if diffLower_in_hours > 0.5 and diffUpper_in_hours < 2:
                lowerTime = upper + timedelta(hours=0) - timedelta(hours=0.5)
                upperTime = upper + timedelta(hours=2)
                mealData.append([lowerTime,upperTime])
                i = i + 1


            if diffLower_in_hours > 0.5 and diffUpper_in_hours >= 4.5:
                lowerTime = current + timedelta(hours=2)
                upperTime = current + timedelta(hours=4)
                noMealData.append([lowerTime,upperTime])

            #6 points rather than 4
            if diffLower_in_hours > 0.5 and diffUpper_in_hours > 2.04:
                lowerTime = current + timedelta(hours=0) - timedelta(hours=0.5)
                upperTime = current + timedelta(hours=2)
                mealData.append([lowerTime,upperTime])

                
            
            #rule 2
            if diffLower_in_hours < 0.5:
                mealData = mealData[:-1]

            if diffLower_in_hours < 0.5:
                noMealData = noMealData[:-1]


                

            #has no impact on score
            if diffLower_in_hours > 0.5 and diffUpper_in_hours >= 2 and diffUpper_in_hours <= 2.04:
                lowerTime = current + timedelta(hours=0) - timedelta(hours=0.5)
                upperTime = current + timedelta(hours=2)
                mealData.append([lowerTime,upperTime])
                
                lowerTime = upper + timedelta(hours=0) - timedelta(hours=0.5)
                upperTime = upper + timedelta(hours=2)
                mealData.append([lowerTime,upperTime])
                
                i = i + 1
                #noMealData.append(data[i])
            
            
            i = i+1



            
    
    return mealData, noMealData
            

# In[176]:
meals, noMeals = extraction(DataClean)
meals2, noMeals2 = extraction(DataClean2)


#print(len(meals) + len(meals2))
#print(len(noMeals) + len(noMeals2))

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

mealpreFeatureData2 = dataForFeatures(CGMDateAndFood2, meals2)
noMealpreFeatureData2 = dataForFeatures(CGMDateAndFood2, noMeals2)

# In[180]:

#print(mealpreFeatureData[0][1])
#print(len(noMealpreFeatureData))
#print(max(mealpreFeatureData[0][1]) - min(mealpreFeatureData[0][1]))


# In[ ]:





# In[181]:


meal = Features(mealpreFeatureData)
noMeal = Features(noMealpreFeatureData)

meal2 = Features(mealpreFeatureData2)
noMeal2 = Features(noMealpreFeatureData2)


noMeal = noMeal[1:29]
noMeal2 = noMeal2[1:29]
#print(meal[0])
#print(features[0])


# In[ ]:





# In[182]:


Trainer(meal,meal2,noMeal,noMeal2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]: