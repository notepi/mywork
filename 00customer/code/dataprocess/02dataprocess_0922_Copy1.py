#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:28:46 2018

@author: pan
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report,precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# 模型评价 混淆矩阵
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint
import math
import matplotlib.pyplot as plt
import seaborn as sns
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def classFir(i):
        if i <= 1000:
            return 0
            pass
        elif i>1000 and i <=1500:
            return 1
            pass
        elif i>1500 and i <=2000:
            return 2
            pass              
        elif i>2000 and i <=2500:
            return 3
            pass 
        elif i>2500 and i <=3000:
            return 4
            pass  
        elif i>3000 and i <=3500:
            return 5
            pass  
        elif i>3500 and i <=4000:
            return 6
            pass  
        elif i>4000 and i <=4500:
            return 7
            pass 
        elif i>4500 and i <=5000:
            return 8
            pass  
        elif i>5000:
            return 9
            pass          
        else:
            print(i)
        pass 
    
def countsCls(ww):
#    j=i
    temp=np.zeros((1, 6))
    for z in range(6):
        if ww <=1000:
            temp[0][z]=ww/100
            break
            pass
        else:
            if z==5:
                temp[0][z]=math.log(ww)
                pass
            else:
                temp[0][z]=1000/100
                pass
            pass
        ww=ww-1000
#        break
        pass
        
#        break
    return pd.DataFrame(temp)
    pass
    
if __name__ == "__main__":

    AllData = pd.read_csv('./finaldata0911.csv')
    
    week=np.zeros((len(AllData), 7))
    for i,j in enumerate(AllData["week"]):
        if j==0:
            week[i][0]=1
            pass
        elif j==1:
            week[i][1]=1
            pass
        elif j==2:
            week[i][2]=1
            pass
        elif j==3:
            week[i][3]=1
            pass
        elif j==4:
            week[i][4]=1
            pass
        elif j==5:
            week[i][5]=1
            pass
        elif j==6:
            week[i][6]=1
            pass
        else:
            print(i,j)
            break        
#        break       
        pass
    week=pd.DataFrame(week,columns=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    AllData=pd.concat([AllData,week],axis=1)
    del AllData["week"]
    
    
    #
    temperatureHightcode=np.zeros((len(AllData), 4))
    for i,j in enumerate(AllData.TemperatureHight):
        if j < 0:
            temperatureHightcode[i][0]=abs(j)
            pass
        elif j < 18:
            temperatureHightcode[i][1]=j
            pass
        elif j < 25:
            temperatureHightcode[i][1]=17
            temperatureHightcode[i][2]=j-17
            pass
        elif j > 24:
            temperatureHightcode[i][1]=17
            temperatureHightcode[i][2]=7
            temperatureHightcode[i][3]=j-24
            pass
    temperatureHightcode=pd.DataFrame(temperatureHightcode,
                                      columns=["TemperatureHight0","TemperatureHight1",
                                               "TemperatureHight2","TemperatureHight3"])
    AllData=pd.concat([AllData,temperatureHightcode],axis=1)
    del AllData["TemperatureHight"]
    
    #
    temperatureLowcode=np.zeros((len(AllData), 4))
    for i,j in enumerate(AllData.TemperatureLow):
        if j < 0:
            temperatureLowcode[i][0]=abs(j)
            pass
        elif j < 18:
            temperatureLowcode[i][1]=j
            pass
        elif j < 25:
            temperatureLowcode[i][1]=17
            temperatureLowcode[i][2]=j-17
            pass
        elif j > 24:
            temperatureLowcode[i][1]=17
            temperatureLowcode[i][2]=7
            temperatureLowcode[i][3]=j-24
            pass        
        pass
    
    temperatureLowcode=pd.DataFrame(temperatureLowcode,
                                    columns=["TemperatureLow0","TemperatureLow1",
                                             "TemperatureLow2","TemperatureLow3"])
    AllData=pd.concat([AllData,temperatureLowcode],axis=1)
    del AllData["TemperatureLow"]
    
    
    temperaturErrorcode=np.zeros((len(AllData), 2))
    for i,j  in enumerate(AllData["TemperaturError"]):
        if j==0:
            temperaturErrorcode[i][0]=1
            pass
        elif j==1:
            temperaturErrorcode[i][0]=2
            pass
        elif j ==2:
            temperaturErrorcode[i][1]=1
            pass
        elif j =="无":
            pass
        else:
            print(i,j)
            pass
    temperaturErrorcode=pd.DataFrame(temperaturErrorcode,columns=["TemperaturError0",
                                                                  "TemperaturError1"])
    AllData=pd.concat([AllData,temperaturErrorcode],axis=1)
    del AllData["TemperaturError"]

    #对结果做编码
    cls=[]
    for i in AllData["counts"]:
        cls.append(classFir(i))
        pass

    AllData["tag"]=cls


    cls=pd.DataFrame()
    for i,j in enumerate(AllData["OneDaysAgo"]):
        cls=pd.concat([cls,countsCls(j)])           
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["OneDaysAgo"]
    
    
    
    cls=pd.DataFrame()
    for i,j in enumerate(AllData["TwoDaysAgo"]):
        cls=pd.concat([cls,countsCls(j)])            
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["TwoDaysAgo"]

    cls=pd.DataFrame()
    for i,j in enumerate(AllData["ThreeDaysAgo"]):
        cls=pd.concat([cls,countsCls(j)])            
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["ThreeDaysAgo"]


    cls=pd.DataFrame()
    for i,j in enumerate(AllData["OneHourAgo"]):
        cls=pd.concat([cls,countsCls(j)])            
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["OneHourAgo"]

    cls=pd.DataFrame()
    for i,j in enumerate(AllData["TneHourAgo"]):
        cls=pd.concat([cls,countsCls(j)])            
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["TneHourAgo"]

    cls=pd.DataFrame()
    for i,j in enumerate(AllData["ThreeHourAgo"]):
        cls=pd.concat([cls,countsCls(j)])            
#        break
        pass
    cls = cls.reset_index(drop=True)
    AllData=pd.concat([AllData,cls],axis=1)
    del AllData["ThreeHourAgo"]
    
    
    temp=np.zeros((len(AllData), 10))
    for i,j in enumerate(AllData["Comfortable"]):
        if j <=25:
            temp[i][0]=1
            pass
        elif j<=38:
            temp[i][1]=1
            pass
        elif j<=50:
            temp[i][2]=1
            pass
        elif j<=58:
            temp[i][3]=1
            pass            
        elif j<=70:
            temp[i][4]=1
            pass 
        elif j<=75:
            temp[i][5]=1
            pass   
        elif j<=79:
            temp[i][6]=1
            pass
        elif j<=85:
            temp[i][7]=1
            pass
        elif j<=89:
            temp[i][8]=1
            pass
        elif j>=90:
            temp[i][9]=1
            pass
#        break
        pass
    temp=pd.DataFrame(temp)
    temp = temp.reset_index(drop=True)
    AllData=pd.concat([AllData,temp],axis=1)
    del AllData["Comfortable"]
    
    del AllData["counts"]
    del AllData["time"]
    
    tag=AllData["tag"]
    del AllData["tag"]
    AllData["tag"]=tag
    AllData.to_csv("finaldata0922.csv",index=False)
