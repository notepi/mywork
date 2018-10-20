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

if __name__ == "__main__":

    AllData = pd.read_csv('../../data/data.csv')
    WeatherData = pd.read_excel('../../天气数据0904.xlsx',encoding = "GBK")
    NewData = pd.read_excel('../../时段温度及相对湿度数据0911.xlsx',encoding = "GBK")
    result = pd.merge(AllData, WeatherData, left_on='data', right_on=u"日期",how='left', sort=False);
    
    del result ['year']
    del result ['日期']
    #没有节假日的数据填充为0
    result["节假日情况"]=result["节假日情况"].fillna(0)
    
    
    dataT=result["data"].value_counts().index.tolist()
    
    
    #只保留早上六点到晚上九点的数据
    result=result[result["hour"] <22]
    result=result[result["hour"] > 5]
    
    result["merge"]=result["time"].apply(str).apply(lambda x:x[4:]).apply(np.int64)
    result=pd.merge(result, NewData, how='left', left_on="merge", right_on=u"时刻")
    del result["merge"]
    del result[u"时刻"]


    
    
    #降序
    dataT.sort()
    
    temp=[]
    for i in dataT:
        temp.append(result[result["data"]==i])
#        break
        pass
        
    for i,j in enumerate(temp):
        if len(j) !=16:
            a=temp[i].copy()
            temp[i]=temp[i-1]
            temp[i].month=a.month.values[0]
            temp[i].day=a.day.values[0]
            temp[i]["星期"]=a["星期"].values[0]
            temp[i].data=np.int64(str(a.month.values[0])+str(a.day.values[0]))
#            break
            pass
        pass
    
    #添加衍生特征
    #1同一时段，前三天的特征
    for i in list(range(len(temp)))[3:]:
        #前一天的
        temp[i]["OneDaysAgo"]=temp[i-1]["counts"].values
        #前两天的
        temp[i]["TwoDaysAgo"]=temp[i-2]["counts"].values
        #前三天的
        temp[i]["ThreeDaysAgo"]=temp[i-3]["counts"].values
#        break
        pass
    
    #前三小时的特征做衍生
    for i in list(range(len(temp)))[3:]:
        cc1=[]
        cc2=[]
        cc3=[]
        for j in range(len(temp[i])):
            if j==0 :
                cc1.append(-1)
                cc2.append(-1)
                cc3.append(-1)
                pass
            elif j==1:
                cc1.append(temp[i].iloc[j-1:j,:]["counts"].values[0])
                cc2.append(-1)
                cc3.append(-1)
                pass
            elif j==2:
                cc1.append(temp[i].iloc[j-1:j,:]["counts"].values[0])
                cc2.append(temp[i].iloc[j-2:j-1,:]["counts"].values[0])
                cc3.append(-1)        
                pass
            else:
                cc1.append(temp[i].iloc[j-1:j,:]["counts"].values[0])
                cc2.append(temp[i].iloc[j-2:j-1,:]["counts"].values[0])
                cc3.append(temp[i].iloc[j-3:j-2,:]["counts"].values[0])  
#            break
            pass
        temp[i]["OneHourAgo"]=cc1
        temp[i]["TneHourAgo"]=cc2
        temp[i]["ThreeHourAgo"]=cc3
#        break
        pass
    
    
    result=pd.DataFrame()
    for i in range(len(temp))[3:]:
        result=pd.concat([result,temp[i]])
#        break
        pass
   
    result=result.fillna(method='pad')
    comfortable=[]
    for i in range(len(result)):
        t=result.iloc[i:i+1,:]["温度"].values[0]
        u=result.iloc[i:i+1,:]["相对湿度"].values[0]/100
        v=result.iloc[i:i+1,:]["风速"].values[0]
        conf=1.8*t-0.55*(1.8*t-26)*(1-u)-3.2*math.sqrt(v)+32
        comfortable.append(conf)
#        if math.isnan(conf):
#            break
#            pass
        
#        break
        pass
    comfortable=pd.DataFrame(comfortable,columns=["Comfortable"])
    
    #非假期为0，假期且不放假为1，假期且放假
    festiva=[]
    for i in result["节假日情况"]:
        if i ==0:
            festiva.append(0)
        elif "-" not in i:
           festiva.append(0)
        else:
            festiva.append(int(i.split("-")[1]))
            pass
#        break
        pass

    
    work=[]
    #0不上班，1上班
    for i,j in enumerate(result["星期"]):
        if j=="周六" or j=="周日":
            if result["节假日情况"].values[i]==u"补班":
                work.append(1)
                pass
            #是非公休的节假日
            else:
                work.append(0)
                pass
            pass
        else:
            if festiva[i]==2:
                work.append(0)
                pass
            else:
                work.append(1)
                pass
#        break
        pass
    festiva=pd.DataFrame(festiva,columns=["FestivaState"])
    work=pd.DataFrame(work,columns=["WorkState"])
    
#    weekdict={ "周一":0,"周二":1,"周三":2,"周四":3,"周五":4,"周六":5,"周日":6}
#    week=[weekdict[x] for x in list(result["星期"])]
    
    
    week=np.zeros((len(result), 1))
    for i,j in enumerate(result["星期"]):
        if "周一" in j:
            week[i]=0
            pass
        elif "周二" in j:
            week[i]=1
            pass
        elif "周三" in j:
            week[i]=2
            pass
        elif "周四" in j:
            week[i]=3
            pass
        elif "周五" in j:
            week[i]=4
            pass
        elif "周六" in j:
            week[i]=5
            pass
        elif "周日" in j:
            week[i]=6
            pass
        else:
            print(i,j)
            break        
#        break       
        pass
    week=pd.DataFrame(week,columns=["week"])

    weatherDay=result["天气类型"].apply(lambda x:x.split("/")[0])

    weatherNight=result["天气类型"].apply(lambda x:x.split("/")[1])
    
    weatrhercode=[]
    weatrherdice={"晴":0,"多云":1,"阴":2,"雷阵雨":3,"阵雨":3,"小雨":4,\
                  "小到中雨":5,"中雨":6,"中到大雨":7,"大雨":8,"大到暴雨":9}
    weatrherDaycode=[weatrherdice[x]  for x in weatherDay]
    weatrherNightcode=[weatrherdice[x]  for x in weatherNight]
    
    datatemp=[]
    datatemp.append(weatrherDaycode)
    datatemp.append(weatrherNightcode)
    #    白天是指8:00-20:00 夜间是20：00到次日早上8:00
    for j,i in enumerate(result["hour"]):
        #白天
        if i > 7 and i < 21:
            weatrhercode.append(datatemp[0][j])
            pass
        #夜里
        else:
            weatrhercode.append(datatemp[1][j])
            pass
        
#        break
        pass
    weatrhercode=pd.DataFrame(weatrhercode,columns=["Weather"])
    
#    temperatureHight=result["最高温"].apply(lambda x:int(x[:-1]))
#    #
#    temperatureHightcode=np.zeros((len(result), 4))
#    for i,j in enumerate(temperatureHight):
#        if j < 0:
#            temperatureHightcode[i][0]=abs(j)
#            pass
#        elif j < 18:
#            temperatureHightcode[i][1]=j
#            pass
#        elif j < 25:
#            temperatureHightcode[i][1]=17
#            temperatureHightcode[i][2]=j-17
#            pass
#        elif j > 24:
#            temperatureHightcode[i][1]=17
#            temperatureHightcode[i][2]=7
#            temperatureHightcode[i][3]=j-24
#            pass
#    temperatureHightcode=pd.DataFrame(temperatureHightcode,
#                                      columns=["TemperatureHight0","TemperatureHight1",
#                                               "TemperatureHight2","TemperatureHight3"])
    temperatureHightcode=result["最高温"].apply(lambda x:int(x[:-1]))
    temperatureHightcode=pd.DataFrame(temperatureHightcode.values,columns=["TemperatureHight"])
#    temperatureLow=result["最低温"].apply(lambda x:int(x[:-1]))
#    #
#    temperatureLowcode=np.zeros((len(result), 4))
#    for i,j in enumerate(temperatureLow):
#        if j < 0:
#            temperatureLowcode[i][0]=abs(j)
#            pass
#        elif j < 18:
#            temperatureLowcode[i][1]=j
#            pass
#        elif j < 25:
#            temperatureLowcode[i][1]=17
#            temperatureLowcode[i][2]=j-17
#            pass
#        elif j > 24:
#            temperatureLowcode[i][1]=17
#            temperatureLowcode[i][2]=7
#            temperatureLowcode[i][3]=j-24
#            pass        
#        
#            
##        break
#        pass
#    
#    temperatureLowcode=pd.DataFrame(temperatureLowcode,
#                                    columns=["TemperatureLow0","TemperatureLow1",
#                                             "TemperatureLow2","TemperatureLow3"])
    temperatureLowcode=pd.DataFrame(result["最低温"].apply(lambda x:int(x[:-1])).values,columns=["TemperatureLow"])
    
#    
#    temperaturErrorcode=np.zeros((len(result), 2))
#    for i,j  in enumerate(result["异常天气（温度类）"]):
#        if j=="黄色高温预警":
#            temperaturErrorcode[i][0]=1
#            pass
#        elif j=="橙色高温预警":
#            temperaturErrorcode[i][0]=2
#            pass
#        elif j =="天气骤降":
#            temperaturErrorcode[i][1]=1
#            pass
#        elif j =="无":
#            pass
#        else:
#            print(i,j)
#            pass
#    temperaturErrorcode=pd.DataFrame(temperaturErrorcode,columns=["TemperaturError0",
#                                                                  "TemperaturError1"])
    

    #    
    temperaturErrorcode=np.zeros((len(result), 1))
    for i,j  in enumerate(result["异常天气（温度类）"]):
        if j=="黄色高温预警":
            temperaturErrorcode[i]=0
            pass
        elif j=="橙色高温预警":
            temperaturErrorcode[i]=1
            pass
        elif j =="天气骤降":
            temperaturErrorcode[i]=2
            pass
        elif j =="无":
            pass
        else:
            print(i,j)
            pass
    temperaturErrorcode=pd.DataFrame(temperaturErrorcode,columns=["TemperaturError"])
            
    Typhoondict={"无":1,"白色台风预警":1,"黄色台风预警":2,"橙色台风预警":3}
    Typhooncode=[Typhoondict[x]  for x in result["异常天气（台风）"]] 
    Typhooncode=pd.DataFrame(Typhooncode,columns=["TyphoonState"])
    
    result=result.reset_index()
    finaldata=pd.concat([comfortable, festiva,work,week,weatrhercode,
                         temperatureHightcode,temperatureLowcode,
                         temperaturErrorcode,Typhooncode,result.iloc[:,-6:],
                         result["time"],result["counts"]],
                        axis=1)
    
    finaldata.to_csv("finaldata0911.csv",index=False)
