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

if __name__ == "__main__":

    AllData = pd.read_csv('../data/data.csv')
    WeatherData = pd.read_excel('../天气数据0904.xlsx',encoding = "GBK")

    result = pd.merge(AllData, WeatherData, left_on='data', right_on=u"日期",how='left', sort=False);
    
    del result ['year']
    del result ['日期']
    #没有节假日的数据填充为0
    result["节假日情况"]=result["节假日情况"].fillna(0)
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
            if result["节假日情况"][i]==u"补班":
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
    
    week=np.zeros((len(result), 7))
    for i,j in enumerate(result["星期"]):
        if "周一" in j:
            week[i][0]=1
            pass
        elif "周二" in j:
            week[i][1]=1
            pass
        elif "周三" in j:
            week[i][2]=1
            pass
        elif "周四" in j:
            week[i][3]=1
            pass
        elif "周五" in j:
            week[i][4]=1
            pass
        elif "周六" in j:
            week[i][5]=1
            pass
        elif "周日" in j:
            week[i][6]=1
            pass
        else:
            print(i,j)
            break        
#        break       
        pass
    weatherDay=result["天气类型"].apply(lambda x:x.split("/")[0])
    weatherNight=result["天气类型"].apply(lambda x:x.split("/")[1])
    ww=weatherDay.value_counts()
    
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
    
    temperatureHight=result["最高温"].apply(lambda x:int(x[:-1]))
    #
    temperatureHightcode=np.zeros((len(result), 4))
    for i,j in enumerate(temperatureHight):
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
        
    temperatureLow=result["最低温"].apply(lambda x:int(x[:-1]))
    #
    temperatureLowcode=np.zeros((len(result), 4))
    for i,j in enumerate(temperatureLow):
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
        
            
#        break
        pass

    winddict={"无持续风向≤3级":0,"北风微风转3-4级":1,"北风3-4级":1,"东北风3-4级":2,"东南风3-4级":2,"北风4-5级":3,"无持续风向微风转3-4级":4,"东风4-5级":5}
    
    windDaycode=[winddict[x]  for x in result["风力风向（白天）"]]
    windNightcode=[winddict[x]  for x in result["风力风向（夜间）"]]
    
    windcode=[]
    datatemp=[]
    datatemp.append(weatrherDaycode)
    datatemp.append(weatrherNightcode)
    #    白天是指8:00-20:00 夜间是20：00到次日早上8:00
    for j,i in enumerate(result["hour"]):
        #白天
        if i > 7 and i < 21:
            windcode.append(datatemp[0][j])
            pass
        #夜里
        else:
            windcode.append(datatemp[1][j])
            pass
        
#        break
        pass
    
    

    temperaturErrorcode=np.zeros((len(result), 2))
    for i,j  in enumerate(result["异常天气（温度类）"]):
        if j=="黄色高温预警":
            temperaturErrorcode[i][0]=1
            pass
        elif j=="橙色高温预警":
            temperaturErrorcode[i][0]=2
            pass
        elif j =="天气骤降":
            temperaturErrorcode[i][1]=1
            pass
        elif j =="无":
            pass
        else:
            print(i,j)
            pass
 
    Typhoondict={"无":1,"白色台风预警":1,"黄色台风预警":2,"橙色台风预警":3}
    Typhooncode=[Typhoondict[x]  for x in result["异常天气（台风）"]] 
    
    
    a=result["异常天气（降雨类）"].value_counts()
    aa=result["异常天气（台风）"].value_counts() 
    dataT=result["data"].value_counts().index.tolist()
    
    
    #只保留早上六点到晚上九点的数据
    result=result[result["hour"] <22]
    result=result[result["hour"] > 5]

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
    tt=pd.DataFrame()
    for i in range(len(temp))[3:]:
        tt=pd.concat([tt,temp[i]])
#        break
        pass
   
    t=0
    for i in range(len(temp))[3:]:
        t=t+len(temp[i]) 
#        break
        pass
    
    
    
    
#    TyphoonErrorcode=np.zeros((len(result), 4))
#    for i,j  in enumerate(result["异常天气（台风）"]):
#        if j=="白色台风预警":
#            
#            pass          
#        elif j=="黄色高温预警":
#            temperaturErrorcode[i][1]=1
#            pass
#        elif j=="橙色高温预警":
#            temperaturErrorcode[i][2]=1
#            pass
#        elif j =="天气骤降":
#            temperaturErrorcode[i][3]=1
#            pass
#        elif j =="无":
#            pass
#        else:
#            print(i,j)
#            pass
    
#    aa=result["风力风向（白天）"].value_counts().index.tolist()
#    aa=result["风力风向（白天）"].value_counts().index.tolist()
    
#    name=AllData.columns.values.tolist().copy()
#    #id
#    idnum=AllData[name[0]].value_counts()
#    #area
#    area=AllData[name[1]].value_counts()
##    .index.tolist()
#    #date
#    date=AllData[name[2]].value_counts()
#    #deviceId
#    deviceId=AllData[name[3]].value_counts()
#    #duration
#    duration=AllData[name[4]].value_counts()   
#    #gait
#    gait=AllData['gait'].value_counts()
#    #height
#    height=AllData[name[6]].value_counts()
#    #humidity
#    humidity=AllData[name[7]].value_counts()
#    #movement
#    movement=AllData[name[8]].value_counts()
#    #rawMovmt
#    rawMovmt=AllData[name[9]].value_counts()
#    #rein
#    rein=AllData[name[10]].value_counts()
#    #skinTemperature
#    skinTemperature=AllData[name[11]].value_counts()
#    #state
#    state=AllData[name[12]].value_counts()
#    #heartRate
#    heartRate=AllData[name[13]].value_counts()
#    #height
#    height=AllData[name[6]].value_counts()
#    Data1617 = pd.read_excel('../data/Advanced System Design ..xlsx','16-17',encoding = "GBK")
#    Data1718 = pd.read_excel('../data/Advanced System Design ..xlsx','17-18',encoding = "GBK")
#    Data=pd.concat([Data1617,Data1718]).reset_index()
#    
#    print("16-17 not in 17-18")
#    for i in Data1617.columns.values.tolist():
#        if i in Data1718.columns.values.tolist():
#            continue
#            pass
#        print(i)
#    print("17-18 not in 16-17")
#    for i in Data1718.columns.values.tolist():
#        if i in Data1617.columns.values.tolist():
#            continue
#            pass
#        print(i)    
#    
#    pd.DataFrame(Data1617.columns.values.tolist()).to_csv('../data/Dataname.csv',encoding = "GBK",index=False)
#    
#
#    #####major
#    major=Data['major']
#    majorCode=np.zeros((len(major), 5))
#    
#    for i,j in enumerate(major):
#        if j=="MSc in Web Applications and Services" or \
#            j=="MSc in Advanced Software Engineering" or \
#            j=="MSc in Cloud Computing":
#            majorCode[i][0]=1    
#            continue
#            pass
#        elif j=="MSc in Advanced Computer Science" or \
#            j=="MSc in Advanced Computational Methods":
#            majorCode[i][1]=1    
#            continue
#            pass
#        elif j=="MSc in Actuarial Science" or \
#            j=="MSc in Applied Computation and Numerical Modelling":
#            majorCode[i][2]=1 
#            continue
#            pass
#        elif j=="MSc in Software Engineering for Financial Services" or \
#            j=="MSc in Data Analysis in Business Intelligence programme" or \
#            j=="MSc in Financial Mathematics and Computation":
#            majorCode[i][3]=1
#            continue
#            pass
#        elif j=="MSc Advanced Electrical and Electronic Engineering":
#            majorCode[i][4]=1 
#            continue
#            pass
#        else:
#            print(j)
#            break
#        pass
#    majorCode=pd.DataFrame(majorCode)
#    
#    #age
#    agecode=Data['age']
#    
#    #academic year,drop out
#    academicyear=Data['academic year']
#    
#    #gender
#    genderdict={"female":0,"male":1}
#    gendercoede=pd.DataFrame([ genderdict[i] for i in Data['gender'].apply(str.lower)])
#    
#    #module selection tendency
#    moduleselectiontendency=Data["module selection tendency"]
#    moduleselectiontendencyCode=np.zeros((len(Data), 4))
#    moduledict={"web-related":0,"Technology theory ":1,"program skill":2,"algorithm-related":3}
#    for i,j in enumerate (moduleselectiontendency):
#        moduleselectiontendencyCode[i][moduledict[j]]=1
#        pass
#    moduleselectiontendencyCode=pd.DataFrame(moduleselectiontendencyCode)
#    
#    #pre-major
#    premajor=Data["pre-major"]
#    #a=Data["pre-major"].value_counts()
#    
#    #language learning ability
#    language=Data["language learning ability"]
#    a=language.value_counts().index.tolist()
#    languagedict={"not so good":0,"general":1,"good":2,"native speaker":3}
#    languagecode=pd.DataFrame([languagedict[i] for i in language])
#
#    #the reason to select this module
#    reason=Data["the reason to select this module"]
#    reasonCode=np.zeros((len(reason), 3))
#    
#    reasondict={"easy to pass":0,"interested in this module":1,"useful for future job":2}
#    for i,j in enumerate (reason):
#        reasonCode[i][reasondict[j]]=1
#        pass
#    reasonCode=pd.DataFrame(reasonCode)
#    #Learned Similar Module  or Ralated Knowledge
#    RalatedKnowledgedict={"no":0,"yes":1}
#    RalatedKnowledgecode=pd.DataFrame([RalatedKnowledgedict[i] for i in Data["Learned Similar Module  or Ralated Knowledge"]])
#    
#    
#    #learning attitude
#    learningattitudedict={"positive":0,"general":1,"passive":2}
#    learningattitudecode=pd.DataFrame([learningattitudedict[i] for i in Data["learning attitude"]])
#    
#    #the goal for the final result
#    goal=Data["the goal for the final result"].value_counts().index.tolist()
#    goaldict={"pass":0,"merit":1,"distinction":2}
#    goalcode=pd.DataFrame([goaldict[i] for i in Data["the goal for the final result"]])
#    
#    #grade
#    grade=Data['grade']
#    gradecode=[]
#    for i in grade:
#        if i < 50:
#            gradecode.append(0)
#            continue
#            pass
#        elif i<61:
#            gradecode.append(1)
#            continue
#            pass
#        elif i<71:
#            gradecode.append(2)
#            continue
#            pass
#        else:
#            gradecode.append(3)
#            continue
#            pass
#        pass
#    gradecode=pd.DataFrame(gradecode)
#    #majorCode,age,gender,moduleselectiontendency,language,RalatedKnowledge,learning attitude,goalcode,grade
#    result=pd.concat([majorCode,agecode,gendercoede,moduleselectiontendencyCode,languagecode,reasonCode,RalatedKnowledgecode,learningattitudecode,goalcode,gradecode],axis=1)
#
#    #balanced data
#            
#
#    #生成训练数据和测试数据
#    testlDataTest, TrainlDataTrain = train_test_split(result, train_size=0.3, random_state=1)
#    
#       
#    XdataTrain = TrainlDataTrain.iloc[:,:-1]
#    TagTrain = TrainlDataTrain.iloc[:,-1]
#    
#    XdataTest = testlDataTest.iloc[:,:-1]
#    TagTest = testlDataTest.iloc[:,-1]
#    
#    #svm
#    #训练模型
#    model = svm.SVC(kernel='rbf')
#    c_can = np.logspace(-2, 2, 10)
#    gamma_can = np.logspace(-2, 2, 10)
#    svc = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
#    svc.fit(XdataTrain, TagTrain)
#    
#
#    #预测结果        
#    print("============================svm=========================================")
#    Result = svc.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
#    
#    #模型结果        
#    print("============================svm=========================================")
#    Result = svc.predict(XdataTrain)
#    print('The accuracy is:',accuracy_score(TagTrain,Result))
#    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
#    print('The confusion matrix is:\n',confusion_matrix(TagTrain,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTrain,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTrain,Result,average='micro')) 
#    
#    
#   #MNB
#    #训练模型    
#    MNB=MultinomialNB()
#    MNB.fit(XdataTrain, TagTrain.values.ravel())
#    #预测结果        
#    Result = MNB.predict(XdataTest)
#    #预测结果        
#    print("============================MNB=========================================")
#    Result = MNB.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
#    
#    #Dtree
#    Dtree = tree.DecisionTreeClassifier()
#    Dtree.fit(XdataTrain, TagTrain)
#    
#    #预测结果        
#    print("============================Dtree=========================================")
#    Result = Dtree.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
#    
#    #REF
#    ref = RandomForestClassifier()
#    ref.fit(XdataTrain, TagTrain)
#    
#    #预测结果        
#    print("============================REF=========================================")
#    Result = ref.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))   
#    
#    # 使用随机森林作为分类器，分类器有20课树
#    clf = RandomForestClassifier(n_estimators=20)
#    
#    
#    # Utility function to report best scores
#    def report(results, n_top=3):
#        for i in range(1, n_top + 1):
#            candidates = np.flatnonzero(results['rank_test_score'] == i)
#            for candidate in candidates:
#                print("Model with rank: {0}".format(i))
#                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                      results['mean_test_score'][candidate],
#                      results['std_test_score'][candidate]))
#                print("Parameters: {0}".format(results['params'][candidate]))
#                print("")
#    
#    
#    # 设置可能学习的参数
#    param_dist = {"max_depth": [3, None],
#                  "max_features": sp_randint(1, 11),
#                  "min_samples_split": sp_randint(2, 11),
#                  "min_samples_leaf": sp_randint(1, 11),
#                  "bootstrap": [True, False],
#                  "criterion": ["gini", "entropy"]}
#    
#    # 随机搜索， randomized search
#    n_iter_search = 35
#    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                       n_iter=n_iter_search)
#    #起始时间
#    start = time()
#    random_search.fit(XdataTrain, TagTrain)
#    print("RandomizedSearchCV took %.2f seconds for %d candidates"
#          " parameter settings." % ((time() - start), n_iter_search))
#    report(random_search.cv_results_)
#    
#    print("=============================================")
#
#    # use a full grid over all parameters
#    param_grid = {"max_depth": [3, None],
#                  "max_features": [1, 3, 10],
#                  "min_samples_split": [2, 3, 10],
#                  "min_samples_leaf": [1, 3, 10],
#                  "bootstrap": [True, False],
#                  "criterion": ["gini", "entropy"]}
#    
#    # 网格搜索， grid search
#    grid_search = GridSearchCV(clf, param_grid=param_grid)
#    start = time()
#    grid_search.fit(XdataTrain, TagTrain)
#    
#    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#          % (time() - start, len(grid_search.cv_results_['params'])))
#    report(grid_search.cv_results_)
#    
#    #对比分类效果
#    #random_search
#    #预测结果        
#    print("============================random_search=========================================")
#    Result = random_search.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #混淆矩阵
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
#   
#    #预测结果        
#    print("============================grid_search=========================================")
#    Result = grid_search.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #混淆矩阵
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
#    
#    
#    
#    
#    
#        
#        
#        
#        
#        
#    
#        
#        
#        
#        
#            
#
#
#    
#    
#    
#
#    