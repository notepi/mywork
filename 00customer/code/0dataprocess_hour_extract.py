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
    #20180904
    #nrows=1000,
    AllData = pd.read_csv('../gd_train_data.txt',header=None)
    AllData = AllData[AllData.iloc[:,1]=="线路10"]
    Data=pd.concat([AllData.iloc[:,0],AllData.iloc[:,-2]],axis=1)
    Data.columns = ['place','time']
    Data.time=Data.time.apply(str)
    FinalData=pd.DataFrame([])
    a=Data.time.value_counts().sort_index(axis=0)
    #根据时间统计次数
    FinalData['time']=a.index.tolist()
    FinalData['counts']=a.values.tolist()

    FinalData["data"]=FinalData.time.apply(lambda x:x[:-2]).apply(int)
    FinalData["year"]=FinalData.time.apply(lambda x:x[:4])
    FinalData["month"]=FinalData.time.apply(lambda x:x[4:6])
    FinalData["day"]=FinalData.time.apply(lambda x:x[6:8])
    FinalData["hour"]=FinalData.time.apply(lambda x:x[8:])
    
    FinalData.to_csv('../data/data.csv',index=False)
#    AllData = pd.read_csv('../data/data.csv')
#    WeatherData = pd.read_excel('../天气数据0904.xlsx',encoding = "GBK")
    pass
    
    
    
    
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