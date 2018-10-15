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
from skrvm import RVC
if __name__ == "__main__":

    AllData = pd.read_csv('finaldata.csv')
##    AllData["counts"]
#    # the histogram of the data
#    n, bins, patches = plt.hist(AllData["counts"], 30, density=True, facecolor='g', alpha=0.75)
#    
#    
#    plt.xlabel('Times')
#    plt.ylabel('Counts')
#    plt.title('Frequency distribution')
#    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#    #plt.axis([40, 160, 0, 0.03])
#    plt.grid(True)
#    plt.show()
    
    cls=np.zeros((len(AllData["counts"]), 7))
    for i in AllData["counts"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    

    AllData["tag"]=cls
    
    
    cls=[]
    for i in AllData["OneDaysAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["OneDaysAgo"]=cls
    
    
    cls=[]
    for i in AllData["TwoDaysAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["TwoDaysAgo"]=cls
    
    cls=[]
    for i in AllData["ThreeDaysAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["ThreeDaysAgo"]=cls
    
    cls=[]
    for i in AllData["OneHourAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["OneHourAgo"]=cls
 
    cls=[]
    for i in AllData["TneHourAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["TneHourAgo"]=cls
    
    cls=[]
    for i in AllData["ThreeHourAgo"]:
        if i <= 1000:
            cls.append(0)
            pass
        elif i>1000 and i <=1500:
            cls.append(1)
            pass
        elif i>1500 and i <=2000:
            cls.append(2)
            pass              
        elif i>2000 and i <=2500:
            cls.append(3)
            pass 
        elif i>2500 and i <=3000:
            cls.append(4)
            pass  
        elif i>3000 and i <=3500:
            cls.append(5)
            pass  
        elif i>3500 and i <=4000:
            cls.append(6)
            pass  
        elif i>4000 and i <=4500:
            cls.append(7)
            pass 
        elif i>4500 and i <=5000:
            cls.append(8)
            pass  
        elif i>5000:
            cls.append(9)
            pass          
        else:
            print(i)
            
            break
        pass
    
    AllData["ThreeHourAgo"]=cls
    
    AllData["Comfortable"]=(AllData["Comfortable"]-50)/10
    
    
    a=AllData["tag"].value_counts().index.tolist()
    temp=[]
    for i in a:
        temp.append(AllData[AllData["tag"]==i])
#        break
        pass

    for i in temp:
        break
        pass
    
    del AllData["counts"]
    del AllData["time"]
    
    #生成数据
#    a=AllData["counts"]
#    del AllData["counts"]
#    AllData["counts"]=a
#    a= AllData["time"]
#    del AllData["time"]
#    AllData["time"]=a    
#    AllData.to_csv("dataStand.csv",index=False)
    
#    # the histogram of the data
#    n, bins, patches = plt.hist(AllData["tag"], 30, density=True, facecolor='g', alpha=0.75)
#    
#    
#    plt.xlabel('Times')
#    plt.ylabel('Counts')
#    plt.title('Frequency distribution')
#    plt.grid(True)
#    plt.show()  
#    
#
    
    #生成训练数据和测试数据
    testlDataTest, TrainlDataTrain = train_test_split(AllData, train_size=0.3, random_state=1)
    
       
    XdataTrain = TrainlDataTrain.iloc[:,:-1]
    TagTrain = TrainlDataTrain.iloc[:,-1]
    
    XdataTest = testlDataTest.iloc[:,:-1]
    TagTest = testlDataTest.iloc[:,-1]
    
    #svm
    #训练模型
    model = svm.SVC(kernel='rbf')
    c_can = np.logspace(-2, 2, 10)
    gamma_can = np.logspace(-2, 2, 10)
    svc = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    svc.fit(XdataTrain, TagTrain)
    

    #预测结果        
    print("============================svm=========================================")
    Result = svc.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    #模型结果        
    print("============================svm=========================================")
    Result = svc.predict(XdataTrain)
    print('The accuracy is:',accuracy_score(TagTrain,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTrain,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTrain,Result))
    #all
    print('The precision are:\n',precision_score(TagTrain,Result,average='micro')) 
    
    
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
