#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:43:16 2018

@author: pan
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def file_name(file_dir):  
  for paths, dirs, files in os.walk(file_dir): 
#    print(paths) #当前目录路径 
#    print(dirs) #当前路径下所有子目录 
#    print(files) #当前路径下所有非目录子文件 
    pass
  return paths,dirs,files  
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
            pass
        pass
    pass
if __name__ == "__main__":
    file="./mergefinal/AQI/AQI.csv"
    Data=pd.read_csv(file,encoding = "GBK")
    del Data['date']
    
    tag=[]
    for i in Data["values"]:
        if i<50:
            tag.append(0)
            pass
        elif i <101:
            tag.append(1)
            pass
        elif i <151:
            tag.append(2)
            pass
        elif i <201:
            tag.append(3)
            pass
        elif i <300:
            tag.append(4)
            pass
        else:
            tag.append(5)
            pass
        pass
    Data['tag']=tag
    del Data["values"]
    
    #画图
    n, bins, patches = plt.hist(Data['tag'], 11, density=True, facecolor='g', alpha=0.75)
    
#    plt.figure()
    plt.xlabel('Times')
    plt.ylabel('Counts')
    plt.title('Frequency distribution')
    plt.grid(True)
    plt.show()  
    
#做样本不均衡处理
#    #生成训练数据和测试数据 
    cc=[]
    a=Data.tag.value_counts().index.tolist() 
    
    #根据种类进行分类
    DataTest = pd.DataFrame()
    DataTrain = pd.DataFrame()
    for i in a:
        temp=Data[Data["tag"]==i]

        TemptestlDataTest, TempTrainlDataTrain = train_test_split(temp, train_size=0.2, random_state=1)
        DataTest=pd.concat([DataTest,TemptestlDataTest])
        DataTrain=pd.concat([DataTrain,TempTrainlDataTrain])
        
        pass
    
    ###    #生成训练数据和测试数据
##    testlDataTest, TrainlDataTrain = train_test_split(AllData, train_size=0.3, random_state=1)
     
    XdataTrain = DataTrain.iloc[:,:-1]
    TagTrain = DataTrain.iloc[:,-1]
    
    XdataTest = DataTest.iloc[:,:-1]
    TagTest = DataTest.iloc[:,-1]
    

    '''
    ##A
    '''
####============================================================================
    print("=================train model")
    # 使用SVM作为分类器
    clf = svm.SVC()
    
    # use a full grid over all parameters
    param_grid = {"kernel": ["linear","rbf","sigmoid"],
                  "C": np.logspace(-2, 2, 10),
                  "gamma": np.logspace(-2, 2, 10)
                  }
    
    print("=====tain=====")
    # 网格搜索， grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(XdataTrain, TagTrain)
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    svc=grid_search
    
#   
#    '''
#    ##B
#    '''
#####============================================================================
#    print("=================train model")
#    #训练模型
#    model = svm.SVC(kernel='rbf')
#    c_can = np.logspace(-2, 0.65, 10)
#    gamma_can = np.logspace(-2, 2, 10)
#    start = time()
#    svc = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
#    svc.fit(XdataTrain, TagTrain)    
#
#    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#      % (time() - start, len(svc.cv_results_['params'])))
#    report(svc.cv_results_)


#"================================================================================================"
 #预测结果        
    print("============================svm_TEST=========================================")
    Result = svc.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
    #模型结果        
    print("============================svm_TRAIN=========================================")
    Result = svc.predict(XdataTrain)
    print('The accuracy is:',accuracy_score(TagTrain,Result))
    #print(confusion_matrix(y_test,y_pred,labels=[0,1,2]))
    print('The confusion matrix is:\n',confusion_matrix(TagTrain,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTrain,Result))
    #all
    print('The precision are:\n',precision_score(TagTrain,Result,average='micro')) 


    
    pass

