# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:05:17 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import os 
#import re
#import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.externals import joblib  
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib  
import time
def train(data):
    #MNB
    #训练模型    
    MNB=MultinomialNB()
    MNB.fit(data.iloc[:,:-1], data.iloc[:,-1]) 
    return MNB
    pass

if __name__ == "__main__":
    
#    #读取数据
#    TrainData = pd.read_excel("traindata.xlsx",encoding = "GBK")
#    del TrainData["XM"]
#    
#    #读取数据
#    TestData = pd.read_excel("testdata.xlsx",encoding = "GBK")
#    del TestData["XM"]    
#    
#    
#    Test, Train = train_test_split(TrainData, train_size=0.3, random_state=1)
#    
################################################################################
##MNB
##    #训练模型    
##    MNB=MultinomialNB()
##    MNB.fit(Train.iloc[:,:-1], Train.iloc[:,-1])
#    MNB=train(Train)
#    
#    #预测结果        
#    Result = MNB.predict(Test.iloc[:,:-1])
#
#    #精确度,召回率,F1值
#    DataTestCtpye = Test.iloc[:,-1].values.reshape(-1)
#    PR= precision_recall_fscore_support(DataTestCtpye, Result, average='binary')
#    print("=================================================================")
#    print("精确度:",PR[0])
#    print("召回率:",PR[1])
#    print("F1值:",PR[2])
#    while(1):
#        print("time test")
#        #预测结果        
#        Result = MNB.predict(Test.iloc[:,:-1])
#        time.sleep(10)
#        pass
##    
##    #保存分类模型
##    joblib.dump(MNB, "../model/MultinomialNBModel.m")
####    FinalKeyData.to_csv('../data/traindata/traindata.csv',encoding = "GBK",index=False) 
##    

    
        
        
    
    
    
    
    
