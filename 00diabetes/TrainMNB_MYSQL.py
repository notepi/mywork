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
from sklearn.externals import joblib  
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import time
import pymysql
import cx_Oracle
def oracel():
    #链接数据库    
    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
    #构造sql语句
    sql_month = " SELECT * FROM tab"
    #数据库中读取数据
    company_month = pd.read_sql_query(sql_month,con=database)
    #数据类型转换成int
    company_month=company_month.astype(int)
    database.close()
    return company_month
    pass
def trainModel(data):
    #MNB
    #训练模型    
    MNB=MultinomialNB()
    MNB.fit(data.iloc[:,:-1], data.iloc[:,-1]) 
    #保存分类模型
    joblib.dump(MNB, "./model/MultinomialNBModel.m")
    return MNB
    pass
def getTrainData(a=1):
    #从本地文件
    if a==1:
        #读取数据
        TrainData = pd.read_excel("traindata.xlsx",encoding = "GBK")
        del TrainData["XM"]
        pass
    #从mysql
    elif a==0:  
        db = pymysql.connect("localhost", "root", "123456", "mysql")
        sql = "select * from help_category"
        TrainData = pd.read_sql_query(sql,con=db)
        db.close()
        pass
    #从oracle
    elif a==2:
        #链接数据库    
        database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
        #构造sql语句
        sql_month = " SELECT * FROM help_category"
        #数据库中读取数据
        TrainData = pd.read_sql_query(sql_month,con=database)
        #数据类型转换成int
        TrainData=TrainData.astype(int)
        database.close()
        del TrainData["XM"]
        pass
    
    return TrainData
    pass
def getPredictData(a=1):
    if a==1:
        #读取数据
        TestData = pd.read_excel("testdata.xlsx",encoding = "GBK")
        pass
    elif a==0:
        db = pymysql.connect("localhost", "root", "123456", "mysql")
        sql = "select * from help_category"
        TestData = pd.read_sql_query(sql,con=db)
        db.close()
        pass
    elif a==2:
          #链接数据库    
        database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
        #构造sql语句
        sql_month = " SELECT * FROM where result is null"
        #数据库中读取数据
        TestData = pd.read_sql_query(sql_month,con=database)
        del TestData["result"]
        #数据类型转换成int
        TestData=TestData.astype(int)
        database.close()
        pass
    del TestData["XM"]   
    return TestData
    pass
if __name__ == "__main__":
    TrainData=getTrainData(a=0)
    
    db = pymysql.connect("localhost", "root", "123456", "mysql")
    sql = "select * from help_category"
    TrainData = pd.read_sql_query(sql,con=db)
    data=zip(TrainData.name, TrainData.name)
    
    TrainData = pd.read_excel("traindata.xlsx",encoding = "GBK")
    del TrainData["XM"]
    a=TrainData.astype(int)
    
#    Test, Train = train_test_split(TrainData, train_size=0.3, random_state=1)
#    
# 
# 
################################################################################
#    #MNB
#    #训练模型    
#    MNB=trainModel(TrainData)
#    
#    #预测结果      
#    bb=joblib.load("./model/MultinomialNBModel.m")
#    Result = bb.predict(Test.iloc[:,:-1])
#
#    #精确度,召回率,F1值
#    DataTestCtpye = Test.iloc[:,-1].values.reshape(-1)
#    PR= precision_recall_fscore_support(DataTestCtpye, Result, average='binary')
#    print("=================================================================")
#    print("精确度:",PR[0])
#    print("召回率:",PR[1])
#    print("F1值:",PR[2])
#    count=0
#    traintime=0
#    while(0):
#        print(count)
#        if not (count%10):
#            traintime=traintime+1
#            print("========TrainModel========count:",traintime)
#            TrainData=getTrainData()
#            MNB=trainModel(TrainData)
#            pass
#        print("+++++++Predict+++++++++count:",count)
#        #预测结果    
#        MNB=joblib.load("./model/MultinomialNBModel.m")
##        Result = MNB.predict(Test.iloc[:,:-1])
# 
#    
#        Result = MNB.predict(Test.iloc[:,:-1])
#        #精确度,召回率,F1值
#        DataTestCtpye = Test.iloc[:,-1].values.reshape(-1)
#        PR= precision_recall_fscore_support(DataTestCtpye, Result, average='binary')
#        print("=================================================================")
#        print("精确度:",PR[0])
#        print("召回率:",PR[1])
#        print("F1值:",PR[2])
#        
#        time.sleep(10)
#        count=count+1
##        break
#        pass
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
