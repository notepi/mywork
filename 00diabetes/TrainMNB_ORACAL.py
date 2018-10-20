# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:05:17 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import os 
from sklearn.externals import joblib  
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import time
import pymysql
import cx_Oracle

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
        #删除多余字符
        del TrainData["XM"]
        del TrainData["XB"]
        del TrainData["RYBH"]
        del TrainData["GMSFHM"]
        del TrainData["XLH"]
        
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
        sql_month = " SELECT * FROM jjj where JG is null"
        #数据库中读取数据
        TestData = pd.read_sql_query(sql_month,con=database)
        del TestData["result"]
        #数据类型转换成int
        TestData=TestData.astype(int)
        database.close()
        pass
            #删除多余字符
        del TrainData["XM"]
        del TrainData["XB"]
        del TrainData["RYBH"]
        del TrainData["GMSFHM"]
        pass
    return TestData
    pass
if __name__ == "__main__":
    count=0
    traintime=0
    while(1):
        print(count)
        
        if not (count%10):
            traintime=traintime+1
            print("========TrainModel========count:",traintime)
            #获取训练数据
            TrainData=getTrainData()
            #训练模型
            MNB=trainModel(TrainData)
            pass
        print("+++++++Predict+++++++++count:",count)
        #预测结果    
        #读取训练好的模型
        MNB=joblib.load("./model/MultinomialNBModel.m")
        #获取要预测的数据
        Test=getPredictData(a=1)
        xlh=Test["XLH"]
        del Test["XLH"]

        
        Result = MNB.predict(Test.iloc[:,:-1])
        
        #构造数据
        final=pd.DataFrame(xlh)
        final['JG']=Result
        
        #推送
        #链接数据库    
        database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
        c = database.cursor()
        for i in range(len(Result)):
            
            sql="UPDATE help_category SET result=%d WHERE XLH=%d" % (Result[i],xlh[i])
            c.execute(sql)
            c.execute('commit')
            pass
        database.close()
        
        time.sleep(1)
        count=count+1
        break
        pass
  

    
        
        
    
    
    
    
    
