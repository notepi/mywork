# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:05:17 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import os 
from sklearn.externals import joblib  
from sklearn.naive_bayes import MultinomialNB
import time
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
def getTrainData():

    #链接数据库    
    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
    #构造sql语句
    sql_month = " SELECT * FROM help_category"
    #数据库中读取数据
    TrainData = pd.read_sql_query(sql_month,con=database)
    #数据类型转换成int
    TrainData=TrainData
    database.close()
    #删除多余字符    
    del TrainData["XM"]
    del TrainData["XB"]
    del TrainData["RYBH"]
    del TrainData["GMSFHM"]
    del TrainData["XLH"]
    return TrainData
    pass
def getPredictData():
    #链接数据库    
    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
    #构造sql语句
    sql_month = " SELECT * FROM jjj where JG is null"
    #数据库中读取数据
    TestData = pd.read_sql_query(sql_month,con=database)
    #数据类型转换成int
    TestData=TestData
    database.close()
    
    #删除多余字符    
    del TestData["XM"]
    del TestData["XB"]
    del TestData["RYBH"]
    del TestData["GMSFHM"]
        
    return TestData
    pass
if __name__ == "__main__":
    count=0
    traintime=0
    while(1):
        print(count)
        
        if not (count%100):
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
        Test=getPredictData()
        if len(Test) < 1:
            time.sleep(1)
            count=count+1
            continue
            pass
        xlh=Test["XLH"]
        del Test["XLH"]

        
        Result = MNB.predict(Test.iloc[:,:-1])
        
        
        #推送
        #链接数据库    
        database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
        c = database.cursor()
        for i in range(len(Result)):     
            sql="UPDATE help_category SET JG=%d WHERE XLH=%d" % (Result[i],xlh[i])
            c.execute(sql)
            c.execute('commit')
            pass
        database.close()
        
        time.sleep(1)
        count=count+1
#        break
        pass
  

    
        
        
    
    
    
    
    
