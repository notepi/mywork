# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:05:17 2018

@author: Administrator
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import cx_Oracle

if __name__ == "__main__":
    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
    sql_month = " SELECT * FROM tab"
    data = pd.read_sql_query(sql_month,con=database)
    #删除多余字符
    del data["XM"]
    del data["XB"]
    del data["RYBH"]
    del data["GMSFHM"]
    del data["XLH"]
    database.close()
    
    #训练模型    
    MNB=MultinomialNB()
    MNB.fit(data.iloc[:,:-1], data.iloc[:,-1]) 
    
    
    #捞预测数据
    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
    sql_month = " SELECT * FROM tab where JG is null"
    test = pd.read_sql_query(sql_month,con=database)
    database.close()
    
    xlh=test["XLH"]
    
    del test["XM"]
    del test["XB"]
    del test["RYBH"]
    del test["GMSFHM"]
    del test["XLH"]

    

    
    #预测
    Result = MNB.predict(test.iloc[:,:-1])
#    #写数据
#    database = cx_Oracle.connect('system', 'oracle', '192.168.32.130:1521/XE')
#    
#    sql="UPDATE help_category SET JG=%d WHERE XLH=%d" % (Result[0],xlh[0])
#    database.execute(sql)
#    database.execute('commit')
#    database.close()
#    pass




