import pymysql
import numpy as np
import pandas as pd
##写
#db = pymysql.connect("localhost", "root", "123456", "medicinediet")
#sql = "update re_test set test_result = ('%s') where test_ID = ('%s')" % ('qweqeqe',1)
#cursor = db.cursor()
#cursor.execute(sql)
#db.commit()
#读
db = pymysql.connect("localhost", "root", "123456", "mysql")
sql = "select * from help_keyword"
cursor = db.cursor()
cursor.execute(sql)
while(1):
    find = cursor.fetchone()
    print(find)
    break
    pass

while(1):
    b = cursor.fetchall()
    print(b)
    break
    pass
#将数据库中的数据进行格式转换
cc=[]
data=pd.DataFrame()
for i in range(len(b[0])):
    cc=[]
    for j in b:
        cc.append(j[i])
        pass
    data=pd.concat([data,pd.DataFrame(cc)],axis=1)
    pass

