#-*- coding:gbk -*-
'''
    author:huliangjun
    date:2017/11/13
    project name:劳动力资源增量预测
'''
import cx_Oracle
import pandas as pd
import sys
import time
import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARMA,ARIMA
from dateutil.relativedelta import relativedelta
import warnings
import copy

class laborResour(object):

    def __init__(self,numMonth=12):
        self.numMonth = numMonth
        self.db = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')

    def createDataset(self,dataset,look_back = 2):
        dataX,dataY = [],[]
        for i in range(len(dataset) - 1):
            a = tuple(dataset[i:(i + look_back)])
            dataX.append(a)
            dataY.append(dataset[i + 1])
        return dataX,dataY

    def allLaborIndex(self):
        # database = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')
        sql_month = "select DISTINCT T_MONTH from AB01_WITH_CD01_COUNT_RESULT order by T_MONTH"
        labor_month = pd.read_sql_query(sql_month,con=self.db)
        labor_month = labor_month.values.ravel().tolist()
        month_X, month_Y = self.createDataset(labor_month)
        sqlAllLabor = 'select count(*) as TOTAL01 from \
                     (select t.ccd001,t.aac001,substr(aae036,1,6) as intime,substr(ccd017,1,6) as outtime from CD01_RESULT t \
                     where substr(aae036,1,6)<{} \
                     and(ccd017 is null or substr(ccd017,1,6)>{}))'
        tmpAllLabor = []
        for (i, j) in month_X:
            tmp = sqlAllLabor.format(j, i)
            allLabor = pd.read_sql_query(tmp, con=self.db)
            tmpAllLabor.append(allLabor['TOTAL01'].values[0])
        laborIncre = pd.DataFrame(tmpAllLabor,columns=['TOTAL01'])
        t_month = pd.DataFrame(month_Y,columns=['T_MONTH'])
        allLaborIncre = pd.concat([t_month,laborIncre],axis=1)
        allLaborIncre = allLaborIncre.set_index('T_MONTH')
        allLaborIncre.index.name = None
        return allLaborIncre

    def typeWorkTotal(self):
        sql_month = "select DISTINCT T_MONTH from AB01_WITH_CD01_COUNT_RESULT order by T_MONTH"
        labor_month = pd.read_sql_query(sql_month,con=self.db)
        labor_month = labor_month.values.ravel().tolist()
        month_X, month_Y = self.createDataset(labor_month)
        sqlTypeWork = "select {} as T_MONTH,ACA111,count(*) as MYSUM from \
                       (select t.ccd001,t.aac001,substr(aae036,1,6) as intime,substr(ccd017,1,6) as outtime,t.aca111 from CD01_RESULT t \
                       where substr(aae036,1,6)<{} and (ccd017 is null or substr(ccd017,1,6)>{})) \
                       group by aca111 \
                       order by mysum desc"
        tmpTypeWork = pd.DataFrame(columns=['T_MONTH','ACA111','MYSUM'])
        for (i, j) in month_X:
            tmp = sqlTypeWork.format(i, j, i)
            typeWork = pd.read_sql_query(tmp, con=self.db)
            tmpTypeWork = tmpTypeWork.append(typeWork,ignore_index=True)
        self.inTypeWorkDatabase(tmpTypeWork)
        return tmpTypeWork

    def inTypeWorkDatabase(self,data):
        # db = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')
        clear = self.db.cursor()
        clear.execute('truncate table LDLZYZL_GZ')
        clear.close()
        sql_export = "insert into LDLZYZL_GZ values(:T_MONTH,:ACA111,:MYSUM)"
        c = self.db.cursor()
        monthParm = [str(int(x)) for x in data['T_MONTH']]
        typeWorkParm = data['ACA111'].tolist()
        sumParm = data['MYSUM'].tolist()
        parm = zip(monthParm,typeWorkParm,sumParm)
        c.executemany(sql_export,parm)
        c.execute('commit')
        c.close()

    def allLaborIncrement(self,data):
        monthList = data.index.tolist()
        monthX,monthY = self.createDataset(monthList)
        laborResource = []
        for (i, j) in monthX:
            tmpLabor = data.loc[j].values[0] - data.loc[i].values[0]
            laborResource.append(tmpLabor)
        laborResourceIncre = pd.DataFrame(laborResource,columns=['INCREMENT'])
        month = pd.DataFrame(monthY,columns=['T_MONTH'])
        allLaborResouIncre = pd.concat([month,laborResourceIncre],axis=1)
        self.inAllLaborDatabase(allLaborResouIncre)
        allLaborResouIncre = allLaborResouIncre.set_index('T_MONTH')
        allLaborResouIncre.index.name = None
        return allLaborResouIncre

    def inAllLaborDatabase(self,data):
        # db = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')
        clear = self.db.cursor()
        clear.execute('truncate table LDLZYZL_All')
        clear.close()
        sql_export = "insert into LDLZYZL_All values(:T_MONTH,:TOTAL01)"
        monthParm = [str(int(x)) for x in data['T_MONTH']]
        totalParm = data['INCREMENT'].astype(np.float).tolist()
        parm = zip(monthParm,totalParm)
        c = self.db.cursor()
        c.executemany(sql_export,parm)
        c.execute('commit')
        c.close()

    def getData(self,data):
        data = self.tranDataset(data)
        data.index = pd.to_datetime(data.index)
        train_data = data['INCREMENT'].astype(np.float)
        arima = ARIMA(endog=train_data[:27], order=(3, 1, 2)).fit(disp=-1)
        predict = arima.forecast(2)[0]
        train_data['2017-04-01'] = float(int(predict[0]))
        train_data['2017-05-01'] = float(int(predict[1]))
        return train_data

    def arimaModel(self,data):
        # roll_data = data.rolling(window=12).mean()
        # data_diff = data - roll_data
        # data_diff.dropna(inplace=True)
        data = self.getData(data)
        arima = ARIMA(endog=data, order=(4,0,2)).fit(disp=-1)
        prediction = arima.forecast(self.numMonth)[0].tolist()
        result = []
        for i in range(len(prediction)):
            predResult = prediction[i]
            result.append(predResult)
        predict_tmp_date = self.toDataFrame(data)
        time = pd.period_range(predict_tmp_date, periods=len(prediction), freq='M')
        timeStrfTime = self.toStrfTime(time)
        incrResult = pd.DataFrame({'T_MONTH': timeStrfTime, 'INCREMENT': result})
        return incrResult

    def toStrfTime(self,date):
        dateTrans = []
        for i in range(len(date)):
            dateTmp = str(date[i].strftime('%Y%m'))
            dateTrans.append(dateTmp)
        return dateTrans

    def toDataFrame(self,trainData):
        time_tmp = trainData.index[-1]
        year = time_tmp.year
        month = time_tmp.month
        date_data = datetime.datetime(year, month, 1)
        predict_date = date_data + relativedelta(months=(1))
        if predict_date.month < 10:
            predict_tmp_date = str(predict_date.year) + '0' + str(predict_date.month)
        else:
            predict_tmp_date = str(predict_date.year) + str(predict_date.month)
        return predict_tmp_date

    def tranDataset(self,dataset):
        month = []
        for i in dataset.index:
            i = i + '01'
            month.append(i)
        allIncr = pd.DataFrame(dataset['INCREMENT'].values, index=month)
        allIncr.rename(columns={0: 'INCREMENT'}, inplace=True)
        return allIncr

    def getAllLaborIncr(self):
        allLabor = self.allLaborIndex()
        allLaborIncre = self.allLaborIncrement(allLabor)
        allIncrResult = self.arimaModel(allLaborIncre)
        return allIncrResult

if __name__ == '__main__':
    m1 = laborResour()
    result = m1.getAllLaborIncr()
    m1.typeWorkTotal()
